# -*- coding: utf-8 -*-
"""
Phân loại rác: huấn luyện 6 mô hình ML cổ điển, chạy 5 lần, xuất Excel.

Chạy sau khi đã chạy tien_xu_ly_anh.py:
    python huan_luyen_mo_hinh.py

Kết quả (Excel, hình, mô hình) lưu vào thư mục ket_qua/.
"""


# ============================================================================
# Phân loại rác: Huấn luyện mô hình ML cổ điển, 5 lần chạy
#
# Bài toán có 5 lớp của nhóm: giay, nhua, huu_co, kim_loai, khac (pin, thủy tinh, gốm, ...).
# Đầu vào là đặc trưng 512 chiều từ ResNet18 (features_512_raw.npz) và metadata.csv, do script tien_xu_ly_anh.py tạo ra.
#
# | # | Thuật toán | Họ | Biểu đồ loss |
# | --- | --- | --- | --- |
# | 1 | Softmax Regression (SGD) | Tuyến tính | Log loss train/val theo epoch |
# | 2a | SVM tuyến tính (SGD + hinge loss) | Dựa trên lề | Hinge loss train/val theo epoch |
# | 2b | SVM kernel RBF | Dựa trên lề | Validation curve, learning curve |
# | 3 | Decision Tree | Cây quyết định | Validation curve: tỉ lệ sai (0-1 loss) train/val theo độ sâu cây |
# | 4 | AdaBoost | Ensemble cây (boosting) | Tỉ lệ sai (0-1 loss) train/val theo vòng boosting |
# | 5 | XGBoost (Gradient Boosting) | Ensemble cây (boosting) | Log loss train/val theo vòng boosting |
#
# XGBoost là một cách cài đặt tối ưu của thuật toán Gradient Boosting (thêm regularization, chạy song song, xử lý nhanh hơn), nên trong báo cáo có thể gọi là Gradient Boosting (XGBoost).
# AdaBoost và Gradient Boosting đều là boosting nhưng khác cách sửa lỗi: AdaBoost tăng trọng số các mẫu bị đoán sai để cây sau tập trung vào chúng; Gradient Boosting cho cây sau học phần dư (gradient của hàm loss) của các cây trước.
#
# Quy trình 5 lần chạy
#
# Lấy 2.000 ảnh (phân tầng theo lớp và nguồn). Mỗi lần chạy dùng một seed khác nhau và làm lại toàn bộ:
#
# 1. Chia lại train / val / test = 70 / 15 / 15 (1.400 / 300 / 300 ảnh).
# 2. Fit lại StandardScaler và PCA (512 -> 128 chiều) chỉ trên train của lần đó.
# 3. Tìm siêu tham số trên val, huấn luyện có early stopping.
# 4. Đánh giá trên test của lần đó.
#
# Kết quả báo cáo là trung bình ± độ lệch chuẩn của 5 lần. Toàn bộ chi tiết từng lần được ghi vào file Excel ket_qua_5_lan_chay.xlsx để làm dẫn chứng.
#
# Cách dùng: chạy python tien_xu_ly_anh.py trước, rồi chạy python huan_luyen_mo_hinh.py. Không cần GPU; mỗi lần chạy mất khoảng 1 đến vài phút tùy máy.
# ============================================================================

# ============================================================================
# 1. Cấu hình
# ============================================================================
import os as _os
try:
    BASE_DIR = _os.path.dirname(_os.path.abspath(__file__)).replace("\\", "/")   # thư mục chứa file script này
except NameError:
    BASE_DIR = _os.getcwd().replace("\\", "/")
FEAT_DIR = f"{BASE_DIR}/features"            # output của script tiền xử lý
OUT_DIR  = f"{BASE_DIR}/ket_qua"             # nơi lưu Excel, hình, mô hình

TOTAL_SAMPLES = 2000          # tổng số ảnh dùng (train + val + test); None = dùng hết
N_RUNS = 5                    # số lần chạy
SEEDS = [0, 1, 2, 3, 4]       # seed của từng lần chạy
SAMPLE_SEED = 42              # seed để chọn 2000 ảnh (cố định, 5 lần dùng chung 2000 ảnh)
VAL_SIZE, TEST_SIZE = 0.15, 0.15
N_COMPONENTS = 128            # số chiều sau PCA

MAX_EPOCHS = 300              # số epoch tối đa cho mô hình SGD
PATIENCE = 20                 # early stopping: dừng nếu loss val không giảm sau chừng này epoch
ALPHAS = [1e-4, 1e-3, 1e-2, 1e-1]
SGD_ETA0 = {"log_loss": 1e-3, "hinge": 1e-4}
SVC_C = [0.1, 1, 10, 100]
SVC_GAMMA = ["scale", 1e-3, 1e-2]
XGB_DEPTHS = [3, 5]
XGB_LR = 0.05
XGB_MAX_ROUNDS = 2000
XGB_EARLY_STOP = 50
DT_DEPTHS = [3, 5, 8, 12, 20, None]   # độ sâu cây quyết định; None = không giới hạn
DT_MIN_LEAF = [1, 5]                  # số mẫu tối thiểu ở mỗi lá
ADA_DEPTHS = [1, 2]                   # độ sâu cây yếu trong AdaBoost (1 = decision stump)
ADA_LR = 0.5
ADA_MAX_ROUNDS = 200

assert len(SEEDS) == N_RUNS


# ============================================================================
# 2. Đọc dữ liệu và chọn 2.000 ảnh
# ============================================================================
import os, sys, copy, time, warnings

import matplotlib
matplotlib.use("Agg")   # lưu hình ra file, không mở cửa sổ
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import (log_loss, hinge_loss, accuracy_score, f1_score, precision_score,
                             recall_score, precision_recall_fscore_support, confusion_matrix)
import xgboost as xgb
import joblib

os.makedirs(OUT_DIR, exist_ok=True)
raw = np.load(f"{FEAT_DIR}/features_512_raw.npz", allow_pickle=True)
meta_all = pd.read_csv(f"{FEAT_DIR}/metadata.csv")
F_all = raw["feats"]
assert len(meta_all) == len(F_all), "metadata.csv không khớp features_512_raw.npz, chạy lại script tiền xử lý"

CLASSES = ["giay", "nhua", "huu_co", "kim_loai", "khac"]
LABELS = np.arange(len(CLASSES))
meta_all["y"] = meta_all.label.map({c: i for i, c in enumerate(CLASSES)})

def strat_split(index, test_size, strat, seed):
    """Chia phân tầng theo (lớp, nguồn); nhóm nào quá ít ảnh thì lùi về phân tầng theo lớp."""
    try:
        return train_test_split(index, test_size=test_size, stratify=strat, random_state=seed)
    except ValueError:
        warnings.warn("Có nhóm (lớp, nguồn) quá ít ảnh, chỉ phân tầng theo lớp.")
        return train_test_split(index, test_size=test_size, stratify=meta_all.loc[index, "label"], random_state=seed)

if TOTAL_SAMPLES is not None and len(meta_all) > TOTAL_SAMPLES:
    strat = meta_all.label + "|" + meta_all.source
    keep, _ = strat_split(meta_all.index, len(meta_all) - TOTAL_SAMPLES, strat, SAMPLE_SEED)
    meta = meta_all.loc[sorted(keep)].reset_index(drop=True)
    F = F_all[sorted(keep)]
else:
    if TOTAL_SAMPLES is not None and len(meta_all) < TOTAL_SAMPLES:
        warnings.warn(f"Chỉ có {len(meta_all)} ảnh, ít hơn {TOTAL_SAMPLES}. Dùng toàn bộ.")
    meta, F = meta_all.reset_index(drop=True), F_all
meta = meta.drop(columns=["split"], errors="ignore")
print(f"Dùng {len(meta)} ảnh, đặc trưng {F.shape[1]} chiều.")
tab = pd.crosstab(meta.label, meta.source, margins=True, margins_name="Tổng").reindex(CLASSES + ["Tổng"])
print((tab).to_string())


# ============================================================================
# 3. Hàm dùng chung
# ============================================================================
# Hình vẽ trắng đen cho báo cáo: train = nét liền, val = nét đứt
TRAIN_COLOR, VAL_COLOR = "black", "black"
TRAIN_LS, VAL_LS = "-", "--"
XGB_DEVICE = "cpu"
MODEL_NAMES = ["Softmax Regression", "SVM tuyến tính", "SVM RBF", "Decision Tree", "AdaBoost", "XGBoost"]

def style(ax):
    ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=0.25)

def prepare_run(seed):
    """Chia lại dữ liệu và fit lại scaler + PCA trên train của lần chạy này."""
    strat = meta.label + "|" + meta.source
    i_tr, i_tmp = strat_split(meta.index, VAL_SIZE + TEST_SIZE, strat, seed)
    i_va, i_te = strat_split(i_tmp, TEST_SIZE / (VAL_SIZE + TEST_SIZE), strat[i_tmp], seed)
    idx = {"train": np.sort(i_tr), "val": np.sort(i_va), "test": np.sort(i_te)}
    s1 = StandardScaler().fit(F[idx["train"]])
    n_comp = min(N_COMPONENTS, len(idx["train"]), F.shape[1])
    pca = PCA(n_components=n_comp, random_state=seed).fit(s1.transform(F[idx["train"]]))
    Z = {s: pca.transform(s1.transform(F[idx[s]])) for s in idx}
    s2 = StandardScaler().fit(Z["train"])
    X = {s: s2.transform(Z[s]) for s in idx}
    y = {s: meta.y.to_numpy()[idx[s]] for s in idx}
    return X, y, idx, {"scaler_512": s1, "pca": pca, "scaler_pca": s2}, pca.explained_variance_ratio_.sum()

def mloss(model, Xs, ys, kind):
    if kind == "log":
        return log_loss(ys, model.predict_proba(Xs), labels=LABELS)
    return hinge_loss(ys, model.decision_function(Xs), labels=LABELS)

def train_sgd(X, y, loss, alpha, seed):
    kind = "log" if loss == "log_loss" else "hinge"
    rng = np.random.default_rng(seed)
    clf = SGDClassifier(loss=loss, alpha=alpha, learning_rate="constant", eta0=SGD_ETA0[loss], random_state=seed)
    h = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best, best_loss, best_ep, wait = None, np.inf, 0, 0
    for ep in range(1, MAX_EPOCHS + 1):
        p = rng.permutation(len(y["train"]))
        clf.partial_fit(X["train"][p], y["train"][p], classes=LABELS)
        h["train_loss"].append(mloss(clf, X["train"], y["train"], kind))
        h["val_loss"].append(mloss(clf, X["val"], y["val"], kind))
        h["train_acc"].append(clf.score(X["train"], y["train"]))
        h["val_acc"].append(clf.score(X["val"], y["val"]))
        if h["val_loss"][-1] < best_loss - 1e-4:
            best, best_loss, best_ep, wait = copy.deepcopy(clf), h["val_loss"][-1], ep, 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    return best, h, best_ep

def tune_sgd(X, y, loss, seed, run, model_name, tuning_rows):
    runs = {}
    for a in ALPHAS:
        m, h, be = train_sgd(X, y, loss, a, seed)
        runs[a] = (m, h, be)
        tuning_rows.append({"Lần chạy": run, "Mô hình": model_name, "Siêu tham số": f"alpha={a}",
                            "Epoch/vòng dừng": be, "Loss train": h["train_loss"][be-1], "Loss val": h["val_loss"][be-1],
                            "Acc train": h["train_acc"][be-1], "Acc val": h["val_acc"][be-1]})
    a_best = min(runs, key=lambda a: runs[a][1]["val_loss"][runs[a][2] - 1])
    m, h, be = runs[a_best]
    return m, h, be, f"alpha={a_best}"

def tune_svc(X, y, seed, run, tuning_rows):
    best, best_key = None, None
    for g in SVC_GAMMA:
        for c in SVC_C:
            m = SVC(kernel="rbf", C=c, gamma=g, random_state=seed).fit(X["train"], y["train"])
            r = {"Lần chạy": run, "Mô hình": "SVM RBF", "Siêu tham số": f"C={c}, gamma={g}", "Epoch/vòng dừng": None,
                 "Loss train": hinge_loss(y["train"], m.decision_function(X["train"]), labels=LABELS),
                 "Loss val": hinge_loss(y["val"], m.decision_function(X["val"]), labels=LABELS),
                 "Acc train": m.score(X["train"], y["train"]), "Acc val": m.score(X["val"], y["val"]),
                 "C": c, "gamma": str(g)}
            tuning_rows.append(r)
            key = (r["Acc val"], -r["Loss val"])
            if best_key is None or key > best_key:
                best, best_key, best_r = m, key, r
    return best, best_r

def tune_xgb(X, y, seed, run, tuning_rows):
    global XGB_DEVICE
    try:
        import torch; dev = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        dev = "cpu"
    XGB_DEVICE = dev
    best = None
    for depth in XGB_DEPTHS:
        m = xgb.XGBClassifier(objective="multi:softprob", n_estimators=XGB_MAX_ROUNDS, learning_rate=XGB_LR,
                              max_depth=depth, subsample=0.8, colsample_bytree=0.5, min_child_weight=2,
                              reg_lambda=1.0, eval_metric="mlogloss", early_stopping_rounds=XGB_EARLY_STOP,
                              tree_method="hist", device=dev, random_state=seed)
        m.fit(X["train"], y["train"], eval_set=[(X["train"], y["train"]), (X["val"], y["val"])], verbose=False)
        ev = m.evals_result(); bi = m.best_iteration
        r = {"Lần chạy": run, "Mô hình": "XGBoost", "Siêu tham số": f"max_depth={depth}", "Epoch/vòng dừng": bi + 1,
             "Loss train": ev["validation_0"]["mlogloss"][bi], "Loss val": ev["validation_1"]["mlogloss"][bi],
             "Acc train": m.score(X["train"], y["train"]), "Acc val": m.score(X["val"], y["val"])}
        tuning_rows.append(r)
        if best is None or r["Loss val"] < best[1]["Loss val"]:
            best = (m, r)
    m, r = best
    ev = m.evals_result()
    h = {"train_loss": ev["validation_0"]["mlogloss"], "val_loss": ev["validation_1"]["mlogloss"]}
    return m, h, r["Epoch/vòng dừng"], r["Siêu tham số"]

def tune_tree(X, y, seed, run, tuning_rows):
    best, best_key = None, None
    for leaf in DT_MIN_LEAF:
        for depth in DT_DEPTHS:
            m = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, random_state=seed).fit(X["train"], y["train"])
            r = {"Lần chạy": run, "Mô hình": "Decision Tree", "Siêu tham số": f"max_depth={depth}, min_samples_leaf={leaf}",
                 "Epoch/vòng dừng": None,
                 "Loss train": 1 - m.score(X["train"], y["train"]), "Loss val": 1 - m.score(X["val"], y["val"]),
                 "Acc train": m.score(X["train"], y["train"]), "Acc val": m.score(X["val"], y["val"]),
                 "depth": str(depth), "leaf": leaf, "depth_thuc_te": m.get_depth()}
            tuning_rows.append(r)
            key = (r["Acc val"], -m.get_depth())          # hòa thì chọn cây nông hơn
            if best_key is None or key > best_key:
                best, best_key, best_r = m, key, r
    return best, best_r

def tune_ada(X, y, seed, run, tuning_rows):
    best = None
    for depth in ADA_DEPTHS:
        m = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=depth, random_state=seed),
                               n_estimators=ADA_MAX_ROUNDS, learning_rate=ADA_LR, random_state=seed).fit(X["train"], y["train"])
        # AdaBoost (SAMME) tối ưu sai số phân loại; xác suất của nó gần như đều nên log loss không phản ánh việc học.
        # Vì vậy theo dõi 0-1 loss (tỉ lệ sai) sau từng vòng.
        acct = [accuracy_score(y["train"], p) for p in m.staged_predict(X["train"])]
        accs = [accuracy_score(y["val"], p) for p in m.staged_predict(X["val"])]
        h = {"train_loss": [1 - a for a in acct], "val_loss": [1 - a for a in accs]}
        bi = int(np.argmin(h["val_loss"])) + 1          # vòng có tỉ lệ sai trên val thấp nhất (hòa thì lấy vòng sớm nhất)
        r = {"Lần chạy": run, "Mô hình": "AdaBoost", "Siêu tham số": f"max_depth={depth}, learning_rate={ADA_LR}",
             "Epoch/vòng dừng": bi, "Loss train": h["train_loss"][bi-1], "Loss val": h["val_loss"][bi-1],
             "Acc train": acct[bi-1], "Acc val": accs[bi-1]}
        tuning_rows.append(r)
        if best is None or r["Loss val"] < best[2]["Loss val"]:
            best = (depth, h, r)
    depth, h, r = best
    # Huấn luyện lại đúng số vòng tốt nhất (AdaBoost tất định nên giống hệt mô hình tại vòng đó)
    m = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=depth, random_state=seed),
                           n_estimators=r["Epoch/vòng dừng"], learning_rate=ADA_LR, random_state=seed).fit(X["train"], y["train"])
    return m, h, r["Epoch/vòng dừng"], r["Siêu tham số"]


# ============================================================================
# 4. Chạy 5 lần
#
# Mỗi lần in ra accuracy test của từng mô hình. Mọi con số (kể cả loss từng epoch và mọi tổ hợp siêu tham số đã thử) được lưu lại để ghi vào Excel.
# ============================================================================
detail_rows, class_rows, cm_rows, loss_rows, tuning_rows, data_rows = [], [], [], [], [], []
hists, saved_models = {}, {}

for run, seed in enumerate(SEEDS, start=1):
    t_run = time.time()
    X, y, idx, prep, var_kept = prepare_run(seed)
    for s in idx:
        sub = meta.loc[idx[s]]
        for (lbl, src), n in sub.groupby(["label", "source"]).size().items():
            data_rows.append({"Lần chạy": run, "Seed": seed, "Tập": s, "Lớp": lbl, "Nguồn": src, "Số ảnh": n})

    models, info = {}, {}
    t = time.time(); m, h, be, hp = tune_sgd(X, y, "log_loss", seed, run, "Softmax Regression", tuning_rows)
    models["Softmax Regression"] = m; info["Softmax Regression"] = (hp, be, h["train_loss"][be-1], h["val_loss"][be-1], "log loss", time.time() - t)
    hists[(run, "Softmax Regression")] = (h, be)

    t = time.time(); m, h, be, hp = tune_sgd(X, y, "hinge", seed, run, "SVM tuyến tính", tuning_rows)
    models["SVM tuyến tính"] = m; info["SVM tuyến tính"] = (hp, be, h["train_loss"][be-1], h["val_loss"][be-1], "hinge loss", time.time() - t)
    hists[(run, "SVM tuyến tính")] = (h, be)

    t = time.time(); m, r = tune_svc(X, y, seed, run, tuning_rows)
    models["SVM RBF"] = m; info["SVM RBF"] = (r["Siêu tham số"], None, r["Loss train"], r["Loss val"], "hinge loss", time.time() - t)

    t = time.time(); m, r = tune_tree(X, y, seed, run, tuning_rows)
    models["Decision Tree"] = m
    info["Decision Tree"] = (r["Siêu tham số"] + f" (sâu thực tế {r['depth_thuc_te']})", None, r["Loss train"], r["Loss val"], "0-1 loss (tỉ lệ sai)", time.time() - t)

    t = time.time(); m, h, be, hp = tune_ada(X, y, seed, run, tuning_rows)
    models["AdaBoost"] = m; info["AdaBoost"] = (hp, be, h["train_loss"][be-1], h["val_loss"][be-1], "0-1 loss (tỉ lệ sai)", time.time() - t)
    hists[(run, "AdaBoost")] = (h, be)

    t = time.time(); m, h, be, hp = tune_xgb(X, y, seed, run, tuning_rows)
    models["XGBoost"] = m; info["XGBoost"] = (hp, be, h["train_loss"][be-1], h["val_loss"][be-1], "log loss", time.time() - t)
    hists[(run, "XGBoost")] = (h, be)

    src_test = meta.source.to_numpy()[idx["test"]]
    for name in MODEL_NAMES:
        m = models[name]
        p = {s: m.predict(X[s]) for s in X}
        hp, be, trl, val, lname, secs = info[name]
        detail_rows.append({
            "Mô hình": name, "Lần chạy": run, "Seed": seed, "Siêu tham số chọn": hp,
            "Số ảnh train": len(y["train"]), "Số ảnh val": len(y["val"]), "Số ảnh test": len(y["test"]),
            "PCA giữ phương sai": var_kept, "Epoch/vòng dừng": be, "Loại loss": lname,
            "Loss train": trl, "Loss val": val,
            "Acc train": accuracy_score(y["train"], p["train"]),
            "Acc val": accuracy_score(y["val"], p["val"]),
            "Acc test": accuracy_score(y["test"], p["test"]),
            "Precision macro test": precision_score(y["test"], p["test"], average="macro", zero_division=0),
            "Recall macro test": recall_score(y["test"], p["test"], average="macro", zero_division=0),
            "F1 macro test": f1_score(y["test"], p["test"], average="macro", zero_division=0),
            "Acc test tự chụp": accuracy_score(y["test"][src_test == "tu_chup"], p["test"][src_test == "tu_chup"]) if (src_test == "tu_chup").any() else None,
            "Acc test RealWaste": accuracy_score(y["test"][src_test == "realwaste"], p["test"][src_test == "realwaste"]) if (src_test == "realwaste").any() else None,
            "Thời gian tìm + huấn luyện (giây)": secs})
        pr, rc, f1, sp = precision_recall_fscore_support(y["test"], p["test"], labels=LABELS, zero_division=0)
        for k, c in enumerate(CLASSES):
            class_rows.append({"Mô hình": name, "Lần chạy": run, "Lớp": c, "Precision": pr[k], "Recall": rc[k], "F1": f1[k], "Số ảnh test": sp[k]})
        cm = confusion_matrix(y["test"], p["test"], labels=LABELS)
        for k, c in enumerate(CLASSES):
            cm_rows.append({"Mô hình": name, "Lần chạy": run, "Thực tế": c, **{f"Dự đoán {cc}": int(cm[k, j]) for j, cc in enumerate(CLASSES)}})
    for (rr, name), (h, be) in hists.items():
        if rr != run: continue
        for e in range(len(h["train_loss"])):
            loss_rows.append({"Mô hình": name, "Lần chạy": run, "Epoch/vòng": e + 1,
                              "Loss train": h["train_loss"][e], "Loss val": h["val_loss"][e],
                              "Được chọn (early stopping)": "x" if e + 1 == be else ""})
    saved_models[run] = {"prep": prep, "models": models}
    accs = " | ".join(f"{n}: {r['Acc test']:.3f}" for n, r in zip(MODEL_NAMES, detail_rows[-len(MODEL_NAMES):]))
    print(f"Lần {run} (seed {seed}, {time.time() - t_run:.0f}s)  acc test  {accs}")

detail = pd.DataFrame(detail_rows)
detail["_o"] = detail["Mô hình"].map({n: i for i, n in enumerate(MODEL_NAMES)})
detail = detail.sort_values(["_o", "Lần chạy"]).drop(columns="_o").reset_index(drop=True)


# ============================================================================
# 5. Kết quả trung bình 5 lần
# ============================================================================
METRICS = ["Acc train", "Acc val", "Acc test", "Precision macro test", "Recall macro test", "F1 macro test", "Acc test tự chụp", "Acc test RealWaste"]
g = detail.groupby("Mô hình", sort=False)
summary = pd.DataFrame({m: g[m].mean().map("{:.4f}".format) + " ± " + g[m].std().map("{:.4f}".format) for m in METRICS})
print((summary).to_string())


# ============================================================================
# 6. Biểu đồ loss của 5 lần chạy
#
# Mỗi đường mảnh là một lần chạy; chấm tròn đánh dấu điểm early stopping chọn. Đường đậm là trung bình 5 lần, tính trên đoạn epoch mà cả 5 lần đều có.
# ============================================================================
def plot_runs(name, xlabel, ylabel, fname):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    curves = [hists[(r, name)] for r in range(1, N_RUNS + 1)]
    for h, be in curves:
        e = np.arange(1, len(h["train_loss"]) + 1)
        ax.plot(e, h["train_loss"], color=TRAIN_COLOR, ls=TRAIN_LS, lw=0.7, alpha=0.3)
        ax.plot(e, h["val_loss"], color=VAL_COLOR, ls=VAL_LS, lw=0.7, alpha=0.3)
        ax.plot(be, h["val_loss"][be - 1], "o", color="white", ms=6, mec="black", mew=1.2)
    L = min(len(h["train_loss"]) for h, _ in curves)
    e = np.arange(1, L + 1)
    ax.plot(e, np.mean([h["train_loss"][:L] for h, _ in curves], axis=0), color=TRAIN_COLOR, ls=TRAIN_LS, lw=2.2, label="train (TB 5 lần)")
    ax.plot(e, np.mean([h["val_loss"][:L] for h, _ in curves], axis=0), color=VAL_COLOR, ls=VAL_LS, lw=2.2, label="val (TB 5 lần)")
    ax.plot([], [], "o", color="white", mec="black", mew=1.2, label="điểm dừng từng lần")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(f"{name}: loss qua 5 lần chạy")
    ax.legend(frameon=False); style(ax)
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/{fname}", dpi=130); plt.close()

plot_runs("Softmax Regression", "Epoch", "Log loss", "loss_softmax_5_lan.png")
plot_runs("SVM tuyến tính", "Epoch", "Hinge loss", "loss_svm_tuyen_tinh_5_lan.png")
plot_runs("AdaBoost", "Vòng boosting (số cây)", "Tỉ lệ sai (0-1 loss)", "loss_adaboost_5_lan.png")
plot_runs("XGBoost", "Vòng boosting (số cây)", "Log loss", "loss_xgboost_5_lan.png")

# SVM RBF: validation curve theo C (tại gamma được chọn của từng lần)
tune = pd.DataFrame(tuning_rows)
rbf = tune[tune["Mô hình"] == "SVM RBF"]
fig, ax = plt.subplots(figsize=(8, 4.5))
for run in range(1, N_RUNS + 1):
    chosen = detail[(detail["Mô hình"] == "SVM RBF") & (detail["Lần chạy"] == run)]["Siêu tham số chọn"].iloc[0]
    g_best = chosen.split("gamma=")[1]
    sub = rbf[(rbf["Lần chạy"] == run) & (rbf.gamma == g_best)].sort_values("C")
    ax.plot(sub.C, sub["Loss train"], "o-", color=TRAIN_COLOR, lw=1, alpha=0.6, label="train" if run == 1 else None)
    ax.plot(sub.C, sub["Loss val"], "s--", color=VAL_COLOR, mfc="white", lw=1, alpha=0.6, label="val" if run == 1 else None)
ax.set_xscale("log"); ax.set_xlabel("C (lớn = ít regularization)"); ax.set_ylabel("Hinge loss")
ax.set_title("SVM RBF: validation curve qua 5 lần chạy"); ax.legend(frameon=False); style(ax)
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/loss_svm_rbf_5_lan.png", dpi=130); plt.close()

# Decision Tree: validation curve theo độ sâu (tại min_samples_leaf được chọn của từng lần)
dt = tune[tune["Mô hình"] == "Decision Tree"]
order = [str(d) for d in DT_DEPTHS]
xlab = ["không giới hạn" if d == "None" else d for d in order]
fig, ax = plt.subplots(figsize=(8, 4.5))
for run in range(1, N_RUNS + 1):
    chosen = detail[(detail["Mô hình"] == "Decision Tree") & (detail["Lần chạy"] == run)]["Siêu tham số chọn"].iloc[0]
    leaf = int(chosen.split("min_samples_leaf=")[1].split(" ")[0])
    sub = dt[(dt["Lần chạy"] == run) & (dt.leaf == leaf)].set_index("depth").reindex(order)
    xs = np.arange(len(order))
    ax.plot(xs, sub["Loss train"], "o-", color=TRAIN_COLOR, lw=1, alpha=0.6, label="train" if run == 1 else None)
    ax.plot(xs, sub["Loss val"], "s--", color=VAL_COLOR, mfc="white", lw=1, alpha=0.6, label="val" if run == 1 else None)
    d_best = chosen.split("max_depth=")[1].split(",")[0]
    k = order.index(d_best)
    ax.plot(k, sub["Loss val"].iloc[k], "D", color="black", ms=7)
ax.plot([], [], "D", color="black", label="độ sâu được chọn")
ax.set_xticks(np.arange(len(order)), xlab); ax.set_xlabel("max_depth (sâu = mô hình phức tạp hơn)")
ax.set_ylabel("Tỉ lệ sai (0-1 loss)"); ax.set_title("Decision Tree: validation curve qua 5 lần chạy")
ax.legend(frameon=False); style(ax)
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/loss_decision_tree_5_lan.png", dpi=130); plt.close()


# ============================================================================
# 7. Ma trận nhầm lẫn (cộng dồn 5 lần) và so sánh mô hình
# ============================================================================
cmdf = pd.DataFrame(cm_rows)
fig, axes = plt.subplots(2, 3, figsize=(17, 10))
for ax, name in zip(axes.ravel(), MODEL_NAMES):
    cm = cmdf[cmdf["Mô hình"] == name].groupby("Thực tế", sort=False)[[f"Dự đoán {c}" for c in CLASSES]].sum().reindex(CLASSES).to_numpy()
    cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    ax.imshow(cmn, cmap="Greys", vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{cm[i, j]}\n{cmn[i, j]:.0%}", ha="center", va="center", fontsize=8, color="white" if cmn[i, j] > 0.6 else "black")
    ax.set_xticks(LABELS, CLASSES, rotation=30); ax.set_yticks(LABELS, CLASSES)
    ax.set_xlabel("Dự đoán"); ax.set_ylabel("Thực tế"); ax.set_title(name, fontsize=10)
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/confusion_matrix_5_lan.png", dpi=130); plt.close()

means = g[["Acc test", "F1 macro test"]].mean().reindex(MODEL_NAMES)
stds = g[["Acc test", "F1 macro test"]].std().reindex(MODEL_NAMES)
ax = means.plot.bar(yerr=stds, capsize=4, figsize=(12, 4.8), color=["#444444", "#bbbbbb"], edgecolor="black", width=0.7)
for cont in ax.containers:
    if hasattr(cont, "patches"):
        ax.bar_label(cont, fmt="%.3f", fontsize=8, padding=3)
ax.set_ylim(0, 1.1); ax.set_xlabel(""); ax.set_title("Trung bình 5 lần chạy trên tập test (thanh lỗi = độ lệch chuẩn)")
plt.xticks(rotation=12); ax.legend(frameon=False, loc="lower right"); style(ax)
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/so_sanh_mo_hinh_5_lan.png", dpi=130); plt.close()


# ============================================================================
# 8. Xuất file Excel
#
# File ket_qua_5_lan_chay.xlsx gồm các sheet:
#
# | Sheet | Nội dung |
# | --- | --- |
# | Tong_hop | Trung bình, độ lệch chuẩn, min, max của từng mô hình qua 5 lần. Tính bằng công thức Excel từ sheet Chi_tiet_5_lan, nên bấm vào ô là thấy nguồn |
# | Chi_tiet_5_lan | Mỗi dòng là 1 mô hình trong 1 lần chạy: seed, siêu tham số chọn, epoch dừng, loss, accuracy, precision, recall, F1, thời gian |
# | Theo_lop | Precision, recall, F1 của từng lớp, từng lần chạy |
# | Ma_tran_nham_lan | Ma trận nhầm lẫn trên test của từng lần chạy |
# | Loss_theo_epoch | Loss train/val ở từng epoch / vòng của từng lần, đánh dấu điểm early stopping |
# | Tim_sieu_tham_so | Mọi tổ hợp siêu tham số đã thử trong từng lần và kết quả trên val |
# | Du_lieu_tung_lan | Số ảnh theo tập, lớp, nguồn của từng lần chia |
# | Cau_hinh | Các tham số cấu hình đã dùng |
# ============================================================================
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

XLSX = f"{OUT_DIR}/ket_qua_5_lan_chay.xlsx"
config = pd.DataFrame([
    ("Tổng số ảnh dùng", len(meta)), ("Số lần chạy", N_RUNS), ("Seed các lần", ", ".join(map(str, SEEDS))),
    ("Seed chọn ảnh", SAMPLE_SEED), ("Tỉ lệ train / val / test", f"{1-VAL_SIZE-TEST_SIZE:.0%} / {VAL_SIZE:.0%} / {TEST_SIZE:.0%}"),
    ("Trích đặc trưng", "ResNet18 (ImageNet), 512 chiều"), ("Giảm chiều", f"StandardScaler + PCA {N_COMPONENTS} chiều, fit trên train từng lần"),
    ("SGD: alpha thử", str(ALPHAS)), ("SGD: learning rate", str(SGD_ETA0)), ("SGD: epoch tối đa / patience", f"{MAX_EPOCHS} / {PATIENCE}"),
    ("SVM RBF: C thử", str(SVC_C)), ("SVM RBF: gamma thử", str(SVC_GAMMA)),
    ("XGBoost: max_depth thử", str(XGB_DEPTHS)), ("XGBoost: learning_rate", XGB_LR),
    ("XGBoost: vòng tối đa / early stopping", f"{XGB_MAX_ROUNDS} / {XGB_EARLY_STOP}"),
    ("Decision Tree: max_depth / min_samples_leaf thử", f"{DT_DEPTHS} / {DT_MIN_LEAF}"),
    ("AdaBoost: max_depth cây yếu / learning_rate / vòng tối đa", f"{ADA_DEPTHS} / {ADA_LR} / {ADA_MAX_ROUNDS}"),
    ("Tiêu chí chọn siêu tham số", "Loss val thấp nhất (SGD, XGBoost); tỉ lệ sai val thấp nhất (AdaBoost); accuracy val cao nhất (SVM RBF, Decision Tree)"),
    ("Ngày chạy", time.strftime("%Y-%m-%d %H:%M")),
    ("Python", sys.version.split()[0]),
    ("scikit-learn / XGBoost", f"{__import__('sklearn').__version__} / {xgb.__version__}"),
    ("Máy", f"{__import__('platform').platform()}; CPU {__import__('platform').processor() or 'không rõ'}; GPU {'có' if XGB_DEVICE == 'cuda' else 'không'}"),
], columns=["Tham số", "Giá trị"])
tune_out = tune.drop(columns=["C", "gamma", "depth", "leaf", "depth_thuc_te"], errors="ignore")

with pd.ExcelWriter(XLSX, engine="openpyxl") as w:
    pd.DataFrame().to_excel(w, sheet_name="Tong_hop")
    detail.to_excel(w, sheet_name="Chi_tiet_5_lan", index=False)
    pd.DataFrame(class_rows).to_excel(w, sheet_name="Theo_lop", index=False)
    cmdf.to_excel(w, sheet_name="Ma_tran_nham_lan", index=False)
    pd.DataFrame(loss_rows).to_excel(w, sheet_name="Loss_theo_epoch", index=False)
    tune_out.to_excel(w, sheet_name="Tim_sieu_tham_so", index=False)
    pd.DataFrame(data_rows).to_excel(w, sheet_name="Du_lieu_tung_lan", index=False)
    config.to_excel(w, sheet_name="Cau_hinh", index=False)

wb = load_workbook(XLSX)
FONT = "Arial"
HEAD_FILL = PatternFill("solid", fgColor="DCE6F5")
thin = Side(style="thin", color="BFBFBF")

# Định dạng chung cho các sheet dữ liệu
for ws in wb.worksheets[1:]:
    for row in ws.iter_rows():
        for c in row:
            c.font = Font(name=FONT, size=10, bold=(c.row == 1))
            if c.row == 1:
                c.fill = HEAD_FILL; c.alignment = Alignment(wrap_text=True, vertical="center")
    for col in ws.columns:
        head = str(col[0].value or "")
        width = max(len(head), *(len(str(c.value)) if c.value is not None else 0 for c in col[1:200]), 6)
        ws.column_dimensions[col[0].column_letter].width = min(width + 2, 40)
        if any(k in head for k in ["Acc", "Precision", "Recall", "F1", "PCA giữ"]):
            for c in col[1:]: c.number_format = "0.0000"
        if "Loss" in head:
            for c in col[1:]: c.number_format = "0.0000"
        if "Thời gian" in head:
            for c in col[1:]: c.number_format = "0.0"
    ws.freeze_panes = "A2"

# Sheet Tong_hop: công thức tham chiếu Chi_tiet_5_lan
ws = wb["Tong_hop"]
src = wb["Chi_tiet_5_lan"]
col_of = {src.cell(1, j).value: get_column_letter(j) for j in range(1, src.max_column + 1)}
ws["A1"] = "TỔNG HỢP KẾT QUẢ 5 LẦN CHẠY"
ws["A2"] = f"{len(meta)} ảnh, 5 lớp, mỗi lần chia lại train/val/test với seed khác. Mọi số trong bảng là công thức tính từ sheet Chi_tiet_5_lan."
ws["A1"].font = Font(name=FONT, size=13, bold=True); ws["A2"].font = Font(name=FONT, size=10, italic=True, color="555555")

SUM_METRICS = ["Acc train", "Acc val", "Acc test", "Precision macro test", "Recall macro test", "F1 macro test", "Acc test tự chụp", "Acc test RealWaste", "Loss train", "Loss val", "Thời gian tìm + huấn luyện (giây)"]
STATS = [("Trung bình", "AVERAGE"), ("Độ lệch chuẩn", "STDEV"), ("Min", "MIN"), ("Max", "MAX")]
r = 4
for name in MODEL_NAMES:
    rows = detail.index[detail["Mô hình"] == name] + 2     # dòng trong Excel (có header)
    r0, r1 = rows.min(), rows.max()
    ws.cell(r, 1, name).font = Font(name=FONT, size=11, bold=True)
    ws.cell(r, 2, f"(dòng {r0} đến {r1} của Chi_tiet_5_lan)").font = Font(name=FONT, size=9, italic=True, color="777777")
    r += 1
    ws.cell(r, 1, "Chỉ số")
    for j, (lbl, _) in enumerate(STATS): ws.cell(r, 2 + j, lbl)
    ws.cell(r, 2 + len(STATS), "Viết vào báo cáo (TB ± ĐLC)")
    for j in range(1, 3 + len(STATS)):
        c = ws.cell(r, j); c.font = Font(name=FONT, size=10, bold=True); c.fill = HEAD_FILL; c.border = Border(bottom=thin)
    r += 1
    for met in SUM_METRICS:
        rng_ = f"Chi_tiet_5_lan!{col_of[met]}{r0}:{col_of[met]}{r1}"
        ws.cell(r, 1, met).font = Font(name=FONT, size=10)
        for j, (_, fn) in enumerate(STATS):
            c = ws.cell(r, 2 + j, f"={fn}({rng_})")
            c.font = Font(name=FONT, size=10); c.number_format = "0.0" if "Thời gian" in met else "0.0000"
        nd = 1 if "Thời gian" in met else 4   # FIXED tự dùng dấu thập phân theo ngôn ngữ máy
        c = ws.cell(r, 2 + len(STATS), f'=FIXED(AVERAGE({rng_}),{nd},TRUE)&" ± "&FIXED(STDEV({rng_}),{nd},TRUE)')
        c.font = Font(name=FONT, size=10)
        r += 1
    r += 1

# Bảng so sánh nhanh accuracy test từng lần
ws.cell(r, 1, "Accuracy test từng lần chạy").font = Font(name=FONT, size=11, bold=True); r += 1
ws.cell(r, 1, "Mô hình")
for k in range(N_RUNS): ws.cell(r, 2 + k, f"Lần {k + 1} (seed {SEEDS[k]})")
ws.cell(r, 2 + N_RUNS, "Trung bình")
for j in range(1, 3 + N_RUNS):
    c = ws.cell(r, j); c.font = Font(name=FONT, size=10, bold=True); c.fill = HEAD_FILL
r += 1
for name in MODEL_NAMES:
    rows = detail.index[detail["Mô hình"] == name] + 2
    ws.cell(r, 1, name).font = Font(name=FONT, size=10)
    for k, rr in enumerate(rows):
        c = ws.cell(r, 2 + k, f"=Chi_tiet_5_lan!{col_of['Acc test']}{rr}"); c.number_format = "0.0000"; c.font = Font(name=FONT, size=10)
    c = ws.cell(r, 2 + N_RUNS, f"=AVERAGE({get_column_letter(2)}{r}:{get_column_letter(1 + N_RUNS)}{r})")
    c.number_format = "0.0000"; c.font = Font(name=FONT, size=10, bold=True)
    r += 1

ws.column_dimensions["A"].width = 24
for j in range(2, 3 + max(len(STATS), N_RUNS)):
    ws.column_dimensions[get_column_letter(j)].width = 16
ws.column_dimensions[get_column_letter(2 + len(STATS))].width = max(ws.column_dimensions[get_column_letter(2 + len(STATS))].width, 26)
wb.save(XLSX)
print("Đã ghi", XLSX)


# ============================================================================
# 9. Lưu mô hình
# ============================================================================
joblib.dump({"runs": saved_models, "classes": CLASSES, "seeds": SEEDS}, f"{OUT_DIR}/mo_hinh_5_lan.joblib")
print("Đã lưu vào", OUT_DIR)
for f in sorted(os.listdir(OUT_DIR)): print("  ", f)


# ============================================================================
# 10. Gợi ý viết báo cáo
#
# Kết quả: lấy cột "Viết vào báo cáo (TB ± ĐLC)" ở sheet Tong_hop. Ví dụ câu viết:
# "Qua 5 lần chạy với 5 cách chia dữ liệu khác nhau, SVM RBF đạt accuracy trên tập test 0,xxx ± 0,xxx".
#
# Độ ổn định: độ lệch chuẩn nhỏ nghĩa là kết quả không phụ thuộc vào một cách chia dữ liệu may mắn.
#
# Biểu đồ loss: dùng các hình loss_*_5_lan.png. Với mỗi hình nêu 3 ý:
#
# 1. Hội tụ: loss train giảm nhanh ở đầu rồi đi ngang, ở cả 5 lần.
# 2. Không overfit: loss val giảm cùng loss train; tại điểm dừng (chấm tròn), khoảng cách hai đường nhỏ. Đối chiếu cột "Acc train" và "Acc val" ở Chi_tiet_5_lan.
# 3. Early stopping: nếu sau chấm tròn loss val đi lên, đó là lúc mô hình bắt đầu học thuộc tập train; mô hình được giữ lại tại chấm tròn. Số epoch dừng của từng lần có ở cột "Epoch/vòng dừng".
#
# Mô hình không lặp theo epoch (SVM RBF, Decision Tree): dùng validation curve. Bên trái (mô hình đơn giản: C nhỏ, cây nông) cả train và val đều sai nhiều, tức underfit. Bên phải (C lớn, cây sâu) train gần như không sai nhưng val sai nhiều hơn, tức overfit. Điểm được chọn nằm ở giữa, nơi loss val thấp nhất. Decision Tree đơn lẻ thường overfit mạnh nhất, đây là lý do AdaBoost và XGBoost ghép nhiều cây lại.
#
# Khi được hỏi về một lần chạy cụ thể: tra sheet Chi_tiet_5_lan (kết quả), Loss_theo_epoch (loss từng epoch) và Tim_sieu_tham_so (vì sao chọn siêu tham số đó).
# ============================================================================
