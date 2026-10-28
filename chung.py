# -*- coding: utf-8 -*-
"""
Cac hang so va ham dung chung cho cac buoc 01 -> 05.

Thu muc du an:
  Data/anh_tu_chup/<lop>/   anh nhom tu chup, 5 thu muc giay, nhua, huu_co, kim_loai, khac
  Data/realwaste/<lop>/     anh RealWaste da xep thu cong vao 5 thu muc cung ten
  vectors/                  dac trung 512 chieu tu ResNet18 (npz)
  ketqua/                   danh sach anh sau lam sach (csv) va cac bang ket qua (xlsx, csv)
  hinh/                     hinh ve cho bao cao, dat ten theo so hinh trong bao cao
  models/                   mo hinh da huan luyen (joblib)
File xuat.py chua phan ghi Excel / CSV va ve hinh; file nay chi co logic.

Cach chia du lieu (moi lan chay mot seed):
  train / test = 85 / 15, phan tang theo (lop, nguon anh)
  - chon sieu tham so: GridSearchCV 5-fold tren tap train
  - huan luyen mo hinh cuoi: giu 15% cua train lam val de early stopping va ve duong loss,
    85% con lai de fit; danh gia mot lan tren test
"""

import copy
import itertools
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (accuracy_score, f1_score, log_loss, hinge_loss,
                             precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BASE = Path(__file__).parent
DATA_ANH = BASE / "Data"                       # anh goc (khong day len git)
VEC, KQ, HINH, MODEL = BASE / "vectors", BASE / "ketqua", BASE / "hinh", BASE / "models"
for d in (VEC, KQ, HINH, MODEL):
    d.mkdir(exist_ok=True)

CLASSES = ["giay", "nhua", "huu_co", "kim_loai", "khac"]
TEN_LOP = {"giay": "Giấy", "nhua": "Nhựa", "huu_co": "Hữu cơ", "kim_loai": "Kim loại", "khac": "Khác"}
LABELS = np.arange(len(CLASSES))
MODEL_NAMES = ["Softmax Regression", "SVM", "XGBoost"]

SEEDS = [0, 1, 2, 3, 4]            # 5 lan chay, moi lan mot cach chia train / test
TEST_SIZE = 0.15                   # train / test = 85 / 15
VAL_FRAC = 0.15                    # phan cua train giu lam val khi huan luyen mo hinh cuoi (early stopping, duong loss)
N_FOLDS = 5                        # so fold cua GridSearchCV tren tap train
TOTAL_SAMPLES = None               # so anh toi da dung (None = dung het anh sau lam sach)
SAMPLE_SEED = 42
N_COMPONENTS = 128                 # so chieu PCA dung cho thuc nghiem 02 (chon sau thuc nghiem 01)

# ----------------------- luoi sieu tham so (thuc nghiem 02) -----------------------
# Chi do cac tham so quan trong nhat cua tung mo hinh; cac tham so con lai co dinh.
MAX_EPOCHS, PATIENCE = 200, 15     # Softmax: SGD theo epoch, early stopping theo loss val
XGB_MAX_ROUNDS, XGB_EARLY_STOP = 2000, 50   # XGBoost khi huan luyen mo hinh cuoi (early stopping theo loss val)
XGB_CV_ROUNDS = 300                         # XGBoost trong GridSearchCV: so cay co dinh
XGB_FIXED = {"subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 2, "reg_lambda": 1.0}
GRID = {
    "Softmax Regression": {"alpha": [1e-4, 1e-3, 1e-2, 1e-1], "eta0": [1e-3, 1e-2]},                 # 8 to hop
    "SVM": {"kernel": ["linear", "rbf"], "C": [0.1, 1, 10, 100]},                                      # 8 (gamma = scale)
    "XGBoost": {"max_depth": [3, 5, 7], "learning_rate": [0.05, 0.1, 0.3]},                          # 9
}
# sieu tham so mac dinh (dung cho thuc nghiem 01)
DEFAULT_HP = {
    "Softmax Regression": {"alpha": 1e-3, "eta0": 1e-3},
    "SVM": {"kernel": "rbf", "C": 10, "gamma": "scale"},
    "XGBoost": {"max_depth": 5, "learning_rate": 0.1},
}
LOSS_NAME = {"Softmax Regression": "log loss", "SVM": "hinge loss", "XGBoost": "log loss"}
# tieu chi chon tren ket qua k-fold: log loss CV thap nhat (Softmax, XGBoost); accuracy CV cao nhat (SVM, khong co xac suat)
TIEU_CHI = {"Softmax Regression": "loss", "SVM": "acc", "XGBoost": "loss"}

try:
    import xgboost as xgb
except ImportError:
    xgb = None


# ============================================================================
# Doc dac trung
# ============================================================================
def doc_dac_trung():
    """Doc vectors/features_512.npz; chon toi da TOTAL_SAMPLES anh (phan tang theo lop + nguon).
    Tra ve (meta: DataFrame co cot label, source, y; F: ma tran N x 512)."""
    f = VEC / "features_512.npz"
    if not f.exists():
        raise SystemExit("Thieu vectors/features_512.npz. Chay python 02_trich_dac_trung.py truoc.")
    raw = np.load(f, allow_pickle=True)
    meta = pd.DataFrame({"path": raw["path"], "label": raw["label"], "source": raw["source"]})
    F = raw["feats"]
    meta["y"] = meta.label.map({c: i for i, c in enumerate(CLASSES)})
    if TOTAL_SAMPLES is not None and len(meta) > TOTAL_SAMPLES:
        strat = meta.label + "|" + meta.source
        keep, _ = _strat_split(meta, meta.index, len(meta) - TOTAL_SAMPLES, strat, SAMPLE_SEED)
        meta, F = meta.loc[sorted(keep)].reset_index(drop=True), F[sorted(keep)]
    return meta, F


def _strat_split(meta, index, test_size, strat, seed):
    """Chia phan tang theo (lop, nguon); nhom nao qua it anh thi lui ve phan tang theo lop."""
    try:
        return train_test_split(index, test_size=test_size, stratify=strat, random_state=seed)
    except ValueError:
        warnings.warn("Co nhom (lop, nguon) qua it anh, chi phan tang theo lop.")
        return train_test_split(index, test_size=test_size, stratify=meta.loc[index, "label"], random_state=seed)


def chia_va_pca(meta, F, seed, n_components):
    """Mot lan chay: chia train / test, tach val trong train, roi fit StandardScaler + PCA tren train.
    n_components >= 512 thi khong PCA.
    Tra ve X, y (dict theo tap: train = toan bo train, fit = train tru val, val, test), idx, bo tien xu ly,
    ti le phuong sai giu lai, va khoa phan tang (lop|nguon) cua tap train de chia fold."""
    strat = meta.label + "|" + meta.source
    i_tr, i_te = _strat_split(meta, meta.index, TEST_SIZE, strat, seed)
    i_fit, i_va = _strat_split(meta, i_tr, VAL_FRAC, strat[i_tr], seed)
    idx = {"train": np.sort(i_tr), "fit": np.sort(i_fit), "val": np.sort(i_va), "test": np.sort(i_te)}
    s1 = StandardScaler().fit(F[idx["train"]])
    if n_components >= F.shape[1]:
        pca, var_kept = None, 1.0
        Z = {s: s1.transform(F[idx[s]]) for s in idx}
    else:
        pca = PCA(n_components=min(n_components, len(idx["train"])), random_state=seed).fit(s1.transform(F[idx["train"]]))
        var_kept = pca.explained_variance_ratio_.sum()
        Z = {s: pca.transform(s1.transform(F[idx[s]])) for s in idx}
    s2 = StandardScaler().fit(Z["train"])
    X = {s: s2.transform(Z[s]) for s in idx}
    y = {s: meta.y.to_numpy()[idx[s]] for s in idx}
    return X, y, idx, {"scaler_512": s1, "pca": pca, "scaler_pca": s2}, var_kept, strat.to_numpy()[idx["train"]]


# ============================================================================
# Chon sieu tham so bang GridSearchCV k-fold tren tap train
# ============================================================================
def _uoc_luong(name, seed):
    """Bo uoc luong dung trong GridSearchCV (tham so chua co trong luoi la gia tri co dinh)."""
    if name == "Softmax Regression":
        return SGDClassifier(loss="log_loss", penalty="l2", learning_rate="constant", max_iter=MAX_EPOCHS,
                             early_stopping=True, validation_fraction=VAL_FRAC, n_iter_no_change=PATIENCE,
                             tol=1e-4, random_state=seed)
    if name == "SVM":
        return SVC(random_state=seed)
    if xgb is None:
        raise SystemExit("Thieu thu vien xgboost. Chay: pip install xgboost")
    return xgb.XGBClassifier(objective="multi:softprob", n_estimators=XGB_CV_ROUNDS, tree_method="hist",
                             random_state=seed, n_jobs=1, **XGB_FIXED)


def chia_fold(X_train, strat_train, seed):
    """N_FOLDS fold phan tang theo (lop, nguon); nhom qua it thi phan tang theo lop."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            return list(skf.split(X_train, strat_train))
    except (ValueError, UserWarning):
        return list(skf.split(X_train, np.array([s.split("|")[0] for s in strat_train])))


def tim_sieu_tham_so(name, X, y, strat_train, seed):
    """GridSearchCV k-fold tren X['train']. Tra ve (sieu tham so tot nhat, bang ket qua moi to hop)."""
    scoring = {"acc": "accuracy"} if name == "SVM" else {"acc": "accuracy", "loss": "neg_log_loss"}
    gs = GridSearchCV(_uoc_luong(name, seed), GRID[name], scoring=scoring, refit=False, cv=chia_fold(X["train"], strat_train, seed),
                      n_jobs=-1, return_train_score=True)
    gs.fit(X["train"], y["train"])
    cv = gs.cv_results_
    rows = []
    for i, p in enumerate(cv["params"]):
        hp = _chuan_hp(dict(p))
        r = {"Mô hình": name, "Siêu tham số": hp_str(hp), **hp,
             "Acc train CV TB": cv["mean_train_acc"][i], "Acc CV TB": cv["mean_test_acc"][i], "Acc CV ĐLC": cv["std_test_acc"][i]}
        for k in range(N_FOLDS):
            r[f"Acc fold {k + 1}"] = cv[f"split{k}_test_acc"][i]
        if "loss" in scoring:
            r["Loss CV TB"], r["Loss CV ĐLC"] = -cv["mean_test_loss"][i], cv["std_test_loss"][i]
        else:
            r["Loss CV TB"], r["Loss CV ĐLC"] = None, None
        r["Hạng"] = int(cv[f"rank_test_{TIEU_CHI[name]}"][i])
        r["Thời gian fit TB (giây)"] = cv["mean_fit_time"][i]
        rows.append(r)
    df = pd.DataFrame(rows)
    best = df[df["Hạng"] == 1].index[0]
    df["Được chọn"] = ""; df.loc[best, "Được chọn"] = "x"
    hp = {k: v for k, v in dict(cv["params"][best]).items()}
    return hp, df


# ============================================================================
# Huan luyen mo hinh cuoi voi sieu tham so da chon (fit tren X['fit'], theo doi loss tren X['val'])
# ============================================================================
def fit_softmax(X, y, hp, seed):
    rng = np.random.default_rng(seed)
    clf = SGDClassifier(loss="log_loss", penalty="l2", alpha=hp["alpha"],
                        learning_rate="constant", eta0=hp["eta0"], random_state=seed)
    h = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best, best_loss, best_ep, wait = None, np.inf, 0, 0
    for ep in range(1, MAX_EPOCHS + 1):
        p = rng.permutation(len(y["fit"]))
        clf.partial_fit(X["fit"][p], y["fit"][p], classes=LABELS)
        h["train_loss"].append(log_loss(y["fit"], clf.predict_proba(X["fit"]), labels=LABELS))
        h["val_loss"].append(log_loss(y["val"], clf.predict_proba(X["val"]), labels=LABELS))
        h["train_acc"].append(clf.score(X["fit"], y["fit"]))
        h["val_acc"].append(clf.score(X["val"], y["val"]))
        if h["val_loss"][-1] < best_loss - 1e-4:
            best, best_loss, best_ep, wait = copy.deepcopy(clf), h["val_loss"][-1], ep, 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    return best, h, best_ep


def fit_svm(X, y, hp, seed):
    kw = {"gamma": hp.get("gamma", "scale")} if hp["kernel"] == "rbf" else {}
    m = SVC(kernel=hp["kernel"], C=hp["C"], random_state=seed, **kw).fit(X["fit"], y["fit"])
    h = {"train_loss": [hinge_loss(y["fit"], m.decision_function(X["fit"]), labels=LABELS)],
         "val_loss": [hinge_loss(y["val"], m.decision_function(X["val"]), labels=LABELS)],
         "train_acc": [m.score(X["fit"], y["fit"])], "val_acc": [m.score(X["val"], y["val"])]}
    return m, h, None


def fit_xgb(X, y, hp, seed):
    if xgb is None:
        raise SystemExit("Thieu thu vien xgboost. Chay: pip install xgboost")
    m = xgb.XGBClassifier(objective="multi:softprob", n_estimators=XGB_MAX_ROUNDS, learning_rate=hp["learning_rate"],
                          max_depth=hp["max_depth"], eval_metric="mlogloss", early_stopping_rounds=XGB_EARLY_STOP,
                          tree_method="hist", random_state=seed, **XGB_FIXED)
    m.fit(X["fit"], y["fit"], eval_set=[(X["fit"], y["fit"]), (X["val"], y["val"])], verbose=False)
    ev, bi = m.evals_result(), m.best_iteration
    h = {"train_loss": ev["validation_0"]["mlogloss"], "val_loss": ev["validation_1"]["mlogloss"],
         "train_acc": [m.score(X["fit"], y["fit"])], "val_acc": [m.score(X["val"], y["val"])]}
    return m, h, bi + 1


FIT = {"Softmax Regression": fit_softmax, "SVM": fit_svm, "XGBoost": fit_xgb}


def hp_str(hp):
    return ", ".join(f"{k}={v}" for k, v in hp.items())


def _chuan_hp(hp):
    """SVM: kernel linear khong dung gamma (ghi '-'); kernel rbf khong co trong luoi thi gamma = scale."""
    if hp.get("kernel") == "linear":
        hp["gamma"] = "-"
    elif hp.get("kernel") == "rbf":
        hp.setdefault("gamma", "scale")
    return hp


def huan_luyen(name, X, y, hp, seed):
    """Huan luyen 1 mo hinh voi 1 bo sieu tham so. Tra ve (mo hinh, lich su loss, dong ket qua train/val)."""
    t = time.time()
    m, h, be = FIT[name](X, y, hp, seed)
    k = (be - 1) if be else -1                                    # vi tri diem dung trong lich su loss
    ka = k if len(h["train_acc"]) == len(h["train_loss"]) else -1  # SVM, XGBoost chi ghi accuracy mo hinh cuoi
    hp_ghi = _chuan_hp(dict(hp))
    r = {"Mô hình": name, "Siêu tham số": hp_str(hp_ghi), **hp_ghi, "Epoch/vòng dừng": be,
         "Loss train": h["train_loss"][k], "Loss val": h["val_loss"][k],
         "Acc train": h["train_acc"][ka], "Acc val": h["val_acc"][ka], "Thời gian (giây)": time.time() - t}
    return m, h, r


def gia_tri_luoi(grid):
    """Gop luoi (dict hoac list dict) thanh {tham so: [cac gia tri]} de thong ke anh huong tung tham so."""
    out = {}
    for g in (grid if isinstance(grid, list) else [grid]):
        for k, vs in g.items():
            for v in vs:
                if v not in out.setdefault(k, []):
                    out[k].append(v)
    return out


def to_hop_luoi(grid):
    """Moi to hop cua luoi (dict hoac list dict). SVM kernel linear khong dung gamma nen gamma ghi '-'."""
    out, seen = [], set()
    for g in (grid if isinstance(grid, list) else [grid]):
        keys = list(g)
        for vals in itertools.product(*[g[k] for k in keys]):
            hp = _chuan_hp(dict(zip(keys, vals)))
            key = tuple(hp.items())
            if key not in seen:
                seen.add(key); out.append(hp)
    return out


def danh_gia_test(m, X, y, src_test):
    """Cac do do tren tap test, ke ca accuracy rieng theo nguon anh."""
    p = m.predict(X["test"])
    out = {"Acc test": accuracy_score(y["test"], p),
           "Precision macro test": precision_score(y["test"], p, average="macro", zero_division=0),
           "Recall macro test": recall_score(y["test"], p, average="macro", zero_division=0),
           "F1 macro test": f1_score(y["test"], p, average="macro", zero_division=0)}
    for src, ten in [("tu_chup", "Acc test tự chụp"), ("realwaste", "Acc test RealWaste")]:
        mask = src_test == src
        out[ten] = accuracy_score(y["test"][mask], p[mask]) if mask.any() else None
    return out, p
