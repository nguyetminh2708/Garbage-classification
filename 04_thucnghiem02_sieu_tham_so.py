# -*- coding: utf-8 -*-
"""
THUC NGHIEM 02: Chon sieu tham so bang GridSearchCV k-fold va ket qua chinh (5 lan chay).

Voi N_COMPONENTS chieu (chon sau thuc nghiem 01), moi lan chay:
  1. chia train / test = 85 / 15, phan tang theo (lop, nguon anh)
  2. fit StandardScaler + PCA tren train
  3. voi moi mo hinh, GridSearchCV 5-fold tren tap train, chi do cac tham so quan trong (GRID trong chung.py):
     Softmax 8, SVM 8, XGBoost 9 to hop; tieu chi: log loss CV thap nhat (Softmax, XGBoost), accuracy CV cao nhat (SVM)
  4. huan luyen lai to hop tot nhat: giu 15% cua train lam val de early stopping va ve duong loss, fit tren 85% con lai
  5. danh gia mot lan tren test
Lap 5 lan (seed 0..4): 25 to hop x 5 fold x 5 lan = 625 luot fit trong CV (chay song song tren moi nhan CPU).

Dau ra:
  ketqua/ket_qua_5_lan_chay.xlsx        - 10 sheet (xem ham ghi_cac_sheet trong xuat.py)
  models/mo_hinh_5_lan.joblib            - bo tien xu ly va 3 mo hinh cua tung lan chay
  hinh/hinh_4_2_so_sanh_mo_hinh.png      - accuracy va Macro F1 trung binh tren test
  hinh/hinh_4_3_loss_softmax.png         - log loss theo epoch, 5 lan
  hinh/hinh_4_4_svm_validation_curve.png - accuracy train va CV theo C, 5 lan
  hinh/hinh_4_5_loss_xgboost.png         - log loss theo vong boosting, 5 lan
  hinh/hinh_4_6_confusion_matrix.png     - ma tran nham lan cong don 5 lan
"""

import sys
import time

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from chung import (CLASSES, GRID, LABELS, LOSS_NAME, MODEL, MODEL_NAMES, N_COMPONENTS, N_FOLDS, SEEDS, chia_va_pca,
                   danh_gia_test, doc_dac_trung, huan_luyen, tim_sieu_tham_so, to_hop_luoi)
from xuat import bang_cau_hinh, ghi_cac_sheet, sheet_tong_hop, tong_hop_sieu_tham_so, ve_hinh_thucnghiem02

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ============================================================================
# 1. Chay thuc nghiem
# ============================================================================
def chay(meta, F):
    n_combo = {n: len(to_hop_luoi(GRID[n])) for n in MODEL_NAMES}
    print("So to hop moi lan chay:", n_combo, f"-> {sum(n_combo.values())} to hop x {N_FOLDS} fold x {len(SEEDS)} lan =",
          sum(n_combo.values()) * N_FOLDS * len(SEEDS), "luot fit trong CV\n")
    detail, class_rows, cm_rows, loss_rows, tune_rows, data_rows = [], [], [], [], [], []
    hists, saved = {}, {}

    for run, seed in enumerate(SEEDS, start=1):
        t_run = time.time()
        X, y, idx, prep, var_kept, strat_train = chia_va_pca(meta, F, seed, N_COMPONENTS)
        src_test = meta.source.to_numpy()[idx["test"]]
        for s in idx:
            for (lbl, src), n in meta.loc[idx[s]].groupby(["label", "source"]).size().items():
                data_rows.append({"Lần chạy": run, "Seed": seed, "Tập": s, "Lớp": lbl, "Nguồn": src, "Số ảnh": n})

        models = {}
        for name in MODEL_NAMES:
            t_model = time.time()
            hp, cvdf = tim_sieu_tham_so(name, X, y, strat_train, seed)
            t_cv = time.time() - t_model
            cvdf.insert(0, "Seed", seed); cvdf.insert(0, "Lần chạy", run)
            tune_rows.append(cvdf)
            chon = cvdf[cvdf["Được chọn"] == "x"].iloc[0]

            m, h, r = huan_luyen(name, X, y, hp, seed)
            models[name] = m; hists[(run, name)] = (h, r["Epoch/vòng dừng"])
            te, p_test = danh_gia_test(m, X, y, src_test)
            detail.append({"Mô hình": name, "Lần chạy": run, "Seed": seed, "Siêu tham số chọn": r["Siêu tham số"],
                           "Số ảnh train": len(y["train"]), "Số ảnh val (trong train)": len(y["val"]), "Số ảnh test": len(y["test"]),
                           "Số chiều": N_COMPONENTS, "PCA giữ phương sai": var_kept,
                           "Acc CV TB (tổ hợp chọn)": chon["Acc CV TB"], "Acc CV ĐLC (tổ hợp chọn)": chon["Acc CV ĐLC"],
                           "Loss CV TB (tổ hợp chọn)": chon["Loss CV TB"], "Epoch/vòng dừng": r["Epoch/vòng dừng"],
                           "Loại loss": LOSS_NAME[name], "Loss train": r["Loss train"], "Loss val": r["Loss val"],
                           "Acc train": r["Acc train"], "Acc val": r["Acc val"], **te,
                           "Số tổ hợp đã thử": n_combo[name], "Thời gian GridSearchCV (giây)": t_cv,
                           "Thời gian tìm + huấn luyện (giây)": time.time() - t_model})
            pr, rc, f1, sp = precision_recall_fscore_support(y["test"], p_test, labels=LABELS, zero_division=0)
            for k, c in enumerate(CLASSES):
                class_rows.append({"Mô hình": name, "Lần chạy": run, "Lớp": c, "Precision": pr[k], "Recall": rc[k], "F1": f1[k], "Số ảnh test": sp[k]})
            cm = confusion_matrix(y["test"], p_test, labels=LABELS)
            for k, c in enumerate(CLASSES):
                cm_rows.append({"Mô hình": name, "Lần chạy": run, "Thực tế": c, **{f"Dự đoán {cc}": int(cm[k, j]) for j, cc in enumerate(CLASSES)}})
            if name != "SVM":
                for e in range(len(h["train_loss"])):
                    loss_rows.append({"Mô hình": name, "Lần chạy": run, "Epoch/vòng": e + 1, "Loss train": h["train_loss"][e],
                                      "Loss val": h["val_loss"][e], "Được chọn (early stopping)": "x" if e + 1 == r["Epoch/vòng dừng"] else ""})
        saved[run] = {"prep": prep, "models": models}
        print(f"Lan {run} (seed {seed}, {time.time() - t_run:.0f}s)  acc test  " +
              " | ".join(f"{d['Mô hình']}: {d['Acc test']:.3f} ({d['Siêu tham số chọn']})" for d in detail[-len(MODEL_NAMES):]))

    detail = pd.DataFrame(detail)
    detail["_o"] = detail["Mô hình"].map({n: i for i, n in enumerate(MODEL_NAMES)})
    detail = detail.sort_values(["_o", "Lần chạy"]).drop(columns="_o").reset_index(drop=True)
    return dict(detail=detail, tune=pd.concat(tune_rows, ignore_index=True), class_rows=pd.DataFrame(class_rows), cm=pd.DataFrame(cm_rows),
                loss=pd.DataFrame(loss_rows), data=pd.DataFrame(data_rows), hists=hists, saved=saved, n_combo=n_combo, meta=meta)


# ============================================================================
# 2. Chay (tong hop, Excel, hinh: xem xuat.py)
# ============================================================================
def main() -> None:
    meta, F = doc_dac_trung()
    print(f"{len(meta)} anh, dac trung {F.shape[1]} chieu, PCA {N_COMPONENTS} chieu")
    R = chay(meta, F)
    hp, marg = tong_hop_sieu_tham_so(R)

    g = R["detail"].groupby("Mô hình", sort=False)
    print("\n=== Ket qua trung binh 5 lan (TB +- DLC) ===")
    for met in ["Acc CV TB (tổ hợp chọn)", "Acc train", "Acc val", "Acc test", "F1 macro test", "Acc test tự chụp", "Acc test RealWaste"]:
        print(f"  {met:26s}" + "  ".join(f"{n}: {g[met].mean()[n]:.3f} ± {g[met].std()[n]:.3f}" for n in MODEL_NAMES))

    ve_hinh_thucnghiem02(R)
    xlsx = ghi_cac_sheet(R, hp, marg, bang_cau_hinh(R))
    sheet_tong_hop(xlsx, R)
    joblib.dump({"runs": R["saved"], "classes": CLASSES, "seeds": SEEDS, "n_components": N_COMPONENTS}, MODEL / "mo_hinh_5_lan.joblib")
    print("\nDa ghi: ketqua/ket_qua_5_lan_chay.xlsx, models/mo_hinh_5_lan.joblib, hinh/hinh_4_2 den hinh_4_6")


if __name__ == "__main__":
    main()
