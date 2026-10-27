# -*- coding: utf-8 -*-
"""
THUC NGHIEM 02: Khao sat sieu tham so va ket qua chinh (5 lan chay).

Voi N_COMPONENTS chieu (chon sau thuc nghiem 01), moi lan chay:
  1. chia train / val / test 70 / 15 / 15, phan tang theo (lop, nguon anh)
  2. fit StandardScaler + PCA tren train
  3. huan luyen MOI to hop trong luoi sieu tham so cua 3 mo hinh (GRID trong chung.py):
     Softmax 30, SVM 30, XGBoost 36 to hop; mo hinh lap dung early stopping theo loss val
  4. to hop tot nhat tren val duoc danh gia tren test, dung mot lan
Lap 5 lan (seed 0..4): 96 to hop x 5 = 480 luot huan luyen, khoang 1 den 2 gio tren CPU.

Dau ra:
  ketqua/ket_qua_5_lan_chay.xlsx        - 10 sheet (xem bang trong ham xuat_excel)
  models/mo_hinh_5_lan.joblib            - bo tien xu ly va 3 mo hinh cua tung lan chay
  hinh/hinh_4_2_so_sanh_mo_hinh.png      - accuracy va Macro F1 trung binh tren test
  hinh/hinh_4_3_loss_softmax.png         - log loss theo epoch, 5 lan
  hinh/hinh_4_4_svm_validation_curve.png - hinge loss theo C, 5 lan
  hinh/hinh_4_5_loss_xgboost.png         - log loss theo vong boosting, 5 lan
  hinh/hinh_4_6_confusion_matrix.png     - ma tran nham lan cong don 5 lan
"""

import sys
import time

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from chung import (CLASSES, GRID, LABELS, LOSS_NAME, MODEL, MODEL_NAMES, N_COMPONENTS, SEEDS, chia_va_pca,
                   danh_gia_test, doc_dac_trung, huan_luyen, tieu_chi, to_hop_luoi)
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
    print("So to hop moi lan chay:", n_combo, "-> tong", sum(n_combo.values()) * len(SEEDS), "luot huan luyen\n")
    detail, class_rows, cm_rows, loss_rows, tune_rows, data_rows = [], [], [], [], [], []
    hists, saved = {}, {}

    for run, seed in enumerate(SEEDS, start=1):
        t_run = time.time()
        X, y, idx, prep, var_kept = chia_va_pca(meta, F, seed, N_COMPONENTS)
        src_test = meta.source.to_numpy()[idx["test"]]
        for s in idx:
            for (lbl, src), n in meta.loc[idx[s]].groupby(["label", "source"]).size().items():
                data_rows.append({"Lần chạy": run, "Seed": seed, "Tập": s, "Lớp": lbl, "Nguồn": src, "Số ảnh": n})

        models = {}
        for name in MODEL_NAMES:
            t_model, best = time.time(), None
            for hp in to_hop_luoi(GRID[name]):
                m, h, r = huan_luyen(name, X, y, hp, seed)
                tune_rows.append({"Lần chạy": run, "Seed": seed, **r})
                if best is None or tieu_chi(name, r) < tieu_chi(name, best[2]):
                    best = (m, h, r)
            m, h, r = best
            models[name] = m; hists[(run, name)] = (h, r["Epoch/vòng dừng"])
            te, p_test = danh_gia_test(m, X, y, src_test)
            detail.append({"Mô hình": name, "Lần chạy": run, "Seed": seed, "Siêu tham số chọn": r["Siêu tham số"],
                           "Số ảnh train": len(y["train"]), "Số ảnh val": len(y["val"]), "Số ảnh test": len(y["test"]),
                           "Số chiều": N_COMPONENTS, "PCA giữ phương sai": var_kept, "Epoch/vòng dừng": r["Epoch/vòng dừng"],
                           "Loại loss": LOSS_NAME[name], "Loss train": r["Loss train"], "Loss val": r["Loss val"],
                           "Acc train": r["Acc train"], "Acc val": r["Acc val"], **te,
                           "Số tổ hợp đã thử": n_combo[name], "Thời gian tìm + huấn luyện (giây)": time.time() - t_model})
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
    return dict(detail=detail, tune=pd.DataFrame(tune_rows), class_rows=pd.DataFrame(class_rows), cm=pd.DataFrame(cm_rows),
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
    for met in ["Acc train", "Acc val", "Acc test", "F1 macro test", "Acc test tự chụp", "Acc test RealWaste"]:
        print(f"  {met:20s}" + "  ".join(f"{n}: {g[met].mean()[n]:.3f} ± {g[met].std()[n]:.3f}" for n in MODEL_NAMES))

    ve_hinh_thucnghiem02(R)
    xlsx = ghi_cac_sheet(R, hp, marg, bang_cau_hinh(R))
    sheet_tong_hop(xlsx, R)
    joblib.dump({"runs": R["saved"], "classes": CLASSES, "seeds": SEEDS, "n_components": N_COMPONENTS}, MODEL / "mo_hinh_5_lan.joblib")
    print("\nDa ghi: ketqua/ket_qua_5_lan_chay.xlsx, models/mo_hinh_5_lan.joblib, hinh/hinh_4_2 den hinh_4_6")


if __name__ == "__main__":
    main()
