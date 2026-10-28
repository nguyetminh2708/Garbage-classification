# -*- coding: utf-8 -*-
"""
THUC NGHIEM 01: Khao sat so chieu dac trung sau PCA.

Thu 5 muc: 32, 64, 128, 256, 512 (512 = giu nguyen, khong PCA). Moi muc chay 5 lan voi 5 cach chia
train / test (seed 0..4); moi lan fit lai StandardScaler + PCA tren tap train. Ba mo hinh dung sieu tham so
mac dinh (DEFAULT_HP trong chung.py), huan luyen tren 85% cua train va do accuracy tren 15% con lai (val).
Muc duoc chon (N_COMPONENTS) dung cho thuc nghiem 02.

Dau ra:
  ketqua/thucnghiem01_pca.csv       - tung muc, tung mo hinh, tung lan chay
  ketqua/thucnghiem01_pca_tb.csv    - trung binh +- do lech chuan theo muc va mo hinh
  hinh/hinh_4_1_khao_sat_pca.png    - accuracy val trung binh theo so chieu
"""

import sys
import time

import pandas as pd

from chung import DEFAULT_HP, MODEL_NAMES, SEEDS, chia_va_pca, danh_gia_test, doc_dac_trung, huan_luyen
from xuat import tong_hop_pca, ve_hinh_thucnghiem01, xuat_thucnghiem01

PCA_DIMS = [32, 64, 128, 256, 512]      # cac muc so chieu thu; 512 = khong PCA

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ------------------------------ chay thuc nghiem ------------------------------
def chay(meta, F) -> pd.DataFrame:
    """Moi muc so chieu x moi lan chay x moi mo hinh: mot dong ket qua."""
    rows = []
    for dim in PCA_DIMS:
        t0 = time.time()
        for run, seed in enumerate(SEEDS, start=1):
            X, y, idx, _, var_kept, _ = chia_va_pca(meta, F, seed, dim)
            src_test = meta.source.to_numpy()[idx["test"]]
            for name in MODEL_NAMES:
                m, h, r = huan_luyen(name, X, y, DEFAULT_HP[name], seed)
                te, _ = danh_gia_test(m, X, y, src_test)
                rows.append({"Số chiều": dim, "Mô hình": name, "Lần chạy": run, "Seed": seed, "PCA giữ phương sai": var_kept,
                             "Siêu tham số": r["Siêu tham số"], "Epoch/vòng dừng": r["Epoch/vòng dừng"],
                             "Loss train": r["Loss train"], "Loss val": r["Loss val"], "Acc train": r["Acc train"],
                             "Acc val": r["Acc val"], "Acc test": te["Acc test"], "F1 macro test": te["F1 macro test"],
                             "Thời gian (giây)": r["Thời gian (giây)"]})
        sub = pd.DataFrame(rows); sub = sub[sub["Số chiều"] == dim]
        print(f"  {dim:>3} chieu ({time.time() - t0:.0f}s)  acc val TB  " +
              " | ".join(f"{n}: {sub[sub['Mô hình'] == n]['Acc val'].mean():.3f}" for n in MODEL_NAMES))
    return pd.DataFrame(rows)





# ------------------------------ chay ------------------------------
def main() -> None:
    meta, F = doc_dac_trung()
    print(f"{len(meta)} anh, dac trung {F.shape[1]} chieu")
    print(pd.crosstab(meta.label, meta.source, margins=True, margins_name="Tong").to_string(), "\n")

    df = chay(meta, F)
    tb = tong_hop_pca(df)
    xuat_thucnghiem01(df, tb)
    ve_hinh_thucnghiem01(df, PCA_DIMS)

    print("\n" + tb.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nDa ghi: ketqua/thucnghiem01_pca.csv, ketqua/thucnghiem01_pca_tb.csv, hinh/hinh_4_1_khao_sat_pca.png")


if __name__ == "__main__":
    main()
