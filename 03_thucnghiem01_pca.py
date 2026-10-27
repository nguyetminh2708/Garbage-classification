# -*- coding: utf-8 -*-
"""
THUC NGHIEM 01: Khao sat so chieu dac trung sau PCA.

Thu 5 muc: 32, 64, 128, 256, 512 (512 = giu nguyen, khong PCA). Moi muc chay 5 lan voi 5 cach chia
du lieu (seed 0..4); moi lan fit lai StandardScaler + PCA tren tap train. Ba mo hinh dung sieu tham so
mac dinh (DEFAULT_HP trong chung.py). Muc duoc chon (N_COMPONENTS) dung cho thuc nghiem 02.

Dau ra:
  ketqua/thucnghiem01_pca.csv       - tung muc, tung mo hinh, tung lan chay
  ketqua/thucnghiem01_pca_tb.csv    - trung binh +- do lech chuan theo muc va mo hinh
  hinh/hinh_4_1_khao_sat_pca.png    - accuracy val trung binh theo so chieu
"""

import sys
import time

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chung import (DEFAULT_HP, HINH, KQ, MODEL_NAMES, SEEDS, chia_va_pca, danh_gia_test,
                   doc_dac_trung, huan_luyen)

PCA_DIMS = [32, 64, 128, 256, 512]

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def main() -> None:
    meta, F = doc_dac_trung()
    print(f"{len(meta)} anh, dac trung {F.shape[1]} chieu")
    print(pd.crosstab(meta.label, meta.source, margins=True, margins_name="Tong").to_string(), "\n")

    rows = []
    for dim in PCA_DIMS:
        t0 = time.time()
        for run, seed in enumerate(SEEDS, start=1):
            X, y, idx, _, var_kept = chia_va_pca(meta, F, seed, dim)
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

    df = pd.DataFrame(rows)
    tb = (df.groupby(["Mô hình", "Số chiều"], sort=False)
          .agg(**{"PCA giữ phương sai": ("PCA giữ phương sai", "mean"), "Acc train TB": ("Acc train", "mean"),
                  "Acc val TB": ("Acc val", "mean"), "Acc val ĐLC": ("Acc val", "std"),
                  "Acc test TB": ("Acc test", "mean"), "Acc test ĐLC": ("Acc test", "std"),
                  "F1 test TB": ("F1 macro test", "mean"), "Thời gian TB (giây)": ("Thời gian (giây)", "mean")}).reset_index())
    df.to_csv(KQ / "thucnghiem01_pca.csv", index=False, encoding="utf-8-sig")
    tb.to_csv(KQ / "thucnghiem01_pca_tb.csv", index=False, encoding="utf-8-sig", float_format="%.4f")

    plt.figure(figsize=(7, 4))
    for name in MODEL_NAMES:
        mu = df[df["Mô hình"] == name].groupby("Số chiều")["Acc val"].mean()
        plt.plot(mu.index, mu.values, marker="o", label=name)
    plt.xscale("log", base=2); plt.xticks(PCA_DIMS, [str(d) for d in PCA_DIMS])
    plt.xlabel("Số chiều sau PCA (512 = không giảm chiều)"); plt.ylabel("Accuracy val (trung bình 5 lần)")
    plt.title("Khảo sát số chiều PCA"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(HINH / "hinh_4_1_khao_sat_pca.png", dpi=120); plt.close()

    print("\n" + tb.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nDa ghi: ketqua/thucnghiem01_pca.csv, ketqua/thucnghiem01_pca_tb.csv, hinh/hinh_4_1_khao_sat_pca.png")


if __name__ == "__main__":
    main()
