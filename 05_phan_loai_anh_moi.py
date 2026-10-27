# -*- coding: utf-8 -*-
"""
Buoc 5 (demo): Phan loai MOT hoac nhieu anh moi, khong co trong bo du lieu.

Anh moi di dung con duong ma 2.213 anh cu da di:
  chuan hoa anh (02) -> ResNet18 -> 512 chieu -> StandardScaler + PCA cua lan chay da chon (04)
  -> 3 mo hinh Softmax / SVM / XGBoost da huan luyen (04) -> du doan

Cach dung (chay sau 04_thucnghiem02_sieu_tham_so.py, can co models/mo_hinh_5_lan.joblib):
    python 05_phan_loai_anh_moi.py anh1.jpg anh2.heic        # mot hoac nhieu anh
    python 05_phan_loai_anh_moi.py thu_muc_anh/              # moi anh trong thu muc
    python 05_phan_loai_anh_moi.py anh.jpg --lan 3           # dung mo hinh cua lan chay 3 (mac dinh lan 1)

Script KHONG ghi de file nao, ket qua chi in ra man hinh.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import joblib
import numpy as np

from chung import BASE, CLASSES, MODEL, MODEL_NAMES, TEN_LOP

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".heic", ".heif"}


def nap_module(ten_file: str, ten: str):
    """Dung lai ham chuan hoa anh cua file 02 (ten file bat dau bang so nen khong import thuong duoc)."""
    spec = importlib.util.spec_from_file_location(ten, BASE / ten_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser(description="Phan loai anh rac moi bang 3 mo hinh da huan luyen")
    ap.add_argument("duong_dan", nargs="+", help="file anh hoac thu muc chua anh")
    ap.add_argument("--lan", type=int, default=1, help="dung bo tien xu ly va mo hinh cua lan chay nao (1 den 5)")
    args = ap.parse_args()

    f = MODEL / "mo_hinh_5_lan.joblib"
    if not f.exists():
        sys.exit("Thieu models/mo_hinh_5_lan.joblib. Chay python 04_thucnghiem02_sieu_tham_so.py truoc.")
    saved = joblib.load(f)
    run = saved["runs"][args.lan]
    prep = run["prep"]
    models = {("SVM" if k == "SVM RBF" else k): v for k, v in run["models"].items()}   # file cu dat ten "SVM RBF"

    paths = []
    for p in args.duong_dan:
        p = Path(p)
        paths += sorted(q for q in p.rglob("*") if q.suffix.lower() in IMG_EXTS) if p.is_dir() else [p]
    if not paths:
        sys.exit("Khong tim thay anh.")

    print("1. Chuan hoa anh va trich dac trung ResNet18 (giong file 02)")
    tx = nap_module("02_trich_dac_trung.py", "trich")
    feats = tx.trich(tx.tao_backbone(), [str(p) for p in paths])

    print(f"2. StandardScaler + PCA cua lan chay {args.lan}")
    Z = prep["scaler_512"].transform(feats)
    if prep["pca"] is not None:
        Z = prep["pca"].transform(Z)
    X = prep["scaler_pca"].transform(Z)

    print("3. Du doan bang 3 mo hinh\n")
    print(f"  {'Anh':34s}" + "".join(f"{n:>22s}" for n in MODEL_NAMES))
    for i, p in enumerate(paths):
        dong = []
        for name in MODEL_NAMES:
            m = models[name]
            k = int(m.predict(X[i:i + 1])[0])
            if hasattr(m, "predict_proba"):
                dong.append(f"{TEN_LOP[CLASSES[k]]} ({m.predict_proba(X[i:i + 1])[0][k]:.0%})")
            else:
                dong.append(TEN_LOP[CLASSES[k]])
        print(f"  {p.name[:34]:34s}" + "".join(f"{s:>22s}" for s in dong))
    print("\n(SVM khong co xac suat; so trong ngoac la do tin cua Softmax va XGBoost. "
          "Mo hinh tot nhat trong bao cao la SVM.)")


if __name__ == "__main__":
    main()
