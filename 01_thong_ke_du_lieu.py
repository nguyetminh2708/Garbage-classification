# -*- coding: utf-8 -*-
"""
Buoc 1: Gom danh sach anh, lam sach va thong ke du lieu.

Nguon anh (anh nam trong thu muc lop nao thi mang nhan do, script khong tu doi nhan):
  Data/anh_tu_chup/<lop>/   anh nhom tu chup
  Data/realwaste/<lop>/     anh RealWaste da xep thu cong vao 5 lop

Dung toan bo anh cua ca hai nguon: hai nguon xap xi 50/50, cac lop chenh nhau khong qua 20% nen khong can can bang.

Lam sach: bo file khong doc duoc; bo anh trung noi dung (so ma bam MD5); xoay theo EXIF roi do kich thuoc.

Dau ra:
  ketqua/danh_sach_anh.csv          - danh sach anh dung cho cac buoc sau (path, label, source, orig_class, w, h, ...)
  ketqua/thong_ke_du_lieu.xlsx      - 6 sheet: Tong_quan, Loai_bo, Kich_thuoc, Dinh_dang, Loai_con, Danh_sach_anh
  hinh/hinh_3_1_phan_bo_lop.png     - so anh moi lop theo nguon
  hinh/hinh_3_2_phan_bo_kich_thuoc.png
"""

import hashlib
import re
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

from chung import BASE, CLASSES, DATA_ANH
from xuat import rel, thong_ke_du_lieu, ve_hinh_du_lieu, xuat_thong_ke_du_lieu

try:
    from pillow_heif import register_heif_opener   # doc anh HEIC cua iPhone
    register_heif_opener()
except ImportError:
    print("(canh bao) thieu pillow-heif, anh .heic se khong doc duoc. Chay: pip install pillow-heif")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OWN_DIR, RW_DIR = DATA_ANH / "anh_tu_chup", DATA_ANH / "realwaste"
MIN_SIDE_WARN = 224         # anh co canh nho hon muc nay se bi danh dau (phong to se mo)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".heic", ".heif"}


# ============================================================================
# 1. Gom danh sach anh
# ============================================================================
def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS and not p.name.startswith("."))



def gom_danh_sach() -> tuple[list[dict], dict, dict]:
    """Gom moi anh trong Data/anh_tu_chup va Data/realwaste. Tra ve (rows, so anh tu chup theo lop, so anh RealWaste theo lop)."""
    rows = []
    for c in CLASSES:
        d = OWN_DIR / c
        if not d.is_dir():
            print(f"(canh bao) chua co thu muc {d.relative_to(BASE)}")
            continue
        for p in list_images(d):
            sub = p.relative_to(d).parts[0] if len(p.relative_to(d).parts) > 1 else c   # loai con = thu muc con (neu co)
            rows.append({"path": str(p), "label": c, "source": "tu_chup", "orig_class": sub})
    own_avail = {c: sum(1 for r in rows if r["label"] == c) for c in CLASSES}

    rw_avail = {}
    for c in CLASSES:
        d = RW_DIR / c
        n = 0
        if d.is_dir():
            for p in list_images(d):
                orig = re.sub(r"[_ ]?\(?\d+\)?$", "", p.stem) or c   # "Glass_176" -> "Glass", chi de thong ke loai con
                rows.append({"path": str(p), "label": c, "source": "realwaste", "orig_class": orig})
                n += 1
        rw_avail[c] = n
    return rows, own_avail, rw_avail


def inspect(path: str) -> dict:
    """Mo thu anh: ma bam MD5, kich thuoc sau khi xoay theo EXIF, che do mau."""
    try:
        md5 = hashlib.md5(Path(path).read_bytes()).hexdigest()
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im); im.load()
            return {"ok": True, "w": im.width, "h": im.height, "mode": im.mode, "md5": md5, "err": ""}
    except Exception as e:
        return {"ok": False, "w": None, "h": None, "mode": None, "md5": None, "err": str(e)[:80]}




# ============================================================================
# 2. Kiem tra va lam sach
# ============================================================================
def kiem_tra_va_lam_sach(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict], pd.DataFrame]:
    """Mo thu tung anh, bo file hong va anh trung. Tra ve (df sach, danh sach bi loai, so anh truoc lam sach)."""
    print(f"Gom {len(df)} anh. Dang kiem tra tung anh...")
    info = pd.DataFrame([inspect(p) for p in df.path])
    df = pd.concat([df, info], axis=1)
    df["ext"] = df.path.str.rsplit(".", n=1).str[-1].str.lower()
    n_truoc = pd.crosstab(df.label, df.source).reindex(CLASSES).fillna(0).astype(int)

    bad = df[~df.ok]
    loai_bo = [{"Lý do": "File hỏng / không đọc được", "Đường dẫn": rel(r.path), "Lớp": r.label, "Nguồn": r.source, "Ghi chú": r.err}
               for _, r in bad.iterrows()]
    df = df[df.ok].copy()
    dup = df.duplicated("md5", keep="first")
    first_of = df[~dup].set_index("md5").path
    loai_bo += [{"Lý do": "Trùng nội dung", "Đường dẫn": rel(r.path), "Lớp": r.label, "Nguồn": r.source, "Ghi chú": "trùng với " + rel(first_of[r.md5])}
                for _, r in df[dup].iterrows()]
    df = df[~dup].reset_index(drop=True)
    df["square"] = df.w == df.h
    df["small"] = df[["w", "h"]].min(axis=1) < MIN_SIDE_WARN
    print(f"Bo {len(bad)} file hong, {int(dup.sum())} anh trung. Con {len(df)} anh.")
    return df, loai_bo, n_truoc


# ============================================================================
# 3. Chay (thong ke, Excel, hinh: xem xuat.py)
# ============================================================================
def main() -> None:
    rows, own_avail, rw_avail = gom_danh_sach()
    if not rows:
        sys.exit("Khong co anh nao. Kiem tra thu muc Data/anh_tu_chup va Data/realwaste.")
    df, loai_bo, n_truoc = kiem_tra_va_lam_sach(pd.DataFrame(rows))
    tk = thong_ke_du_lieu(df, n_truoc, own_avail, rw_avail, MIN_SIDE_WARN)
    xuat_thong_ke_du_lieu(df, tk, loai_bo)
    ve_hinh_du_lieu(df, MIN_SIDE_WARN)
    print("\n" + tk["tong_quan"].to_string(index=False))
    print("\nDa ghi: ketqua/danh_sach_anh.csv, ketqua/thong_ke_du_lieu.xlsx, hinh/hinh_3_1, hinh_3_2")


if __name__ == "__main__":
    main()
