# -*- coding: utf-8 -*-
"""
Buoc 1: Gom danh sach anh, can bang so anh moi lop, lam sach va thong ke du lieu.

Nguon anh (anh nam trong thu muc lop nao thi mang nhan do, script khong tu doi nhan):
  Data/anh_tu_chup/<lop>/   anh nhom tu chup
  Data/realwaste/<lop>/     anh RealWaste da xep thu cong vao 5 lop

Can bang: moi lop lay HET anh tu chup, phan con thieu toi TARGET_PER_CLASS bu bang anh RealWaste
chon ngau nhien (seed co dinh), lay deu giua cac loai con (Glass, Textile Trash, ...).

Lam sach: bo file khong doc duoc; bo anh trung noi dung (so ma bam MD5); xoay theo EXIF roi do kich thuoc.

Dau ra:
  ketqua/danh_sach_anh.csv          - danh sach anh dung cho cac buoc sau (path, label, source, orig_class, w, h, ...)
  ketqua/thong_ke_du_lieu.xlsx      - 6 sheet: Tong_quan, Loai_bo, Kich_thuoc, Dinh_dang, Loai_con, Danh_sach_anh
  hinh/hinh_3_1_phan_bo_lop.png     - so anh moi lop theo nguon
  hinh/hinh_3_2_phan_bo_kich_thuoc.png
"""

import hashlib
import random
import re
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chung import BASE, CLASSES, DATA_ANH, HINH, KQ

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
TARGET_PER_CLASS = 450      # muc can bang moi lop
MIN_SIDE_WARN = 224         # anh co canh nho hon muc nay se bi danh dau (phong to se mo)
SEED = 42
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".heic", ".heif"}


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS and not p.name.startswith("."))


def sample_even(groups: dict, n: int, rng: random.Random) -> list:
    """Lay n anh chia deu cho cac loai con; loai nao het thi phan du don sang loai khac."""
    pools = {k: rng.sample(v, len(v)) for k, v in groups.items()}
    picked = []
    while len(picked) < n and any(pools.values()):
        for k in list(pools):
            if pools[k] and len(picked) < n:
                picked.append((pools[k].pop(), k))
    return picked


def gom_danh_sach() -> tuple[list[dict], dict, dict]:
    """Tra ve (rows, so anh tu chup co san theo lop, so anh RealWaste co san theo lop)."""
    rng = random.Random(SEED)
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

    # lop nao nhieu hon muc can bang thi lay ngau nhien cho vua
    kept = []
    for c in CLASSES:
        mine = [r for r in rows if r["label"] == c]
        if len(mine) > TARGET_PER_CLASS:
            print(f"(luu y) lop {c} co {len(mine)} anh tu chup, chi lay ngau nhien {TARGET_PER_CLASS}")
            mine = rng.sample(mine, TARGET_PER_CLASS)
        kept += mine
    rows = kept
    own_count = {c: sum(1 for r in rows if r["label"] == c) for c in CLASSES}

    rw_avail = {}
    for c in CLASSES:
        d = RW_DIR / c
        groups = {}
        if d.is_dir():
            for p in list_images(d):
                orig = re.sub(r"[_ ]?\(?\d+\)?$", "", p.stem) or c   # "Glass_176" -> "Glass", chi de thong ke
                groups.setdefault(orig, []).append(p)
        rw_avail[c] = sum(len(v) for v in groups.values())
        need = max(TARGET_PER_CLASS - own_count[c], 0)
        if need > rw_avail[c]:
            print(f"(luu y) lop {c} can {need} anh RealWaste nhung chi co {rw_avail[c]}")
            need = rw_avail[c]
        rows += [{"path": str(p), "label": c, "source": "realwaste", "orig_class": k}
                 for p, k in sample_even(groups, need, rng)]
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


def rel(p: str) -> str:
    try:
        return str(Path(p).relative_to(DATA_ANH))
    except ValueError:
        return p


def main() -> None:
    rows, own_avail, rw_avail = gom_danh_sach()
    if not rows:
        sys.exit("Khong co anh nao. Kiem tra thu muc Data/anh_tu_chup va Data/realwaste.")
    df = pd.DataFrame(rows)
    print(f"Gom {len(df)} anh. Dang kiem tra tung anh...")
    info = pd.DataFrame([inspect(p) for p in df.path])
    df = pd.concat([df, info], axis=1)
    df["ext"] = df.path.str.rsplit(".", n=1).str[-1].str.lower()
    n_truoc = pd.crosstab(df.label, df.source).reindex(CLASSES).fillna(0).astype(int)

    # ------------------------------ lam sach ------------------------------
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

    # ------------------------------ thong ke ------------------------------
    n_sau = pd.crosstab(df.label, df.source).reindex(CLASSES).fillna(0).astype(int)
    tong_quan = pd.DataFrame({"Lớp": CLASSES})
    for src in ["tu_chup", "realwaste"]:
        tong_quan[f"{src}: trước làm sạch"] = n_truoc[src].to_numpy() if src in n_truoc else 0
        tong_quan[f"{src}: sau làm sạch"] = n_sau[src].to_numpy() if src in n_sau else 0
    tong_quan["Tổng sau làm sạch"] = n_sau.sum(axis=1).to_numpy()
    tong_quan["Ảnh tự chụp có sẵn"] = [own_avail[c] for c in CLASSES]
    tong_quan["Ảnh RealWaste có sẵn"] = [rw_avail[c] for c in CLASSES]
    tong_quan.loc[len(tong_quan)] = ["Tổng"] + [tong_quan[c].sum() for c in tong_quan.columns[1:]]

    kich_thuoc = (df.groupby(["label", "source"])
                  .agg(**{"Số ảnh": ("path", "size"), "Rộng min": ("w", "min"), "Rộng trung vị": ("w", "median"), "Rộng max": ("w", "max"),
                          "Cao min": ("h", "min"), "Cao trung vị": ("h", "median"), "Cao max": ("h", "max"),
                          "Tỉ lệ ảnh vuông": ("square", "mean"), f"Ảnh có cạnh < {MIN_SIDE_WARN}": ("small", "sum")})
                  .reset_index().rename(columns={"label": "Lớp", "source": "Nguồn"}))
    dinh_dang = pd.crosstab([df.source, df.ext], df.label).reindex(columns=CLASSES).fillna(0).astype(int).reset_index().rename(columns={"source": "Nguồn", "ext": "Định dạng"})
    che_do_mau = pd.crosstab(df.source, df["mode"]).reset_index().rename(columns={"source": "Nguồn"})
    loai_con = df.groupby(["label", "source", "orig_class"]).size().rename("Số ảnh").reset_index().rename(columns={"label": "Lớp", "source": "Nguồn", "orig_class": "Loại con"})
    danh_sach = df[["path", "label", "source", "orig_class", "w", "h", "ext", "mode"]].assign(path=df.path.map(rel)).rename(
        columns={"path": "Đường dẫn", "label": "Lớp", "source": "Nguồn", "orig_class": "Loại con", "w": "Rộng", "h": "Cao", "ext": "Định dạng", "mode": "Chế độ màu"})

    xlsx = KQ / "thong_ke_du_lieu.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        tong_quan.to_excel(w, sheet_name="Tong_quan", index=False)
        pd.DataFrame(loai_bo, columns=["Lý do", "Đường dẫn", "Lớp", "Nguồn", "Ghi chú"]).to_excel(w, sheet_name="Loai_bo", index=False)
        kich_thuoc.to_excel(w, sheet_name="Kich_thuoc", index=False)
        dinh_dang.to_excel(w, sheet_name="Dinh_dang", index=False)
        che_do_mau.to_excel(w, sheet_name="Dinh_dang", index=False, startrow=len(dinh_dang) + 3)
        loai_con.to_excel(w, sheet_name="Loai_con", index=False)
        danh_sach.to_excel(w, sheet_name="Danh_sach_anh", index=False)
    from chung import dinh_dang_excel
    dinh_dang_excel(xlsx)

    df.drop(columns=["ok", "md5", "err"]).to_csv(KQ / "danh_sach_anh.csv", index=False, encoding="utf-8-sig")

    # -------------------------------- hinh --------------------------------
    counts = pd.crosstab(df.label, df.source).reindex(CLASSES).fillna(0)
    ax = counts.plot.bar(stacked=True, figsize=(7, 4), rot=0)
    ax.set_xlabel(""); ax.set_ylabel("Số ảnh"); ax.set_title("Số ảnh mỗi lớp theo nguồn"); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(HINH / "hinh_3_1_phan_bo_lop.png", dpi=120); plt.close()

    plt.figure(figsize=(7, 4))
    for src, sub in df.groupby("source"):
        plt.hist(sub[["w", "h"]].min(axis=1), bins=30, alpha=0.7, label=f"{src} ({len(sub)} ảnh)")
    plt.axvline(MIN_SIDE_WARN, color="red", ls="--", label=f"ngưỡng {MIN_SIDE_WARN}px")
    plt.xlabel("Cạnh ngắn của ảnh (điểm ảnh)"); plt.ylabel("Số ảnh"); plt.title("Phân bố kích thước ảnh theo nguồn")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(HINH / "hinh_3_2_phan_bo_kich_thuoc.png", dpi=120); plt.close()

    print("\n" + tong_quan.to_string(index=False))
    print("\nDa ghi: ketqua/danh_sach_anh.csv, ketqua/thong_ke_du_lieu.xlsx, hinh/hinh_3_1, hinh_3_2")


if __name__ == "__main__":
    main()
