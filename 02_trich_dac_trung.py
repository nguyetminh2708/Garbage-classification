# -*- coding: utf-8 -*-
"""
Buoc 2: Chuan hoa anh va trich dac trung bang ResNet18 (huan luyen san tren ImageNet).

Moi anh di qua: xoay theo EXIF -> RGB (nen trong suot ghep len nen trang) -> dem thanh hinh vuong
bang mau trung binh ImageNet -> resize 224x224 (bicubic) -> chuan hoa theo mean/std ImageNet
-> ResNet18 bo lop phan loai cuoi -> vector 512 chieu.

Mang chi la bo trich dac trung: che do eval, khong cap nhat trong so, khong tang cuong du lieu.
Buoc nay chay MOT lan cho toan bo anh; cac thuc nghiem sau chi doc lai file npz.

Dau ra:
  vectors/features_512.npz           - feats (N x 512), path, label, source
  hinh/hinh_3_3_anh_truoc_sau.png     - moi lop mot anh, truoc va sau chuan hoa
  hinh/hinh_3_4a_pca_phuong_sai.png   - phuong sai tich luy theo so thanh phan chinh (minh hoa, fit tren 70% anh)
  hinh/hinh_3_4b_pca_2d.png           - du lieu tren 2 thanh phan chinh dau
Lan chay dau can mang de tai trong so ResNet18 (khoang 45 MB).
"""

import sys

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image, ImageOps
from torchvision import transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chung import CLASSES, KQ, HINH, VEC, N_COMPONENTS, chia_va_pca

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

IMG_SIZE = 224
BATCH_SIZE = 64
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
PAD_COLOR = tuple(int(m * 255) for m in MEAN)       # sau chuan hoa gan bang 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------ chuan hoa anh ------------------------------
def load_rgb(path: str) -> Image.Image:
    im = ImageOps.exif_transpose(Image.open(path))
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        return bg
    return im.convert("RGB")


def pad_to_square(im: Image.Image) -> Image.Image:
    w, h = im.size
    if w == h:
        return im
    s = max(w, h)
    canvas = Image.new("RGB", (s, s), PAD_COLOR)
    canvas.paste(im, ((s - w) // 2, (s - h) // 2))
    return canvas


to_tensor = transforms.Compose([
    transforms.Lambda(pad_to_square),
    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, paths): self.paths = list(paths)
    def __len__(self): return len(self.paths)
    def __getitem__(self, i): return to_tensor(load_rgb(self.paths[i]))


# ------------------------------ bo trich dac trung ------------------------------
def tao_backbone() -> torch.nn.Module:
    net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Identity()             # bo lop phan loai 1000 lop, giu dau ra 512 chieu
    return net.to(DEVICE).eval()


@torch.no_grad()
def trich(net: torch.nn.Module, paths) -> np.ndarray:
    dl = torch.utils.data.DataLoader(ImageDataset(paths), batch_size=BATCH_SIZE)
    out = []
    for i, x in enumerate(dl):
        out.append(net(x.to(DEVICE)).cpu().numpy())
        print(f"\r  batch {i + 1}/{len(dl)}", end="")
    print()
    return np.concatenate(out).astype(np.float32)


def main() -> None:
    f = KQ / "danh_sach_anh.csv"
    if not f.exists():
        sys.exit("Thieu ketqua/danh_sach_anh.csv. Chay python 01_thong_ke_du_lieu.py truoc.")
    df = pd.read_csv(f)
    print(f"{len(df)} anh | thiet bi {DEVICE} | torch {torch.__version__}")

    # hinh truoc / sau chuan hoa: moi lop mot anh
    def denorm(t): return (t.permute(1, 2, 0).numpy() * STD + MEAN).clip(0, 1)
    sample = df.groupby("label").sample(1, random_state=42).set_index("label").reindex(CLASSES)
    fig, axes = plt.subplots(2, len(CLASSES), figsize=(3 * len(CLASSES), 6.4))
    for j, (label, r) in enumerate(sample.iterrows()):
        axes[0, j].imshow(load_rgb(r.path)); axes[0, j].set_title(f"{label}\n{r.w}x{r.h} ({r.source})", fontsize=9)
        axes[1, j].imshow(denorm(to_tensor(load_rgb(r.path)))); axes[1, j].set_title(f"sau xử lý {IMG_SIZE}x{IMG_SIZE}", fontsize=9)
    for a in axes.ravel(): a.axis("off")
    plt.tight_layout(); plt.savefig(HINH / "hinh_3_3_anh_truoc_sau.png", dpi=120); plt.close()

    print("Dang trich dac trung bang ResNet18...")
    feats = trich(tao_backbone(), df.path)
    np.savez_compressed(VEC / "features_512.npz", feats=feats, path=df.path.to_numpy(),
                        label=df.label.to_numpy(), source=df.source.to_numpy())
    print(f"Ma tran dac trung: {feats.shape} -> vectors/features_512.npz")

    # minh hoa PCA tren mot lan chia (seed 42): phuong sai tich luy va 2 thanh phan dau
    meta = df[["path", "label", "source"]].copy(); meta["y"] = meta.label.map({c: i for i, c in enumerate(CLASSES)})
    X, y, _, prep, kept = chia_va_pca(meta, feats, 42, N_COMPONENTS)
    cum = np.cumsum(prep["pca"].explained_variance_ratio_)
    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(1, len(cum) + 1), cum)
    plt.axhline(kept, color="red", ls="--", label=f"{len(cum)} chiều giữ {kept:.1%}")
    plt.xlabel("Số thành phần chính"); plt.ylabel("Phương sai tích lũy được giữ lại")
    plt.title("PCA: phương sai giữ lại theo số chiều"); plt.ylim(0, 1.02); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(HINH / "hinh_3_4a_pca_phuong_sai.png", dpi=120); plt.close()

    Z = prep["pca"].transform(prep["scaler_512"].transform(feats))   # toa do PCA chua chuan hoa lan 2, chi de ve
    plt.figure(figsize=(7, 5.5))
    for i, c in enumerate(CLASSES):
        pts = Z[meta.y.to_numpy() == i]
        plt.scatter(pts[:, 0], pts[:, 1], s=12, alpha=0.6, label=c)
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("Dữ liệu trên 2 thành phần chính đầu"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(HINH / "hinh_3_4b_pca_2d.png", dpi=120); plt.close()
    print(f"PCA {N_COMPONENTS} chieu giu {kept:.1%} phuong sai. Da ghi hinh/hinh_3_3, hinh_3_4a, hinh_3_4b")


if __name__ == "__main__":
    main()
