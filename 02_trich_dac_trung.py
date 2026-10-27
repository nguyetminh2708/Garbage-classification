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

from chung import KQ, VEC, N_COMPONENTS
from xuat import ve_hinh_pca, ve_hinh_truoc_sau

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




# ------------------------------ chay ------------------------------
def main() -> None:
    f = KQ / "danh_sach_anh.csv"
    if not f.exists():
        sys.exit("Thieu ketqua/danh_sach_anh.csv. Chay python 01_thong_ke_du_lieu.py truoc.")
    df = pd.read_csv(f)
    print(f"{len(df)} anh | thiet bi {DEVICE} | torch {torch.__version__}")

    ve_hinh_truoc_sau(df, load_rgb, to_tensor, MEAN, STD, IMG_SIZE)

    print("Dang trich dac trung bang ResNet18...")
    feats = trich(tao_backbone(), df.path)
    np.savez_compressed(VEC / "features_512.npz", feats=feats, path=df.path.to_numpy(),
                        label=df.label.to_numpy(), source=df.source.to_numpy())
    print(f"Ma tran dac trung: {feats.shape} -> vectors/features_512.npz")

    kept = ve_hinh_pca(df, feats)
    print(f"PCA {N_COMPONENTS} chieu giu {kept:.1%} phuong sai. Da ghi hinh/hinh_3_3, hinh_3_4a, hinh_3_4b")


if __name__ == "__main__":
    main()
