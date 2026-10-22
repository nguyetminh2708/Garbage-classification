# -*- coding: utf-8 -*-
"""
Phân loại rác: tiền xử lý ảnh và trích xuất đặc trưng.

Cách chạy (mở Terminal trong thư mục ML):
    python tien_xu_ly_anh.py

Kết quả lưu vào thư mục features/. Sửa đường dẫn trong phần CẤU HÌNH nếu cần.
"""


# ============================================================================
# Phân loại rác: Tiền xử lý ảnh và trích xuất đặc trưng
#
# Script này biến mỗi ảnh thành một vector 128 đặc trưng để dùng cho các thuật toán ML cơ bản ở bước sau.
#
# Quy trình:
#
# 1. Gom ảnh từ RealWaste (tự tải) và ảnh nhóm tự chụp (nếu có), gộp về 5 lớp của nhóm: giay, nhua, huu_co, kim_loai, khac (pin, thủy tinh, gốm, ...).
# 2. Kiểm tra ảnh: bỏ file hỏng, thống kê kích thước, cảnh báo ảnh quá nhỏ.
# 3. Chia train / val / test (phân tầng theo lớp và nguồn ảnh).
# 4. Chuẩn hóa ảnh: xoay đúng chiều theo EXIF, đưa về RGB, đệm thành hình vuông, resize 224x224, chuẩn hóa theo ImageNet.
# 5. Dùng ResNet18 đã huấn luyện sẵn (bỏ lớp phân loại cuối) để lấy vector 512 chiều. Deep learning chỉ dùng ở bước này.
# 6. Chuẩn hóa (StandardScaler) rồi PCA xuống 128 chiều. Scaler và PCA chỉ fit trên tập train để tránh rò rỉ dữ liệu.
# 7. Lưu kết quả vào thư mục features.
#
# Cách chạy:
#
# 1. Cài Python 3.10 trở lên, rồi mở Terminal trong thư mục ML và cài thư viện (chỉ cần làm 1 lần):
#    pip install -r requirements.txt
# 2. Chạy: python tien_xu_ly_anh.py
# 3. Lần chạy đầu cần mạng để tải trọng số ResNet18 (khoảng 45 MB), các lần sau không cần.
#
# Máy có card NVIDIA thì PyTorch tự dùng GPU; không có thì chạy bằng CPU, 2.000 ảnh mất khoảng vài phút.
# ============================================================================

# ============================================================================
# 1. Cấu hình
#
# Ảnh nhóm tự chụp đặt trong thư mục Data theo cấu trúc dưới đây (tên thư mục con phải đúng như vậy; thiếu thư mục nào thì script bỏ qua lớp đó):
#
#
# ML/Data/anh_tu_chup/
#     giay/       *.jpg, *.png, *.heic, ...
#     nhua/
#     huu_co/
#     kim_loai/
#     khac/
#         pin/        (được phép chia thư mục con theo loại, không bắt buộc)
#         thuy_tinh/
#         gom/
#
#
# Lớp khac gồm mọi loại rác không thuộc 4 lớp còn lại: pin, thủy tinh, gốm, vải, ... Loại con được ghi lại trong cột orig_class để phân tích lỗi sau này.
#
# Chưa có ảnh tự chụp thì cứ chạy, script sẽ chỉ dùng RealWaste.
# ============================================================================
# ===== CẤU HÌNH: chỉ cần sửa ở đây =====
import os as _os
try:
    BASE_DIR = _os.path.dirname(_os.path.abspath(__file__)).replace("\\", "/")   # thư mục chứa file script này
except NameError:
    BASE_DIR = _os.getcwd().replace("\\", "/")
DATA_DIR      = f"{BASE_DIR}/Data"           # thư mục dữ liệu tải từ Drive, gồm anh_tu_chup/ và realwaste/
OWN_DATA_DIR  = f"{DATA_DIR}/anh_tu_chup"    # ảnh nhóm tự chụp, 5 thư mục con giay, nhua, huu_co, kim_loai, khac
OUT_DIR       = f"{BASE_DIR}/features"       # nơi lưu kết quả
REALWASTE_DIR = f"{DATA_DIR}/realwaste"      # RealWaste đã xếp theo 5 lớp; nếu chưa có và không có link Drive, script tự tải từ UCI

# ===== Tải dữ liệu từ Google Drive (tùy chọn) =====
# Dán link chia sẻ của file zip trên Drive. Link phải mở quyền "Bất kỳ ai có đường liên kết đều xem được".
# Cũng có thể dán đường dẫn tới file zip có sẵn trên máy.
# Để trống "" thì dùng thư mục trên máy như cũ (OWN_DATA_DIR, REALWASTE_DIR ở trên).
# Ảnh tải về được giải nén vào thư mục _drive/, không đụng tới thư mục Data trên máy.
# Khi nhóm cập nhật bộ ảnh: tải file zip mới lên Drive rồi dán link mới vào đây, script sẽ tự tải lại.
DRIVE_ZIP_TU_CHUP   = ""   # link tới anh_tu_chup.zip
DRIVE_ZIP_REALWASTE = ""   # link tới realwaste.zip

USE_REALWASTE = True
# Cách lấy ảnh RealWaste cho mỗi lớp:
#   "bu_du"    : bù cho mỗi lớp đủ TARGET_PER_CLASS ảnh (ảnh tự chụp + RealWaste). Các lớp có tổng bằng nhau.
#   "mot_mot"  : số ảnh RealWaste bằng số ảnh tự chụp của lớp đó (mỗi lớp 50% / 50%).
#   "co_dinh"  : cố định N_REALWASTE_PER_CLASS ảnh mỗi lớp.
REALWASTE_MODE = "bu_du"
TARGET_PER_CLASS = 400        # dùng cho "bu_du": 5 lớp x 400 = 2.000 ảnh
N_REALWASTE_PER_CLASS = 200   # dùng cho "co_dinh"
EVEN_SUBFOLDER = True         # trong một lớp, lấy đều từ các thư mục RealWaste (vd Khác: Misc, Glass, Textile mỗi loại 1/3)
USE_OWN_DATA  = True

IMG_SIZE     = 224    # kích thước đầu vào của ResNet18
MIN_SIDE_WARN = 224   # ảnh có cạnh nhỏ hơn mức này sẽ bị cảnh báo (phóng to sẽ mờ)
N_COMPONENTS = 128    # số chiều sau PCA
VAL_SIZE, TEST_SIZE = 0.15, 0.15
SEED = 42
BATCH_SIZE = 64
PRETRAINED = True     # luôn để True khi chạy thật

CLASSES = ["giay", "nhua", "huu_co", "kim_loai", "khac"]

# Lớp "khac" = mọi loại rác không thuộc 4 lớp còn lại (pin, thủy tinh, gốm, vải, ...)
# RealWaste có 9 thư mục, gộp về 5 lớp như sau:
REALWASTE_MAP = {
    "Paper": "giay",
    "Cardboard": "giay",            # bìa carton tính là giấy
    "Plastic": "nhua",
    "Food Organics": "huu_co",
    "Vegetation": "huu_co",
    "Metal": "kim_loai",
    "Miscellaneous Trash": "khac",
    "Glass": "khac",                # thủy tinh thuộc nhóm khác
    "Textile Trash": "huu_co",      # vải xếp vào hữu cơ theo quy ước của nhóm (có thể phân hủy khi chôn, đốt)
}

# Ảnh tự chụp: nếu để loại con thành thư mục riêng ngang hàng 5 lớp, khai báo ở đây để gộp vào lớp lớn.
# (Cách khác: đặt thẳng vào khac/pin/, khac/thuy_tinh/, ... script cũng tự đọc được.)
OWN_ALIAS = {
    "pin": "khac",
    "thuy_tinh": "khac",
    "gom": "khac",
    "carton": "giay",
}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".heic", ".heif"}


# ============================================================================
# 2. Thư viện
# ============================================================================
import os, sys, glob, random, warnings
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules

import matplotlib
matplotlib.use("Agg")   # lưu hình ra file, không mở cửa sổ
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import torch, torchvision
from torchvision import transforms
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUT_DIR, exist_ok=True)
print("Thiết bị:", DEVICE, "| torch", torch.__version__)


# ============================================================================
# 3. RealWaste
#
# RealWaste (UCI, giấy phép CC BY 4.0): 4.752 ảnh rác thật, 524x524. Cần ghi nguồn trong báo cáo:
# Single, S., Iranmanesh, S., & Raad, R. (2023). *RealWaste: A Novel Real-Life Data Set for Landfill Waste Classification Using Deep Learning.* Information, 14(12), 633.
#
# RealWaste có 9 lớp, được gộp về 5 lớp của nhóm:
#
# | Lớp của nhóm | Thư mục RealWaste |
# | --- | --- |
# | giay | Paper, Cardboard |
# | nhua | Plastic |
# | huu_co | Food Organics, Vegetation |
# | kim_loai | Metal |
# | khac | Miscellaneous Trash, Glass, Textile Trash |
# ============================================================================
def fetch_zip(link, dest_dir, name):
    """Tải file zip từ Google Drive (hoặc lấy file zip trên máy) rồi giải nén vào dest_dir.
    Nếu dest_dir đã chứa đúng bản của link này thì bỏ qua, không tải lại."""
    import shutil, zipfile
    marker = Path(dest_dir) / ".nguon_tai.txt"
    if marker.exists() and marker.read_text(encoding="utf-8").strip() == link:
        print(f"{name}: đã có bản tải từ link này, bỏ qua bước tải.")
        return
    os.makedirs(f"{BASE_DIR}/_drive", exist_ok=True)
    zip_path = f"{BASE_DIR}/_drive/{name}.zip"
    if os.path.isfile(link):
        shutil.copy(link, zip_path)
    else:
        try:
            import gdown
        except ImportError:
            raise SystemExit("Thiếu thư viện gdown. Chạy: pip install gdown")
        print(f"Đang tải {name} từ Google Drive...")
        if gdown.download(link, zip_path, quiet=False, fuzzy=True) is None:
            raise SystemExit(f"Không tải được {name}. Kiểm tra link đã mở quyền 'Bất kỳ ai có đường liên kết đều xem được'.")
    if os.path.isdir(dest_dir):
        shutil.rmtree(dest_dir)          # chỉ xóa thư mục trong _drive/, do script tự quản lý
    print(f"Đang giải nén {name}...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest_dir)
    os.remove(zip_path)
    marker.write_text(link, encoding="utf-8")

def find_own_root(base):
    """Tìm thư mục chứa các thư mục lớp (giay, nhua, ...) dù file zip lồng mấy cấp."""
    names = set(CLASSES) | set(OWN_ALIAS)
    for root, dirs, _ in sorted(os.walk(base), key=lambda t: t[0].count(os.sep)):
        if names & set(dirs):
            return root
    return None

if DRIVE_ZIP_TU_CHUP:
    fetch_zip(DRIVE_ZIP_TU_CHUP, f"{BASE_DIR}/_drive/data_tu_chup", "data_tu_chup")
    OWN_DATA_DIR = find_own_root(f"{BASE_DIR}/_drive/data_tu_chup")
    assert OWN_DATA_DIR, "Trong file zip ảnh tự chụp không thấy các thư mục giay, nhua, huu_co, kim_loai, khac."
    print("Ảnh tự chụp:", OWN_DATA_DIR)
if DRIVE_ZIP_REALWASTE:
    REALWASTE_DIR = f"{BASE_DIR}/_drive/realwaste"
    fetch_zip(DRIVE_ZIP_REALWASTE, REALWASTE_DIR, "realwaste")

def find_realwaste_root(base):
    """Tìm thư mục RealWaste, dù zip giải nén lồng mấy cấp. Chấp nhận 2 kiểu sắp xếp:
    - "goc": 9 thư mục gốc của RealWaste (Paper, Metal, ...), gộp về 5 lớp theo REALWASTE_MAP
    - "theo_lop": đã xếp sẵn vào 5 thư mục giay, nhua, huu_co, kim_loai, khac; ảnh ở thư mục nào thuộc lớp đó"""
    hits = glob.glob(f"{base}/**/Food Organics", recursive=True)
    if hits:
        return str(Path(hits[0]).parent), "goc"
    if os.path.isdir(base):
        for root, dirs, _ in sorted(os.walk(base), key=lambda t: t[0].count(os.sep)):
            if len(set(CLASSES) & set(dirs)) >= 3:
                return root, "theo_lop"
    return None, None

if USE_REALWASTE:
    rw_root, rw_layout = find_realwaste_root(REALWASTE_DIR)
    if rw_root is None:
        os.makedirs(REALWASTE_DIR, exist_ok=True)
        import urllib.request, zipfile
        zip_path = f"{REALWASTE_DIR}/realwaste.zip"
        print("Chưa có RealWaste, đang tải (khoảng 650 MB)...")
        urllib.request.urlretrieve("https://archive.ics.uci.edu/static/public/908/realwaste.zip", zip_path)
        zipfile.ZipFile(zip_path).extractall(REALWASTE_DIR)
        # zip của UCI có thể chứa thêm 1 file zip bên trong
        for z in glob.glob(f"{REALWASTE_DIR}/**/*.zip", recursive=True):
            if os.path.abspath(z) != os.path.abspath(zip_path):
                zipfile.ZipFile(z).extractall(REALWASTE_DIR)
        rw_root, rw_layout = find_realwaste_root(REALWASTE_DIR)
    assert rw_root, "Không tìm thấy RealWaste sau khi giải nén, kiểm tra lại link tải."
    print("RealWaste:", rw_root, "| kiểu thư mục:", rw_layout)
    print(sorted(os.listdir(rw_root)))


# ============================================================================
# 4. Gom danh sách ảnh
# ============================================================================
def list_images(folder):
    return sorted(p for p in Path(folder).rglob("*")
                  if p.is_file() and p.suffix.lower() in IMG_EXTS and not p.name.startswith("."))

rows = []

# --- Ảnh tự chụp (đọc trước để biết mỗi lớp cần bù bao nhiêu ảnh RealWaste) ---
if USE_OWN_DATA:
    if not os.path.isdir(OWN_DATA_DIR):
        warnings.warn(f"Chưa có thư mục ảnh tự chụp: {OWN_DATA_DIR}. Tạm thời chỉ dùng RealWaste.")
    else:
        for d in sorted(Path(OWN_DATA_DIR).iterdir()):
            if not d.is_dir() or d.name.startswith("."):
                continue
            label = d.name if d.name in CLASSES else OWN_ALIAS.get(d.name)
            if label is None:
                warnings.warn(f"Thư mục '{d.name}' không thuộc 5 lớp và không có trong OWN_ALIAS nên bị bỏ qua.")
                continue
            for p in list_images(d):
                # loại con: thư mục con bên trong lớp (vd khac/pin/a.jpg -> pin), hoặc tên alias
                sub = p.relative_to(d).parts[0] if len(p.relative_to(d).parts) > 1 else d.name
                rows.append({"path": str(p), "label": label, "source": "tu_chup", "orig_class": sub})
        missing = [c for c in CLASSES if c not in {r["label"] for r in rows if r["source"] == "tu_chup"}]
        if missing:
            warnings.warn(f"Ảnh tự chụp chưa có lớp: {missing}")


rng = random.Random(SEED)
own_count = {c: sum(1 for r in rows if r["label"] == c) for c in CLASSES}

# Ở chế độ "bu_du", lớp nào có nhiều hơn TARGET_PER_CLASS ảnh tự chụp thì chọn ngẫu nhiên cho vừa
if REALWASTE_MODE == "bu_du":
    kept = []
    for c in CLASSES:
        mine = [r for r in rows if r["label"] == c]
        if len(mine) > TARGET_PER_CLASS:
            warnings.warn(f"Lớp {c} có {len(mine)} ảnh tự chụp, chỉ lấy ngẫu nhiên {TARGET_PER_CLASS}.")
            mine = rng.sample(mine, TARGET_PER_CLASS)
        kept += mine
    rows = kept
    own_count = {c: sum(1 for r in rows if r["label"] == c) for c in CLASSES}

def sample_even(groups, n):
    """Lấy n ảnh, chia đều cho các thư mục con; thư mục nào thiếu thì phần dư chuyển sang thư mục khác."""
    pools = {k: rng.sample(v, len(v)) for k, v in groups.items()}
    picked = []
    while len(picked) < n and any(pools.values()):
        for k in list(pools):
            if pools[k] and len(picked) < n:
                picked.append((pools[k].pop(), k))
    return picked

# --- Ảnh RealWaste ---
if USE_REALWASTE:
    groups = {c: {} for c in CLASSES}
    if rw_layout == "goc":
        for rw_name, label in REALWASTE_MAP.items():
            groups[label][rw_name] = list_images(Path(rw_root) / rw_name)
    else:
        # Thư mục đã gộp theo lớp: ảnh nằm trong thư mục nào thì thuộc lớp đó (nhóm tự xếp, script không đổi).
        # Tên file chỉ dùng để ghi loại gốc (vd "Food Organics_12.jpg" -> "Food Organics") cho phần phân tích lỗi.
        import re
        for label in CLASSES:
            d = Path(rw_root) / label
            if not d.is_dir():
                warnings.warn(f"RealWaste chưa có thư mục {label}.")
                continue
            for p in list_images(d):
                orig = re.sub(r"[_ ]?\(?\d+\)?$", "", p.stem) or label
                groups[label].setdefault(orig, []).append(p)
    for label in CLASSES:
        if REALWASTE_MODE == "bu_du":
            need = max(TARGET_PER_CLASS - own_count[label], 0)
        elif REALWASTE_MODE == "mot_mot":
            need = own_count[label]
        else:
            need = N_REALWASTE_PER_CLASS
        avail = sum(len(v) for v in groups[label].values())
        if need > avail:
            warnings.warn(f"Lớp {label} cần {need} ảnh RealWaste nhưng chỉ có {avail}.")
            need = avail
        if EVEN_SUBFOLDER:
            items = sample_even(groups[label], need)
        else:
            flat = [(p, k) for k, v in groups[label].items() for p in v]
            items = rng.sample(flat, need)
        rows += [{"path": str(p), "label": label, "source": "realwaste", "orig_class": rw} for p, rw in items]

df = pd.DataFrame(rows)
assert len(df), "Không có ảnh nào. Kiểm tra lại đường dẫn ở ô Cấu hình."
print((pd.crosstab(df.label, df.source, margins=True, margins_name="Tổng").reindex(CLASSES + ["Tổng"])).to_string())


# ============================================================================
# 5. Kiểm tra ảnh
#
# Mở thử từng ảnh để:
#
# - Bỏ file hỏng hoặc không đọc được.
# - Xoay đúng chiều theo EXIF (ảnh điện thoại hay bị lưu nằm ngang).
# - Thống kê kích thước. Ảnh vuông nhưng kích thước khác nhau không ảnh hưởng gì, vì bước sau resize tất cả về 224x224.
# - Cảnh báo ảnh có cạnh nhỏ hơn 224 (phóng to lên sẽ mờ, nên chụp lại nếu nhiều).
# - Phát hiện ảnh trùng lặp (cùng nội dung file), tránh một ảnh vừa ở train vừa ở test.
# ============================================================================
import hashlib

def inspect(path):
    try:
        with open(path, "rb") as f:
            md5 = hashlib.md5(f.read()).hexdigest()
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            im.load()
            return {"ok": True, "w": im.width, "h": im.height, "mode": im.mode, "md5": md5}
    except Exception as e:
        return {"ok": False, "w": None, "h": None, "mode": None, "md5": None, "err": str(e)[:80]}

info = pd.DataFrame([inspect(p) for p in df.path])
df = pd.concat([df.reset_index(drop=True), info], axis=1)

bad = df[~df.ok]
if len(bad):
    print(f"Bỏ {len(bad)} file không đọc được:"); print(bad[["path", "err"]].to_string())
df = df[df.ok].copy()

dup = df.duplicated("md5", keep="first")
if dup.any():
    print(f"Bỏ {dup.sum()} ảnh trùng lặp nội dung.")
df = df[~dup].reset_index(drop=True)

df["square"] = df.w == df.h
df["small"] = df[["w", "h"]].min(axis=1) < MIN_SIDE_WARN
print(f"Còn {len(df)} ảnh hợp lệ.")
print("\nKích thước ảnh theo nguồn:")
print(df.groupby("source").agg(so_anh=("path", "size"),
                               rong_min=("w", "min"), rong_max=("w", "max"),
                               cao_min=("h", "min"), cao_max=("h", "max"),
                               ti_le_vuong=("square", "mean"), anh_nho=("small", "sum")).to_string())
print("\nLoại con trong từng lớp:")
print(df.groupby(["label", "source"]).orig_class.value_counts().to_string())
print("\nChế độ màu:", df["mode"].value_counts().to_dict())
if df.small.any():
    warnings.warn(f"{df.small.sum()} ảnh có cạnh < {MIN_SIDE_WARN}px. Xem danh sách: df[df.small]")


# ============================================================================
# 6. Chia train / val / test
#
# Chia theo tỉ lệ 70 / 15 / 15, phân tầng theo cặp (lớp, nguồn). Nhờ vậy mỗi tập có tỉ lệ ảnh RealWaste và ảnh tự chụp như nhau ở mọi lớp, mô hình không thể "ăn gian" bằng cách nhận ra nguồn ảnh.
# Phải chia trước khi fit scaler và PCA.
# ============================================================================
strat = df.label + "|" + df.source
idx_train, idx_tmp = train_test_split(df.index, test_size=VAL_SIZE + TEST_SIZE,
                                      stratify=strat, random_state=SEED)
idx_val, idx_test = train_test_split(idx_tmp, test_size=TEST_SIZE / (VAL_SIZE + TEST_SIZE),
                                     stratify=strat[idx_tmp], random_state=SEED)
df["split"] = "train"
df.loc[idx_val, "split"] = "val"
df.loc[idx_test, "split"] = "test"
print((pd.crosstab([df.split, df.source], df.label)[CLASSES]).to_string())


# ============================================================================
# 7. Chuẩn hóa ảnh
#
# Mỗi ảnh đi qua các bước:
#
# | Bước | Lý do |
# | --- | --- |
# | Xoay theo EXIF | Ảnh điện thoại thường lưu chiều xoay trong metadata |
# | Đưa về RGB (ảnh có nền trong suốt thì ghép lên nền trắng) | CNN cần đúng 3 kênh màu |
# | Đệm thành hình vuông bằng màu trung bình ImageNet | Ảnh không vuông không bị méo, không bị cắt mất vật. Ảnh đã vuông thì bỏ qua bước này |
# | Resize về 224x224 | Kích thước đầu vào của ResNet18; ảnh vuông kích thước nào cũng về cùng một cỡ |
# | Chuyển về [0, 1] rồi chuẩn hóa theo mean/std ImageNet | Khớp với cách ResNet18 được huấn luyện |
#
# Không dùng tăng cường dữ liệu (lật, xoay ngẫu nhiên) ở đây vì mục tiêu là lấy đặc trưng ổn định cho mỗi ảnh.
# ============================================================================
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
PAD_COLOR = tuple(int(m * 255) for m in MEAN)  # sau chuẩn hóa sẽ gần bằng 0

def load_rgb(path):
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        return bg
    return im.convert("RGB")

def pad_to_square(im):
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


# ============================================================================
# Xem thử ảnh trước và sau chuẩn hóa
# ============================================================================
def denorm(t):
    return (t.permute(1, 2, 0).numpy() * STD + MEAN).clip(0, 1)

sample = df.groupby("label").sample(1, random_state=SEED).set_index("label").reindex([c for c in CLASSES if c in set(df.label)])
fig, axes = plt.subplots(2, len(CLASSES), figsize=(3 * len(CLASSES), 6.4))
for j, (label, r) in enumerate(sample.iterrows()):
    axes[0, j].imshow(load_rgb(r.path)); axes[0, j].set_title(f"{label}\n{r.w}x{r.h} ({r.source})", fontsize=9)
    axes[1, j].imshow(denorm(to_tensor(load_rgb(r.path)))); axes[1, j].set_title(f"sau xử lý {IMG_SIZE}x{IMG_SIZE}", fontsize=9)
for a in axes.ravel(): a.axis("off")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/anh_truoc_sau_xu_ly.png", dpi=120); plt.close()


# ============================================================================
# Phân bố số ảnh theo lớp
# ============================================================================
counts = pd.crosstab(df.label, df.source).reindex(CLASSES).fillna(0)
ax = counts.plot.bar(stacked=True, figsize=(7, 4), color=["#444444", "#bbbbbb"][:counts.shape[1]], edgecolor="black", width=0.7)
ax.set_xlabel(""); ax.set_ylabel("Số ảnh"); ax.set_title("Số ảnh mỗi lớp theo nguồn")
ax.spines[["top", "right"]].set_visible(False); plt.xticks(rotation=0)
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/phan_bo_lop.png", dpi=120); plt.close()


# ============================================================================
# 8. Trích xuất đặc trưng bằng ResNet18
#
# ResNet18 đã huấn luyện trên ImageNet. Bỏ lớp fc cuối cùng, đầu ra của lớp average pooling là vector 512 chiều mô tả nội dung ảnh.
# Mạng ở chế độ eval() và không cập nhật trọng số: đây chỉ là bộ trích đặc trưng cố định, không phải mô hình phân loại.
# ============================================================================
weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if PRETRAINED else None
backbone = torchvision.models.resnet18(weights=weights)
backbone.fc = torch.nn.Identity()
backbone = backbone.to(DEVICE).eval()

@torch.no_grad()
def extract(paths):
    dl = torch.utils.data.DataLoader(ImageDataset(paths), batch_size=BATCH_SIZE,
                                     num_workers=2 if IN_COLAB else 0)
    out = []
    for i, x in enumerate(dl):
        out.append(backbone(x.to(DEVICE)).cpu().numpy())
        print(f"\r  batch {i + 1}/{len(dl)}", end="")
    print()
    return np.concatenate(out).astype(np.float32)

feats_512 = extract(df.path)
print("Kích thước ma trận đặc trưng:", feats_512.shape)


# ============================================================================
# 9. Chuẩn hóa và PCA xuống 128 chiều
#
# - StandardScaler: đưa mỗi đặc trưng về trung bình 0, độ lệch chuẩn 1 (quan trọng với SVM, KNN, Logistic Regression).
# - PCA: giảm từ 512 xuống 128 chiều, giữ phần lớn phương sai, giúp mô hình ML huấn luyện nhanh hơn và ít overfit hơn.
# - Cả hai chỉ fit trên tập train, rồi áp dụng (transform) cho val và test.
# ============================================================================
y_all = df.label.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
m = {s: (df.split == s).to_numpy() for s in ["train", "val", "test"]}

n_comp = min(N_COMPONENTS, m["train"].sum(), feats_512.shape[1])
if n_comp < N_COMPONENTS:
    warnings.warn(f"Tập train chỉ có {m['train'].sum()} ảnh nên PCA tối đa {n_comp} chiều.")

scaler = StandardScaler().fit(feats_512[m["train"]])
pca = PCA(n_components=n_comp, random_state=SEED).fit(scaler.transform(feats_512[m["train"]]))
X = {s: pca.transform(scaler.transform(feats_512[m[s]])).astype(np.float32) for s in m}
y = {s: y_all[m[s]] for s in m}

kept = pca.explained_variance_ratio_.sum()
print(f"PCA {feats_512.shape[1]} -> {n_comp} chiều, giữ lại {kept:.1%} phương sai.")
for s in m: print(f"  X_{s}: {X[s].shape}")

cum = np.cumsum(pca.explained_variance_ratio_)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(np.arange(1, n_comp + 1), cum, color="black", lw=2)
ax.axhline(kept, color="black", ls="--", lw=1)
ax.text(n_comp, kept, f" {kept:.1%}", va="bottom", ha="right", color="black")
ax.set_xlabel("Số thành phần chính"); ax.set_ylabel("Phương sai tích lũy được giữ lại")
ax.set_title("PCA: phương sai giữ lại theo số chiều"); ax.set_ylim(0, 1.02)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/pca_phuong_sai.png", dpi=120); plt.close()


# ============================================================================
# Kiểm tra nhanh: 2 thành phần chính đầu tiên
#
# Nếu các lớp đã tách nhau phần nào ngay trên 2 chiều, đặc trưng từ ResNet18 là hữu ích cho bài toán.
# ============================================================================
palette = ["black"] * 5
markers = ["o", "s", "^", "D", "x"]   # trắng đen: phân biệt lớp bằng hình dạng điểm
fig, ax = plt.subplots(figsize=(7, 5.5))
for i, c in enumerate(CLASSES):
    pts = X["train"][y["train"] == i]
    ax.scatter(pts[:, 0], pts[:, 1], s=14, alpha=0.55, color=palette[i], marker=markers[i], label=c)
ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title("Tập train trên 2 thành phần chính đầu")
ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/pca_2d.png", dpi=120); plt.close()


# ============================================================================
# 10. Lưu kết quả
#
# | File | Nội dung |
# | --- | --- |
# | features_128.npz | X_train, y_train, X_val, y_val, X_test, y_test, classes: dùng trực tiếp cho bước huấn luyện ML |
# | features_512_raw.npz | Đặc trưng 512 chiều gốc chưa PCA, để đổi số chiều PCA mà không phải chạy lại ResNet18 |
# | metadata.csv | Đường dẫn, lớp, nguồn, tập (train/val/test), kích thước từng ảnh |
# | preprocess.joblib | Scaler và PCA đã fit, dùng để xử lý ảnh mới khi demo |
# | *.png | Các hình để đưa vào báo cáo |
# ============================================================================
np.savez_compressed(f"{OUT_DIR}/features_128.npz",
                    X_train=X["train"], y_train=y["train"],
                    X_val=X["val"], y_val=y["val"],
                    X_test=X["test"], y_test=y["test"],
                    classes=np.array(CLASSES))
np.savez_compressed(f"{OUT_DIR}/features_512_raw.npz", feats=feats_512, y=y_all, split=df.split.to_numpy())
df.drop(columns=["ok", "md5"] + [c for c in ["err"] if c in df]).to_csv(f"{OUT_DIR}/metadata.csv", index=False)
joblib.dump({"scaler": scaler, "pca": pca, "classes": CLASSES, "img_size": IMG_SIZE}, f"{OUT_DIR}/preprocess.joblib")
print("Đã lưu vào", OUT_DIR)
for f in sorted(os.listdir(OUT_DIR)): print("  ", f)


# ============================================================================
# 11. Dùng ở bước tiếp theo
#
# Muốn thử nhanh một mô hình thì đọc file features_128.npz (đã chia sẵn 1 lần). Script huấn luyện chính dùng features_512_raw.npz và metadata.csv, rồi PCA lại trong từng lần chạy:
#
# python
# import numpy as np
# d = np.load("features/features_128.npz")
# X_train, y_train = d["X_train"], d["y_train"]
# X_val, y_val     = d["X_val"], d["y_val"]
# X_test, y_test   = d["X_test"], d["y_test"]
# classes = list(d["classes"])
#
#
# Dự đoán một ảnh mới (khi demo):
#
# python
# pp = joblib.load(f"{OUT_DIR}/preprocess.joblib")
# v = backbone(to_tensor(load_rgb("anh_moi.jpg")).unsqueeze(0).to(DEVICE)).detach().cpu().numpy()
# v128 = pp["pca"].transform(pp["scaler"].transform(v))
# model.predict(v128)
#
# ============================================================================
