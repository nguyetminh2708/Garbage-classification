# Phân loại rác thải từ ảnh bằng các thuật toán học máy cổ điển

Tiểu luận môn Học máy. Ảnh rác được đưa qua ResNet18 (huấn luyện sẵn, giữ cố định) để lấy vector đặc trưng 512 chiều, sau đó phân loại bằng 3 thuật toán học máy cổ điển (Softmax Regression, SVM, XGBoost) vào 5 nhóm: giấy, nhựa, hữu cơ, kim loại, khác.

## Cài đặt (làm một lần)

1. Cài Python 3.10 trở lên.
2. Mở Terminal trong thư mục này, chạy: `pip install -r requirements.txt`

## Dữ liệu

Dữ liệu không nằm trong repo. Tải thư mục Machine Learning trên Google Drive của nhóm về, giải nén rồi đặt vào thư mục `Data` sao cho có cấu trúc:

```
Data/
    anh_tu_chup/   giay/  nhua/  huu_co/  kim_loai/  khac/
    realwaste/     giay/  nhua/  huu_co/  kim_loai/  khac/
```

Ảnh nằm trong thư mục nào thì thuộc lớp đó. Quy ước của nhóm: vải quần áo thuộc hữu cơ; giày dép, khẩu trang y tế thuộc khác.

## Cấu trúc thư mục

```
chung.py                            hằng số, lưới siêu tham số và hàm dùng chung
01_thong_ke_du_lieu.py              gom ảnh, cân bằng lớp, làm sạch, thống kê      -> ketqua/danh_sach_anh.csv, ketqua/thong_ke_du_lieu.xlsx, hinh/hinh_3_*
02_trich_dac_trung.py               chuẩn hóa ảnh, ResNet18 -> vector 512 chiều   -> vectors/features_512.npz, hinh/hinh_3_*
03_thucnghiem01_pca.py              thực nghiệm 01: khảo sát số chiều PCA          -> ketqua/thucnghiem01_pca*.csv, hinh/hinh_4_1
04_thucnghiem02_sieu_tham_so.py     thực nghiệm 02: quét lưới siêu tham số, 5 lần  -> ketqua/ket_qua_5_lan_chay.xlsx, models/, hinh/hinh_4_2..4_6
05_phan_loai_anh_moi.py             demo: phân loại ảnh mới bằng mô hình đã huấn luyện
ketqua/    danh sách ảnh sau làm sạch và các bảng kết quả xlsx, csv
vectors/   đặc trưng 512 chiều (không đẩy lên git)
hinh/      hình cho báo cáo, đặt tên theo số hình trong báo cáo
models/    mô hình đã huấn luyện (không đẩy lên git)
```

## Chạy

Chạy lần lượt theo số thứ tự:

```
python 01_thong_ke_du_lieu.py          # vài phút
python 02_trich_dac_trung.py           # vài phút trên CPU; lần đầu cần mạng để tải trọng số ResNet18 (45 MB)
python 03_thucnghiem01_pca.py          # khoảng 5 phút
python 04_thucnghiem02_sieu_tham_so.py # khoảng 1 đến 2 giờ trên CPU (96 tổ hợp x 5 lần = 480 lượt huấn luyện)
python 05_phan_loai_anh_moi.py anh.jpg # demo với ảnh mới
```

Máy không có GPU vẫn chạy được. Muốn chạy nhanh để thử: trong `chung.py` giảm `SEEDS` còn `[0]` và thu nhỏ `GRID`.

## Kết quả

- `ketqua/danh_sach_anh.csv`: danh sách ảnh sau làm sạch mà các bước sau dùng.
- `ketqua/thong_ke_du_lieu.xlsx`: số ảnh theo lớp và nguồn trước / sau làm sạch, file bị loại, kích thước, định dạng.
- `ketqua/thucnghiem01_pca_tb.csv`: accuracy theo số chiều PCA.
- `ketqua/ket_qua_5_lan_chay.xlsx`: sheet Tong_hop (trung bình và độ lệch chuẩn 5 lần), Sieu_tham_so_TB (mọi tổ hợp đã thử), Chi_tiet_5_lan, Loss_theo_epoch, ...
- `hinh/`: hình 3.1 đến 4.6 dùng trong báo cáo.

## Nguồn dữ liệu

RealWaste: S. Single, S. Iranmanesh, R. Raad (2023), Information 14(12), 633. Giấy phép CC BY 4.0.
