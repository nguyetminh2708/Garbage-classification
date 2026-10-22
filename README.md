# Phân loại rác thải từ ảnh bằng các thuật toán học máy cổ điển

Tiểu luận môn Học máy. Ảnh rác được đưa qua ResNet18 (huấn luyện sẵn) để lấy vector đặc trưng, sau đó phân loại bằng 6 thuật toán học máy cổ điển vào 5 nhóm: giấy, nhựa, hữu cơ, kim loại, khác.

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

## Chạy

```
python tien_xu_ly_anh.py        # làm sạch ảnh, trích đặc trưng, lưu vào features/
python huan_luyen_mo_hinh.py    # chạy 5 lần, xuất Excel và hình vào ket_qua/
```

Lần chạy đầu cần mạng để tải trọng số ResNet18 (khoảng 45 MB). Máy không có GPU vẫn chạy được, mỗi script mất vài phút.

## Kết quả

- `ket_qua/ket_qua_5_lan_chay.xlsx`: số liệu của 5 lần chạy, sheet Tong_hop có trung bình và độ lệch chuẩn.
- `ket_qua/*.png`: biểu đồ loss, ma trận nhầm lẫn, so sánh mô hình.

## Nguồn dữ liệu

RealWaste: S. Single, S. Iranmanesh, R. Raad (2023), Information 14(12), 633. Giấy phép CC BY 4.0.
