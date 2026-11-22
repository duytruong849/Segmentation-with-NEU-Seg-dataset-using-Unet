# U-Net Segment Detection - Ảnh và Webcam

Chương trình dùng checkpoint `best.pth` và kiến trúc U-Net trong project để phân đoạn 3 lớp `In`, `Pa`, `Sc`. Kết quả gồm mask màu, bounding box và confidence trung bình của xác suất softmax trên từng vùng liên thông.

## 1. Cài thư viện

```bash
pip install -r requirements.txt
```

## 2. Nhận diện một ảnh

```bash
python main.py --image "duong_dan/anh_test.jpg"
```

Ảnh kết quả được lưu trong thư mục `results` với hậu tố `_result.jpg`.

Muốn mở cửa sổ xem ảnh kết quả:

```bash
python main.py --image "duong_dan/anh_test.jpg" --show
```

## 3. Nhận diện tất cả ảnh trong một thư mục

```bash
python main.py --input-dir "duong_dan/thu_muc_anh"
```

Chương trình xử lý lần lượt JPG, JPEG, PNG, BMP, TIF, TIFF và WEBP; lưu kết quả vào `results`. Có thể dùng `--show` để xem lần lượt từng ảnh.

## 4. Tùy chỉnh

```bash
python main.py --input-dir "images" --output-dir "ket_qua" --conf 0.60 --min-area 150
```

- `--conf`: ngưỡng confidence từ 0 đến 1 (mặc định 0.50).
- `--min-area`: diện tích vùng tối thiểu theo pixel ảnh gốc (mặc định 100).
- `--cpu`: ép chạy CPU.

## 5. Chạy webcam như phiên bản trước

```bash
python main.py
python main.py --camera 1
```

- `Q` hoặc `Esc`: thoát.
- `+` / `-`: tăng / giảm ngưỡng confidence.

## Thông tin model

- U-Net, RGB input 640x640, 4 lớp, `base_c=16`.
- Nhãn: Background, In, Pa, Sc.
- Chuẩn hóa ImageNet: mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`.
- Confidence trên box là trung bình xác suất softmax của lớp trên các pixel trong component; không phải validation accuracy.
