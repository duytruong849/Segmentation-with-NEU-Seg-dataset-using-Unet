import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from UNet import UNet

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def load_model(project_dir: Path, device: torch.device):
    config_path = project_dir / 'config.json'
    weights_path = project_dir / 'last.pth'
    if not config_path.exists():
        raise FileNotFoundError(f'Không tìm thấy {config_path}')
    if not weights_path.exists():
        raise FileNotFoundError(f'Không tìm thấy {weights_path}')

    with config_path.open('r', encoding='utf-8') as f:
        cfg = json.load(f)

    img_size = cfg.get('img_size', [640, 640])
    num_classes = int(cfg.get('num_classes', 4))
    base_c = int(cfg.get('base_c', 16))
    class_names = {int(k): v for k, v in cfg.get('class_names', {}).items()}
    colors = {int(k): tuple(v) for k, v in cfg.get('colors', {}).items()}

    model = UNet(in_channels=3, num_classes=num_classes, base_c=base_c).to(device)
    try:
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, cfg, (int(img_size[0]), int(img_size[1])), class_names, colors, checkpoint


def read_image(path: Path):
    # Dùng imdecode để hỗ trợ đường dẫn có dấu tiếng Việt trên Windows.
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Không đọc được ảnh: {path}')
    return image


def write_image(path: Path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or '.jpg'
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f'Không mã hóa được ảnh đầu ra: {path}')
    encoded.tofile(str(path))


def preprocess(frame_bgr: np.ndarray, input_size, device):
    width, height = input_size
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    arr = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)


def make_overlay_and_boxes(frame_bgr, logits, class_names, colors,
                           confidence_threshold=0.50, min_area=100, alpha=0.35):
    orig_h, orig_w = frame_bgr.shape[:2]
    logits = F.interpolate(logits, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred = np.argmax(probs, axis=0).astype(np.uint8)

    overlay = frame_bgr.copy()
    boxes = []
    for class_id in sorted(class_names):
        if class_id == 0:
            continue
        class_mask = (pred == class_id).astype(np.uint8)
        if not np.any(class_mask):
            continue
        count, labels, stats, _ = cv2.connectedComponentsWithStats(class_mask, connectivity=8)
        color_rgb = colors.get(class_id, (255, 255, 0))
        color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
        for component_id in range(1, count):
            x, y, w, h, area = stats[component_id]
            if area < min_area:
                continue
            component = labels == component_id
            confidence = float(probs[class_id][component].mean())
            if confidence < confidence_threshold:
                continue
            overlay[component] = color_bgr
            boxes.append({
                'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h),
                'class_id': int(class_id), 'label': class_names.get(class_id, str(class_id)),
                'confidence': confidence, 'area': int(area), 'color': color_bgr,
            })

    result = cv2.addWeighted(overlay, alpha, frame_bgr, 1.0 - alpha, 0)
    for box in boxes:
        x, y, w, h = box['x'], box['y'], box['width'], box['height']
        color = box['color']
        cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
        label = f"{box['label']} {box['confidence'] * 100:.1f}%"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
        label_y = max(0, y - th - baseline - 7)
        cv2.rectangle(result, (x, label_y), (x + tw + 8, label_y + th + baseline + 7), color, -1)
        cv2.putText(result, label, (x + 4, label_y + th + 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, (255, 255, 255), 2, cv2.LINE_AA)

    return result, boxes


def predict_one(image_path: Path, output_path: Path, model, input_size, class_names, colors,
                device, confidence, min_area, show=False):
    image = read_image(image_path)
    tensor = preprocess(image, input_size, device)
    with torch.inference_mode():
        logits = model(tensor)
    result, boxes = make_overlay_and_boxes(
        image, logits, class_names, colors,
        confidence_threshold=confidence, min_area=min_area,
    )
    write_image(output_path, result)

    print(f'\n[ẢNH] {image_path.name}')
    print(f'     Số vùng phát hiện: {len(boxes)}')
    for i, box in enumerate(boxes, start=1):
        print(f"     {i}. {box['label']} | Confidence: {box['confidence'] * 100:.1f}% | Area: {box['area']} px")
    print(f'[ĐÃ LƯU] {output_path}')

    if show:
        window = f'Ket qua - {image_path.name}'
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.imshow(window, result)
        print('     Nhấn phím bất kỳ trong cửa sổ ảnh để xem ảnh tiếp theo; nhấn Q để dừng xem.')
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow(window)
        if key in (ord('q'), ord('Q'), 27):
            return False
    return True


def run_image_mode(args, model, input_size, class_names, colors, device):
    output_dir = Path(args.output_dir).expanduser().resolve()
    if args.image:
        image_paths = [Path(args.image).expanduser()]
    else:
        input_dir = Path(args.input_dir).expanduser()
        if not input_dir.is_dir():
            raise NotADirectoryError(f'Không tìm thấy thư mục ảnh: {input_dir}')
        image_paths = sorted(p for p in input_dir.iterdir()
                             if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)

    if not image_paths:
        raise RuntimeError('Không tìm thấy ảnh hợp lệ trong thư mục đầu vào.')

    output_dir.mkdir(parents=True, exist_ok=True)
    report = []
    for image_path in image_paths:
        if not image_path.is_file():
            print(f'[BỎ QUA] Không tìm thấy file: {image_path}')
            continue
        output_path = output_dir / f'{image_path.stem}_result.jpg'
        keep_going = predict_one(
            image_path, output_path, model, input_size, class_names, colors,
            device, args.conf, args.min_area, show=args.show,
        )
        report.append({'image': str(image_path), 'result': str(output_path)})
        if not keep_going:
            break

    report_path = output_dir / 'results.json'
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\nHoàn tất. Ảnh kết quả nằm trong: {output_dir}')
    print(f'Báo cáo danh sách ảnh: {report_path}')


def run_camera(args, model, input_size, class_names, colors, device):
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if hasattr(cv2, 'CAP_DSHOW') else 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise RuntimeError(f'Không mở được camera index {args.camera}. Thử --camera 1 hoặc kiểm tra quyền camera.')

    window_name = 'U-Net Segment Detection | Q: Quit | +/-: Confidence'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    conf_threshold = args.conf
    print('[INFO] Nhấn Q/ESC để thoát, + hoặc - để chỉnh confidence.')
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print('[WARN] Không đọc được frame từ camera.')
                break
            tensor = preprocess(frame, input_size, device)
            with torch.inference_mode():
                logits = model(tensor)
            display, boxes = make_overlay_and_boxes(
                frame, logits, class_names, colors,
                confidence_threshold=conf_threshold, min_area=args.min_area,
            )
            cv2.rectangle(display, (8, 8), (360, 76), (25, 25, 25), -1)
            cv2.putText(display, f'Detected regions: {len(boxes)}', (18, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(display, f'Confidence threshold: {conf_threshold:.2f}', (18, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break
            if key in (ord('+'), ord('=')):
                conf_threshold = min(0.99, conf_threshold + 0.05)
            elif key in (ord('-'), ord('_')):
                conf_threshold = max(0.05, conf_threshold - 0.05)
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='U-Net nhận diện segmentation trên ảnh hoặc webcam')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--image', type=str, help='Đường dẫn đến một ảnh cần nhận diện')
    mode.add_argument('--input-dir', type=str, help='Thư mục chứa nhiều ảnh cần nhận diện')
    parser.add_argument('--output-dir', type=str, default='results', help='Thư mục lưu ảnh kết quả')
    parser.add_argument('--show', action='store_true', help='Hiển thị từng ảnh kết quả sau khi nhận diện')
    parser.add_argument('--camera', type=int, default=0, help='Camera index khi không dùng --image/--input-dir')
    parser.add_argument('--conf', type=float, default=0.50, help='Ngưỡng confidence từ 0 đến 1')
    parser.add_argument('--min-area', type=int, default=100, help='Diện tích vùng nhỏ nhất (pixel)')
    parser.add_argument('--width', type=int, default=1280, help='Chiều rộng camera')
    parser.add_argument('--height', type=int, default=720, help='Chiều cao camera')
    parser.add_argument('--cpu', action='store_true', help='Ép chạy CPU dù có GPU')
    args = parser.parse_args()
    if not 0 <= args.conf <= 1:
        parser.error('--conf phải nằm trong khoảng 0..1')
    if args.min_area < 1:
        parser.error('--min-area phải lớn hơn 0')

    project_dir = Path(__file__).resolve().parent
    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    print(f'[INFO] Device: {device}')
    model, cfg, input_size, class_names, colors, checkpoint = load_model(project_dir, device)
    metrics = checkpoint.get('metrics', {}) if isinstance(checkpoint, dict) else {}
    if metrics:
        print(f"[INFO] Checkpoint epoch: {checkpoint.get('epoch', '?')} | "
              f"Val pixel accuracy: {metrics.get('val_pixel_acc', 0) * 100:.2f}% | "
              f"Val mIoU: {metrics.get('val_mIoU', 0) * 100:.2f}%")
    print('[INFO] Class labels:', ', '.join(f'{k}:{v}' for k, v in class_names.items()))

    if args.image or args.input_dir:
        run_image_mode(args, model, input_size, class_names, colors, device)
    else:
        run_camera(args, model, input_size, class_names, colors, device)


if __name__ == '__main__':
    main()
