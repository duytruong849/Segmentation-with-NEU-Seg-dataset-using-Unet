import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image
from UNet import *
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN & MÀU MASK
# ==========================================
BASE_DATASET_DIR = Path("SegDataset")
TRAIN_DIR = BASE_DATASET_DIR / "train"
VAL_DIR = BASE_DATASET_DIR / "valid"

IMAGE_SUBDIR = "images"
LABEL_SUBDIR = "labels"
MASK_SUBDIR = "masks"
COLOR_SUBDIR = "colored_masks"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

COLORS = {
    0: (0, 0, 0),        # Background
    1: (255, 0, 0),      # In - Red
    2: (0, 255, 0),      # Pa - Green
    3: (0, 0, 255),      # Sc - Blue
}
CLASS_NAMES = {
    0: "Background",
    1: "In",
    2: "Pa",
    3: "Sc",
}

IMG_SIZE = (640, 640)     # (Width, Height) cho PIL resize
NUM_CLASSES = len(COLORS)
BATCH_SIZE = 4
EPOCHS = 20
LR = 1e-4
BASE_C = 16              # Giảm xuống 32 để dễ chạy 640x640 trên GPU vừa phải; có thể tăng lên 64
NUM_VIS_SAMPLES = 4      # Số lượng ảnh validation lưu ra để trực quan hóa
OUTPUT_DIR = Path("runs/unet_640")


# ==========================================
# 2. CHUYỂN ĐỔI MASK
# ==========================================
def rgb_mask_to_class_mask(rgb_mask: np.ndarray, colors: dict = COLORS) -> np.ndarray:
    """Chuyển mask RGB (H, W, 3) thành class index mask (H, W) giá trị 0..3."""
    h, w, _ = rgb_mask.shape
    class_mask = np.zeros((h, w), dtype=np.int64)
    for class_idx, rgb in colors.items():
        match = np.all(rgb_mask == np.array(rgb, dtype=rgb_mask.dtype), axis=-1)
        class_mask[match] = class_idx
    return class_mask


def class_mask_to_rgb_mask(class_mask: np.ndarray, colors: dict = COLORS) -> np.ndarray:
    """Chuyển class index mask (H, W) thành ảnh mask RGB (H, W, 3)."""
    h, w = class_mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_idx, rgb in colors.items():
        rgb_mask[class_mask == class_idx] = rgb
    return rgb_mask


# ==========================================
# 3. DATASET
# ==========================================
class SegmentationDataset(Dataset):
    def __init__(self, split_dir: Path, img_size: tuple[int, int] = (640, 640), is_train: bool = True):
        self.split_dir = Path(split_dir)
        self.image_dir = self.split_dir / IMAGE_SUBDIR
        self.mask_dir = self.split_dir / MASK_SUBDIR
        self.color_dir = self.split_dir / COLOR_SUBDIR
        self.img_size = img_size
        self.is_train = is_train

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Không tìm thấy thư mục ảnh: {self.image_dir}")

        self.image_files = sorted([
            p for p in self.image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ])
        if not self.image_files:
            raise RuntimeError(f"Không có ảnh hợp lệ trong: {self.image_dir}")

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    def _load_mask(self, stem: str) -> np.ndarray:
        """
        Ưu tiên đọc class mask từ `masks/<stem>.png` (1 kênh 0..3).
        Nếu chưa có, tự động đọc từ `colored_masks/<stem>.png` và chuyển sang class mask.
        """
        mask_path = self.mask_dir / f"{stem}.png"
        color_path = self.color_dir / f"{stem}.png"

        if mask_path.exists():
            m = Image.open(mask_path).resize(self.img_size, resample=Image.NEAREST)
            arr = np.array(m)
            if arr.ndim == 3:
                return rgb_mask_to_class_mask(arr[:, :, :3])
            return arr.astype(np.int64)

        if color_path.exists():
            cm = Image.open(color_path).convert("RGB").resize(self.img_size, resample=Image.NEAREST)
            return rgb_mask_to_class_mask(np.array(cm))

        raise FileNotFoundError(f"Không tìm thấy mask cho '{stem}' tại {mask_path} hoặc {color_path}")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int):
        img_path = self.image_files[idx]
        stem = img_path.stem

        img_pil = Image.open(img_path).convert("RGB").resize(self.img_size, resample=Image.BILINEAR)
        img_np = np.array(img_pil, dtype=np.uint8)
        mask_np = self._load_mask(stem)

        # Augmentation cơ bản giữ đúng nhãn mask (chỉ flip ngang cho train)
        if self.is_train and np.random.rand() < 0.5:
            img_np = np.ascontiguousarray(img_np[:, ::-1, :])
            mask_np = np.ascontiguousarray(mask_np[:, ::-1])

        img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float() / 255.0
        img_tensor = torch.from_numpy((img_tensor.numpy() - self.mean) / self.std).float()
        mask_tensor = torch.from_numpy(mask_np).long()

        return img_tensor, mask_tensor, img_np, stem

# ==========================================
# 5. METRICS & SINH ẢNH VALIDATION
# ==========================================
def update_confusion_matrix(conf_mat: np.ndarray, preds: np.ndarray, targets: np.ndarray, num_classes: int):
    valid = (targets >= 0) & (targets < num_classes)
    idx = num_classes * targets[valid].astype(np.int64) + preds[valid].astype(np.int64)
    conf_mat += np.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def compute_iou_from_conf_mat(conf_mat: np.ndarray):
    tp = np.diag(conf_mat).astype(np.float64)
    fp = conf_mat.sum(axis=0).astype(np.float64) - tp
    fn = conf_mat.sum(axis=1).astype(np.float64) - tp
    denom = tp + fp + fn

    per_class_iou = {}
    valid_ious = []
    for c in range(len(tp)):
        if denom[c] > 0:
            iou = float(tp[c] / denom[c])
            valid_ious.append(iou)
        else:
            iou = None
        per_class_iou[CLASS_NAMES.get(c, str(c))] = iou

    mean_iou = float(np.mean(valid_ious)) if valid_ious else 0.0
    pixel_acc = float(tp.sum() / max(conf_mat.sum(), 1))
    return mean_iou, pixel_acc, per_class_iou


@torch.no_grad()
def save_val_predictions(
    model: nn.Module,
    val_dataset: SegmentationDataset,
    device: torch.device,
    out_dir: Path,
    num_samples: int = 6,
):
    """Tạo ảnh ghép ngang: [Ảnh gốc | Ground Truth Color Mask | Predicted Color Mask]."""
    model.eval()
    vis_dir = out_dir / "val_predictions"
    vis_dir.mkdir(parents=True, exist_ok=True)

    count = min(num_samples, len(val_dataset))
    for i in range(count):
        img_tensor, gt_mask, orig_img_np, stem = val_dataset[i]
        logits = model(img_tensor.unsqueeze(0).to(device))
        pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

        gt_rgb = class_mask_to_rgb_mask(gt_mask.numpy())
        pred_rgb = class_mask_to_rgb_mask(pred_mask)

        # Ghép ngang 3 ảnh (640x640) -> (640, 1920, 3)
        panel = np.concatenate([orig_img_np, gt_rgb, pred_rgb], axis=1)
        Image.fromarray(panel).save(vis_dir / f"{stem}_val_pred.png")


def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    metrics: dict,
    config: dict,
):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": config,
        },
        path,
    )


# ==========================================
# 6. VÒNG LẶP TRAIN & VALIDATION
# ==========================================
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, epoch: int | None = None,
    total_epochs: int | None = None):
    model.eval()
    total_loss = 0.0
    conf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    desc = (
        f"Val [{epoch:02d}/{total_epochs:02d}]"
        if epoch is not None and total_epochs is not None
        else "Val"
    )
    pbar = tqdm(loader, desc=desc, leave=False)

    with torch.no_grad():
        for imgs, masks, _, _ in pbar:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            logits = model(imgs)
            loss = criterion(logits, masks)
            total_loss += loss.item() * imgs.size(0)

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            update_confusion_matrix(conf_mat, preds, masks.cpu().numpy(), NUM_CLASSES)

            pbar.set_postfix(val_loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(loader.dataset)
    mean_iou, pixel_acc, per_class_iou = compute_iou_from_conf_mat(conf_mat)
    return avg_loss, mean_iou, pixel_acc, per_class_iou


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = {
        "img_size": IMG_SIZE,
        "num_classes": NUM_CLASSES,
        "colors": {str(k): v for k, v in COLORS.items()},
        "class_names": CLASS_NAMES,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "lr": LR,
        "base_c": BASE_C,
    }
    with open(OUTPUT_DIR / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    train_dataset = SegmentationDataset(TRAIN_DIR, img_size=IMG_SIZE, is_train=True)
    val_dataset = SegmentationDataset(VAL_DIR, img_size=IMG_SIZE, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    model = UNet(in_channels=3, num_classes=NUM_CLASSES, base_c=BASE_C).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    best_miou = -1.0
    history = []
    csv_path = OUTPUT_DIR / "metrics.csv"
    csv_fields = ["epoch", "lr", "train_loss", "val_loss", "val_mIoU", "val_pixel_acc"] + [
        f"IoU_{CLASS_NAMES[c]}" for c in range(NUM_CLASSES)
    ]

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=csv_fields)
        writer.writeheader()

        for epoch in range(1, EPOCHS + 1):
            model.train()
            train_loss_sum = 0.0

            train_pbar = tqdm(
                train_loader,
                desc=f"Train [{epoch:02d}/{EPOCHS:02d}]",
                leave=False,
            )

            for imgs, masks, _, _ in train_pbar:
                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                optimizer.zero_grad()
                logits = model(imgs)
                loss = criterion(logits, masks)
                loss.backward()
                optimizer.step()

                batch_loss = loss.item()
                train_loss_sum += batch_loss * imgs.size(0)
                train_pbar.set_postfix(loss=f"{batch_loss:.4f}")

            train_loss = train_loss_sum / len(train_dataset)
            val_loss, val_miou, val_acc, per_class_iou = evaluate(
                model, val_loader, criterion, device, epoch=epoch, total_epochs=EPOCHS
            )
            
            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(val_miou)

            epoch_metrics = {
                "epoch": epoch,
                "lr": current_lr,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "val_mIoU": round(val_miou, 6),
                "val_pixel_acc": round(val_acc, 6),
                "per_class_IoU": per_class_iou,
            }
            history.append(epoch_metrics)

            row = {
                "epoch": epoch,
                "lr": current_lr,
                "train_loss": epoch_metrics["train_loss"],
                "val_loss": epoch_metrics["val_loss"],
                "val_mIoU": epoch_metrics["val_mIoU"],
                "val_pixel_acc": epoch_metrics["val_pixel_acc"],
            }
            for c in range(NUM_CLASSES):
                cname = CLASS_NAMES[c]
                val_c = per_class_iou[cname]
                row[f"IoU_{cname}"] = round(val_c, 6) if val_c is not None else ""
            writer.writerow(row)
            f_csv.flush()

            print(
                f"Epoch [{epoch:02d}/{EPOCHS:02d}] | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val mIoU: {val_miou:.4f} | Val Acc: {val_acc:.4f}"
            )

            # Lưu checkpoint last mỗi epoch
            save_checkpoint(
                OUTPUT_DIR / "last.pth",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                metrics=epoch_metrics,
                config=config,
            )

            # Lưu checkpoint best khi Val mIoU cải thiện
            if val_miou > best_miou:
                best_miou = val_miou
                save_checkpoint(
                    OUTPUT_DIR / "best.pth",
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    metrics=epoch_metrics,
                    config=config,
                )
                save_val_predictions(
                    model=model,
                    val_dataset=val_dataset,
                    device=device,
                    out_dir=OUTPUT_DIR,
                    num_samples=NUM_VIS_SAMPLES,
                )

    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f_json:
        json.dump(history, f_json, indent=2, ensure_ascii=False)

    print(f"\nHoàn tất! Best Val mIoU: {best_miou:.4f}")
    print(f"Weights & thông số lưu tại: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()