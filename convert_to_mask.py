
from pathlib import Path
import cv2
import numpy as np

# ============== CONFIGURATION ==============

DATASET_DIR = Path("SegDataset/valid")

IMAGE_DIR = DATASET_DIR / "images"
LABEL_DIR = DATASET_DIR / "labels"

MASK_DIR = DATASET_DIR / "masks"
COLOR_DIR = DATASET_DIR / "colored_masks"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp"
}

# YOLO Class ID -> Mask Pixel Value
CLASS_MAPPING = {
    0: 1,  # In
    1: 2,  # Pa
    2: 3   # Sc
}

# RGB colors for visualization
COLORS = {
    0: (0, 0, 0),        # Background
    1: (255, 0, 0),      # In - Red
    2: (0, 255, 0),      # Pa - Green
    3: (0, 0, 255)       # Sc - Blue
}

# ===========================================


def convert_label(label_path, width, height):

    # Single-channel semantic mask
    mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    with open(label_path, "r") as file:

        for line_number, line in enumerate(file, 1):

            values = line.strip().split()

            if len(values) < 7:
                print(
                    f"Invalid polygon: {label_path}, "
                    f"line {line_number}"
                )
                continue

            try:
                class_id = int(values[0])

                if class_id not in CLASS_MAPPING:
                    print(f"Unknown class ID: {class_id}")
                    continue

                coords = np.array(
                    list(map(float, values[1:])),
                    dtype=np.float32
                )

                if len(coords) % 2 != 0:
                    continue

                if not np.all(np.isfinite(coords)):
                    continue

                points = coords.reshape(-1, 2)

                # Normalized coordinates -> pixels
                points[:, 0] *= width
                points[:, 1] *= height

                points = np.rint(points).astype(np.int32)

                points[:, 0] = np.clip(
                    points[:, 0], 0, width - 1
                )

                points[:, 1] = np.clip(
                    points[:, 1], 0, height - 1
                )

                if len(points) < 3:
                    continue

                pixel_value = CLASS_MAPPING[class_id]

                cv2.fillPoly(
                    mask,
                    [points],
                    color=pixel_value
                )

            except (ValueError, OverflowError):
                print(
                    f"Error in {label_path}, "
                    f"line {line_number}"
                )

    return mask


def create_color_mask(mask):

    height, width = mask.shape

    color_mask = np.zeros(
        (height, width, 3),
        dtype=np.uint8
    )

    for pixel_value, color in COLORS.items():

        color_mask[mask == pixel_value] = color

    # Convert RGB to BGR for OpenCV saving
    color_mask = cv2.cvtColor(
        color_mask,
        cv2.COLOR_RGB2BGR
    )

    return color_mask


def batch_convert():

    MASK_DIR.mkdir(parents=True, exist_ok=True)
    COLOR_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = [
        p for p in IMAGE_DIR.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    converted = 0
    missing = 0
    failed = 0

    print(f"Found {len(image_paths)} images.")

    for image_path in image_paths:

        label_path = LABEL_DIR / (
            image_path.stem + ".txt"
        )

        if not label_path.exists():
            print(f"Missing label: {image_path.name}")
            missing += 1
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            failed += 1
            continue

        height, width = image.shape[:2]

        try:

            mask = convert_label(
                label_path,
                width,
                height
            )

            color_mask = create_color_mask(mask)

            # Save grayscale class mask
            cv2.imwrite(
                str(MASK_DIR / f"{image_path.stem}.png"),
                mask
            )

            # Save RGB visualization
            cv2.imwrite(
                str(COLOR_DIR / f"{image_path.stem}.png"),
                color_mask
            )

            converted += 1

        except Exception as error:

            print(f"Failed: {image_path.name}: {error}")
            failed += 1

    print("\n========== REPORT ==========")
    print(f"Total images: {len(image_paths)}")
    print(f"Converted: {converted}")
    print(f"Missing labels: {missing}")
    print(f"Failed: {failed}")
    print("============================")


if __name__ == "__main__":
    batch_convert()
