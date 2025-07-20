from pathlib import Path
from typing import Tuple
import os

import cv2
import numpy as np

from tire_vision.thread.segmentator.model import ThreadSegmentator
from tire_vision.config import ThreadSegmentatorConfig

SRC_DIR = Path("data/dataset_synthetic")
DEST_DIR = Path("data/thread/depth/synthetic")

DEST_DIR.mkdir(parents=True, exist_ok=True)

if not SRC_DIR.exists():
    raise FileNotFoundError(f"Source directory {SRC_DIR} does not exist")


def _get_image_and_label(path: Path) -> Tuple[np.ndarray, float]:
    image = cv2.imread(str(path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    label = float(path.stem.split("_")[2]) * 10
    return image, label


def main():
    cfg = ThreadSegmentatorConfig()
    segmentator = ThreadSegmentator(cfg)

    for image_path in SRC_DIR.glob("*.jpg"):
        print(f"Processing {image_path}")
        try:
            image, label = _get_image_and_label(image_path)

            result = segmentator.crop_tire(image)

            save_path = DEST_DIR / f"{len(os.listdir(DEST_DIR))}_{label}.png"

            result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(save_path), result)

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue


if __name__ == "__main__":
    main()
