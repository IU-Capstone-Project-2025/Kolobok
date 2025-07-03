from pathlib import Path
from typing import Tuple
import os
import sys

import torch
from torchvision.io import read_image, write_png

sys.path.append(str(Path(__file__).parent.parent))

from tire_vision.thread.segmentation.segmentator import SegmentationInferencer
from tire_vision.config import SegmentationConfig

SRC_DIR = Path("data/dataset")
DEST_DIR = Path("data/thread/depth/orig")

DEST_DIR.mkdir(parents=True, exist_ok=True)

if not SRC_DIR.exists():
    raise FileNotFoundError(f"Source directory {SRC_DIR} does not exist")


def _get_image_and_label(path: Path) -> Tuple[torch.Tensor, float]:
    image = read_image(str(path))
    label = float(path.parent.stem[:-2].replace(",", "."))
    return image, label


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = SegmentationConfig(device=device)
    segmentator = SegmentationInferencer(cfg)

    for images_path in SRC_DIR.iterdir():
        for image_path in images_path.iterdir():
            print(f"Processing {image_path}")
            try:
                image, label = _get_image_and_label(image_path)

                result = segmentator.crop_tire(image.to(device)).cpu()

                save_path = DEST_DIR / f"{len(os.listdir(DEST_DIR))}_{label}.png"

                write_png(result, save_path)

            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue


if __name__ == "__main__":
    main()
