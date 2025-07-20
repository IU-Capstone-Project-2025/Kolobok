from pathlib import Path
import shutil
import sys
from dataclasses import dataclass

import tyro

from sklearn.model_selection import train_test_split


sys.path.append(str(Path(__file__).parent.parent))


@dataclass
class Args:
    src_dir: str
    train_dir: str
    val_dir: str
    images_name: str = "images"
    masks_name: str = "masks"
    test_size: float = 0.2


def main():
    args = tyro.cli(Args)

    pairs = []

    src_images = Path(args.src_dir) / args.images_name
    src_masks = Path(args.src_dir) / args.masks_name

    dest_images_train = Path(args.train_dir) / args.images_name
    dest_masks_train = Path(args.train_dir) / args.masks_name

    dest_images_val = Path(args.val_dir) / args.images_name
    dest_masks_val = Path(args.val_dir) / args.masks_name

    dest_images_train.mkdir(parents=True, exist_ok=True)
    dest_masks_train.mkdir(parents=True, exist_ok=True)
    dest_images_val.mkdir(parents=True, exist_ok=True)
    dest_masks_val.mkdir(parents=True, exist_ok=True)

    for image_path in src_images.iterdir():
        mask_path = src_masks / image_path.name

        if not mask_path.exists():
            print(f"Mask not found for {image_path}")
            continue

        pairs.append((image_path, mask_path))

    train_pairs, val_pairs = train_test_split(
        pairs, test_size=args.test_size, random_state=42
    )

    for image_path, mask_path in train_pairs:
        shutil.copy(image_path, dest_images_train / image_path.name)
        shutil.copy(mask_path, dest_masks_train / mask_path.name)

    for image_path, mask_path in val_pairs:
        shutil.copy(image_path, dest_images_val / image_path.name)
        shutil.copy(mask_path, dest_masks_val / mask_path.name)


if __name__ == "__main__":
    main()
