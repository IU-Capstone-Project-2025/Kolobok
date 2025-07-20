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
    test_size: float = 0.2


def main():
    args = tyro.cli(Args)

    src_images = Path(args.src_dir)

    dest_images_train = Path(args.train_dir)

    dest_images_val = Path(args.val_dir)

    dest_images_train.mkdir(parents=True, exist_ok=True)
    dest_images_val.mkdir(parents=True, exist_ok=True)

    image_paths = list(src_images.glob("*.png"))

    train_paths, val_paths = train_test_split(
        image_paths, test_size=args.test_size, random_state=42
    )

    for image_path in train_paths:
        shutil.copy(image_path, dest_images_train / image_path.name)

    for image_path in val_paths:
        shutil.copy(image_path, dest_images_val / image_path.name)


if __name__ == "__main__":
    main()
