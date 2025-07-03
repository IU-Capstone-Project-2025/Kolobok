from pathlib import Path
from typing import Tuple
import os
import shutil
import sys

from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).parent.parent))


SRC_DIR = Path("data/thread/depth/orig")
TRAIN_DIR = Path("data/thread/depth/orig_train")
TEST_DIR = Path("data/thread/depth/orig_test")

TRAIN_DIR.mkdir(parents=True, exist_ok=True)
TEST_DIR.mkdir(parents=True, exist_ok=True)

if not SRC_DIR.exists():
    raise FileNotFoundError(f"Source directory {SRC_DIR} does not exist")


def main():
    all_paths = list(SRC_DIR.iterdir())
    train_paths, test_paths = train_test_split(
        all_paths, test_size=0.3, random_state=42
    )

    for path in train_paths:
        shutil.copy(path, TRAIN_DIR / path.name)

    for path in test_paths:
        shutil.copy(path, TEST_DIR / path.name)


if __name__ == "__main__":
    main()
