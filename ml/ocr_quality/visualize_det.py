from pathlib import Path
import sys

import cv2
import numpy as np
from tqdm import tqdm

from scipy.ndimage import uniform_filter

# from tire_vision.text.preprocessor.model import TireDetector
# from tire_vision.config import TireDetectorConfig

input_data_root = Path.cwd() / "data" / "annotations"
output_data_root = Path.cwd() / "data" / "annotations_contrast"

output_data_root.mkdir(parents=True, exist_ok=True)

# cfg = TireDetectorConfig()
# detector = TireDetector(cfg)

# def visualize_det(img, polygon_rim, polygon_wheel):
#     img_clone = img.copy()
#     cv2.polylines(img_clone, [polygon_rim], True, (0, 0, 255), 2)
#     cv2.polylines(img_clone, [polygon_wheel], True, (0, 255, 0), 2)
#     return img_clone


clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(64, 64))


def contrast_enhance(img):
    l, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
    l = clahe.apply(l)

    l = cv2.bilateralFilter(l, d=9, sigmaColor=75, sigmaSpace=75)

    g1 = cv2.GaussianBlur(l, (0, 0), sigmaX=1.0)
    g2 = cv2.GaussianBlur(l, (0, 0), sigmaX=3.0)
    dog = cv2.subtract(g1, g2)  # mid-frequency details
    l = cv2.add(l, dog)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tophat = cv2.morphologyEx(l, cv2.MORPH_TOPHAT, kernel)
    l = cv2.add(l, tophat)

    l = cv2.normalize(l, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def main():
    for img_path in tqdm(input_data_root.iterdir()):
        img = cv2.imread(str(img_path))
        contrast_img = contrast_enhance(img)
        cv2.imwrite(str(output_data_root / f"{img_path.stem}.jpg"), contrast_img)


if __name__ == "__main__":
    main()
