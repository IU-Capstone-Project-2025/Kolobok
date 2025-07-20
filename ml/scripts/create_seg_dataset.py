from pathlib import Path
from dataclasses import dataclass
import json

import tyro

import cv2
import numpy as np


@dataclass
class Args:
    src_dir: str
    dest_dir: str
    annot_path: str
    images_name: str = "images"
    masks_name: str = "masks"
    tire_name: str = "Wheel"
    rim_name: str = "Rim"


def main():
    args = tyro.cli(Args)
    src_dir = Path(args.src_dir)
    dest_dir = Path(args.dest_dir)

    with open(args.annot_path, "r") as f:
        data = json.load(f)

    for item in data:
        image_name = Path(item["data"]["image"]).name.split("-", maxsplit=1)[-1]

        src_image_path = src_dir / image_name
        dest_image_path = dest_dir / args.images_name / image_name
        dest_mask_path = dest_dir / args.masks_name / image_name

        dest_image_path.parent.mkdir(parents=True, exist_ok=True)
        dest_mask_path.parent.mkdir(parents=True, exist_ok=True)

        annots = item["annotations"][-1]["result"]

        tire_annot = list(
            filter(lambda x: x["value"]["polygonlabels"][0] == args.tire_name, annots)
        )[-1]
        # rim_annot = list(
        #     filter(lambda x: x["value"]["polygonlabels"][0] == args.rim_name, annots)
        # )[-1]

        tire_poly = np.array(tire_annot["value"]["points"])
        # rim_poly = np.array(rim_annot["value"]["points"])

        tire_x_scale = tire_annot["original_width"] / 100
        tire_y_scale = tire_annot["original_height"] / 100
        # rim_x_scale = rim_annot["original_width"] / 100
        # rim_y_scale = rim_annot["original_height"] / 100

        tire_poly[:, 0] *= tire_x_scale
        tire_poly[:, 1] *= tire_y_scale
        # rim_poly[:, 0] *= rim_x_scale
        # rim_poly[:, 1] *= rim_y_scale

        tire_poly = tire_poly.astype(np.int32)
        # rim_poly = rim_poly.astype(np.int32)

        image = cv2.imread(str(src_image_path))

        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        mask = cv2.fillPoly(mask, [tire_poly], 255)
        # mask = cv2.fillPoly(mask, [rim_poly], 0)

        cv2.imwrite(str(dest_image_path), image)
        cv2.imwrite(str(dest_mask_path), mask)


if __name__ == "__main__":
    main()
