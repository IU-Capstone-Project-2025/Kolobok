from pathlib import Path
import pickle as pkl
import asyncio
import json

from numpy import absolute
from rapidfuzz import fuzz
import cv2
import polars as pl
from tqdm import tqdm

from tire_vision.text.preprocessor.model import SidewallSegmentator
from tire_vision.text.preprocessor.unwrapper import SidewallUnwrapper
from tire_vision.text.ocr.pipeline import OCRPipeline
from tire_vision.text.index.pipeline import IndexPipeline
from tire_vision.config import (
    SidewallSegmentatorConfig,
    SidewallUnwrapperConfig,
    OCRConfig,
    IndexConfig,
)

import logging

logging.basicConfig(level=logging.CRITICAL)


def get_cached_results(
    pkl_path: str,
    segmentator: SidewallSegmentator,
    unwrapper: SidewallUnwrapper,
    input_dir: Path,
):
    if Path(pkl_path).exists():
        logging.info(f"Loading cached results from {pkl_path}")
        with open(pkl_path, "rb") as f:
            return pkl.load(f)

    logging.info("No cached results found, processing from scratch")
    results = []

    for img_path in tqdm(sorted(list(input_dir.iterdir()))):
        logging.info(f"Processing {img_path}")
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = segmentator.forward(img)
        unwrapped = unwrapper.get_unwrapped_tire(img, mask)
        unwrapped = cv2.cvtColor(unwrapped, cv2.COLOR_RGB2BGR)

        images = [img, unwrapped]
        results.append(
            {
                "img_path": str(img_path),
                "images": images,
            }
        )

    with open(pkl_path, "wb") as f:
        pkl.dump(results, f)

    return results


def get_cached_ocr_results(pkl_path: str):
    if not Path(pkl_path).exists():
        logging.info(f"No cached OCR results found, processing from scratch")
        return []

    logging.info(f"Loading cached OCR results from {pkl_path}")
    with open(pkl_path, "rb") as f:
        return pkl.load(f)


def read_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def check_correct(gt: dict[str, object], candidates: list[dict[str, object]]):
    result = dict(
        absolutely_correct=0,
        max_ratio_brand=0,
        max_ratio_model=0,
    )
    for candidate in candidates:
        if (
            candidate["model_id"] == gt["model_id"]
            and candidate["brand_id"] == gt["brand_id"]
        ):
            result["absolutely_correct"] = 1
            result["max_ratio_brand"] = 1
            result["max_ratio_model"] = 1
            return result

        ratio_brand = fuzz.ratio(candidate["brand_name"], gt["brand_name"]) / 100
        ratio_model = fuzz.ratio(candidate["model_name"], gt["model_name"]) / 100
        result["max_ratio_brand"] = max(result["max_ratio_brand"], ratio_brand)
        result["max_ratio_model"] = max(result["max_ratio_model"], ratio_model)

    return result


def main():
    cfg_segmentator = SidewallSegmentatorConfig()
    cfg_unwrapper = SidewallUnwrapperConfig()
    cfg_ocr = OCRConfig()
    cfg_index = IndexConfig()
    cfg_index.db_host = "localhost"

    model = SidewallSegmentator(cfg_segmentator)
    unwrapper = SidewallUnwrapper(cfg_unwrapper)
    ocr = OCRPipeline(cfg_ocr)
    index = IndexPipeline(cfg_index)
    db = index.database

    input_dir = Path("/Users/n-zagainov/kolobok/ml/data/annotations")
    gt_paths = Path("/Users/n-zagainov/kolobok/ml/data/annotations_processed")

    results = get_cached_results(
        pkl_path="results.pkl", 
        segmentator=model,
        unwrapper=unwrapper,
        input_dir=input_dir,
    )

    for result in results:
        img_path = Path(result["img_path"])
        img_name = img_path.stem
        Path("/Users/n-zagainov/kolobok/ml/data/annotations_unwrapped").mkdir(parents=True, exist_ok=True)
        save_path = Path("/Users/n-zagainov/kolobok/ml/data/annotations_unwrapped") / f"{img_name}.png"
        second_image = result["images"][1]
        cv2.imwrite(str(save_path), second_image)


    # joined_table = (
    #     db.table.filter(pl.col("parent_id") != 0)
    #     .join(
    #         db.table.filter(pl.col("parent_id") == 0),
    #         left_on="parent_id",
    #         right_on="id",
    #         how="left",
    #         suffix="_right",
    #     )
    #     .select(
    #         pl.col("id").alias("model_id"),
    #         pl.col("name").alias("model_name"),
    #         pl.col("parent_id").alias("brand_id"),
    #         pl.col("name_right").alias("brand_name"),
    #     )
    # )
    # num_samples = 0
    # total_correct = 0
    # for result in results:
    #     img_path = result["img_path"]
    #     gt_path = gt_paths / f"{Path(img_path).stem}.json"
    #     strings = result["ocr_result"]["strings"]
    #     gt = read_json(gt_path)
    #     if gt["model"] is None or gt["brand"] is None:
    #         continue
    #     gt = joined_table.filter(
    #         pl.col("model_id") == int(gt["model"]),
    #         pl.col("brand_id") == int(gt["brand"]),
    #     ).to_dicts()[0]

    #     index_results = index.get_best_matches(strings)
    #     index_results = [
    #         {
    #             k: item[k]
    #             for k in [
    #                 "model_id",
    #                 "model_name",
    #                 "brand_id",
    #                 "brand_name",
    #                 "combined_score",
    #             ]
    #         }
    #         for item in index_results
    #     ]

    #     print(f"FILES: {img_path}, {gt_path}")
    #     # logging.info(f"GT:\n{json.dumps(gt, indent=4)}")
    #     # logging.info(f"INDEX RESULTS:\n{json.dumps(index_results, indent=4)}")
    #     check_results = check_correct(gt, index_results)
    #     absolute_correct = check_results["absolutely_correct"]
    #     max_ratio_brand = check_results["max_ratio_brand"]
    #     max_ratio_model = check_results["max_ratio_model"]
    #     print(
    #         f"ABSOLUTE CORRECT: {absolute_correct}, MAX RATIO BRAND: {max_ratio_brand}, MAX RATIO MODEL: {max_ratio_model}"
    #     )
    #     if absolute_correct == 0:
    #         print(f"STRINGS:\n{strings}")
    #         print(f"GT:\n{json.dumps(gt, indent=4)}")
    #         print(f"INDEX RESULTS:\n{json.dumps(index_results, indent=4)}")

    #     print("-" * 100)
    #     total_correct += absolute_correct
    #     num_samples += 1

    # print(f"TOTAL CORRECT: {total_correct}, NUM SAMPLES: {num_samples}")


if __name__ == "__main__":
    main()
