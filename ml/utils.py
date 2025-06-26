from typing import Any

import numpy as np
import cv2
import torch

from tire_vision.thread.pipeline import TireVisionPipeline
from tire_vision.text import TireOCR
from tire_vision.config import TireVisionConfig, CLASS_MAPPING, CLASS_COLORS


cfg = TireVisionConfig()
pipeline = TireVisionPipeline(cfg)
ocr_pipeline = TireOCR(cfg.ocr)


def get_thread_stats(image: np.ndarray) -> dict[str, Any]:
    image = torch.from_numpy(image).permute(2, 0, 1)
    result = pipeline(image)
    return result


def extract_tire_info(image: np.ndarray) -> dict[str, Any]:
    """Extract tire information using OCR."""
    return ocr_pipeline.extract_tire_info(image)


def add_annotations(image: np.ndarray, annotations: list[dict[str, Any]]) -> np.ndarray:
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for annotation in annotations:
        x, y, w, h = annotation["box"]
        class_name = CLASS_MAPPING[annotation["class"]]
        color = CLASS_COLORS[class_name]
        cv2.rectangle(image_bgr, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            image_bgr, class_name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )

    return image_bgr
