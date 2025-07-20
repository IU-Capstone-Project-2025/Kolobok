from typing import Dict, Any, Optional
import time
from traceback import format_exc
import json

import numpy as np

from tire_vision.text.preprocessor.model import SidewallSegmentator
from tire_vision.text.preprocessor.unwrapper import SidewallUnwrapper
from tire_vision.text.ocr.pipeline import OCRPipeline
from tire_vision.text.index.pipeline import IndexPipeline
from tire_vision.config import (
    OCRConfig,
    SidewallUnwrapperConfig,
    SidewallSegmentatorConfig,
    IndexConfig,
)

import logging


def _json_format(data):
    return json.dumps(data, indent=4)


class TireAnnotationPipeline:
    def __init__(
        self,
        sidewall_segmentator_config: SidewallSegmentatorConfig,
        sidewall_unwrapper_config: SidewallUnwrapperConfig,
        ocr_config: OCRConfig,
        index_config: IndexConfig,
    ):
        self.detector = SidewallSegmentator(sidewall_segmentator_config)
        self.unwrapper = SidewallUnwrapper(sidewall_unwrapper_config)
        self.ocr = OCRPipeline(ocr_config)
        self.index = IndexPipeline(index_config)

        self.logger = logging.getLogger("tire_annotation_pipeline")
        self.logger.info("TireAnnotationPipeline initialized")

    def __call__(self, image: np.ndarray) -> Dict[str, Any]:
        self.logger.info("Running TireAnnotationPipeline")
        start_time = time.perf_counter()

        unwrap_success = False
        unwrapped_image: Optional[np.ndarray] = None

        self.logger.info("Running SidewallSegmentator and SidewallUnwrapper")
        try:
            self.logger.info("Running TireDetector")
            tire_mask = self.detector.forward(image)
            self.logger.info(f"TireDetector result shape: {tire_mask.shape}")
            unwrapped_image = self.unwrapper.forward(image, tire_mask)
            unwrap_success = True

        except Exception:
            self.logger.error(format_exc())
            self.logger.error("Error unwrapping tire. Falling back to original image")

        self.logger.info(
            f"Original image shape: {image.shape}, unwrapped image shape: {getattr(unwrapped_image, 'shape', None)}"
        )
        images_for_ocr = [image]
        if unwrap_success:
            self.logger.info("Unwrap successful, using both images for OCR")
            images_for_ocr.append(unwrapped_image)
        else:
            self.logger.warning(
                "Unwrap failed, using only original image for OCR. This could lead to less accurate results."
            )

        self.logger.info("Running OCRPipeline")
        ocr_result = self.ocr.extract_tire_info(images_for_ocr)
        self.logger.info(f"OCRPipeline result:\n{_json_format(ocr_result)}")

        self.logger.info("Running IndexPipeline")
        index_results = self.index.get_best_matches(ocr_result["strings"])
        self.logger.info(f"IndexPipeline result:\n{_json_format(index_results)}")

        combined_result = {
            "strings": ocr_result["strings"],
            "tire_size": ocr_result["tire_size"],
            "index_results": index_results,
        }

        latency = time.perf_counter() - start_time
        self.logger.info(f"TireAnnotationPipeline completed in {latency:.4f} seconds")

        return combined_result
