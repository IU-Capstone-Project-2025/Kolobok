from typing import Dict, Any
import time
from traceback import format_exc

import numpy as np

from tire_vision.text.preprocessor.model import TireDetector
from tire_vision.text.preprocessor.unwrapper import TireUnwrapper
from tire_vision.text.ocr.pipeline import TireOCR
from tire_vision.config import (
    OCRConfig,
    TireUnwrapperConfig,
    TireDetectorConfig,
)

import logging


class TireAnnotationPipeline:
    def __init__(
        self,
        detector_config: TireDetectorConfig,
        unwrapper_config: TireUnwrapperConfig,
        ocr_config: OCRConfig,
    ):
        self.detector = TireDetector(detector_config)
        self.unwrapper = TireUnwrapper(unwrapper_config)
        self.ocr = TireOCR(ocr_config)

        self.logger = logging.getLogger("tire_annotation_pipeline")
        self.logger.info("TireAnnotationPipeline initialized")

    def __call__(self, image: np.ndarray) -> Dict[str, Any]:
        self.logger.info("Running TireAnnotationPipeline")
        start_time = time.perf_counter()

        unwrap_success = False
        unwrapped_image: np.ndarray | None = None

        self.logger.info("Running TireUnwrapper")
        try:
            self.logger.info("Running TireDetector")
            detection_result = self.detector.detect(image)
            self.logger.info(f"TireDetector result keys: {detection_result.keys()}")
            unwrapped_image = self.unwrapper.get_unwrapped_tire(
                image,
                detection_result[self.detector.tire_class_name],
                detection_result[self.detector.rim_class_name],
            )
            unwrap_success = True
        except Exception:
            self.logger.error(format_exc())
            self.logger.error(
                "Error running TireUnwrapper. Falling back to original image"
            )

        self.logger.info(
            f"Original image shape: {image.shape}, unwrapped image shape: {getattr(unwrapped_image, 'shape', None)}"
        )

        # Prepare images and prompt for OCR
        if unwrap_success and unwrapped_image is not None:
            self.logger.info("Unwrap successful, using both images for OCR")
            images_for_ocr = [image, unwrapped_image]
            prompt_suffix = (
                "You will receive TWO images: the first is the original tire photo, "
                "the second is the unwrapped tread/sidewall strip. Use both images to "
                "extract the tire information."
            )
        else:
            self.logger.info("Unwrap failed, using only original image for OCR")
            images_for_ocr = [image]
            prompt_suffix = (
                "You will receive ONE image: the original tire photo. Use it to extract "
                "the tire information."
            )

        prompt = f"{self.ocr.config.prompt}\n\n{prompt_suffix}"

        self.logger.info("Running TireOCR")
        ocr_result = self.ocr.extract_tire_info(images_for_ocr, prompt)
        self.logger.info(f"TireOCR result:\n {ocr_result}")

        latency = time.perf_counter() - start_time
        self.logger.info(f"TireAnnotationPipeline completed in {latency:.4f} seconds")

        return ocr_result
