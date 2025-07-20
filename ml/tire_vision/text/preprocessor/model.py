import logging
import time

import numpy as np
import torch

from tire_vision.config import SidewallSegmentatorConfig
from tire_vision.segmentation.onnx import OnnxSegmentator


class SidewallSegmentator:
    def __init__(self, config: SidewallSegmentatorConfig):
        self.config = config
        self.logger = logging.getLogger("sidewall_segmentator")

        self.segmentator = OnnxSegmentator(
            self.config.sidewall_segmentator_onnx,
            self.config.resize_shape,
            self.config.confidence_threshold,
        )

        self.logger.info("SidewallSegmentator initialized successfully")

    @torch.no_grad()
    def forward(self, image: np.ndarray):
        start_time = time.perf_counter()

        mask = self.segmentator(image)

        end_time = time.perf_counter()
        self.logger.info(
            f"Completed sidewall segmentation in {end_time - start_time} seconds"
        )

        return mask
