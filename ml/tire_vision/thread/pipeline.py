from typing import Dict, Any
import time

import numpy as np

from tire_vision.thread.segmentator.model import ThreadSegmentator
from tire_vision.thread.spikes.pipeline import SpikePipeline
from tire_vision.thread.depth.model import DepthRegressor
from tire_vision.config import (
    ThreadSegmentatorConfig,
    SpikePipelineConfig,
    DepthRegressorConfig,
)

import logging


class TireThreadPipeline:
    def __init__(
        self,
        segmentator_config: ThreadSegmentatorConfig,
        spike_pipeline_config: SpikePipelineConfig,
        depth_regressor_config: DepthRegressorConfig,
    ):
        self.segmentator = ThreadSegmentator(segmentator_config)
        self.spike_pipeline = SpikePipeline(spike_pipeline_config)
        self.depth_regressor = DepthRegressor(depth_regressor_config)

        self.logger = logging.getLogger("tire_thread_pipeline")

    def __call__(self, image: np.ndarray) -> Dict[str, Any]:
        self.logger.info("Starting tire thread pipeline")
        start_time = time.perf_counter()

        cropped_image = self.segmentator.crop_tire(image)
        if cropped_image is None:
            self.logger.warning("Tire not found on the image, or it is too small")
            return {
                "success": 0,
                "detail": "Tire not found on the image, or it is too small",
            }

        spikes = self.spike_pipeline.detect_spikes(cropped_image)

        depth = self.depth_regressor(cropped_image)

        latency = time.perf_counter() - start_time
        self.logger.info(f"Tire thread pipeline completed in {latency:.4f} seconds")

        result = {
            "success": 1,
            "cropped_image": cropped_image,
            "depth": depth,
            "spikes": spikes,
        }

        self.logger.info(
            f"Cropped image shape: {cropped_image.shape} "
            f"Depth: {depth} "
            f"Number of spikes: {len(spikes)}"
        )

        return result
