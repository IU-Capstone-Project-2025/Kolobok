from typing import Dict, Any

import torch
import numpy as np

from tire_vision.thread.segmentation.segmentator import SegmentationInferencer
from tire_vision.thread.spikes.pipeline import SpikePipeline
from tire_vision.thread.depth.pipeline import DepthEstimatorPipeline
from tire_vision.config import (
    SegmentationConfig,
    SpikePipelineConfig,
    DepthEstimatorConfig,
)

import logging


class TireThreadPipeline:
    def __init__(
        self,
        segmentation_config: SegmentationConfig,
        spikes_config: SpikePipelineConfig,
        depth_config: DepthEstimatorConfig,
    ):
        self.segmentator = SegmentationInferencer(segmentation_config)
        self.spike_pipeline = SpikePipeline(spikes_config)
        self.depth_pipeline = DepthEstimatorPipeline(depth_config)

    def __call__(self, image: torch.Tensor) -> Dict[str, Any]:
        cropped_image = self.segmentator.crop_tire(image)
        if cropped_image is None:
            return {
                "success": 0,
                "detail": "Tire not found on the image, or it is too small",
            }

        cropped_image = cropped_image.to(torch.float32) / 255

        spikes = self.spike_pipeline.detect_spikes(cropped_image)

        depth = self.depth_pipeline.estimate_depth(cropped_image)

        cropped_image = cropped_image.cpu().numpy().transpose(1, 2, 0)
        cropped_image = cropped_image * 255
        cropped_image = cropped_image.astype(np.uint8)

        return {
            "success": 1,
            "cropped_image": cropped_image,
            "depth": depth,
            "spikes": spikes,
        }
