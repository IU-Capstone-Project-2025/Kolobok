from typing import Dict, Any

import torch
import numpy as np

from tire_vision.thread.segmentation.segmentator import SegmentationInferencer
from tire_vision.thread.spikes.pipeline import SpikePipeline
from tire_vision.thread.depth.pipeline import DepthEstimatorPipeline
from tire_vision.config import TireVisionConfig


class TireVisionPipeline:
    def __init__(self, config: TireVisionConfig):
        self.config = config
        self.segmentator = SegmentationInferencer(config.segmentation)
        self.spike_pipeline = SpikePipeline(config.spikes)
        self.depth_pipeline = DepthEstimatorPipeline(config.depth)

    def __call__(self, image: torch.Tensor) -> Dict[str, Any]:
        cropped_image = self.segmentator.crop_tire(image)
        cropped_image = cropped_image.to(torch.float32) / 255

        spikes = self.spike_pipeline.detect_spikes(cropped_image)

        depth = self.depth_pipeline.estimate_depth(cropped_image)

        cropped_image = cropped_image.cpu().numpy().transpose(1, 2, 0)
        cropped_image = cropped_image * 255
        cropped_image = cropped_image.astype(np.uint8)

        return {
            "cropped_image": cropped_image,
            "depth": depth,
            "spikes": spikes,
        }
