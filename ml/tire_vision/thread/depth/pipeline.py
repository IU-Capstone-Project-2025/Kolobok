import time

import torch
from torchvision import transforms

from tire_vision.thread.depth.model import get_depth_estimator
from tire_vision.config import DepthEstimatorConfig

import logging


class DepthEstimatorPipeline:
    def __init__(self, config: DepthEstimatorConfig):
        self.config = config
        self.transform = transforms.Resize(
            self.config.resize_shape, interpolation=transforms.InterpolationMode.BICUBIC
        )
        self.model = get_depth_estimator(self.config.model_name, self.config.checkpoint)
        self.model.to(self.config.device)
        self.model.eval()
        self.logger = logging.getLogger("depth")

        self.logger.info("DepthEstimatorPipeline module initialized")

    @torch.no_grad()
    def estimate_depth(self, image: torch.Tensor) -> float:
        image_device = image.to(self.config.device)
        start_time = time.perf_counter()

        self.logger.info("Applying transform")
        image_device = self.transform(image_device)

        self.logger.info("Estimating depth")
        result = self.model(image_device[None])[0, 0].cpu().exp().item()

        latency = time.perf_counter() - start_time
        self.logger.info(f"Depth estimation completed in {latency:.4f} seconds")

        return result
