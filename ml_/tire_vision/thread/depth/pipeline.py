import torch
from torchvision import transforms

from tire_vision.thread.depth.model import get_depth_estimator
from tire_vision.config import DepthEstimatorConfig


class DepthEstimatorPipeline:
    def __init__(self, config: DepthEstimatorConfig):
        self.config = config
        self.transform = transforms.Resize(
            self.config.resize_shape, interpolation=transforms.InterpolationMode.BICUBIC
        )
        self.model = get_depth_estimator(self.config.model_name, self.config.checkpoint)
        self.model.to(self.config.device)
        self.model.eval()

    @torch.no_grad()
    def estimate_depth(self, image: torch.Tensor) -> float:
        image_device = image.to(self.config.device)
        image_device = self.transform(image_device)

        result = self.model(image_device[None])[0, 0].cpu().exp().item()

        return result
