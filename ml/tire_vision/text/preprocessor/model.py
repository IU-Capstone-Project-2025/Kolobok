import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import logging

from torchvision.transforms import functional as VF, InterpolationMode

from transformers import SegformerConfig, SegformerForSemanticSegmentation


from tire_vision.config import SidewallSegmentatorConfig


class SegformerWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor):
        logits = self.model(images).logits
        return logits


class SidewallSegmentator:
    def __init__(self, config: SidewallSegmentatorConfig):
        self.config = config

        model_config = SegformerConfig.from_pretrained(self.config.hf_model_id)
        model_config.num_labels = 1

        base_model = SegformerForSemanticSegmentation._from_config(model_config)
        self.logger = logging.getLogger("sidewall_segmentator")

        self.model = SegformerWrapper(base_model)
        if self.config.segmentator_checkpoint:
            self.model.load_state_dict(
                torch.load(
                    self.config.segmentator_checkpoint,
                    map_location=self.config.device,
                    weights_only=True,
                )
            )
        else:
            self.logger.warning(
                "Sidewall segmentator checkpoint not found, using random weights"
            )
        self.model.to(self.config.device)
        self.model.eval()

    @torch.no_grad()
    def forward(self, image: np.ndarray):
        torch_image = (
            torch.from_numpy(image)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.config.device, torch.float32)
            / 255
        )
        *_, h, w = torch_image.shape

        torch_image = VF.resize(
            torch_image,
            self.config.resize_shape,
            interpolation=InterpolationMode.BICUBIC,
        )
        logits = self.model(torch_image)
        logits = VF.resize(
            logits, (h, w), interpolation=InterpolationMode.BICUBIC
        ).squeeze()

        mask = (F.sigmoid(logits) > self.config.confidence_threshold).to(
            torch.uint8
        ).numpy() * 255

        return mask
