import os

import torch
from torch import nn
from torchvision.transforms import functional as VF, InterpolationMode
from torchvision.models import googlenet, GoogLeNet_Weights

from transformers import SegformerForSemanticSegmentation, SegformerConfig


SEGFORMER_MODEL_NAME = "nvidia/segformer-b2-finetuned-ade-512-512"


class SegformerWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def _upscale_logits(self, logits: torch.Tensor, target_shape: tuple = (512, 512)):
        logits = VF.resize(
            logits,
            size=target_shape,
            interpolation=InterpolationMode.BILINEAR,
        )
        return logits

    def forward(self, images: torch.Tensor):
        logits = self.model(images).logits
        logits = self._upscale_logits(logits, target_shape=images.shape[2:])
        return logits


def get_spike_detector(ckpt_path: str):
    config = SegformerConfig.from_pretrained(SEGFORMER_MODEL_NAME)
    config.num_labels = 1

    base_model = SegformerForSemanticSegmentation._from_config(config)

    model = SegformerWrapper(base_model)

    model.load_state_dict(torch.load(ckpt_path, weights_only=True, map_location="cpu"))
    model.eval()

    return model


def get_spike_classifier(ckpt_path: str):
    model = googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(1024, 3)

    model.load_state_dict(torch.load(ckpt_path, weights_only=True, map_location="cpu"))
    model.eval()

    return model
