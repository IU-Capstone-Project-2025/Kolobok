import logging
from dataclasses import dataclass
from typing import Tuple

import tyro
import torch
from torch import nn
from torchvision.models import (
    densenet201,
    googlenet,
    swin_v2_t,
    swin_s,
    efficientnet_b7,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("torch2onnx")


def get_model(model_name: str) -> nn.Module:
    if model_name == "swin_s":
        model = swin_s()
        model.head = nn.Linear(768, 1)
    elif model_name == "effnet_b7":
        model = efficientnet_b7()
        model.classifier = nn.Linear(2560, 1)
    elif model_name == "swin_v2_t":
        model = swin_v2_t()
        model.head = nn.Linear(768, 1)
    elif model_name == "densenet201":
        model = densenet201()
        model.classifier = nn.Linear(1920, 1)
    elif model_name == "googlenet":
        model = googlenet()
        model.fc = nn.Linear(1024, 1)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model


def load_checkpoint(ckpt_path: str, pl_prefix: str = "model."):
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "state_dict" in state_dict:
        state_dict = {
            k[len(pl_prefix) :] if k.startswith(pl_prefix) else k: v
            for k, v in state_dict["state_dict"].items()
        }

    return state_dict


def get_depth_estimator(model_name: str, checkpoint_path: str):
    model = get_model(model_name)
    checkpoint = load_checkpoint(checkpoint_path)
    model.load_state_dict(checkpoint)
    model.eval()
    return model


@dataclass
class ModelConfig:
    model_name: str
    ckpt_path: str
    onnx_path: str
    input_shape: Tuple[int, int] = (512, 512)


@dataclass
class Args:
    depth: ModelConfig


@torch.no_grad()
def main():
    args = tyro.cli(Args)

    dummy_input = torch.rand(1, 3, *args.depth.input_shape)

    logger.info(f"Converting {args.depth.ckpt_path} to ONNX...")
    model = get_depth_estimator(args.depth.model_name, args.depth.ckpt_path)

    model.eval()

    torch.onnx.export(
        model,
        dummy_input,
        args.depth.onnx_path,
        verbose=True,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    logger.info(f"Saved {args.depth.onnx_path}")


if __name__ == "__main__":
    main()
