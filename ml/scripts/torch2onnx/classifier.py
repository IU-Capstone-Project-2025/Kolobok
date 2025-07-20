import logging
from dataclasses import dataclass
from typing import Tuple

import tyro
import torch
from torch import nn
from torchvision.models import GoogLeNet_Weights, googlenet

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("torch2onnx")


@dataclass
class ModelConfig:
    ckpt_path: str
    onnx_path: str
    input_shape: Tuple[int, int] = (32, 32)


@dataclass
class Args:
    spike: ModelConfig


def load_checkpoint(ckpt_path: str, pl_prefix: str = "model."):
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "state_dict" in state_dict:
        state_dict = {
            k[len(pl_prefix) :] if k.startswith(pl_prefix) else k: v
            for k, v in state_dict["state_dict"].items()
        }

    return state_dict


@torch.no_grad()
def main():
    args = tyro.cli(Args)

    dummy_input = torch.rand(1, 3, *args.spike.input_shape)

    logger.info(f"Converting {args.spike.ckpt_path} to ONNX...")
    state_dict = load_checkpoint(args.spike.ckpt_path)

    model = googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(1024, 3)

    model.load_state_dict(state_dict)
    model.eval()

    torch.onnx.export(
        model,
        dummy_input,
        args.spike.onnx_path,
        verbose=True,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    logger.info(f"Saved {args.spike.onnx_path}")


if __name__ == "__main__":
    main()
