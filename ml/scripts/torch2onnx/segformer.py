import logging
from dataclasses import dataclass

import tyro
import torch
from torch import nn

from transformers import SegformerConfig, SegformerForSemanticSegmentation


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("torch2onnx")


@dataclass
class ModelConfig:
    ckpt_path: str
    onnx_path: str
    hf_model_id: str = "nvidia/segformer-b1-finetuned-ade-512-512"


@dataclass
class Args:
    model: ModelConfig


def load_checkpoint(ckpt_path: str, pl_prefix: str = "model."):
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "state_dict" in state_dict:
        state_dict = {
            k[len(pl_prefix) :] if k.startswith(pl_prefix) else k: v
            for k, v in state_dict["state_dict"].items()
        }

    return state_dict


class SegformerWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor):
        logits = self.model(images).logits
        return logits


@torch.no_grad()
def main():
    args = tyro.cli(Args)

    dummy_input = torch.rand(1, 3, 512, 512)

    logger.info(f"Converting {args.model.ckpt_path} to ONNX...")
    state_dict = load_checkpoint(args.model.ckpt_path)

    hf_model_config = SegformerConfig.from_pretrained(args.model.hf_model_id)
    hf_model_config.num_labels = 1

    base_model = SegformerForSemanticSegmentation._from_config(hf_model_config)

    model = SegformerWrapper(base_model)

    model.load_state_dict(state_dict)
    model.eval()

    torch.onnx.export(
        model,
        dummy_input,
        args.model.onnx_path,
        verbose=True,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    logger.info(f"Saved {args.model.onnx_path}")


if __name__ == "__main__":
    main()
