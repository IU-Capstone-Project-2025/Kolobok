import logging

import torch
from torch import nn

from transformers import SegformerConfig, SegformerForSemanticSegmentation

from tire_vision.config import TireVisionConfig

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("torch2onnx")


class SegformerWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor):
        logits = self.model(images).logits
        return logits


@torch.no_grad()
def main():
    logger.info("Converting model to ONNX...")

    cfg = TireVisionConfig()
    ckpt_path = cfg.sidewall_segmentator_config.segmentator_checkpoint
    if ckpt_path is None:
        raise ValueError("Checkpoint not found")
    onnx_path = cfg.sidewall_segmentator_config.segmentator_onnx

    logger.info("Loading checkpoint...")

    state_dict = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=True,
    )

    logger.info("Loading model config...")

    model_config = SegformerConfig.from_pretrained(
        cfg.sidewall_segmentator_config.hf_model_id
    )
    model_config.num_labels = 1

    base_model = SegformerForSemanticSegmentation._from_config(model_config)

    model = SegformerWrapper(base_model)

    model.load_state_dict(state_dict)
    model.eval()

    logger.info("Exporting model to ONNX...")

    dummy_input = torch.rand(1, 3, 512, 512)

    _ = torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        verbose=True,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    logger.info("Model exported to ONNX successfully!")


if __name__ == "__main__":
    main()
