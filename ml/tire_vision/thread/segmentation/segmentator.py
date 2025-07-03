from pathlib import Path
import sys
import os
from typing import Literal, Union
import time

import torch
from torchvision.transforms import functional as VF, InterpolationMode

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg, CfgNode as CN
from detectron2.engine import DefaultTrainer
from detectron2.projects.deeplab import add_deeplab_config

from huggingface_hub import hf_hub_download, login

from tire_vision.thread.segmentation.coco_stuff import COCO_CATEGORIES
from tire_vision.config import SegmentationConfig

import logging


cur_dir = Path(__file__)
root_dir = cur_dir.parent.parent.parent.parent
san_dir = root_dir / "external" / "SAN"

try:
    sys.path.append(str(san_dir))
    import san  # noqa: F401

except ImportError:
    raise ImportError(
        "This project requires SAN repository to be initialized. "
        + "Please, run `git submodule update --init --recursive`"
    )

CFG_PATH = san_dir / "configs/san_clip_vit_res4_coco.yaml"
CKPT_PATH = "huggingface:san_vit_b_16.pth"

AugmentationMode = Literal["COCO-stuff", "COCO-all"]
SegmentationMode = Literal["accurate", "efficient"]


def download_model(model_path: str):
    model_path = model_path.split(":")[1]
    if "HF_TOKEN" in os.environ:
        login(token=os.environ["HF_TOKEN"])

    model_path = hf_hub_download("Mendel192/san", filename=model_path)

    return model_path


def add_san_config(cfg):
    # copied from maskformer2
    cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_semantic"
    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    # solver config
    # optimizer
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    cfg.SOLVER.WEIGHT_DECAY_EMBED_GROUP = [
        "absolute_pos_embed",
        "positional_embedding",
        "pos_embed",
        "query_embed",
        "relative_position_bias_table",
    ]
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0
    cfg.SOLVER.CLIP_MULTIPLIER = 1.0
    cfg.SOLVER.TEST_IMS_PER_BATCH = 1

    # san
    cfg.MODEL.SAN = CN()
    cfg.MODEL.SAN.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.SAN.CLASS_WEIGHT = 2.0
    cfg.MODEL.SAN.DICE_WEIGHT = 5.0
    cfg.MODEL.SAN.MASK_WEIGHT = 5.0
    cfg.MODEL.SAN.TRAIN_NUM_POINTS = 112 * 112
    cfg.MODEL.SAN.NUM_CLASSES = 171
    cfg.MODEL.SAN.OVERSAMPLE_RATIO = 3.0
    cfg.MODEL.SAN.IMPORTANCE_SAMPLE_RATIO = 0.75
    cfg.MODEL.SAN.CLIP_MODEL_NAME = "ViT-B/16"
    cfg.MODEL.SAN.CLIP_PRETRAINED_NAME = "openai"
    cfg.MODEL.SAN.CLIP_TEMPLATE_SET = "vild"
    cfg.MODEL.SAN.FEATURE_LAST_LAYER_IDX = 9
    cfg.MODEL.SAN.CLIP_FROZEN_EXCLUDE = ["positional_embedding"]
    cfg.MODEL.SAN.CLIP_DEEPER_FROZEN_EXCLUDE = []
    cfg.MODEL.SAN.REC_CROSS_ATTN = False
    cfg.MODEL.SAN.REC_DOWNSAMPLE_METHOD = "max"
    cfg.MODEL.SAN.SOS_TOKEN_FORMAT = "cls_token"
    cfg.MODEL.SAN.SIZE_DIVISIBILITY = 32
    cfg.MODEL.SAN.ASYMETRIC_INPUT = True
    cfg.MODEL.SAN.CLIP_RESOLUTION = 0.5

    cfg.MODEL.SAN.SEM_SEG_POSTPROCESS_BEFORE_INFERENCE = True
    # side adapter
    cfg.MODEL.SIDE_ADAPTER = CN()
    cfg.MODEL.SIDE_ADAPTER.NAME = "RegionwiseSideAdapterNetwork"
    cfg.MODEL.SIDE_ADAPTER.VIT_NAME = "vit_w240n6d8_patch16"
    cfg.MODEL.SIDE_ADAPTER.PRETRAINED = False
    cfg.MODEL.SIDE_ADAPTER.IMAGE_SIZE = 640
    cfg.MODEL.SIDE_ADAPTER.DROP_PATH_RATE = 0.0
    cfg.MODEL.SIDE_ADAPTER.NUM_QUERIES = 100
    cfg.MODEL.SIDE_ADAPTER.FUSION_TYPE = "add"
    cfg.MODEL.SIDE_ADAPTER.FUSION_MAP = ["0->0", "3->1", "6->2", "9->3"]
    cfg.MODEL.SIDE_ADAPTER.DEEP_SUPERVISION_IDXS = [7, 8]

    cfg.MODEL.SIDE_ADAPTER.ATTN_BIAS = CN()
    cfg.MODEL.SIDE_ADAPTER.ATTN_BIAS.NUM_HEADS = 12
    cfg.MODEL.SIDE_ADAPTER.ATTN_BIAS.NUM_LAYERS = 1
    cfg.MODEL.SIDE_ADAPTER.ATTN_BIAS.EMBED_CHANNELS = 256
    cfg.MODEL.SIDE_ADAPTER.ATTN_BIAS.MLP_CHANNELS = 256
    cfg.MODEL.SIDE_ADAPTER.ATTN_BIAS.MLP_NUM_LAYERS = 3
    cfg.MODEL.SIDE_ADAPTER.ATTN_BIAS.RESCALE_ATTN_BIAS = True

    # wandb
    cfg.WANDB = CN()
    cfg.WANDB.PROJECT = "san"
    cfg.WANDB.NAME = None
    # use flash attention
    cfg.MODEL.FLASH = False


def setup(config_file: str, device: str):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_san_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.MODEL.DEVICE = device
    return cfg


class SegmentationInferencer:
    def __init__(
        self,
        config: SegmentationConfig,
    ):
        self.config = config
        self.device = config.device
        self.target = config.target
        self.vocab = [self.target]
        self.vocab_aug_mode = config.vocab_aug_mode
        self.segmentation_mode = config.segmentation_mode
        self._augment_vocabulary()
        self.logger = logging.getLogger("san")

        cfg = setup(str(CFG_PATH), device=self.device)
        self.model = DefaultTrainer.build_model(cfg)
        self.ckpt_path = download_model(str(CKPT_PATH))
        DetectionCheckpointer(self.model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            self.ckpt_path
        )
        self.model.eval()
        self.model.to(self.device)

        self.logger.info("SegmentationInferencer module initialized")

    @staticmethod
    def _resize_map(
        map: torch.Tensor,
        h: int,
        w: int,
        interpolation: Literal["bicubic", "nearest"],
    ):
        if interpolation == "bicubic":
            return VF.resize(map, (h, w), interpolation=InterpolationMode.BICUBIC)
        elif interpolation == "nearest":
            return VF.resize(map, (h, w), interpolation=InterpolationMode.NEAREST)
        else:
            raise ValueError(f"Invalid interpolation mode: {interpolation}")

    @staticmethod
    def _get_binary_mask(logits: torch.Tensor):
        result = torch.where(torch.argmax(logits, dim=0, keepdim=True) == 0, 1, 0)
        return result.to(torch.uint8)

    @torch.no_grad()
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        self.logger.info("Running SAN forward pass")
        start_time = time.perf_counter()
        *_, h, w = img.shape

        if h > w:
            img_resized = VF.resize(
                img, (640, int(640 * w / h)), interpolation=InterpolationMode.BICUBIC
            )
        else:
            img_resized = VF.resize(
                img, (int(640 * h / w), 640), interpolation=InterpolationMode.BICUBIC
            )
        self.logger.info(
            f"Original shape: {img.shape}, Resized shape: {img_resized.shape}"
        )

        self.logger.info("Inferencing model")
        result = self.model([{"image": img_resized, "vocabulary": self.vocab_aug}])

        seg = result[0]["sem_seg"]

        if self.segmentation_mode == "efficient":
            self.logger.info("Getting segmentation mask in 'efficient' mode")
            mask = self._get_binary_mask(seg)
            mask = self._resize_map(mask, h, w, "nearest").squeeze(0)

        elif self.segmentation_mode == "accurate":
            self.logger.info("Getting segmentation mask in 'accurate' mode")
            seg = self._resize_map(seg, h, w, "bicubic")
            mask = self._get_binary_mask(seg).squeeze(0)
        else:
            raise ValueError(
                "segmentation_˝mode must be one of ['accurate', 'efficient']"
            )

        latency = time.perf_counter() - start_time
        self.logger.info(f"SAN forward pass completed in {latency:.4f} seconds")

        return mask

    def _augment_vocabulary(self):
        default_voc = [c["name"] for c in COCO_CATEGORIES]
        stuff_voc = [
            c["name"]
            for c in COCO_CATEGORIES
            if "isthing" not in c or c["isthing"] == 0
        ]
        vocab_set = set(self.vocab)

        if self.vocab_aug_mode == "COCO-all":
            self.vocab_aug = self.vocab + [c for c in default_voc if c not in vocab_set]
        elif self.vocab_aug_mode == "COCO-stuff":
            self.vocab_aug = self.vocab + [c for c in stuff_voc if c not in vocab_set]
        else:
            self.vocab_aug = self.vocab

    def crop_tire(self, img: torch.Tensor) -> Union[torch.Tensor, None]:
        img = img.to(self.device)
        *_, h, w = img.shape

        padding = (
            int(h * self.config.padding_frac),
            int(w * self.config.padding_frac),
        )

        mask = self.forward(img)
        if mask.sum() == 0:
            self.logger.error("No tire found on the image. Returning None")
            return None

        i, j = torch.where(mask == 1)

        min_i, max_i = torch.min(i), torch.max(i)
        min_j, max_j = torch.min(j), torch.max(j)

        if (
            max_i - min_i < self.config.min_tire_pixels
            or max_j - min_j < self.config.min_tire_pixels
        ):
            self.logger.error("Tire is too small. Returning None")
            return None

        img_rembg = img * mask + 255 * (1 - mask)

        min_i = max(0, min_i - padding[0])
        max_i = min(h, max_i + padding[0])
        min_j = max(0, min_j - padding[1])
        max_j = min(w, max_j + padding[1])

        return img_rembg[..., min_i:max_i, min_j:max_j]

    def rembg_tire(self, img: torch.Tensor) -> torch.Tensor:
        mask = self.forward(img)
        return img * mask + 255 * (1 - mask)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return self.forward(img)
