from dataclasses import dataclass
import os
from typing import Literal, Tuple
import warnings

import torch

DEVICE = os.environ["DEVICE"]
if DEVICE.startswith("cuda") and not torch.cuda.is_available():
    warnings.warn("CUDA is not available, using CPU")
    DEVICE = "cpu"


CLASS_MAPPING = {
    0: "good",
    1: "bad",
}

CLASS_COLORS = {
    "good": (0, 255, 0),
    "bad": (0, 0, 255),
}


@dataclass
class SegmentationConfig:
    device: str = DEVICE
    target: str = "wheel-tire-thread"
    vocab_aug_mode: Literal["COCO-stuff", "COCO-all", "none"] = "COCO-stuff"
    segmentation_mode: Literal["accurate", "efficient"] = "accurate"
    padding_frac: float = 0.01


@dataclass
class SpikePipelineConfig:
    detector_checkpoint: str = os.environ["SPIKE_DETECTOR_CHECKPOINT"]
    classifier_checkpoint: str = os.environ["SPIKE_CLASSIFIER_CHECKPOINT"]
    device: str = DEVICE
    detection_threshold: float = 0.5
    erosion_iterations: int = 3
    dilation_iterations: int = 3
    crop_size: int = 32


@dataclass
class DepthEstimatorConfig:
    model_name: str = os.environ["DEPTH_ESTIMATOR_MODEL_NAME"]
    checkpoint: str = os.environ["DEPTH_ESTIMATOR_CHECKPOINT"]
    device: str = DEVICE
    resize_shape: Tuple[int, int] = (512, 512)


OCR_PROMPT = """You are provided with an image of a tire. Your task is to determine the following parameters of the tire:
1) manufacturer - manufacturer of the tire, if provided, else null
2) model - model of the tire, string
3) tire_size_string - this size of the tire as a single string in format "<width, int>/<aspect-ratio, int><construction-type, char>/<diameter, int> <max-weight, int><speed-rating, char>", for example, "205/55R/16 91V". usually, the format is always the same

each field must be either the answer to the question, or null, if you cannot find this information on the image.
Provide output strictly in json format, without anything else 
Example:
{
    "manufacturer": "...",
    "model": "...",
    "tire_size_string": "..."
}"""


@dataclass
class TireDetectorConfig:
    model_id: str = os.environ["TIRE_DETECTOR_MODEL_ID"]
    roboflow_api_key: str = os.environ["ROBOFLOW_API_KEY"]


@dataclass
class TireVisionConfig:
    segmentation = SegmentationConfig()
    spikes = SpikePipelineConfig()
    depth = DepthEstimatorConfig()
    tire_detector = TireDetectorConfig()
