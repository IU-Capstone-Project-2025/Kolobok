from dataclasses import dataclass
import os
from typing import Literal, Tuple, Optional
import warnings

import torch
import cv2

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
    "bad": (255, 0, 0),
}


@dataclass
class SegmentationConfig:
    device: str = DEVICE
    target: str = "wheel-tire-thread"
    vocab_aug_mode: Literal["COCO-stuff", "COCO-all", "none"] = "COCO-stuff"
    segmentation_mode: Literal["accurate", "efficient"] = "efficient"
    padding_frac: float = 0.01
    min_tire_pixels: int = 96


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
3) tire_size_string - this size of the tire as a single string in format "<width, int>/<aspect-ratio, int><construction-type, char>/<diameter, int> <max-weight, int><speed-rating, char>", for example, "205/55R/16 91V". usually, the format is always the same.

each field must be either the answer to the question, or null, if you cannot find this information on the image.
Answer precisely what is written on the tire.
Note about tire_size_string: sometimes, the format of the size is not as described above, in such cases, return tire_size_string as it is written on the tire.
Note about model and manufacturer: usually, the manufacturer and model names have the largest font size among all the text on the image, but distinguishing model and manufacturer name is up to you.
Note about the problem as a whole: some information might be duplicated multiple times, which makes the task easier, but overwhelming to resolve conflicts. Try to focus on the most prominent options.
You are allowed and encouraged to reason about the output, but your final answer has to be in JSON format. My recommendation is to use the following format:

Example:
On the image (images) I can see the following text: ... <here you can list all the text you see on the image>
some of those repeat multiple times, which introduces multiple candidates for manufacturer, model, and tire_size_string. To avoid inaccuracies, I will focus on the most prominent ones.
<text>, <text> have the largest font size, so they might be manufacturer and model names.
I am not confident about tire size, as I can see that it is written in multiple spots. Hence, I will return the most clear writing as tire_size_string.
All in all, most likely, manufacturer is ..., model is ..., tire_size_string is ...,
Final answer:
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
class TireUnwrapperConfig:
    crop_enlarge_factor: float = 1.1
    polar_angle_steps: int = 360
    polar_flags: int = cv2.WARP_POLAR_LINEAR | cv2.WARP_FILL_OUTLIERS
    perspective_margin: int = 10
    clahe_mode: Literal["disabled", "luminance", "colors", "black-and-white"] = (
        "black-and-white"
    )
    clahe_clip_limit: float = 5.0
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)
    cut_strip: bool = True
    cut_mask_threshold: float = 0.5
    concat_strip: bool = True


@dataclass
class OCRConfig:
    model_name: str = "openai/gpt-4o-mini"
    # model_name: str = "cuuupid/glm-4v-9b:69196a237cdc310988a4b12ad64f4b36d10189428c19a18526af708546e1856f"
    prompt: str = OCR_PROMPT
    top_p: float = 0.9
    temperature: float = 0.7
    presence_penalty: float = 0
    frequency_penalty: float = 0
    max_completion_tokens: int = 4096


@dataclass
class TireVisionConfig:
    segmentation = SegmentationConfig()
    spikes = SpikePipelineConfig()
    depth = DepthEstimatorConfig()
    tire_detector = TireDetectorConfig()
    tire_unwrapper = TireUnwrapperConfig()
    ocr = OCRConfig()
