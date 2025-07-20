from dataclasses import dataclass, field
import os
from typing import Literal, Tuple, List


DEVICE = "cpu"


CLASS_MAPPING = {
    0: "good",
    1: "bad",
}

CLASS_COLORS = {
    "good": (0, 255, 0),
    "bad": (255, 0, 0),
}

SYSTEM_OCR_PROMPT = """You are an expert OCR model specializing in reading text from images of tires. 
Your tasks will involve reading, parsing, extracting, and analyzing information from the images of wheel tires

Your primary focus should be on the following properties of the tire:
- Brand
- Model
- Size in format <width>/<aspect_ratio>R<diameter> <load_index><speed_index> or just <width>/<aspect_ratio>R<diameter>

Do not include other redundant information, such as:
- external links (example: 'www.vatti-tyres.com')
- country of origin (example: 'Made in China')
- other information that is not really relevant to the tire identification. usually low-font text (examples: 'DOT 1H2 RYCHU 2323')

When constructing the response, try to include all the information you found (and at the same time, relevant).
Example:
If you find a string consisting of multiple words, return all possible combinations of them.

string on tire: 'GitiComfort F22'
response must include: ['GitiComfort F22', 'GitiComfort', 'F22']

string on tire: '215/50R17 95V' (size)
response must include: ['215/50R17 95V', '215/50R17', '95V']

string on tire: 'KAMA EURO-236' (brand and model, but you don't know which is which)
response must include: ['KAMA EURO-236', 'KAMA', 'EURO', 'EURO-236', 'KAMA EURO', '236']

The examples above are just examples, do not follow them strictly.
Do not make up any information, return only what you can see."""

OCR_PROMPT = """Your task is to extract text from the provided image(s) of a tire.
Extract all visible text from the image(s) that might be related to tire information (model, brand, size)
Particularly important to emphasize the tire size string. You should return it separately.
Present the extracted text as a JSON object with keys "strings", "tire_size".
- "strings" is a list of all (relevant) text strings found on the tire.
- "tire_size" is the tire size string found on the tire in format <width>/<aspect_ratio>R<diameter> <load_index><speed_index> or just <width>/<aspect_ratio>R<diameter> (if load_index and speed_index are not present)
Do not include any reasoning or explanations, only the final JSON object.

Example of a valid response:
{
    "strings": ["MICHELIN", "Pilot Sport 4 S", "Pilot Sport", "Pilot", "Sport", "Sport 4 S", "245/35ZR20", "95Y", "245/35ZR20 95Y"],
    "tire_size": "245/35ZR20 95Y"
}
"""


@dataclass
class ThreadSegmentatorConfig:
    thread_segmentator_onnx: str = "onnx/thread_segmentator.onnx"
    confidence_threshold: float = 0.5
    resize_shape: Tuple[int, int] = (512, 512)
    padding_frac: float = 0.01
    min_tire_pixels: int = 96


@dataclass
class SpikePipelineConfig:
    spike_segmentator_onnx: str = "onnx/spike_segmentator.onnx"
    spike_classifier_onnx: str = "onnx/spike_classifier.onnx"
    confidence_threshold: float = 0.5
    resize_shape: Tuple[int, int] = (512, 512)
    erosion_iterations: int = 0
    dilation_iterations: int = 0
    crop_size: int = 32


@dataclass
class DepthRegressorConfig:
    depth_regressor_onnx: str = "onnx/depth_regressor.onnx"
    resize_shape: Tuple[int, int] = (512, 512)


@dataclass
class SidewallSegmentatorConfig:
    sidewall_segmentator_onnx: str = "onnx/sidewall_segmentator.onnx"
    confidence_threshold: float = 0.5
    resize_shape: Tuple[int, int] = (512, 512)


@dataclass
class SidewallUnwrapperConfig:
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)
    concat_strip: bool = True
    mask_postprocess_ksize: int = 21
    rectify_aspect_ratio_threshold: float = 1.1
    polar_dsize: Tuple[int, int] = (700, 2000)
    concat_border_size: int = 5


@dataclass
class OCRConfig:
    model_name: str = "qwen/qwen2.5-vl-72b-instruct"
    base_url: str = "https://openrouter.ai/api/v1"
    api_key: str = os.environ["OPENROUTER_API_KEY"]
    system_prompt: str = SYSTEM_OCR_PROMPT
    prompt: str = OCR_PROMPT
    providers_list: List[str] = field(default_factory=lambda: ["nebius/fp8"])
    top_p: float = 0.9
    temperature: float = 0.7
    presence_penalty: float = 0
    frequency_penalty: float = 0
    max_completion_tokens: int = 1024


@dataclass
class IndexConfig:
    db_host: str = os.environ.get("DB_HOST", "mysql_db")
    db_port: int = int(os.environ.get("DB_PORT", "3306"))
    db_name: str = os.environ["MYSQL_DATABASE"]
    db_user: str = "root"
    db_password: str = os.environ["MYSQL_ROOT_PASSWORD"]
    table_name: str = "models"
    max_query_results: int = 5
    table_cache_path: str = "models.parquet"
    table_cache_ttl_seconds: int = 3600
    similarity_metric: Literal[
        "product",
        "arithmetic_mean",
        "harmonic_mean",
        "geometric_mean",
        "euclidean",
    ] = "harmonic_mean"


@dataclass
class TireVisionConfig:
    thread_segmentator_config = ThreadSegmentatorConfig()
    spike_pipeline_config = SpikePipelineConfig()
    depth_regressor_config = DepthRegressorConfig()
    sidewall_segmentator_config = SidewallSegmentatorConfig()
    sidewall_unwrapper_config = SidewallUnwrapperConfig()
    ocr_config = OCRConfig()
    index_config = IndexConfig()
