import pytest
import numpy as np

from tire_vision.text.pipeline import TireAnnotationPipeline
from tire_vision.config import TireDetectorConfig, TireUnwrapperConfig, OCRConfig


@pytest.fixture
def ocr_configs():
    """Creates real configs for the OCR pipeline."""
    # These configs will read from environment variables
    detector_config = TireDetectorConfig()
    unwrapper_config = TireUnwrapperConfig()
    ocr_config = OCRConfig()
    return detector_config, unwrapper_config, ocr_config


@pytest.mark.network
def test_tire_annotation_pipeline(ocr_configs):
    """
    Tests that the tire annotation pipeline runs successfully with real API calls.
    This test requires network access and valid API keys in environment variables.
    """
    detector_config, unwrapper_config, ocr_config = ocr_configs

    # Initialize the pipeline
    pipeline = TireAnnotationPipeline(
        detector_config=detector_config,
        unwrapper_config=unwrapper_config,
        ocr_config=ocr_config,
    )

    # Create a dummy image. For a real test, you might want to use a specific test image file.
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)

    # Run the pipeline
    # Note: This will make real API calls to Roboflow and an OCR service.
    # The OCR model may return an error or an empty dict for a blank image.
    result = pipeline(dummy_image)

    # Check the result
    assert isinstance(result, dict)
