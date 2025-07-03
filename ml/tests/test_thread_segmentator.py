import pytest
import torch
from tire_vision.thread.segmentation.segmentator import SegmentationInferencer
from tire_vision.config import SegmentationConfig


@pytest.fixture
def segmentation_config():
    """Creates a real config for the segmentation inferencer."""
    return SegmentationConfig(device="cpu")


@pytest.mark.network
def test_segmentation_inferencer(segmentation_config):
    """
    Tests that the segmentation inferencer runs successfully.
    This test requires network access to download the model checkpoint.
    """
    # Initialize the pipeline with the real config
    # This will download the model from Hugging Face Hub
    pipeline = SegmentationInferencer(config=segmentation_config)

    # Create a dummy image
    dummy_image = torch.randn(3, 480, 640)

    # Run the pipeline
    result = pipeline(dummy_image)

    # Check the result
    assert isinstance(result, torch.Tensor)
    assert result.shape == (480, 640)
    assert result.dtype == torch.uint8
