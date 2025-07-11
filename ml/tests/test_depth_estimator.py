import pytest
import torch

from tire_vision.thread.depth.pipeline import DepthEstimatorPipeline
from tire_vision.config import DepthEstimatorConfig


@pytest.fixture
def depth_estimator_config():
    """Creates a real config for the depth estimator."""
    return DepthEstimatorConfig(device="cpu")


def test_depth_estimator_pipeline(depth_estimator_config):
    """Tests that the depth estimator pipeline runs successfully with a real model."""
    # This test will fail if the environment variables for the checkpoint are not set.
    pipeline = DepthEstimatorPipeline(config=depth_estimator_config)

    # Create a dummy image
    # The image size should match the model's expected input size from the config
    dummy_image = torch.randn(3, *depth_estimator_config.resize_shape)

    # Run the pipeline
    result = pipeline.estimate_depth(dummy_image)

    # Check the result
    assert isinstance(result, float)
