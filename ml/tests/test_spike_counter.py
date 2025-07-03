import pytest
import torch

from tire_vision.thread.spikes.pipeline import SpikePipeline
from tire_vision.config import SpikePipelineConfig


@pytest.fixture
def spike_pipeline_config():
    """Creates a real config for the spike pipeline."""
    return SpikePipelineConfig(device="cpu")


def test_spike_pipeline(spike_pipeline_config):
    """Tests that the spike pipeline runs successfully with real models."""
    
    pipeline = SpikePipeline(config=spike_pipeline_config)

    # Create a dummy image
    dummy_image = torch.randn(3, 256, 256)

    # Run the pipeline
    result = pipeline.detect_spikes(dummy_image)

    # Check the result
    assert isinstance(result, list)
