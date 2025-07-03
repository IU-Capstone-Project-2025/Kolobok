import pytest
import numpy as np

from tire_vision.text.preprocessor.unwrapper import TireUnwrapper
from tire_vision.config import TireUnwrapperConfig


@pytest.fixture
def tire_unwrapper():
    """Creates a TireUnwrapper with a default config."""
    config = TireUnwrapperConfig()
    return TireUnwrapper(config)


def test_tire_unwrapper(tire_unwrapper):
    """Tests that the tire unwrapper runs successfully."""
    # Create a dummy image
    image = np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8)

    # Create dummy polygons that are somewhat realistic
    # A larger rectangle for the tire
    tire_polygon = np.array([
        [10, 10], [630, 10], [630, 470], [10, 470]
    ])
    # A smaller rectangle for the rim inside the tire
    rim_polygon = np.array([
        [100, 100], [540, 100], [540, 380], [100, 380]
    ])

    # Run the unwrapper
    result = tire_unwrapper.get_unwrapped_tire(image, tire_polygon, rim_polygon)

    # Check the result
    assert isinstance(result, np.ndarray)
    assert len(result.shape) == 3
    assert result.shape[2] == 3 # Check for 3 channels (BGR)
