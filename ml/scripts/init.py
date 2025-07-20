import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("initializer")

logger.info("Initializing service...")

from tire_vision.config import TireVisionConfig  # noqa: E402, F403
from utils import *  # noqa: E402, F403

cfg = TireVisionConfig()

logger.info("Service initialized successfully!")
