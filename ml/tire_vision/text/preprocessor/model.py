from collections import defaultdict

import numpy as np

from inference import get_roboflow_model

from tire_vision.config import TireDetectorConfig


class TireDetector:
    def __init__(self, config: TireDetectorConfig):
        self.config = config
        self.model = get_roboflow_model(
            model_id=self.config.model_id,
            api_key=self.config.roboflow_api_key,
        )

        self.rim_class_name = "rim"
        self.tire_class_name = "wheel"

    def detect(self, image: np.ndarray):
        result = self.model.infer(image)

        output = defaultdict(lambda: None)

        for prediction in result[0].predictions:
            class_name = prediction.class_name
            output[class_name] = np.array(
                [[p.x, p.y] for p in prediction.points], dtype=np.int32
            )

        assert isinstance(output[self.rim_class_name], np.ndarray)
        assert isinstance(output[self.tire_class_name], np.ndarray)

        return output
