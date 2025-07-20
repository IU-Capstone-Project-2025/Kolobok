import onnxruntime
import numpy as np

import torch
from torchvision.transforms import functional as VF, InterpolationMode

from tire_vision.config import DepthRegressorConfig


class DepthRegressor:
    def __init__(self, config: DepthRegressorConfig):
        self.config = config
        self.session = onnxruntime.InferenceSession(self.config.depth_regressor_onnx)

    def forward(self, image: np.ndarray):
        h, w, _ = image.shape
        image = image.transpose(2, 0, 1)[None].astype(np.float32) / 255
        image_torch = torch.from_numpy(image)
        image_torch = VF.resize(
            image_torch,
            self.config.resize_shape,
            interpolation=InterpolationMode.BILINEAR,
        )

        result = self.session.run(None, {"input": image_torch.numpy()})[0]

        result_scaled = 10 / (1 + np.exp(-np.squeeze(result)))

        return float(result_scaled)

    def __call__(self, image: np.ndarray):
        return self.forward(image)
