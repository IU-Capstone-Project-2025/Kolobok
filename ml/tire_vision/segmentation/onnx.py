from typing import Tuple, Optional

import numpy as np
import cv2
import onnxruntime


class OnnxSegmentator:
    def __init__(
        self,
        onnx_path: str,
        resize_shape: Tuple[int, int],
        threshold: Optional[float] = None,
    ):
        self.onnx_path = onnx_path
        self.resize_shape = resize_shape
        self.threshold = threshold
        self.session = onnxruntime.InferenceSession(self.onnx_path)

    def forward(self, image: np.ndarray):
        """
        Args:
            image: np.ndarray (H, W, 3), np.uint8,

        Returns:
            mask: np.ndarray (H, W), np.uint8 if threshold is None else np.float32 (normalized to [0, 1]),
        """
        h, w, _ = image.shape

        resized_image = (
            cv2.resize(image, self.resize_shape, interpolation=cv2.INTER_LINEAR)
            .transpose(2, 0, 1)[None]
            .astype(np.float32)
            / 255
        )

        logits = self.session.run(
            None,
            {"input": resized_image},
        )[0]

        logits_squeezed = np.squeeze(logits, axis=(0, 1))
        probs = 1 / (1 + np.exp(-logits_squeezed))
        probs = cv2.resize(probs, (w, h), interpolation=cv2.INTER_LINEAR)

        if self.threshold is not None:
            return (probs > self.threshold).astype(np.uint8) * 255

        return probs

    def __call__(self, image: np.ndarray):
        return self.forward(image)
