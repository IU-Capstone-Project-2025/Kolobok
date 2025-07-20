from typing import List, Dict, Any
import time

import numpy as np
import cv2
import torch

from tire_vision.config import SpikePipelineConfig
from tire_vision.segmentation.onnx import OnnxSegmentator
from tire_vision.thread.spikes.classifier import SpikeClassifier

import logging


class SpikePipeline:
    def __init__(self, config: SpikePipelineConfig):
        self.config = config
        self.segmentator = OnnxSegmentator(
            self.config.spike_segmentator_onnx,
            self.config.resize_shape,
            self.config.confidence_threshold,
        )

        self.classifier = SpikeClassifier(self.config.spike_classifier_onnx)

        self.kernel = np.ones(shape=(3, 3), dtype=np.uint8)

        self.logger = logging.getLogger("spikes")

        self.logger.info("Spike Pipeline module initialized")

    @torch.no_grad()
    def detect_spikes(self, image: np.ndarray) -> List[Dict[str, Any]]:
        h, w, _ = image.shape
        self.logger.info("Running spike detection")
        start_time = time.perf_counter()

        self.logger.info("Running spike detection model")
        detection_mask = self.segmentator(image)

        self.logger.info("Generating detection mask")

        self.logger.info("Applying morphology")
        detection_mask = cv2.erode(
            detection_mask, self.kernel, iterations=self.config.erosion_iterations
        )
        detection_mask = cv2.dilate(
            detection_mask, self.kernel, iterations=self.config.dilation_iterations
        )

        self.logger.info("Running connected components")
        _, _, _, centroids = cv2.connectedComponentsWithStats(
            detection_mask, connectivity=8, ltype=cv2.CV_32S
        )

        centroids = centroids.astype(np.int32)[1:]
        boxes = []

        crop_half = self.config.crop_size // 2

        self.logger.info("Extracting spike crops")
        crops = []
        for c1, c2 in centroids:
            x1 = np.clip(c1 - crop_half, 0, w)
            y1 = np.clip(c2 - crop_half, 0, h)
            x2 = np.clip(c1 + crop_half, 0, w)
            y2 = np.clip(c2 + crop_half, 0, h)

            spike_image = image[y1:y2, x1:x2, :]

            crop_h, crop_w = spike_image.shape[:2]
            if crop_h != self.config.crop_size or crop_w != self.config.crop_size:
                dh = self.config.crop_size - crop_h
                dw = self.config.crop_size - crop_w
                pad_width = (
                    (dh // 2, dh // 2 + dh % 2),
                    (dw // 2, dw // 2 + dw % 2),
                    (0, 0),
                )
                spike_image = np.pad(
                    spike_image, pad_width, mode="constant", constant_values=0
                )

            crops.append(spike_image)

            boxes.append(
                {
                    "box": (
                        int(x1),
                        int(y1),
                        int(x2 - x1),
                        int(y2 - y1),
                    ),
                }
            )

        spikes = []
        if len(crops) > 0:
            crops = np.stack(crops).astype(np.float32).transpose(0, 3, 1, 2) / 255

            spike_classes = self.classifier(crops).argmax(axis=1)

            for box, spike_class in zip(boxes, spike_classes):
                if spike_class == 2:
                    continue
                spikes.append({**box, "class": int(spike_class)})

        latency = time.perf_counter() - start_time
        self.logger.info(f"Spike detection completed in {latency:.4f} seconds")

        return spikes
