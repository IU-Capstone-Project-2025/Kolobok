from typing import List, Dict, Any
import time

import numpy as np
import cv2
import torch
from torch.nn import functional as F

from tire_vision.thread.spikes.model import get_spike_detector, get_spike_classifier
from tire_vision.config import SpikePipelineConfig

import logging


class SpikePipeline:
    def __init__(self, config: SpikePipelineConfig):
        self.config = config
        self.detector = get_spike_detector(self.config.detector_checkpoint)
        self.detector.to(self.config.device)
        self.detector.eval()

        self.classifier = get_spike_classifier(self.config.classifier_checkpoint)
        self.classifier.to(self.config.device)
        self.classifier.eval()

        self.threshold = self.config.detection_threshold

        self.kernel = np.ones(shape=(3, 3), dtype=np.uint8)

        self.logger = logging.getLogger("spikes")

        self.logger.info("Spike Pipeline module initialized")

    @torch.no_grad()
    def detect_spikes(self, image: torch.Tensor) -> List[Dict[str, Any]]:
        *_, h, w = image.shape
        self.logger.info("Running spike detection")
        start_time = time.perf_counter()

        image_device = image.to(self.config.device)

        self.logger.info("Running spike detection model")
        detection_probs = self.detector(image_device[None])[0, 0]

        self.logger.info("Generating detection mask")
        detection_mask = (
            (detection_probs > self.config.detection_threshold)
            .cpu()
            .numpy()
            .astype(np.uint8)
        )

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
        spikes = []

        crop_half = self.config.crop_size // 2

        self.logger.info("Extracting spike crops")
        for c1, c2 in centroids:
            x1 = max(0, c1 - crop_half)
            y1 = max(0, c2 - crop_half)
            x2 = min(w, c1 + crop_half)
            y2 = min(h, c2 + crop_half)

            spike_image = image_device[:, y1:y2, x1:x2]

            crop_h, crop_w = spike_image.shape[1:]
            if crop_h != self.config.crop_size or crop_w != self.config.crop_size:
                dh = self.config.crop_size - crop_h
                dw = self.config.crop_size - crop_w
                to_pad = (dw // 2, dw // 2 + dw % 2, dh // 2, dh // 2 + dh % 2)
                spike_image = F.pad(spike_image, to_pad)

            spike_class_logits = self.classifier(spike_image[None])[0]

            spike_class = torch.argmax(spike_class_logits).item()
            if spike_class == 2:
                continue

            spikes.append(
                {
                    "box": (
                        x1.item(),
                        y1.item(),
                        (x2 - x1).item(),
                        (y2 - y1).item(),
                    ),
                    "class": spike_class,
                }
            )

        latency = time.perf_counter() - start_time
        self.logger.info(f"Spike detection completed in {latency:.4f} seconds")

        return spikes
