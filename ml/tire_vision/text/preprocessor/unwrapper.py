import numpy as np
import cv2

from tire_vision.config import SidewallUnwrapperConfig


class SidewallUnwrapper:
    """Takes a tire mask, unwraps and enhances the sidewall."""

    def __init__(self, config: SidewallUnwrapperConfig):
        self.config = config
        self.clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_tile_grid_size,
        )

    def _postprocess_mask(self, mask: np.ndarray):
        """Performs morphological operations and keeps the largest connected component."""
        ksize = self.config.mask_postprocess_ksize
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        processed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern)
        processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_OPEN, kern)

        n, labels, stats, _ = cv2.connectedComponentsWithStats(processed_mask)
        if n <= 1:
            return processed_mask
        areas = stats[1:, cv2.CC_STAT_AREA]
        best = 1 + int(np.argmax(areas))
        return (labels == best).astype(np.uint8) * 255

    def _ellipse_params_from_mask(self, mask_cc):
        cnts, _ = cv2.findContours(mask_cc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            raise RuntimeError("no contour")
        cnt = max(cnts, key=cv2.contourArea)
        return cv2.fitEllipse(cnt)

    def get_unwrapped_tire(self, image: np.ndarray, mask: np.ndarray):
        h, w = image.shape[:2]
        mask_cc = self._postprocess_mask(mask)
        (cx, cy), (MA, ma), angle = self._ellipse_params_from_mask(mask_cc)
        cx, cy, MA, ma = float(cx), float(cy), float(MA), float(ma)
        r_minor = ma / 2.0

        do_rectify = (MA / ma) > self.config.rectify_aspect_ratio_threshold

        if do_rectify:
            M_rot = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
            image = cv2.warpAffine(image, M_rot, (w, h), flags=cv2.INTER_CUBIC)

            scale_y = ma / MA
            M_scale = np.array(
                [[1, 0, 0], [0, scale_y, cy * (1 - scale_y)]], dtype=np.float32
            )
            image = cv2.warpAffine(image, M_scale, (w, h), flags=cv2.INTER_CUBIC)

        polar_image = cv2.warpPolar(
            image,
            self.config.polar_dsize,
            (cx, cy),
            r_minor,
            flags=cv2.INTER_CUBIC,
        )

        polar_image = cv2.rotate(polar_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if self.config.concat_strip:
            _, w, c = polar_image.shape
            strip_slide = np.concatenate(
                [polar_image[:, w // 2 :], polar_image[:, : w // 2]], axis=1
            )
            border = np.zeros(
                (self.config.concat_border_size, w, c), dtype=polar_image.dtype
            )
            polar_image = np.concatenate([polar_image, border, strip_slide], axis=0)

        lab_image = cv2.cvtColor(polar_image, cv2.COLOR_RGB2LAB)
        lab_image[:, :, 0] = self.clahe.apply(lab_image[:, :, 0])

        return cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)

    def forward(self, image: np.ndarray, mask: np.ndarray):
        return self.get_unwrapped_tire(image, mask)
