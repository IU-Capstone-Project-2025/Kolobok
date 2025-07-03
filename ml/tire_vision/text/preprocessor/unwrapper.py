from typing import Optional

import numpy as np
import cv2
from shapely.geometry import Polygon

from tire_vision.config import TireUnwrapperConfig


class TireUnwrapper:
    """Takes detected tire polygons, unwraps and enhances the image."""

    def __init__(self, config: TireUnwrapperConfig):
        self.config = config

        self.clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_tile_grid_size,
        )

        self.clahe_map = {
            "disabled": lambda x: x,
            "luminance": self._clahe_luminance,
            "colors": self._clahe_colors,
            "black-and-white": self._clahe_black_and_white,
        }

        self.clahe_fn = self.clahe_map[self.config.clahe_mode]

    def _clahe_luminance(self, image: np.ndarray):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        merged = cv2.merge((l, a, b))
        result = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        return result

    def _clahe_colors(self, image: np.ndarray):
        channels = cv2.split(image)
        channels = [self.clahe.apply(c) for c in channels]

        result = cv2.merge(channels)
        return result

    def _clahe_black_and_white(self, image: np.ndarray):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        result_single_channel = self.clahe.apply(gray)
        result = cv2.cvtColor(result_single_channel, cv2.COLOR_GRAY2BGR)
        return result

    def _get_tire_center_and_radius(self, polygon: np.ndarray):
        """
        Compute the center and radius of the minimal enclosing circle for the tire polygon.
        """
        (x, y), radius = cv2.minEnclosingCircle(polygon)
        return (int(x), int(y)), int(radius)

    def _center_tire(self, image: np.ndarray, polygon_tire: np.ndarray):
        """
        Crop and center the tire disc based on the computed center and radius.
        """
        center, radius = self._get_tire_center_and_radius(polygon_tire)
        radius = int(radius * self.config.crop_enlarge_factor)
        h, w = image.shape[:2]
        x1 = max(center[0] - radius, 0)
        y1 = max(center[1] - radius, 0)
        x2 = min(center[0] + radius, w)
        y2 = min(center[1] + radius, h)
        crop = image[y1:y2, x1:x2]
        new_center = (crop.shape[1] // 2, crop.shape[0] // 2)
        return crop, new_center, radius

    def _rectify_tire(
        self,
        image: np.ndarray,
        polygon_tire: np.ndarray,
        polygon_rim: np.ndarray,
    ):
        """Apply perspective transform to align the tire if it is inclined.

        Returns the warped image and transformed polygons.
        """
        pts = polygon_tire.astype(np.float32)
        x_coords = pts[:, 0]
        y_coords = pts[:, 1]

        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()

        src_bb = np.array(
            [
                [x_min, y_min],
                [x_max, y_min],
                [x_max, y_max],
                [x_min, y_max],
            ],
            dtype=np.float32,
        )

        target_w = int(x_max - x_min)
        target_h = int(y_max - y_min)

        m = self.config.perspective_margin
        dst_bb = np.array(
            [
                [m, m],
                [target_w - m, m],
                [target_w - m, target_h - m],
                [m, target_h - m],
            ],
            dtype=np.float32,
        )

        transform = cv2.getPerspectiveTransform(src_bb, dst_bb)
        warped_img = cv2.warpPerspective(image, transform, (target_w, target_h))

        # transform polygons
        tire_h = polygon_tire.reshape(-1, 1, 2).astype(np.float32)
        rim_h = polygon_rim.reshape(-1, 1, 2).astype(np.float32)

        warped_tire = (
            cv2.perspectiveTransform(tire_h, transform).reshape(-1, 2).astype(np.int32)
        )
        warped_rim = (
            cv2.perspectiveTransform(rim_h, transform).reshape(-1, 2).astype(np.int32)
        )

        return warped_img, warped_tire, warped_rim

    def _unwrap_polar_tire(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray],
        center: tuple[int, int],
        radius_tire: int,
        radius_rim: int,
    ):
        """
        Unwrap the tire disc into polar coordinates.
        """
        height = radius_tire
        width = int(np.ceil(2 * np.pi * radius_rim))

        dsize = (height, width)
        flags = self.config.polar_flags
        polar_img = cv2.warpPolar(image, dsize, center, radius_tire, flags)

        if self.config.cut_strip:
            polar_mask = (
                cv2.warpPolar(mask, dsize, center, radius_tire, flags).astype(
                    np.float32
                )
                / 255
            )

            keep = np.where(
                np.mean(polar_mask, axis=0) > self.config.cut_mask_threshold
            )[0]

            polar_img = polar_img[:, keep]

        polar_img = cv2.rotate(polar_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        return polar_img

    def _apply_clahe(self, image: np.ndarray):
        """
        Apply CLAHE to enhance local contrast, using L-channel in LAB color space.
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_tile_grid_size,
        )
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        result = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        return result

    def _get_polygon_center(self, polygon: np.ndarray):
        shp_centroid = Polygon(polygon).centroid
        cx, cy = int(shp_centroid.x), int(shp_centroid.y)
        return cx, cy

    def _get_max_radius(self, polygon: np.ndarray, center: tuple[int, int]):
        center_arr = np.array(center)
        distances = np.linalg.norm(polygon - center_arr, axis=1)
        return int(np.max(distances))

    def _crop_around_center(
        self, image: np.ndarray, center: tuple[int, int], radius: int
    ) -> tuple[np.ndarray, tuple[int, int]]:
        """Crop square ROI around center with given radius."""
        h, w = image.shape[:2]
        x1 = max(center[0] - radius, 0)
        y1 = max(center[1] - radius, 0)
        x2 = min(center[0] + radius, w)
        y2 = min(center[1] + radius, h)

        crop = image[y1:y2, x1:x2]
        new_center = (center[0] - x1, center[1] - y1)
        return crop, new_center

    def _get_tire_mask(
        self, image: np.ndarray, polygon_tire: np.ndarray, polygon_rim: np.ndarray
    ):
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [polygon_tire.astype(np.int32)], 255)
        cv2.fillPoly(mask, [polygon_rim.astype(np.int32)], 0)

        return mask

    def _concat_strip(self, image: np.ndarray):
        _, w, c = image.shape
        strip_slide = np.concatenate([image[:, w // 2 :], image[:, : w // 2]], axis=1)

        border = np.zeros((5, w, c), dtype=image.dtype)
        result = np.concatenate([image, border, strip_slide], axis=0)

        return result

    def get_unwrapped_tire(
        self, image: np.ndarray, polygon_tire: np.ndarray, polygon_rim: np.ndarray
    ):
        """
        Full pipeline: crop, unwrap, and enhance the tire image.

        Parameters:
        image : np.ndarray
            Original BGR image containing the tire.
        polygon_tire : np.ndarray
            Detected outer polygon of the tire (N×2 array of int coordinates).
        """
        rect_img, rect_tire, rect_rim = self._rectify_tire(
            image, polygon_tire, polygon_rim
        )

        center_rim = self._get_polygon_center(rect_rim)
        radius_tire = int(
            self._get_max_radius(rect_tire, center_rim)
            * self.config.crop_enlarge_factor
        )
        radius_rim = int(
            self._get_max_radius(rect_rim, center_rim) * self.config.crop_enlarge_factor
        )

        crop, new_center = self._crop_around_center(rect_img, center_rim, radius_tire)

        mask = (
            self._get_tire_mask(crop, rect_tire, rect_rim)
            if self.config.cut_strip
            else None
        )

        polar = self._unwrap_polar_tire(crop, mask, new_center, radius_tire, radius_rim)

        if self.config.concat_strip:
            polar = self._concat_strip(polar)

        enhanced = self.clahe_fn(polar)

        return enhanced
