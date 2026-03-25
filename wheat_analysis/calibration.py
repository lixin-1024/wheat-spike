"""
图像标定模块。

第一版实现基于黑色背景上的白色圆片进行尺度标定：
- 圆片真实直径默认 5 cm
- 输出像素/厘米、毫米/像素等换算关系
- 色卡接口先预留，不参与主流程计算
"""
from __future__ import annotations

import cv2
import numpy as np


class ScaleCalibrator:
    """基于白色圆片的尺度标定器"""

    def __init__(self, disc_diameter_cm: float = 5.0):
        self.disc_diameter_cm = float(disc_diameter_cm)

    def calibrate(self, image: np.ndarray) -> dict:
        """
        从图像中检测白色圆片并输出尺度换算参数。
        """
        circle = self._detect_white_disc(image)
        result = {
            'calibration_ok': False,
            'disc_diameter_cm': self.disc_diameter_cm,
            'disc_center': None,
            'disc_radius_px': None,
            'disc_diameter_px': None,
            'px_per_cm': None,
            'mm_per_px': None,
            'color_card_bbox': None,
        }

        if circle is None:
            return result

        center_x, center_y, radius_px = circle
        diameter_px = float(radius_px * 2.0)
        px_per_cm = diameter_px / self.disc_diameter_cm if self.disc_diameter_cm > 0 else None
        mm_per_px = 10.0 / px_per_cm if px_per_cm and px_per_cm > 0 else None

        result.update({
            'calibration_ok': px_per_cm is not None and px_per_cm > 0,
            'disc_center': (float(center_x), float(center_y)),
            'disc_radius_px': float(radius_px),
            'disc_diameter_px': diameter_px,
            'px_per_cm': float(px_per_cm) if px_per_cm is not None else None,
            'mm_per_px': float(mm_per_px) if mm_per_px is not None else None,
        })
        return result

    def _detect_white_disc(self, image: np.ndarray):
        if image is None or image.size == 0:
            return None

        blurred = cv2.GaussianBlur(image, (9, 9), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # 白色圆片通常表现为高亮、低饱和度。
        mask = cv2.inRange(hsv, (0, 0, 150), (180, 70, 255))
        kernel = np.ones((5, 5), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_candidate = None
        best_score = -1.0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 1e-6:
                continue

            circularity = 4.0 * np.pi * area / (perimeter ** 2)
            if circularity < 0.65:
                continue

            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            if radius < 10:
                continue

            circle_area = np.pi * (radius ** 2)
            fill_ratio = area / max(circle_area, 1e-6)
            score = circularity * fill_ratio * area
            if score > best_score:
                best_score = score
                best_candidate = (cx, cy, radius)

        return best_candidate
