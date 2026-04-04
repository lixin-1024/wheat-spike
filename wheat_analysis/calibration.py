"""
图像标定模块。

第一版实现基于黑色背景上的白色圆片进行尺度标定：
- 圆片真实直径默认 5 cm
- 输出像素/厘米、毫米/像素等换算关系
- 新增 ColorChecker Classic 24（D65）色彩标定：
    - 自动检测色卡区域
    - 24 色块采样
    - 颜色矩阵拟合与 DeltaE 评估
"""
from __future__ import annotations

import cv2
import numpy as np


# 参考色值（ColorChecker Classic 24, sRGB D65），按 4x6 从左到右、从上到下顺序。
COLORCHECKER_CLASSIC_24_D65_RGB = np.asarray(
        [
                [115, 82, 68],
                [194, 150, 130],
                [98, 122, 157],
                [87, 108, 67],
                [133, 128, 177],
                [103, 189, 170],
                [214, 126, 44],
                [80, 91, 166],
                [193, 90, 99],
                [94, 60, 108],
                [157, 188, 64],
                [224, 163, 46],
                [56, 61, 150],
                [70, 148, 73],
                [175, 54, 60],
                [231, 199, 31],
                [187, 86, 149],
                [8, 133, 161],
                [243, 243, 242],
                [200, 200, 200],
                [160, 160, 160],
                [122, 122, 121],
                [85, 85, 85],
                [52, 52, 52],
        ],
        dtype=np.float32,
)
COLORCHECKER_CLASSIC_24_D65_BGR = COLORCHECKER_CLASSIC_24_D65_RGB[:, ::-1]


class ScaleCalibrator:
    """基于白色圆片的尺度标定器"""

    def __init__(self, disc_diameter_cm: float = 5.0, enable_color_calibration: bool = True):
        self.disc_diameter_cm = float(disc_diameter_cm)
        self.enable_color_calibration = bool(enable_color_calibration)
        self.color_reference_bgr = COLORCHECKER_CLASSIC_24_D65_BGR.copy()
        self.color_card_reference = "ColorChecker Classic 24 (D65)"

    def calibrate(self, image: np.ndarray) -> dict:
        """
        从图像中检测白色圆片并输出尺度换算参数，同时尝试色彩标定。
        """
        circle = self._detect_white_disc(image)
        result = {
            'calibration_ok': False,
            'scale_calibration_ok': False,
            'disc_diameter_cm': self.disc_diameter_cm,
            'disc_center': None,
            'disc_radius_px': None,
            'disc_diameter_px': None,
            'px_per_cm': None,
            'mm_per_px': None,
            'color_calibration_ok': False,
            'color_card_reference': self.color_card_reference,
            'color_card_bbox': None,
            'color_card_quad': None,
            'color_card_confidence': None,
            'color_matrix': None,
            'color_bias': None,
            'color_patch_means_bgr': None,
            'color_patch_corrected_bgr': None,
            'color_reference_bgr': self.color_reference_bgr.tolist(),
            'color_delta_e_mean': None,
            'color_delta_e_max': None,
            'color_quality_score': None,
            'color_error': None,
        }

        if circle is not None:
            center_x, center_y, radius_px = circle
            diameter_px = float(radius_px * 2.0)
            px_per_cm = diameter_px / self.disc_diameter_cm if self.disc_diameter_cm > 0 else None
            mm_per_px = 10.0 / px_per_cm if px_per_cm and px_per_cm > 0 else None

            result.update({
                'calibration_ok': px_per_cm is not None and px_per_cm > 0,
                'scale_calibration_ok': px_per_cm is not None and px_per_cm > 0,
                'disc_center': (float(center_x), float(center_y)),
                'disc_radius_px': float(radius_px),
                'disc_diameter_px': diameter_px,
                'px_per_cm': float(px_per_cm) if px_per_cm is not None else None,
                'mm_per_px': float(mm_per_px) if mm_per_px is not None else None,
            })

        if not self.enable_color_calibration:
            result['color_error'] = '色彩标定已禁用'
            return result
        if image is None or image.size == 0:
            result['color_error'] = '输入图像为空'
            return result

        card = self._detect_color_card(image)
        if card is None:
            result['color_error'] = '未检测到色卡'
            return result

        sampled = self._sample_color_card(image, card['quad'])
        if sampled is None:
            result['color_error'] = '色卡采样失败'
            result['color_card_bbox'] = card['bbox']
            result['color_card_quad'] = card['quad'].tolist()
            result['color_card_confidence'] = float(card['confidence'])
            return result

        fitted = self._fit_color_matrix(sampled['patch_means_bgr'])
        if fitted is None:
            result['color_error'] = '颜色矩阵拟合失败'
            result['color_card_bbox'] = card['bbox']
            result['color_card_quad'] = card['quad'].tolist()
            result['color_card_confidence'] = float(card['confidence'])
            return result

        delta_mean = fitted['delta_e_mean']
        quality = float(np.clip((1.0 - delta_mean / 25.0) * 0.7 + card['confidence'] * 0.3, 0.0, 1.0))
        result.update({
            'color_calibration_ok': True,
            'color_card_bbox': card['bbox'],
            'color_card_quad': card['quad'].tolist(),
            'color_card_confidence': float(card['confidence']),
            'color_matrix': fitted['color_matrix'].tolist(),
            'color_bias': fitted['color_bias'].tolist(),
            'color_patch_means_bgr': sampled['patch_means_bgr'].tolist(),
            'color_patch_corrected_bgr': fitted['corrected_patch_means_bgr'].tolist(),
            'color_delta_e_mean': float(fitted['delta_e_mean']),
            'color_delta_e_max': float(fitted['delta_e_max']),
            'color_quality_score': quality,
            'color_error': None,
        })
        return result

    def apply_color_correction(self, image: np.ndarray, calibration: dict | None) -> np.ndarray:
        """
        按标定结果对图像执行颜色校正。若无法校正，则返回原图。
        """
        if image is None or calibration is None:
            return image
        if not calibration.get('color_calibration_ok'):
            return image

        matrix = calibration.get('color_matrix')
        bias = calibration.get('color_bias')
        if matrix is None or bias is None:
            return image

        matrix_arr = np.asarray(matrix, dtype=np.float32)
        bias_arr = np.asarray(bias, dtype=np.float32)
        if matrix_arr.shape != (3, 3) or bias_arr.shape != (3,):
            return image

        image_float = image.astype(np.float32) / 255.0
        flat = image_float.reshape(-1, 3)
        corrected = flat @ matrix_arr.T + bias_arr
        corrected = np.clip(corrected, 0.0, 1.0)
        return (corrected.reshape(image.shape) * 255.0).astype(np.uint8)

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

    def _detect_color_card(self, image: np.ndarray):
        """
        检测色卡四边形区域。
        """
        height, width = image.shape[:2]
        image_area = float(height * width)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        sat_mask = cv2.inRange(hsv, (0, 35, 25), (180, 255, 255))
        kernel = np.ones((9, 9), dtype=np.uint8)
        sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(sat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_score = -1.0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < image_area * 0.003:
                continue
            if area > image_area * 0.45:
                continue

            rect = cv2.minAreaRect(contour)
            w_rect, h_rect = rect[1]
            if w_rect < 20 or h_rect < 20:
                continue

            long_edge = max(w_rect, h_rect)
            short_edge = max(min(w_rect, h_rect), 1e-6)
            aspect = long_edge / short_edge
            if aspect < 1.15 or aspect > 2.25:
                continue

            rect_area = w_rect * h_rect
            extent = area / max(rect_area, 1e-6)
            if extent < 0.35:
                continue

            quad = cv2.boxPoints(rect).astype(np.float32)
            quad = self._order_quad_points(quad)

            roi_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(roi_mask, [quad.astype(np.int32)], 255)
            sat_ratio = float(np.mean((hsv[:, :, 1] > 35)[roi_mask > 0])) if np.any(roi_mask > 0) else 0.0

            area_ratio = area / image_area
            score = area_ratio * 0.45 + extent * 0.35 + sat_ratio * 0.20
            if score > best_score:
                best_score = score
                best = {
                    'quad': quad,
                    'bbox': self._quad_to_bbox(quad),
                    'confidence': float(np.clip(score * 2.5, 0.0, 1.0)),
                }

        return best

    def _sample_color_card(self, image: np.ndarray, quad: np.ndarray):
        """
        将色卡透视到标准平面并采样 4x6=24 个色块。
        """
        if quad is None or len(quad) != 4:
            return None

        target_w, target_h = 600, 400
        dst = np.array(
            [[0, 0], [target_w - 1, 0], [target_w - 1, target_h - 1], [0, target_h - 1]],
            dtype=np.float32,
        )
        transform = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
        warped = cv2.warpPerspective(image, transform, (target_w, target_h))
        if warped is None or warped.size == 0:
            return None

        rows, cols = 4, 6
        margin_x = int(target_w * 0.08)
        margin_y = int(target_h * 0.08)
        inner_w = target_w - 2 * margin_x
        inner_h = target_h - 2 * margin_y
        cell_w = inner_w / cols
        cell_h = inner_h / rows

        patch_means = []
        for row in range(rows):
            for col in range(cols):
                x0 = int(margin_x + col * cell_w + cell_w * 0.22)
                x1 = int(margin_x + (col + 1) * cell_w - cell_w * 0.22)
                y0 = int(margin_y + row * cell_h + cell_h * 0.22)
                y1 = int(margin_y + (row + 1) * cell_h - cell_h * 0.22)

                roi = warped[y0:y1, x0:x1]
                if roi.size == 0:
                    return None
                patch_means.append(roi.reshape(-1, 3).mean(axis=0))

        patch_means_bgr = np.asarray(patch_means, dtype=np.float32)
        return {
            'patch_means_bgr': patch_means_bgr,
            'warped': warped,
        }

    def _fit_color_matrix(self, observed_patch_means_bgr: np.ndarray):
        """
        拟合颜色仿射映射：BGR' = M * BGR + b。
        """
        if observed_patch_means_bgr is None or observed_patch_means_bgr.shape != (24, 3):
            return None

        observed = np.asarray(observed_patch_means_bgr, dtype=np.float32) / 255.0
        reference = self.color_reference_bgr.astype(np.float32) / 255.0

        design = np.hstack([observed, np.ones((observed.shape[0], 1), dtype=np.float32)])
        coeff, _, _, _ = np.linalg.lstsq(design, reference, rcond=None)

        color_matrix = coeff[:3, :].T.astype(np.float32)
        color_bias = coeff[3, :].astype(np.float32)

        corrected = observed @ color_matrix.T + color_bias
        corrected = np.clip(corrected, 0.0, 1.0)

        delta = self._compute_delta_e_stats(corrected * 255.0, self.color_reference_bgr)
        return {
            'color_matrix': color_matrix,
            'color_bias': color_bias,
            'corrected_patch_means_bgr': corrected * 255.0,
            'delta_e_mean': delta['delta_e_mean'],
            'delta_e_max': delta['delta_e_max'],
        }

    def _compute_delta_e_stats(self, sample_bgr: np.ndarray, reference_bgr: np.ndarray):
        sample = np.clip(np.asarray(sample_bgr, dtype=np.float32), 0, 255).astype(np.uint8)
        reference = np.clip(np.asarray(reference_bgr, dtype=np.float32), 0, 255).astype(np.uint8)
        sample_lab = cv2.cvtColor(sample.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
        reference_lab = cv2.cvtColor(reference.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
        delta_e = np.linalg.norm(sample_lab - reference_lab, axis=1)
        return {
            'delta_e_mean': float(delta_e.mean()) if len(delta_e) else None,
            'delta_e_max': float(delta_e.max()) if len(delta_e) else None,
        }

    def _order_quad_points(self, points: np.ndarray):
        """将四边形点排序为 [tl, tr, br, bl]。"""
        pts = np.asarray(points, dtype=np.float32)
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).reshape(-1)
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(d)]
        bl = pts[np.argmax(d)]
        return np.array([tl, tr, br, bl], dtype=np.float32)

    def _quad_to_bbox(self, quad: np.ndarray):
        x_coords = quad[:, 0]
        y_coords = quad[:, 1]
        x1 = int(np.floor(np.min(x_coords)))
        y1 = int(np.floor(np.min(y_coords)))
        x2 = int(np.ceil(np.max(x_coords)))
        y2 = int(np.ceil(np.max(y_coords)))
        return [x1, y1, x2, y2]
