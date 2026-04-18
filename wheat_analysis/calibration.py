"""
图像标定模块。

当前版本同时支持两类标定：
- 基于白色圆片的尺度标定
- 基于 ColorChecker 24 的色彩校正
"""
from __future__ import annotations

import cv2
import numpy as np


class ScaleCalibrator:
    """同时处理尺度标定与色彩校正。"""

    # Macbeth / ColorChecker Classic 常用 sRGB 参考值，按标准横向 6x4 排列。
    COLORCHECKER_SRGB = np.asarray(
        [
            [[115, 82, 68], [194, 150, 130], [98, 122, 157], [87, 108, 67], [133, 128, 177], [103, 189, 170]],
            [[214, 126, 44], [80, 91, 166], [193, 90, 99], [94, 60, 108], [157, 188, 64], [224, 163, 46]],
            [[56, 61, 150], [70, 148, 73], [175, 54, 60], [231, 199, 31], [187, 86, 149], [8, 133, 161]],
            [[243, 243, 242], [200, 200, 200], [160, 160, 160], [122, 122, 121], [85, 85, 85], [52, 52, 52]],
        ],
        dtype=np.float32,
    )

    def __init__(self, disc_diameter_cm: float = 5.0):
        self.disc_diameter_cm = float(disc_diameter_cm)

    def calibrate(self, image: np.ndarray) -> dict:
        """
        从图像中检测白色圆片和 ColorChecker 24，输出尺度与色彩标定参数。
        """
        result = {
            'calibration_ok': False,
            'disc_diameter_cm': self.disc_diameter_cm,
            'disc_center': None,
            'disc_radius_px': None,
            'disc_diameter_px': None,
            'px_per_cm': None,
            'mm_per_px': None,
            'color_card_bbox': None,
            'color_card_corners': None,
            'color_card_size': None,
            'color_calibration_ok': False,
            'color_card_orientation': None,
            'color_matrix': None,
            'color_patch_error': None,
            'color_patch_means': None,
        }

        if image is None or image.size == 0:
            return result

        circle = self._detect_white_disc(image)
        if circle is not None:
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

        color_card = self._detect_colorchecker(image)
        if color_card is not None:
            observed_patches = color_card['patch_means_rgb']
            target_patches = color_card['reference_rgb']
            color_matrix = self._fit_color_matrix(observed_patches, target_patches)
            corrected_patches = self._apply_color_matrix_to_rgb(observed_patches, color_matrix)
            patch_error = float(np.mean(np.linalg.norm(corrected_patches - target_patches, axis=1)))
            color_ok = bool(np.isfinite(patch_error) and patch_error < 45.0)

            result.update({
                'color_card_bbox': color_card['bbox'],
                'color_card_corners': color_card['corners'],
                'color_card_size': color_card['grid_shape'],
                'color_calibration_ok': color_ok,
                'color_card_orientation': color_card['orientation'],
                'color_matrix': color_matrix.tolist(),
                'color_patch_error': patch_error,
                'color_patch_means': observed_patches.tolist(),
            })

        return result

    def apply_color_correction(self, image: np.ndarray, calibration: dict | None) -> np.ndarray | None:
        """根据标定结果生成校正后的图像。"""
        if image is None or calibration is None:
            return None
        matrix = calibration.get('color_matrix')
        if not matrix:
            return image.copy()
        return self._apply_color_matrix_to_bgr(image, np.asarray(matrix, dtype=np.float32))

    def draw_calibration_overlay(self, image: np.ndarray, calibration: dict | None) -> np.ndarray | None:
        """绘制尺度与色卡定位结果，供前端或导出查看。"""
        if image is None:
            return None

        vis = image.copy()
        calibration = calibration or {}

        disc_center = calibration.get('disc_center')
        disc_radius = calibration.get('disc_radius_px')
        if disc_center is not None and disc_radius is not None:
            center = tuple(int(round(v)) for v in disc_center)
            radius = int(round(float(disc_radius)))
            cv2.circle(vis, center, radius, (255, 255, 255), 3)
            cv2.circle(vis, center, 4, (255, 255, 255), -1)
            label = f"{float(calibration.get('px_per_cm') or 0):.2f} px/cm"
            cv2.putText(vis, label, (center[0] - radius, center[1] - radius - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)

        corners = calibration.get('color_card_corners')
        if corners is not None:
            quad = np.asarray(corners, dtype=np.int32).reshape(-1, 1, 2)
            color = (70, 230, 255) if calibration.get('color_calibration_ok') else (0, 165, 255)
            cv2.polylines(vis, [quad], True, color, 3)
            text = f"ColorChecker {'OK' if calibration.get('color_calibration_ok') else 'Detected'}"
            anchor = tuple(np.asarray(corners[0], dtype=int))
            cv2.putText(vis, text, (anchor[0], max(24, anchor[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return vis

    def _detect_white_disc(self, image: np.ndarray):
        blurred = cv2.GaussianBlur(image, (9, 9), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

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

    def _detect_colorchecker(self, image: np.ndarray) -> dict | None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 40, 140)
        edges = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        image_area = float(image.shape[0] * image.shape[1])
        best_candidate = None
        best_score = -1.0

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < image_area * 0.01 or area > image_area * 0.45:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 1e-6:
                continue

            approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue

            corners = self._order_quad_points(approx.reshape(4, 2).astype(np.float32))
            bbox = self._quad_bbox(corners)
            if self._touches_image_border(bbox, image.shape):
                continue

            warped, grid_shape = self._warp_quad(image, corners)
            if warped is None:
                continue

            patch_means = self._sample_colorchecker_patches(warped, grid_shape)
            if patch_means is None:
                continue

            colorfulness = float(np.mean(np.std(patch_means.reshape(-1, 3), axis=1)))
            if colorfulness < 12.0:
                continue

            border_mean = self._estimate_border_brightness(warped)
            score = area * (1.0 + colorfulness / 50.0) * (1.0 + border_mean / 255.0)
            if score > best_score:
                best_reference, orientation = self._match_reference_layout(patch_means)
                best_candidate = {
                    'corners': corners.tolist(),
                    'bbox': bbox,
                    'grid_shape': list(grid_shape),
                    'orientation': orientation,
                    'patch_means_rgb': patch_means.reshape(-1, 3),
                    'reference_rgb': best_reference.reshape(-1, 3),
                }
                best_score = score

        return best_candidate

    def _warp_quad(self, image: np.ndarray, corners: np.ndarray) -> tuple[np.ndarray | None, tuple[int, int]]:
        width_top = np.linalg.norm(corners[1] - corners[0])
        width_bottom = np.linalg.norm(corners[2] - corners[3])
        height_left = np.linalg.norm(corners[3] - corners[0])
        height_right = np.linalg.norm(corners[2] - corners[1])
        width = int(round(max(width_top, width_bottom)))
        height = int(round(max(height_left, height_right)))

        if width < 40 or height < 40:
            return None, (0, 0)

        if width >= height:
            target_w, target_h = 600, 400
            grid_shape = (4, 6)
        else:
            target_w, target_h = 400, 600
            grid_shape = (6, 4)

        destination = np.asarray(
            [[0, 0], [target_w - 1, 0], [target_w - 1, target_h - 1], [0, target_h - 1]],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), destination)
        warped = cv2.warpPerspective(image, matrix, (target_w, target_h))
        return warped, grid_shape

    def _sample_colorchecker_patches(self, warped: np.ndarray, grid_shape: tuple[int, int]) -> np.ndarray | None:
        rows, cols = grid_shape
        if rows <= 0 or cols <= 0:
            return None

        height, width = warped.shape[:2]
        cell_h = height / rows
        cell_w = width / cols
        samples = []
        for row in range(rows):
            row_samples = []
            for col in range(cols):
                x0 = int(round((col + 0.22) * cell_w))
                x1 = int(round((col + 0.78) * cell_w))
                y0 = int(round((row + 0.22) * cell_h))
                y1 = int(round((row + 0.78) * cell_h))
                patch = warped[max(0, y0):min(height, y1), max(0, x0):min(width, x1)]
                if patch.size == 0:
                    return None
                patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                row_samples.append(patch_rgb.reshape(-1, 3).mean(axis=0))
            samples.append(row_samples)
        return np.asarray(samples, dtype=np.float32)

    def _match_reference_layout(self, sampled_rgb: np.ndarray) -> tuple[np.ndarray, str]:
        sampled_shape = sampled_rgb.shape[:2]
        candidates = []
        base = self.COLORCHECKER_SRGB

        for rotation in range(4):
            rotated = np.rot90(base, rotation)
            if rotated.shape[:2] != sampled_shape:
                continue
            orientation = {0: 'upright', 1: 'rot90', 2: 'rot180', 3: 'rot270'}[rotation]
            candidates.append((rotated, orientation))

        best_reference = candidates[0][0]
        best_orientation = candidates[0][1]
        best_error = float('inf')
        sampled_flat = sampled_rgb.reshape(-1, 3)

        for reference, orientation in candidates:
            ref_flat = reference.reshape(-1, 3)
            matrix = self._fit_color_matrix(sampled_flat, ref_flat)
            corrected = self._apply_color_matrix_to_rgb(sampled_flat, matrix)
            error = float(np.mean(np.linalg.norm(corrected - ref_flat, axis=1)))
            if error < best_error:
                best_error = error
                best_reference = reference
                best_orientation = orientation

        return best_reference, best_orientation

    def _fit_color_matrix(self, observed_rgb: np.ndarray, target_rgb: np.ndarray) -> np.ndarray:
        observed = np.asarray(observed_rgb, dtype=np.float32).reshape(-1, 3)
        target = np.asarray(target_rgb, dtype=np.float32).reshape(-1, 3)
        augmented = np.concatenate([observed, np.ones((observed.shape[0], 1), dtype=np.float32)], axis=1)
        matrix, _, _, _ = np.linalg.lstsq(augmented, target, rcond=None)
        return np.asarray(matrix, dtype=np.float32)

    def _apply_color_matrix_to_rgb(self, rgb_pixels: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        rgb = np.asarray(rgb_pixels, dtype=np.float32).reshape(-1, 3)
        augmented = np.concatenate([rgb, np.ones((rgb.shape[0], 1), dtype=np.float32)], axis=1)
        corrected = augmented @ np.asarray(matrix, dtype=np.float32)
        return np.clip(corrected, 0.0, 255.0).astype(np.float32)

    def _apply_color_matrix_to_bgr(self, image: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32).reshape(-1, 3)
        corrected_rgb = self._apply_color_matrix_to_rgb(rgb, matrix).reshape(image.shape[0], image.shape[1], 3)
        corrected_rgb_u8 = np.clip(corrected_rgb, 0.0, 255.0).astype(np.uint8)
        return cv2.cvtColor(corrected_rgb_u8, cv2.COLOR_RGB2BGR)

    def _estimate_border_brightness(self, image: np.ndarray) -> float:
        h, w = image.shape[:2]
        strip = max(4, min(h, w) // 24)
        border_pixels = np.concatenate([
            image[:strip, :, :].reshape(-1, 3),
            image[-strip:, :, :].reshape(-1, 3),
            image[:, :strip, :].reshape(-1, 3),
            image[:, -strip:, :].reshape(-1, 3),
        ], axis=0)
        return float(np.mean(border_pixels))

    def _order_quad_points(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float32)
        sums = pts.sum(axis=1)
        diffs = np.diff(pts, axis=1).reshape(-1)
        ordered = np.zeros((4, 2), dtype=np.float32)
        ordered[0] = pts[np.argmin(sums)]
        ordered[2] = pts[np.argmax(sums)]
        ordered[1] = pts[np.argmin(diffs)]
        ordered[3] = pts[np.argmax(diffs)]
        return ordered

    def _quad_bbox(self, corners: np.ndarray) -> list[float]:
        x_min = float(np.min(corners[:, 0]))
        y_min = float(np.min(corners[:, 1]))
        x_max = float(np.max(corners[:, 0]))
        y_max = float(np.max(corners[:, 1]))
        return [x_min, y_min, x_max, y_max]

    def _touches_image_border(self, bbox: list[float], image_shape: tuple[int, ...], margin: int = 8) -> bool:
        height, width = image_shape[:2]
        x0, y0, x1, y1 = bbox
        return x0 <= margin or y0 <= margin or x1 >= width - margin or y1 >= height - margin
