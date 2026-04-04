"""
表型特征提取模块。

仅保留当前任务需要的表型：
- 小穗级：长度、宽度、长宽比、着生角度
- 穗级：平均小穗表型、穗长、小穗数、着生密度、对称度、重心偏移度
"""
from __future__ import annotations

import cv2
import numpy as np


class PhenotypeExtractor:
    """表型特征提取器"""

    def extract_spikelet_phenotypes(
        self,
        detection: dict,
        skeleton: dict,
        calibration: dict | None = None,
    ) -> dict:
        """
        提取小穗级表型。

        着生角度定义：
        小穗长轴方向 与 该小穗基节点处主茎切线方向 的夹角，
        结果折叠到 [0, 90] 度。
        """
        lengths = np.asarray(detection['heights'], dtype=float)
        widths = np.asarray(detection['widths'], dtype=float)
        aspect_ratios = lengths / np.maximum(widths, 1e-6)

        # 直接使用“最高点 - 最低点”作为小穗主轴方向，确保与基节点定义一致。
        axis_vectors = (
            np.asarray(skeleton['spikelet_highest_points'], dtype=float)
            - np.asarray(skeleton['spikelet_lowest_points'], dtype=float)
        )
        tangent_vectors = np.asarray(skeleton['spikelet_tangent'], dtype=float)
        attachment_angles_deg = self._compute_attachment_angles(axis_vectors, tangent_vectors)

        result = {
            'lengths': lengths,
            'widths': widths,
            'aspect_ratios': aspect_ratios,
            'attachment_angles_deg': attachment_angles_deg,
        }

        if calibration and calibration.get('calibration_ok'):
            mm_per_px = float(calibration['mm_per_px'])
            result['lengths_mm'] = lengths * mm_per_px
            result['widths_mm'] = widths * mm_per_px

        return result

    def extract_ear_phenotypes(
        self,
        detection: dict,
        skeleton: dict,
        spikelet_pheno: dict,
        calibration: dict | None = None,
        image: np.ndarray | None = None,
    ) -> dict:
        """
        提取穗级表型。
        """
        lengths = np.asarray(spikelet_pheno['lengths'], dtype=float)
        widths = np.asarray(spikelet_pheno['widths'], dtype=float)
        aspect_ratios = np.asarray(spikelet_pheno['aspect_ratios'], dtype=float)
        attachment_angles = np.asarray(spikelet_pheno['attachment_angles_deg'], dtype=float)
        spikelet_side = np.asarray(skeleton['spikelet_side'], dtype=float)
        spikelet_s = np.asarray(skeleton['spikelet_s'], dtype=float)

        spike_length_px = float(skeleton['stem_length'])
        spikelet_count = int(detection['count'])

        result = {
            'spikelet_count': spikelet_count,
            'mean_spikelet_length': float(lengths.mean()) if len(lengths) else 0.0,
            'mean_spikelet_width': float(widths.mean()) if len(widths) else 0.0,
            'mean_aspect_ratio': float(aspect_ratios.mean()) if len(aspect_ratios) else 0.0,
            'mean_attachment_angle': float(attachment_angles.mean()) if len(attachment_angles) else 0.0,
            'spike_length_px': spike_length_px,
            'spikelet_density_px': spikelet_count / spike_length_px if spike_length_px > 0 else 0.0,
            'asymmetry_index': self._compute_asymmetry_index(
                spikelet_side,
                lengths,
                widths,
                aspect_ratios,
                attachment_angles,
            ),
            'centroid_offset': float(spikelet_s.mean()) if len(spikelet_s) else 0.0,
            'calibration_ok': False,
            'px_per_cm': None,
            'mm_per_px': None,
            'spike_length_cm': None,
            'spikelet_density_per_cm': None,
            'mean_spikelet_length_mm': None,
            'mean_spikelet_width_mm': None,
            'mean_color_l': None,
            'mean_color_a': None,
            'mean_color_b': None,
            'color_std_l': None,
            'left_right_color_delta_e': None,
        }

        if calibration and calibration.get('calibration_ok'):
            px_per_cm = float(calibration['px_per_cm'])
            mm_per_px = float(calibration['mm_per_px'])
            result.update({
                'calibration_ok': True,
                'px_per_cm': px_per_cm,
                'mm_per_px': mm_per_px,
                'spike_length_cm': spike_length_px / px_per_cm if px_per_cm > 0 else None,
                'spikelet_density_per_cm': spikelet_count / (spike_length_px / px_per_cm) if spike_length_px > 0 else None,
                'mean_spikelet_length_mm': result['mean_spikelet_length'] * mm_per_px,
                'mean_spikelet_width_mm': result['mean_spikelet_width'] * mm_per_px,
            })

        color_metrics = self._extract_color_metrics(image, detection, skeleton)
        if color_metrics is not None:
            result.update(color_metrics)

        return result

    def build_feature_vector(self, ear_pheno: dict) -> tuple[list[str], np.ndarray]:
        """
        组装聚类特征向量。

        尽量优先使用物理尺度，保证跨图像更可比。
        """
        feature_map = {
            'spikelet_count': float(ear_pheno['spikelet_count']),
            'mean_spikelet_length': float(
                ear_pheno['mean_spikelet_length_mm']
                if ear_pheno.get('mean_spikelet_length_mm') is not None
                else ear_pheno['mean_spikelet_length']
            ),
            'mean_spikelet_width': float(
                ear_pheno['mean_spikelet_width_mm']
                if ear_pheno.get('mean_spikelet_width_mm') is not None
                else ear_pheno['mean_spikelet_width']
            ),
            'mean_aspect_ratio': float(ear_pheno['mean_aspect_ratio']),
            'mean_attachment_angle': float(ear_pheno['mean_attachment_angle']),
            'spike_length': float(
                ear_pheno['spike_length_cm']
                if ear_pheno.get('spike_length_cm') is not None
                else ear_pheno['spike_length_px']
            ),
            'spikelet_density': float(
                ear_pheno['spikelet_density_per_cm']
                if ear_pheno.get('spikelet_density_per_cm') is not None
                else ear_pheno['spikelet_density_px']
            ),
            'asymmetry_index': float(ear_pheno['asymmetry_index']),
            'centroid_offset': float(ear_pheno['centroid_offset']),
            'mean_color_l': float(ear_pheno.get('mean_color_l') if ear_pheno.get('mean_color_l') is not None else 0.0),
            'mean_color_a': float(ear_pheno.get('mean_color_a') if ear_pheno.get('mean_color_a') is not None else 0.0),
            'mean_color_b': float(ear_pheno.get('mean_color_b') if ear_pheno.get('mean_color_b') is not None else 0.0),
            'color_std_l': float(ear_pheno.get('color_std_l') if ear_pheno.get('color_std_l') is not None else 0.0),
            'left_right_color_delta_e': float(
                ear_pheno.get('left_right_color_delta_e')
                if ear_pheno.get('left_right_color_delta_e') is not None
                else 0.0
            ),
        }
        return list(feature_map.keys()), np.asarray(list(feature_map.values()), dtype=float)

    def build_spikelet_records(self, detection: dict, skeleton: dict, spikelet_pheno: dict) -> list[dict]:
        """
        构造前端交互需要的小穗级结构化记录。
        """
        centers = np.asarray(detection['centers'], dtype=float)
        corners = np.asarray(detection['xyxyxyxy'], dtype=float)
        records = []
        for idx in range(len(centers)):
            records.append({
                'index': idx,
                'center': centers[idx].tolist(),
                'corners': corners[idx].tolist(),
                'length': float(spikelet_pheno['lengths'][idx]),
                'width': float(spikelet_pheno['widths'][idx]),
                'aspect_ratio': float(spikelet_pheno['aspect_ratios'][idx]),
                'attachment_angle': float(spikelet_pheno['attachment_angles_deg'][idx]),
                'side': 'right' if float(skeleton['spikelet_side'][idx]) >= 0 else 'left',
                'order': int(np.where(skeleton['spikelet_order'] == idx)[0][0]) + 1,
            })
        return records

    def _compute_attachment_angles(self, axis_vectors: np.ndarray, tangent_vectors: np.ndarray) -> np.ndarray:
        axis_unit = axis_vectors / np.maximum(np.linalg.norm(axis_vectors, axis=1, keepdims=True), 1e-8)
        tangent_unit = tangent_vectors / np.maximum(np.linalg.norm(tangent_vectors, axis=1, keepdims=True), 1e-8)
        cos_theta = np.clip(np.abs(np.sum(axis_unit * tangent_unit, axis=1)), 0.0, 1.0)
        return np.degrees(np.arccos(cos_theta))

    def _compute_asymmetry_index(
        self,
        spikelet_side: np.ndarray,
        lengths: np.ndarray,
        widths: np.ndarray,
        aspect_ratios: np.ndarray,
        attachment_angles: np.ndarray,
    ) -> float:
        """
        左右两侧在四项小穗级表型上的均值差异，按整体均值归一化后取平均。
        """
        left_mask = spikelet_side < 0
        right_mask = spikelet_side > 0
        metrics = [lengths, widths, aspect_ratios, attachment_angles]

        normalized_gaps = []
        for metric in metrics:
            left_mean = float(metric[left_mask].mean()) if np.any(left_mask) else 0.0
            right_mean = float(metric[right_mask].mean()) if np.any(right_mask) else 0.0
            overall_mean = float(metric.mean()) if len(metric) else 0.0
            normalized_gaps.append(abs(left_mean - right_mean) / overall_mean if overall_mean > 1e-8 else 0.0)

        return float(np.mean(normalized_gaps)) if normalized_gaps else 0.0

    def _extract_color_metrics(self, image: np.ndarray | None, detection: dict, skeleton: dict) -> dict | None:
        """
        从小穗 OBB 区域提取颜色统计（Lab）。
        """
        if image is None or image.size == 0:
            return None
        if detection.get('xyxyxyxy') is None:
            return None

        polygons = np.asarray(detection['xyxyxyxy'], dtype=float)
        if len(polygons) == 0:
            return None

        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        side = np.asarray(skeleton.get('spikelet_side', np.zeros(len(polygons), dtype=float)), dtype=float)

        patch_means = []
        patch_sides = []
        for index, polygon in enumerate(polygons):
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [polygon.astype(np.int32)], 255)
            values = lab_image[mask > 0]
            if values.size == 0:
                continue
            patch_means.append(values.mean(axis=0))
            patch_sides.append(side[index] if index < len(side) else 0.0)

        if not patch_means:
            return None

        patch_means_arr = np.asarray(patch_means, dtype=np.float32)
        patch_sides_arr = np.asarray(patch_sides, dtype=np.float32)

        left_mask = patch_sides_arr < 0
        right_mask = patch_sides_arr > 0

        left_right_delta_e = 0.0
        if np.any(left_mask) and np.any(right_mask):
            left_mean = patch_means_arr[left_mask].mean(axis=0)
            right_mean = patch_means_arr[right_mask].mean(axis=0)
            left_right_delta_e = float(np.linalg.norm(left_mean - right_mean))

        return {
            'mean_color_l': float(patch_means_arr[:, 0].mean()),
            'mean_color_a': float(patch_means_arr[:, 1].mean()),
            'mean_color_b': float(patch_means_arr[:, 2].mean()),
            'color_std_l': float(patch_means_arr[:, 0].std()),
            'left_right_color_delta_e': left_right_delta_e,
        }
