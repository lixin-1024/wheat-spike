"""
表型特征提取模块。
- 小穗级：长度、宽度、长宽比、着生角度
- 穗级：平均小穗表型、穗长、小穗数、着生密度、对称度、重心偏移度
"""
from __future__ import annotations

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
        小穗中轴方向 与 该小穗基节点处主茎切线方向 的夹角，
        结果折叠到 [0, 90] 度。
        """
        lengths = np.asarray(detection['heights'], dtype=float)
        widths = np.asarray(detection['widths'], dtype=float)
        aspect_ratios = lengths / np.maximum(widths, 1e-6)

        axis_vectors = np.asarray(skeleton['spikelet_axis_dirs'], dtype=float)
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
            'symmetry_index': self._compute_symmetry_index(
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
            'symmetry_index': float(ear_pheno['symmetry_index']),
            'centroid_offset': float(ear_pheno['centroid_offset']),
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

    def _compute_symmetry_index(
        self,
        spikelet_side: np.ndarray,
        lengths: np.ndarray,
        widths: np.ndarray,
        aspect_ratios: np.ndarray,
        attachment_angles: np.ndarray,
    ) -> float:
        """
        采用 Gap = |L - R| / (L + R) 将差异严格映射到 [0, 1] 区间，最后通过 1.0 - mean(Gap) 得到最终的对称度指数。
        """
        left_mask = spikelet_side < 0
        right_mask = spikelet_side > 0

        normalized_gaps = []

        # 1. 计算左右侧小穗数量差异
        left_count = float(np.sum(left_mask))
        right_count = float(np.sum(right_mask))
        count_denominator = left_count + right_count

        if count_denominator > 1e-8:
            count_gap = abs(left_count - right_count) / count_denominator
        else:
            count_gap = 0.0
        normalized_gaps.append(count_gap)

        # 2. 计算各项形态学表型特征的平均差异
        metrics = [lengths, widths, aspect_ratios, attachment_angles]
        for metric in metrics:
            left_mean = float(metric[left_mask].mean()) if np.any(left_mask) else 0.0
            right_mean = float(metric[right_mask].mean()) if np.any(right_mask) else 0.0

            # 以 L + R 为分母，天然约束结果在 0 到 1 之间
            denominator = left_mean + right_mean
            if denominator > 1e-8:
                gap = abs(left_mean - right_mean) / denominator
            else:
                # 如果两边特征均值都为0，视为无差异，完美对称
                gap = 0.0

            normalized_gaps.append(gap)

        # mean_gap 越大代表越不对称，因此用 1 减去它，得到正向的对称度指数
        mean_gap = float(np.mean(normalized_gaps)) if normalized_gaps else 0.0
        return 1.0 - mean_gap
