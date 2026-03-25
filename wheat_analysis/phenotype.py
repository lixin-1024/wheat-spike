"""
表型特征提取模块。

支持：
- 小穗级表型：长度、宽度、长宽比、面积、检测角度、着生角度
- 穗级表型：平均小穗表型、穗长、小穗数、着生密度、对称度、重心偏移度
- 兼容旧版扩展指标：ECI / SDU / SCO / SHI
"""
import numpy as np


class PhenotypeExtractor:
    """表型特征提取器"""

    def __init__(self, uniformity_segments: int = 5):
        self.K = uniformity_segments

    def extract_spikelet_phenotypes(
        self,
        detection: dict,
        skeleton: dict = None,
        calibration: dict = None,
    ) -> dict:
        """
        提取单个小穗级表型参数。

        Args:
            detection: 检测结果
            skeleton: 骨架结果，可选。提供时计算着生角度
            calibration: 标定结果，可选。成功时输出物理尺度
        """
        heights = np.asarray(detection['heights'], dtype=float)
        widths = np.asarray(detection['widths'], dtype=float)
        aspect_ratios = heights / np.maximum(widths, 1e-6)
        areas = heights * widths
        angles_deg = np.degrees(np.asarray(detection['angles'], dtype=float))

        attachment_angles_deg = self._compute_attachment_angles(detection, skeleton)

        result = {
            'lengths': heights,
            'widths': widths,
            'aspect_ratios': aspect_ratios,
            'areas': areas,
            'angles_deg': angles_deg,
            'attachment_angles_deg': attachment_angles_deg,
        }

        if calibration and calibration.get('calibration_ok'):
            mm_per_px = calibration['mm_per_px']
            result.update({
                'lengths_mm': heights * mm_per_px,
                'widths_mm': widths * mm_per_px,
                'areas_mm2': areas * (mm_per_px ** 2),
            })

            if skeleton is not None and 'spikelet_dist' in skeleton:
                result['dist_to_stem_mm'] = np.asarray(skeleton['spikelet_dist'], dtype=float) * mm_per_px

        return result

    def extract_ear_phenotypes(
        self,
        detection: dict,
        skeleton: dict,
        spikelet_pheno: dict = None,
        calibration: dict = None,
    ) -> dict:
        """
        提取穗级（整穗）表型参数。
        """
        if spikelet_pheno is None:
            spikelet_pheno = self.extract_spikelet_phenotypes(detection, skeleton, calibration)

        spikelet_s = np.asarray(skeleton['spikelet_s'], dtype=float)
        spikelet_dist = np.asarray(skeleton['spikelet_dist'], dtype=float)
        spikelet_side = np.asarray(skeleton['spikelet_side'], dtype=float)
        stem_length = float(skeleton['stem_length'])
        count = int(detection['count'])

        lengths = np.asarray(spikelet_pheno['lengths'], dtype=float)
        widths = np.asarray(spikelet_pheno['widths'], dtype=float)
        aspect_ratios = np.asarray(spikelet_pheno['aspect_ratios'], dtype=float)
        attachment_angles = np.asarray(spikelet_pheno['attachment_angles_deg'], dtype=float)

        density_px = count / stem_length if stem_length > 0 else 0.0
        centroid_offset = float(spikelet_s.mean()) if len(spikelet_s) > 0 else 0.0
        asymmetry_index = self._compute_asymmetry_index(
            spikelet_side,
            lengths,
            widths,
            aspect_ratios,
            attachment_angles,
        )
        sdu = self._compute_distribution_uniformity(spikelet_s, count)
        ear_vector = self._build_ear_vector(
            spikelet_s,
            np.asarray(spikelet_pheno['areas'], dtype=float),
            spikelet_side,
            self.K,
        )
        shi = float(np.var(ear_vector)) if len(ear_vector) > 0 else 0.0

        result = {
            'spikelet_count': count,
            'stem_length': stem_length,  # 兼容旧字段
            'spike_length_px': stem_length,
            'mean_spikelet_length': float(lengths.mean()) if len(lengths) > 0 else 0.0,
            'mean_spikelet_width': float(widths.mean()) if len(widths) > 0 else 0.0,
            'mean_aspect_ratio': float(aspect_ratios.mean()) if len(aspect_ratios) > 0 else 0.0,
            'mean_attachment_angle': float(attachment_angles.mean()) if len(attachment_angles) > 0 else 0.0,
            'spikelet_density_px': density_px,
            'asymmetry_index': asymmetry_index,
            'centroid_offset': centroid_offset,
            'ECI': density_px,
            'SDU': sdu,
            'SCO': centroid_offset,
            'SHI': shi,
            'mean_dist_to_stem': float(spikelet_dist.mean()) if len(spikelet_dist) > 0 else 0.0,
            'left_count': int(np.sum(spikelet_side < 0)),
            'right_count': int(np.sum(spikelet_side > 0)),
            'calibration_ok': False,
            'px_per_cm': None,
            'mm_per_px': None,
            'spike_length_cm': None,
            'mean_spikelet_length_mm': None,
            'mean_spikelet_width_mm': None,
            'mean_dist_to_stem_mm': None,
            'spikelet_density_per_cm': None,
        }

        if calibration and calibration.get('calibration_ok'):
            px_per_cm = float(calibration['px_per_cm'])
            mm_per_px = float(calibration['mm_per_px'])
            result.update({
                'calibration_ok': True,
                'px_per_cm': px_per_cm,
                'mm_per_px': mm_per_px,
                'spike_length_cm': stem_length / px_per_cm if px_per_cm > 0 else None,
                'mean_spikelet_length_mm': result['mean_spikelet_length'] * mm_per_px,
                'mean_spikelet_width_mm': result['mean_spikelet_width'] * mm_per_px,
                'mean_dist_to_stem_mm': result['mean_dist_to_stem'] * mm_per_px,
                'spikelet_density_per_cm': count / (stem_length / px_per_cm) if stem_length > 0 else None,
            })

        return result

    def compute_ear_vector_full(self, detection: dict, skeleton: dict, segments: int = 10) -> np.ndarray:
        """
        计算完整穗型向量。
        """
        spikelet_s = np.asarray(skeleton['spikelet_s'], dtype=float)
        spikelet_side = np.asarray(skeleton['spikelet_side'], dtype=float)
        spikelet_dist = np.asarray(skeleton['spikelet_dist'], dtype=float)
        areas = np.asarray(detection['heights'], dtype=float) * np.asarray(detection['widths'], dtype=float)

        total = len(spikelet_s)
        if total == 0:
            return np.array([], dtype=float)

        s_min, s_max = spikelet_s.min(), spikelet_s.max()
        bin_edges = np.linspace(s_min, s_max, segments + 1)
        max_area = areas.max() if len(areas) > 0 and areas.max() > 0 else 1.0
        max_dist = spikelet_dist.max() if len(spikelet_dist) > 0 and spikelet_dist.max() > 0 else 1.0

        vector = []
        for i in range(segments):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            if i == segments - 1:
                mask = (spikelet_s >= lo) & (spikelet_s <= hi)
            else:
                mask = (spikelet_s >= lo) & (spikelet_s < hi)

            n = int(mask.sum())
            avg_area = areas[mask].mean() / max_area if n > 0 else 0.0
            avg_dist = spikelet_dist[mask].mean() / max_dist if n > 0 else 0.0
            left_n = int(np.sum(spikelet_side[mask] < 0))
            right_n = int(np.sum(spikelet_side[mask] > 0))
            lr_ratio = (left_n - right_n) / max(n, 1)

            vector.extend([
                n / total,
                avg_area,
                avg_dist,
                lr_ratio,
            ])

        return np.asarray(vector, dtype=float)

    def build_feature_vector(self, ear_pheno: dict, ear_vector: np.ndarray = None) -> tuple[list[str], np.ndarray]:
        """
        拼接用于聚类的特征向量。

        优先使用物理尺度；若无标定，则回退到像素尺度。
        """
        scalar_features = {
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
            'mean_dist_to_stem': float(
                ear_pheno['mean_dist_to_stem_mm']
                if ear_pheno.get('mean_dist_to_stem_mm') is not None
                else ear_pheno['mean_dist_to_stem']
            ),
            'SDU': float(ear_pheno['SDU']),
            'SHI': float(ear_pheno['SHI']),
        }

        feature_names = list(scalar_features.keys())
        feature_values = list(scalar_features.values())

        if ear_vector is not None:
            ear_vector = np.asarray(ear_vector, dtype=float)
            for idx, value in enumerate(ear_vector):
                feature_names.append(f"ear_vector_{idx}")
                feature_values.append(float(value))

        return feature_names, np.asarray(feature_values, dtype=float)

    def _compute_attachment_angles(self, detection: dict, skeleton: dict = None) -> np.ndarray:
        count = int(detection['count']) if 'count' in detection else len(detection['heights'])
        if skeleton is None or 'spikelet_tangent' not in skeleton:
            return np.full(count, np.nan, dtype=float)

        if 'spikelet_highest_points' in skeleton and 'spikelet_lowest_points' in skeleton:
            axis_vectors = np.asarray(skeleton['spikelet_highest_points'], dtype=float) - np.asarray(
                skeleton['spikelet_lowest_points'], dtype=float
            )
        else:
            angles = np.asarray(detection['angles'], dtype=float)
            axis_vectors = np.column_stack([np.cos(angles), np.sin(angles)])

        tangent_vectors = np.asarray(skeleton['spikelet_tangent'], dtype=float)
        axis_norm = np.linalg.norm(axis_vectors, axis=1, keepdims=True)
        tangent_norm = np.linalg.norm(tangent_vectors, axis=1, keepdims=True)
        axis_unit = axis_vectors / np.maximum(axis_norm, 1e-8)
        tangent_unit = tangent_vectors / np.maximum(tangent_norm, 1e-8)

        cos_theta = np.sum(axis_unit * tangent_unit, axis=1)
        cos_theta = np.clip(np.abs(cos_theta), 0.0, 1.0)
        return np.degrees(np.arccos(cos_theta))

    def _compute_distribution_uniformity(self, spikelet_s: np.ndarray, count: int) -> float:
        if count <= 0:
            return 0.0

        segments = max(1, min(self.K, count))
        bin_edges = np.linspace(0.0, 1.0, segments + 1)
        counts_per_segment = np.zeros(segments, dtype=float)

        for idx in range(segments):
            lo, hi = bin_edges[idx], bin_edges[idx + 1]
            if idx == segments - 1:
                mask = (spikelet_s >= lo) & (spikelet_s <= hi)
            else:
                mask = (spikelet_s >= lo) & (spikelet_s < hi)
            counts_per_segment[idx] = np.sum(mask)

        mean_count = counts_per_segment.mean()
        return float(counts_per_segment.std() / mean_count) if mean_count > 0 else 0.0

    def _compute_asymmetry_index(
        self,
        spikelet_side: np.ndarray,
        lengths: np.ndarray,
        widths: np.ndarray,
        aspect_ratios: np.ndarray,
        attachment_angles: np.ndarray,
    ) -> float:
        left_mask = spikelet_side < 0
        right_mask = spikelet_side > 0
        metrics = [lengths, widths, aspect_ratios, np.nan_to_num(attachment_angles, nan=0.0)]

        diffs = []
        for metric in metrics:
            left_mean = float(metric[left_mask].mean()) if np.any(left_mask) else 0.0
            right_mean = float(metric[right_mask].mean()) if np.any(right_mask) else 0.0
            global_mean = float(metric.mean()) if len(metric) > 0 else 0.0
            if global_mean <= 1e-8:
                diffs.append(0.0)
            else:
                diffs.append(abs(left_mean - right_mean) / global_mean)

        return float(np.mean(diffs)) if diffs else 0.0

    def _build_ear_vector(self, spikelet_s, areas, sides, segments):
        spikelet_s = np.asarray(spikelet_s, dtype=float)
        areas = np.asarray(areas, dtype=float)
        total = len(spikelet_s)
        if total == 0:
            return np.array([], dtype=float)

        s_min, s_max = spikelet_s.min(), spikelet_s.max()
        bin_edges = np.linspace(s_min, s_max, segments + 1)
        max_area = areas.max() if len(areas) > 0 and areas.max() > 0 else 1.0

        vector = []
        for idx in range(segments):
            lo, hi = bin_edges[idx], bin_edges[idx + 1]
            if idx == segments - 1:
                mask = (spikelet_s >= lo) & (spikelet_s <= hi)
            else:
                mask = (spikelet_s >= lo) & (spikelet_s < hi)

            count = int(mask.sum())
            avg_area = areas[mask].mean() if count > 0 else 0.0
            vector.append(count / total)
            vector.append(avg_area / max_area)

        return np.asarray(vector, dtype=float)
