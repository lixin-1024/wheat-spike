"""
表型特征提取模块：从检测结果和骨架信息中提取小穗级表型和穗级表型。
包括：
 - 小穗级表型：长度、宽度、长宽比、面积、角度
 - 穗级表型：穗长、小穗数、紧密度指数(ECI)、分布均匀度(SDU)、
              重心偏移度(SCO)、穗型异质性指数(SHI)
"""
import numpy as np


class PhenotypeExtractor:
    """表型特征提取器"""

    def __init__(self, uniformity_segments: int = 5):
        """
        Args:
            uniformity_segments: 计算分布均匀度时将主茎分成的段数
        """
        self.K = uniformity_segments

    def extract_spikelet_phenotypes(self, detection: dict) -> dict:
        """
        提取单个小穗级的表型参数。

        Returns:
            dict: {
                'lengths': np.ndarray (N,),       # 小穗长度(像素)
                'widths': np.ndarray (N,),        # 小穗宽度(像素)
                'aspect_ratios': np.ndarray (N,), # 长宽比
                'areas': np.ndarray (N,),         # 面积(像素²)
                'angles_deg': np.ndarray (N,),    # 旋转角度(度)
            }
        """
        heights = detection['heights']  # 长边 = 小穗长度
        widths = detection['widths']    # 短边 = 小穗宽度

        return {
            'lengths': heights,
            'widths': widths,
            'aspect_ratios': heights / np.maximum(widths, 1e-6),
            'areas': heights * widths,
            'angles_deg': np.degrees(detection['angles']),
        }

    def extract_ear_phenotypes(self, detection: dict, skeleton: dict) -> dict:
        """
        提取穗级（整穗）表型参数。

        Returns:
            dict: {
                'spikelet_count': int,                 # 小穗数
                'mean_spikelet_length': float,         # 平均小穗长度
                'mean_spikelet_width': float,          # 平均小穗宽度
                'mean_aspect_ratio': float,            # 平均长宽比
                'ECI': float,                          # 穗型紧密度指数
                'SDU': float,                          # 小穗分布均匀度
                'SCO': float,                          # 穗型重心偏移度
                'SHI': float,                          # 穗型异质性指数
                'mean_dist_to_stem': float,            # 平均到主茎距离
                'left_count': int,                     # 左侧小穗数
                'right_count': int,                    # 右侧小穗数
            }
        """
        N = detection['count']
        spikelet_s = skeleton['spikelet_s']                # 归一化弧长位置
        stem_length = skeleton['stem_length']
        spikelet_dist = skeleton['spikelet_dist']
        spikelet_side = skeleton['spikelet_side']

        # 有效穗段长度：从第一个小穗到最后一个小穗在主茎上的实际距离
        s_min, s_max = spikelet_s.min(), spikelet_s.max()
        effective_length = (s_max - s_min) * stem_length

        # ---- ECI: 穗型紧密度指数 ----
        # ECI = N / effective_length
        ECI = N / effective_length if effective_length > 0 else 0.0

        # ---- SDU: 小穗分布均匀度 ----
        # 将 [s_min, s_max] 分成 K 段，统计每段小穗数，计算变异系数
        K = min(self.K, N)
        bin_edges = np.linspace(s_min, s_max, K + 1)
        counts_per_segment = np.zeros(K)
        for i in range(K):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            if i == K - 1:
                counts_per_segment[i] = np.sum((spikelet_s >= lo) & (spikelet_s <= hi))
            else:
                counts_per_segment[i] = np.sum((spikelet_s >= lo) & (spikelet_s < hi))
        mean_count = counts_per_segment.mean()
        std_count = counts_per_segment.std()
        SDU = std_count / mean_count if mean_count > 0 else 0.0

        # ---- SCO: 穗型重心偏移度 ----
        # 将 s 归一化到 [0, 1] 范围（相对于有效穗段）
        if s_max > s_min:
            s_normalized = (spikelet_s - s_min) / (s_max - s_min)
        else:
            s_normalized = np.full(N, 0.5)
        SCO = s_normalized.mean()

        # ---- SHI: 穗型异质性指数 ----
        # 构建穗型向量：每段的 [小穗数, 平均大小, 平均间距]
        spikelet_phenotypes = self.extract_spikelet_phenotypes(detection)
        ear_vector = self._build_ear_vector(
            spikelet_s, spikelet_phenotypes['areas'],
            spikelet_side, s_min, s_max, K
        )
        SHI = np.var(ear_vector) if len(ear_vector) > 0 else 0.0

        return {
            'spikelet_count': N,
            'effective_stem_length': effective_length,
            'mean_spikelet_length': detection['heights'].mean(),
            'mean_spikelet_width': detection['widths'].mean(),
            'mean_aspect_ratio': (detection['heights'] / np.maximum(detection['widths'], 1e-6)).mean(),
            'ECI': ECI,
            'SDU': SDU,
            'SCO': SCO,
            'SHI': SHI,
            'mean_dist_to_stem': spikelet_dist.mean(),
            'left_count': int(np.sum(spikelet_side < 0)),
            'right_count': int(np.sum(spikelet_side > 0)),
        }

    def _build_ear_vector(self, spikelet_s, areas, sides, s_min, s_max, K):
        """
        构建穗型向量：沿主茎分段统计特征。

        穗型向量 V = [n1, a1, n2, a2, ..., nK, aK]
        其中 ni = 第i段小穗数(归一化), ai = 第i段平均面积(归一化)
        """
        bin_edges = np.linspace(s_min, s_max, K + 1)
        vector = []

        # 归一化参数
        total = len(spikelet_s)
        max_area = areas.max() if len(areas) > 0 and areas.max() > 0 else 1.0

        for i in range(K):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            if i == K - 1:
                mask = (spikelet_s >= lo) & (spikelet_s <= hi)
            else:
                mask = (spikelet_s >= lo) & (spikelet_s < hi)

            n = mask.sum()
            avg_area = areas[mask].mean() if n > 0 else 0.0

            vector.append(n / total)              # 归一化小穗数
            vector.append(avg_area / max_area)    # 归一化平均面积

        return np.array(vector)

    def compute_ear_vector_full(self, detection: dict, skeleton: dict,
                                 segments: int = 10) -> np.ndarray:
        """
        计算完整的穗型向量（用于对比分析）。

        穗型向量维度: segments * 4
        每段特征: [归一化小穗数, 归一化平均面积, 归一化平均间距, 左右比]

        该向量具有尺度不变性和旋转不变性。
        """
        spikelet_s = skeleton['spikelet_s']
        spikelet_side = skeleton['spikelet_side']
        spikelet_dist = skeleton['spikelet_dist']
        areas = detection['heights'] * detection['widths']

        s_min, s_max = spikelet_s.min(), spikelet_s.max()
        bin_edges = np.linspace(s_min, s_max, segments + 1)

        total = len(spikelet_s)
        max_area = areas.max() if len(areas) > 0 and areas.max() > 0 else 1.0
        max_dist = spikelet_dist.max() if len(spikelet_dist) > 0 and spikelet_dist.max() > 0 else 1.0

        vector = []
        for i in range(segments):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            if i == segments - 1:
                mask = (spikelet_s >= lo) & (spikelet_s <= hi)
            else:
                mask = (spikelet_s >= lo) & (spikelet_s < hi)

            n = mask.sum()
            avg_area = areas[mask].mean() / max_area if n > 0 else 0.0
            avg_dist = spikelet_dist[mask].mean() / max_dist if n > 0 else 0.0

            left_n = np.sum(spikelet_side[mask] < 0)
            right_n = np.sum(spikelet_side[mask] > 0)
            lr_ratio = (left_n - right_n) / max(n, 1)

            vector.extend([
                n / total,       # 归一化小穗数密度
                avg_area,        # 归一化平均面积
                avg_dist,        # 归一化平均到茎距离
                lr_ratio,        # 左右偏向
            ])

        return np.array(vector)
