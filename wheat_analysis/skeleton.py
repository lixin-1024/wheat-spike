"""
茎-穗骨架生成模块。

处理流程：
1. 从 OBB 四点恢复每个小穗的长轴与“基节点”（长轴最低点）
2. 使用全部基节点拟合主茎样条曲线
3. 在每个基节点对应的样条参数处计算主茎切线
4. 基于基节点与主茎切线，计算小穗沿穗长位置与左右侧信息
"""
import numpy as np
from scipy.interpolate import UnivariateSpline


class SkeletonBuilder:
    """从小穗检测结果构建茎-穗骨架"""

    def __init__(self, spline_smoothing: float = None):
        self.spline_smoothing = spline_smoothing

    def build(self, detection: dict) -> dict:
        """
        从检测结果构建骨架。

        Returns:
            dict: {
                'spikelet_highest_points': np.ndarray (N, 2),
                'spikelet_lowest_points': np.ndarray (N, 2),
                'spikelet_tangent': np.ndarray (N, 2),
                'spikelet_s': np.ndarray (N,),
                'spikelet_side': np.ndarray (N,),
                'spikelet_order': np.ndarray (N,),
                'stem_points': np.ndarray (M, 2),
                'stem_length': float,
            }
        """
        centers = np.asarray(detection['centers'], dtype=float)
        xyxyxyxy = detection.get('xyxyxyxy')
        count = len(centers)

        if count < 2:
            raise ValueError("至少需要2个小穗才能构建骨架")
        if xyxyxyxy is None or len(xyxyxyxy) != count:
            raise ValueError("缺少有效的 OBB 角点 xyxyxyxy，无法构建骨架")

        long_dirs, highest_points, lowest_points = self._extract_spikelet_axes(centers, xyxyxyxy)

        # 基节点就是拟合主茎的输入点。
        stem_fit_points = lowest_points.copy()

        mean = stem_fit_points.mean(axis=0)
        centered = stem_fit_points - mean
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        main_dir = eigvecs[:, np.argmax(eigvals)]
        # 在图像坐标系中，y < 0 代表指向上方
        if main_dir[1] > 0:
            main_dir = -main_dir  # 如果指向下了，就把它翻转过来

        projections = centered @ main_dir
        order = np.argsort(projections)
        sorted_fit_points = stem_fit_points[order]

        diffs = np.diff(sorted_fit_points, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        cum_arc = np.zeros(count, dtype=float)
        cum_arc[1:] = np.cumsum(seg_lengths)
        total_length = float(cum_arc[-1])
        t = cum_arc / total_length if total_length > 0 else np.linspace(0.0, 1.0, count)

        k = min(3, count - 1)
        smoothing = self.spline_smoothing if self.spline_smoothing is not None else count * (max(total_length, 1.0) * 0.01)

        weights = np.ones(count, dtype=float)
        edge_count = max(2, count // 5)
        for edge_idx in range(edge_count):
            weight = 0.2 + 0.8 * (edge_idx / edge_count)
            weights[edge_idx] = min(weights[edge_idx], weight)
            weights[-(edge_idx + 1)] = min(weights[-(edge_idx + 1)], weight)

        spline_x = UnivariateSpline(t, sorted_fit_points[:, 0], k=k, s=smoothing, w=weights)
        spline_y = UnivariateSpline(t, sorted_fit_points[:, 1], k=k, s=smoothing, w=weights)

        sample_count = max(200, count * 10)
        t_fine = np.linspace(0.0, 1.0, sample_count)
        stem_points = np.column_stack([spline_x(t_fine), spline_y(t_fine)])
        stem_length = float(np.sum(np.linalg.norm(np.diff(stem_points, axis=0), axis=1)))

        # 将排序后的样条参数映射回原始小穗索引。
        spikelet_s = np.zeros(count, dtype=float)
        spikelet_tangent = np.zeros((count, 2), dtype=float)
        spikelet_side = np.zeros(count, dtype=float)

        original_t = np.zeros(count, dtype=float)
        original_t[order] = t
        spikelet_s[:] = original_t

        for idx in range(count):
            t_base = float(original_t[idx])
            tx = float(spline_x.derivative()(t_base))
            ty = float(spline_y.derivative()(t_base))
            tangent = np.array([tx, ty], dtype=float)
            tangent_norm = np.linalg.norm(tangent)
            if tangent_norm < 1e-8:
                tangent = np.array(main_dir, dtype=float)
                tangent_norm = np.linalg.norm(tangent)
            tangent = tangent / max(tangent_norm, 1e-8)
            spikelet_tangent[idx] = tangent

            base_point = lowest_points[idx]
            vector_to_center = centers[idx] - base_point
            cross = tangent[0] * vector_to_center[1] - tangent[1] * vector_to_center[0]
            spikelet_side[idx] = 1.0 if cross >= 0 else -1.0

        return {
            'spikelet_highest_points': highest_points,
            'spikelet_lowest_points': lowest_points,
            'spikelet_tangent': spikelet_tangent,
            'spikelet_s': spikelet_s,
            'spikelet_side': spikelet_side,
            'spikelet_order': np.argsort(spikelet_s),
            'stem_points': stem_points,
            'stem_length': stem_length,
            'stem_fit_points': stem_fit_points,
            'long_axis_dirs': long_dirs,
        }

    def _extract_spikelet_axes(self, centers: np.ndarray, xyxyxyxy: np.ndarray):
        """从 OBB 四点恢复长轴方向与高/低端点。"""
        count = len(centers)
        long_dirs = np.zeros((count, 2), dtype=float)
        highest_points = np.zeros((count, 2), dtype=float)
        lowest_points = np.zeros((count, 2), dtype=float)

        for idx in range(count):
            corners = np.asarray(xyxyxyxy[idx], dtype=float)
            edges = np.roll(corners, -1, axis=0) - corners
            edge_lengths = np.linalg.norm(edges, axis=1)
            long_edge = edges[np.argmax(edge_lengths)]
            edge_norm = np.linalg.norm(long_edge)
            long_dir = long_edge / edge_norm if edge_norm > 1e-8 else np.array([1.0, 0.0], dtype=float)

            relative = corners - centers[idx]
            projection = relative @ long_dir
            half_length = float(np.max(np.abs(projection)))
            endpoint_1 = centers[idx] + half_length * long_dir
            endpoint_2 = centers[idx] - half_length * long_dir

            long_dirs[idx] = long_dir
            highest_points[idx] = endpoint_1 if endpoint_1[1] <= endpoint_2[1] else endpoint_2
            lowest_points[idx] = endpoint_1 if endpoint_1[1] >= endpoint_2[1] else endpoint_2

        return long_dirs, highest_points, lowest_points
