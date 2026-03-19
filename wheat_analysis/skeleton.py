"""
茎-穗骨架生成模块：从OBB检测结果中拟合主茎骨架线，
建立每个小穗到主茎的连接关系，形成"茎-穗"骨架。
"""
import numpy as np
from scipy.interpolate import UnivariateSpline


class SkeletonBuilder:
    """从小穗检测结果构建茎-穗骨架"""

    def __init__(self, spline_smoothing: float = None):
        """
        Args:
            spline_smoothing: 样条平滑参数，None 为自动选择
        """
        self.spline_smoothing = spline_smoothing

    def build(self, detection: dict) -> dict:
        """
        从检测结果构建骨架。

        算法步骤：
        1. 提取每个小穗长轴（最高点-最低点）
        2. 用小穗长轴最低点做 PCA，确定主茎方向
        3. 将长轴最低点投影到主方向，按投影排序
        4. 用样条曲线拟合主茎骨架线
        5. 计算小穗到主茎的最近点统计量（s、dist、side）
        6. 计算实际弧长

        Returns:
            dict: {
                'spikelet_highest_points': np.ndarray (N, 2), # 小穗长轴最高点
                'spikelet_lowest_points': np.ndarray (N, 2),  # 小穗长轴最低点
                'spikelet_s': np.ndarray (N,),           # 小穗归一化弧长(裁剪到[0,1])
                'spikelet_dist': np.ndarray (N,),        # 小穗中心到主茎的距离
                'spikelet_side': np.ndarray (N,),        # 小穗在主茎左(-1)或右(+1)侧
                'spikelet_order': np.ndarray (N,),       # 小穗沿主茎的排列序号
                'stem_points': np.ndarray (M, 2),        # 主茎骨架采样点
                'stem_length': float,                    # 主茎弧长(像素)
            }
        """
        centers = detection['centers']  # (N, 2)
        N = len(centers)
        xyxyxyxy = detection.get('xyxyxyxy', None)

        if N < 2:
            raise ValueError("至少需要2个小穗才能构建骨架")
        if xyxyxyxy is None or len(xyxyxyxy) != N:
            raise ValueError("缺少有效的 OBB 角点 xyxyxyxy，无法构建骨架")
        
        # 1. 从角点提取长轴方向、长轴半长、最高/最低端点
        long_dirs = np.zeros((N, 2), dtype=float)
        highest_points = np.zeros((N, 2), dtype=float)
        lowest_points = np.zeros((N, 2), dtype=float)
        for i in range(N):
            corners = xyxyxyxy[i]  # (4, 2)
            edges = np.roll(corners, -1, axis=0) - corners
            edge_lengths = np.linalg.norm(edges, axis=1)
            long_edge = edges[np.argmax(edge_lengths)]
            dnorm = np.hypot(long_edge[0], long_edge[1])
            if dnorm < 1e-8:
                long_dir = np.array([1.0, 0.0], dtype=float)
            else:
                long_dir = long_edge / dnorm

            rel = corners - centers[i]
            proj = rel @ long_dir
            half_len = float(np.max(np.abs(proj)))
            p1 = centers[i] + half_len * long_dir
            p2 = centers[i] - half_len * long_dir

            long_dirs[i] = long_dir
            highest_points[i] = p1 if p1[1] <= p2[1] else p2
            lowest_points[i] = p1 if p1[1] >= p2[1] else p2

        # 主茎拟合输入点：小穗长轴最低点
        stem_fit_points = lowest_points.copy()

        # ========== 2. 用小穗长轴最低点做 PCA，确定主茎方向 ==========
        mean = stem_fit_points.mean(axis=0)
        centered = stem_fit_points - mean
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        # 主方向 = 最大特征值对应的特征向量
        main_dir = eigvecs[:, np.argmax(eigvals)]

        # 统一方向：让投影值从小到大（从穗基部到顶部）
        projections = centered @ main_dir
        if projections[np.argmin(projections[:, None].flatten())] > projections[np.argmax(projections[:, None].flatten())]:
            main_dir = -main_dir
            projections = -projections

        # ========== 3. 将长轴最低点投影到主方向，按投影排序 ==========
        order = np.argsort(projections)
        sorted_fit_points = stem_fit_points[order]

        # ========== 4. 用样条曲线拟合主茎骨架线 ==========
        # 用累计弧长作为参数化变量
        diffs = np.diff(sorted_fit_points, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        cum_arc = np.zeros(N)
        cum_arc[1:] = np.cumsum(seg_lengths)
        total_length = cum_arc[-1]

        # 归一化弧长
        t = cum_arc / total_length if total_length > 0 else np.linspace(0, 1, N)

        # 拟合样条
        k = min(3, N - 1)  # 样条阶数不超过 N-1
        smoothing = self.spline_smoothing
        if smoothing is None:
            # 自动平滑：允许一定偏差
            smoothing = N * (total_length * 0.01)

        # 给首尾部分节点渐进降权，减少端部偏差
        weights = np.ones(N)
        n_edge = max(2, N // 5)  # 首尾各约20%的节点降权
        for ei in range(n_edge):
            w = 0.7 + 0.3 * (ei / n_edge)
            weights[ei] = min(weights[ei], w)
            weights[-(ei + 1)] = min(weights[-(ei + 1)], w)

        spline_x = UnivariateSpline(t, sorted_fit_points[:, 0], k=k, s=smoothing, w=weights)
        spline_y = UnivariateSpline(t, sorted_fit_points[:, 1], k=k, s=smoothing, w=weights)

        # 生成高密度主茎采样点
        num_samples = max(200, N * 10)
        t_fine = np.linspace(0, 1, num_samples)
        stem_x = spline_x(t_fine)
        stem_y = spline_y(t_fine)
        stem_points = np.column_stack([stem_x, stem_y])

        # ========== 5. 计算小穗到主茎的最近点统计量（s、dist、side） ==========
        spikelet_s = np.zeros(N)
        spikelet_dist = np.zeros(N)
        spikelet_side = np.zeros(N)

        for i in range(N):
            cx, cy = centers[i]
            dx, dy = long_dirs[i]
            norm = np.hypot(dx, dy)
            if norm < 1e-8:
                dx, dy = float(main_dir[0]), float(main_dir[1])
                norm = np.hypot(dx, dy)
            dx, dy = dx / norm, dy / norm

            fit_x, fit_y = stem_fit_points[i]
            d2 = (stem_x - fit_x) ** 2 + (stem_y - fit_y) ** 2
            idx_min = int(np.argmin(d2))
            t_opt = float(t_fine[idx_min])
            px, py = float(stem_x[idx_min]), float(stem_y[idx_min])

            spikelet_s[i] = t_opt
            spikelet_dist[i] = np.hypot(cx - px, cy - py)

            # 判断左右侧：使用叉积
            dt = 0.001
            tx = float(spline_x(min(t_opt + dt, 1))) - float(spline_x(max(t_opt - dt, 0)))
            ty = float(spline_y(min(t_opt + dt, 1))) - float(spline_y(max(t_opt - dt, 0)))
            cross = tx * (cy - py) - ty * (cx - px)
            spikelet_side[i] = 1.0 if cross >= 0 else -1.0

        # 按弧长排序的序号
        spikelet_order = np.argsort(spikelet_s)

        # ========== 6. 计算实际弧长 ==========
        stem_diffs = np.diff(stem_points, axis=0)
        stem_seg_lengths = np.linalg.norm(stem_diffs, axis=1)
        stem_arc_length = np.sum(stem_seg_lengths)

        return {
            'spikelet_highest_points': highest_points,
            'spikelet_lowest_points': lowest_points,
            'spikelet_s': spikelet_s,
            'spikelet_dist': spikelet_dist,
            'spikelet_side': spikelet_side,
            'spikelet_order': spikelet_order,
            'stem_points': stem_points,
            'stem_length': stem_arc_length,
        }
