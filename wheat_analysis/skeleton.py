"""
茎-穗骨架生成模块：从OBB检测结果中拟合主茎骨架线，
建立每个小穗到主茎的连接关系，形成"茎-穗"骨架。
"""
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize_scalar


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
        1. PCA 确定主茎方向
        2. 将小穗中心投影到主方向，按投影排序
        3. 用样条曲线拟合主茎骨架线
        4. 计算每个小穗在主茎上的投影点（足点）
        5. 归一化弧长参数 s∈[0,1]

        Returns:
            dict: {
                'stem_points': np.ndarray (M, 2),       # 主茎骨架采样点
                'stem_parameter': np.ndarray (M,),       # 对应的归一化弧长参数
                'spikelet_proj': np.ndarray (N, 2),      # 小穗在主茎上的投影足点
                'spikelet_s': np.ndarray (N,),           # 小穗的归一化弧长位置 s∈[0,1]
                'spikelet_dist': np.ndarray (N,),        # 小穗中心到主茎的距离
                'spikelet_side': np.ndarray (N,),        # 小穗在主茎左(-1)或右(+1)侧
                'spikelet_order': np.ndarray (N,),       # 小穗沿主茎的排列序号
                'stem_length': float,                     # 主茎有效长度(像素)
                'stem_direction': np.ndarray (2,),        # 主茎主方向(单位向量)
                'spline_x': UnivariateSpline,            # 主茎x样条函数
                'spline_y': UnivariateSpline,            # 主茎y样条函数
            }
        """
        centers = detection['centers']  # (N, 2)
        N = len(centers)

        if N < 2:
            raise ValueError("至少需要2个小穗才能构建骨架")

        # ========== 1. PCA 确定主方向 ==========
        mean = centers.mean(axis=0)
        centered = centers - mean
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        # 主方向 = 最大特征值对应的特征向量
        main_dir = eigvecs[:, np.argmax(eigvals)]

        # 统一方向：让投影值从小到大（从穗基部到顶部）
        projections = centered @ main_dir
        if projections[np.argmin(projections[:, None].flatten())] > projections[np.argmax(projections[:, None].flatten())]:
            main_dir = -main_dir
            projections = -projections

        # ========== 2. 按投影排序 ==========
        order = np.argsort(projections)
        sorted_centers = centers[order]
        sorted_proj = projections[order]

        # ========== 3. 样条拟合主茎 ==========
        # 用累计弧长作为参数化变量
        diffs = np.diff(sorted_centers, axis=0)
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
            smoothing = N * (total_length * 0.03) ** 2 # ✅ 修复：增大平滑系数，让曲线更“远离”原始点

        # 给首尾部分节点渐进降权，减少端部偏差
        weights = np.ones(N)
        n_edge = max(2, N // 5)  # 首尾各约20%的节点降权
        for ei in range(n_edge):
            w = 0.3 + 0.7 * (ei / n_edge)
            weights[ei] = min(weights[ei], w)
            weights[-(ei + 1)] = min(weights[-(ei + 1)], w)

        spline_x = UnivariateSpline(t, sorted_centers[:, 0], k=k, s=smoothing, w=weights)
        spline_y = UnivariateSpline(t, sorted_centers[:, 1], k=k, s=smoothing, w=weights)
        # ========== 4. 生成高密度骨架采样点 ==========
        num_samples = max(200, N * 10)
        t_fine = np.linspace(0, 1, num_samples)
        stem_x = spline_x(t_fine)
        stem_y = spline_y(t_fine)
        stem_points = np.column_stack([stem_x, stem_y])

        # ========== 5. 计算OBB上下中点并投影到主茎 ==========
        xyxyxyxy = detection['xyxyxyxy']  # (N, 4, 2)
        spikelet_proj = np.zeros((N, 2))
        spikelet_s = np.zeros(N)
        spikelet_dist = np.zeros(N)
        spikelet_side = np.zeros(N)
        spikelet_near_mid = np.zeros((N, 2))
        spikelet_far_mid = np.zeros((N, 2))

        for i in range(N):
            # 计算OBB上下中点
            corners = xyxyxyxy[i]  # (4, 2)
            sorted_by_y = corners[np.argsort(corners[:, 1])]
            top_mid = sorted_by_y[:2].mean(axis=0)      # 最上面两个点的中点
            bottom_mid = sorted_by_y[2:].mean(axis=0)    # 最下面两个点的中点

            # 判断哪个中点离主茎更近
            d_top = np.min((stem_x - top_mid[0]) ** 2 + (stem_y - top_mid[1]) ** 2)
            d_bottom = np.min((stem_x - bottom_mid[0]) ** 2 + (stem_y - bottom_mid[1]) ** 2)
            if d_top < d_bottom:
                near_mid, far_mid = top_mid, bottom_mid
            else:
                near_mid, far_mid = bottom_mid, top_mid

            spikelet_near_mid[i] = near_mid
            spikelet_far_mid[i] = far_mid

            # 使用 near_mid 投影到主茎
            nmx, nmy = near_mid
            dists = np.sqrt((stem_x - nmx) ** 2 + (stem_y - nmy) ** 2)
            idx_min = np.argmin(dists)
            t_init = t_fine[idx_min]

            def dist_func(tt):
                return (spline_x(tt) - nmx) ** 2 + (spline_y(tt) - nmy) ** 2

            result = minimize_scalar(
                dist_func,
                bounds=(max(0, t_init - 0.05), min(1, t_init + 0.05)),
                method='bounded'
            )
            t_opt = result.x

            px, py = float(spline_x(t_opt)), float(spline_y(t_opt))
            spikelet_proj[i] = [px, py]
            spikelet_s[i] = t_opt
            spikelet_dist[i] = np.sqrt((nmx - px) ** 2 + (nmy - py) ** 2)

            # 判断左右侧：使用叉积
            dt = 0.001
            tx = float(spline_x(min(t_opt + dt, 1))) - float(spline_x(max(t_opt - dt, 0)))
            ty = float(spline_y(min(t_opt + dt, 1))) - float(spline_y(max(t_opt - dt, 0)))
            cross = tx * (nmy - py) - ty * (nmx - px)
            spikelet_side[i] = 1.0 if cross >= 0 else -1.0

        # 按弧长排序的序号
        spikelet_order = np.argsort(spikelet_s)

        # ========== 6. 计算实际弧长 ==========
        stem_diffs = np.diff(stem_points, axis=0)
        stem_seg_lengths = np.linalg.norm(stem_diffs, axis=1)
        actual_stem_length = np.sum(stem_seg_lengths)

        return {
            'stem_points': stem_points,
            'stem_parameter': t_fine,
            'spikelet_proj': spikelet_proj,
            'spikelet_s': spikelet_s,
            'spikelet_dist': spikelet_dist,
            'spikelet_side': spikelet_side,
            'spikelet_order': spikelet_order,
            'spikelet_dist_to_stem': spikelet_dist,
            'spikelet_near_mid': spikelet_near_mid,
            'spikelet_far_mid': spikelet_far_mid,
            'stem_length': actual_stem_length,
            'stem_direction': main_dir,
            'spline_x': spline_x,
            'spline_y': spline_y,
        }
