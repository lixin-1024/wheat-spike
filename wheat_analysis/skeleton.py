"""
茎-穗骨架生成模块：从OBB检测结果中拟合主茎骨架线，
建立每个小穗到主茎的连接关系，形成"茎-穗"骨架。
"""
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize_scalar


class SkeletonBuilder:
    """从小穗检测结果构建茎-穗骨架"""

    def __init__(
        self,
        spline_smoothing: float = None,
        perp_ratio: float = 0.12,
        dist_ratio: float = 1.20,
        min_perp_px: float = 2.0,
        min_dist_px: float = 20.0,
        stem_extension_t: float = 0.12,
    ):
        """
        Args:
            spline_smoothing: 样条平滑参数，None 为自动选择
            perp_ratio: 交点候选的垂距阈值系数（相对小穗长轴）
            dist_ratio: 交点候选的距离阈值系数（相对小穗长轴）
            min_perp_px: 垂距阈值下限（像素）
            min_dist_px: 距离阈值下限（像素）
            stem_extension_t: 主茎尾端(基部)的参数外推量（用于连接底端侧枝）
        """
        self.spline_smoothing = spline_smoothing
        self.perp_ratio = perp_ratio
        self.dist_ratio = dist_ratio
        self.min_perp_px = min_perp_px
        self.min_dist_px = min_dist_px
        self.stem_extension_t = stem_extension_t

    def build(self, detection: dict) -> dict:
        """
        从检测结果构建骨架。

        算法步骤：
        1. 用小穗长轴最低点做 PCA，确定主茎方向
        2. 将长轴最低点投影到主方向，按投影排序
        3. 用样条曲线拟合主茎骨架线
        4. 计算每个小穗引线与主茎曲线的交点
        5. 归一化弧长参数 s∈[0,1]

        Returns:
            dict: {
                'stem_points': np.ndarray (M, 2),       # 主茎骨架采样点
                'stem_parameter': np.ndarray (M,),       # 对应的归一化弧长参数
                'spikelet_intersections': np.ndarray (N, 2),  # 引线与主茎曲线交点
                'spikelet_anchor_points': np.ndarray (N, 2),  # 引线起点(长轴最高端点)
                'spikelet_s': np.ndarray (N,),           # 小穗归一化弧长(裁剪到[0,1])
                'spikelet_t_raw': np.ndarray (N,),       # 小穗在扩展主茎上的原始参数
                'spikelet_dist': np.ndarray (N,),        # 小穗中心到主茎的距离
                'spikelet_side': np.ndarray (N,),        # 小穗在主茎左(-1)或右(+1)侧
                'spikelet_order': np.ndarray (N,),       # 小穗沿主茎的排列序号
                'spikelet_centers': np.ndarray (N, 2),   # 小穗中心点
                'spikelet_guide_dir': np.ndarray (N, 2), # 引线方向单位向量
                'stem_fit_points': np.ndarray (N, 2),   # 主茎拟合点(长轴最低点)
                'stem_length': float,                     # 主茎弧长(像素)
                'stem_direction': np.ndarray (2,),        # 主茎主方向(单位向量)
                'spline_x': UnivariateSpline,            # 主茎x样条函数
                'spline_y': UnivariateSpline,            # 主茎y样条函数
            }
        """
        centers = detection['centers']  # (N, 2)
        N = len(centers)
        xyxyxyxy = detection.get('xyxyxyxy', None)
        angles = detection.get('angles', None)
        xywhr = detection.get('xywhr', None)

        if N < 2:
            raise ValueError("至少需要2个小穗才能构建骨架")

        # 用于主茎拟合的输入点：优先采用每个小穗长轴最低点；缺失时回退中心点。
        stem_fit_points = centers.copy()
        if xyxyxyxy is not None and len(xyxyxyxy) == N:
            for i in range(N):
                corners = xyxyxyxy[i]  # (4, 2)
                edges = np.roll(corners, -1, axis=0) - corners
                edge_lengths = np.linalg.norm(edges, axis=1)
                long_edge = edges[np.argmax(edge_lengths)]
                dnorm = np.hypot(long_edge[0], long_edge[1])
                if dnorm < 1e-8:
                    continue
                long_dir = long_edge / dnorm
                rel = corners - centers[i]
                proj = rel @ long_dir
                half_len = float(np.max(np.abs(proj)))
                p1 = centers[i] + half_len * long_dir
                p2 = centers[i] - half_len * long_dir
                lowest = p1 if p1[1] >= p2[1] else p2
                stem_fit_points[i] = lowest
        elif angles is not None and len(angles) == N:
            guide_angles = angles.copy()
            if xywhr is not None and len(xywhr) == N:
                use_perp = xywhr[:, 3] >= xywhr[:, 2]
                guide_angles[use_perp] += np.pi / 2.0
            long_dirs = np.column_stack([np.cos(guide_angles), np.sin(guide_angles)])
            if xywhr is not None and len(xywhr) == N:
                half_lens = np.maximum(xywhr[:, 2], xywhr[:, 3]) * 0.5
            elif 'heights' in detection and len(detection['heights']) == N:
                half_lens = detection['heights'] * 0.5
            else:
                half_lens = np.full(N, 10.0)

            for i in range(N):
                p1 = centers[i] + half_lens[i] * long_dirs[i]
                p2 = centers[i] - half_lens[i] * long_dirs[i]
                lowest = p1 if p1[1] >= p2[1] else p2
                stem_fit_points[i] = lowest

        # ========== 1. PCA 确定主方向 ==========
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

        # ========== 2. 按投影排序 ==========
        order = np.argsort(projections)
        sorted_fit_points = stem_fit_points[order]
        sorted_proj = projections[order]

        # ========== 3. 样条拟合主茎 ==========
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
            smoothing = N * (total_length * 0.03) ** 2 # ✅ 修复：增大平滑系数，让曲线更“远离”原始点

        # 给首尾部分节点渐进降权，减少端部偏差
        weights = np.ones(N)
        n_edge = max(2, N // 5)  # 首尾各约20%的节点降权
        for ei in range(n_edge):
            w = 0.5 + 0.5 * (ei / n_edge)
            weights[ei] = min(weights[ei], w)
            weights[-(ei + 1)] = min(weights[-(ei + 1)], w)

        spline_x = UnivariateSpline(t, sorted_fit_points[:, 0], k=k, s=smoothing, w=weights)
        spline_y = UnivariateSpline(t, sorted_fit_points[:, 1], k=k, s=smoothing, w=weights)
        # ========== 4. 生成高密度骨架采样点 ==========
        # 基础主茎（用于长度度量）
        num_samples = max(220, N * 12)
        t_base = np.linspace(0, 1, num_samples)
        stem_x_base = spline_x(t_base)
        stem_y_base = spline_y(t_base)
        stem_points_base = np.column_stack([stem_x_base, stem_y_base])

        # 扩展主茎（仅尾端/基部扩展，用于底端侧枝连接）
        ext_t = max(0.0, float(self.stem_extension_t))
        # t_start, t_end = -ext_t, 1.0
        t_start, t_end = 0, 1.0
        t_fine = np.linspace(t_start, t_end, num_samples)
        stem_x = spline_x(t_fine)
        stem_y = spline_y(t_fine)
        stem_points = np.column_stack([stem_x, stem_y])

        # ========== 5. 基于 OBB 中心点与方向角，求引线与主茎交点 ==========
        spikelet_intersections = np.zeros((N, 2))
        spikelet_anchor_points = np.zeros((N, 2))
        spikelet_t_raw = np.zeros(N)
        spikelet_s = np.zeros(N)
        spikelet_dist = np.zeros(N)
        spikelet_side = np.zeros(N)
        spikelet_centers = centers.copy()
        spikelet_guide_dir = np.zeros((N, 2))

        # 优先基于 OBB 角点计算长轴方向，避免角度定义差异导致方向错误。
        if xyxyxyxy is not None and len(xyxyxyxy) == N:
            for i in range(N):
                corners = xyxyxyxy[i]  # (4, 2)
                edges = np.roll(corners, -1, axis=0) - corners
                edge_lengths = np.linalg.norm(edges, axis=1)
                long_edge = edges[np.argmax(edge_lengths)]
                norm = np.hypot(long_edge[0], long_edge[1])
                if norm > 1e-8:
                    spikelet_guide_dir[i] = long_edge / norm
                else:
                    spikelet_guide_dir[i] = main_dir
        elif angles is not None and len(angles) == N:
            # 后备方案：当仅有 xywhr 时，按长轴方向修正角度。
            guide_angles = angles.copy()
            if xywhr is not None and len(xywhr) == N:
                use_perp = xywhr[:, 3] >= xywhr[:, 2]
                guide_angles[use_perp] += np.pi / 2.0
            spikelet_guide_dir[:, 0] = np.cos(guide_angles)
            spikelet_guide_dir[:, 1] = np.sin(guide_angles)
        else:
            # 兼容缺失角度的场景：退化为"中心点到主茎最近点"连接。
            spikelet_guide_dir[:] = main_dir[None, :]

        for i in range(N):
            cx, cy = centers[i]
            dx, dy = spikelet_guide_dir[i]
            norm = np.hypot(dx, dy)
            if norm < 1e-8:
                dx, dy = float(main_dir[0]), float(main_dir[1])
                norm = np.hypot(dx, dy)
            dx, dy = dx / norm, dy / norm
            spikelet_guide_dir[i] = [dx, dy]

            # 引线起点：沿长轴方向的最高端点（图像坐标系中 y 更小）
            if xyxyxyxy is not None and len(xyxyxyxy) == N:
                corners = xyxyxyxy[i]
                rel = corners - centers[i]
                proj = rel @ spikelet_guide_dir[i]
                half_len = float(np.max(np.abs(proj)))
            elif xywhr is not None and len(xywhr) == N:
                half_len = float(max(xywhr[i, 2], xywhr[i, 3]) * 0.5)
            elif 'heights' in detection and len(detection['heights']) == N:
                half_len = float(detection['heights'][i] * 0.5)
            else:
                half_len = 10.0

            p1 = np.array([cx, cy]) + half_len * spikelet_guide_dir[i]
            p2 = np.array([cx, cy]) - half_len * spikelet_guide_dir[i]
            anchor = p1 if p1[1] <= p2[1] else p2
            sx, sy = float(anchor[0]), float(anchor[1])
            spikelet_anchor_points[i] = [sx, sy]

            # 候选过滤：先限制在“离引线近且离中心不过远”的主茎点集合，
            # 再在候选里选初值，避免交点跳到错误分支。
            vx = stem_x - sx
            vy = stem_y - sy
            perp = np.abs(vx * dy - vy * dx)
            anchor_dist = np.hypot(vx, vy)

            if 'heights' in detection and len(detection['heights']) == N:
                long_axis = float(detection['heights'][i])
            elif xywhr is not None and len(xywhr) == N:
                long_axis = float(max(xywhr[i, 2], xywhr[i, 3]))
            else:
                long_axis = 20.0

            tau_perp = max(self.min_perp_px, self.perp_ratio * long_axis)
            tau_dist = max(self.min_dist_px, self.dist_ratio * long_axis)

            valid = (perp <= tau_perp) & (anchor_dist <= tau_dist)
            if np.any(valid):
                score = anchor_dist + 2.0 * perp
                valid_idx = np.where(valid)[0]
                idx_min = valid_idx[np.argmin(score[valid_idx])]
            else:
                idx_min = int(np.argmin(anchor_dist))

            t_init = t_fine[idx_min]

            def line_intersection_obj(tt):
                vx = float(spline_x(tt)) - sx
                vy = float(spline_y(tt)) - sy
                cross = vx * dy - vy * dx
                return cross * cross

            result = minimize_scalar(
                line_intersection_obj,
                bounds=(max(t_start, t_init - 0.08), min(t_end, t_init + 0.08)),
                method='bounded'
            )
            t_opt = result.x

            px, py = float(spline_x(t_opt)), float(spline_y(t_opt))

            # 优化后验收：残差过大或距离过远时，回退到最近点连接。
            # residual = abs((px - sx) * dy - (py - sy) * dx)
            # final_dist = np.hypot(sx - px, sy - py)
            # if residual > 1.5 * tau_perp or final_dist > 1.8 * tau_dist:
            #     idx_fb = int(np.argmin(anchor_dist))
            #     t_opt = float(t_fine[idx_fb])
            #     px, py = float(stem_x[idx_fb]), float(stem_y[idx_fb])

            spikelet_intersections[i] = [px, py]
            spikelet_t_raw[i] = t_opt
            spikelet_dist[i] = np.hypot(cx - px, cy - py)

            # 判断左右侧：使用叉积
            dt = 0.001
            tx = float(spline_x(min(t_opt + dt, 1))) - float(spline_x(max(t_opt - dt, 0)))
            ty = float(spline_y(min(t_opt + dt, 1))) - float(spline_y(max(t_opt - dt, 0)))
            cross = tx * (cy - py) - ty * (cx - px)
            spikelet_side[i] = 1.0 if cross >= 0 else -1.0

        # ========== 5.5 同侧单调约束：减少侧枝交叉 ==========
        # 对每一侧的小穗，按中心沿主方向的顺序，约束其主茎弧长位置为非递减。
        # 这可抑制“上位小穗连到更下方主茎点”导致的同侧交叉。
        # t_before = spikelet_t_raw.copy()
        # for side_sign in (-1.0, 1.0):
        #     side_idx = np.where(spikelet_side == side_sign)[0]
        #     if len(side_idx) < 2:
        #         continue

        #     idx_sorted = side_idx[np.argsort(projections[side_idx])]
        #     prev_t = t_start - 1.0
        #     for idx in idx_sorted:
        #         t_val = float(spikelet_t_raw[idx])
        #         if t_val < prev_t:
        #             t_val = prev_t
        #         t_val = float(np.clip(t_val, t_start, t_end))
        #         spikelet_t_raw[idx] = t_val
        #         prev_t = t_val

        # changed_idx = np.where(np.abs(spikelet_t_raw - t_before) > 1e-6)[0]
        # for idx in changed_idx:
        #     t_adj = float(spikelet_t_raw[idx])
        #     px, py = float(spline_x(t_adj)), float(spline_y(t_adj))
        #     spikelet_intersections[idx] = [px, py]
        #     cx, cy = centers[idx]
        #     spikelet_dist[idx] = np.hypot(cx - px, cy - py)

        # 表型统计使用裁剪后的归一化参数，保持范围在 [0, 1]
        spikelet_s = np.clip(spikelet_t_raw, 0.0, 1.0)

        # 按弧长排序的序号
        spikelet_order = np.argsort(spikelet_t_raw)

        # ========== 6. 计算实际弧长 ==========
        stem_diffs = np.diff(stem_points_base, axis=0)
        stem_seg_lengths = np.linalg.norm(stem_diffs, axis=1)
        stem_arc_length = np.sum(stem_seg_lengths)

        return {
            'stem_points': stem_points,
            'stem_parameter': t_fine,
            'spikelet_intersections': spikelet_intersections,
            'spikelet_anchor_points': spikelet_anchor_points,
            'spikelet_s': spikelet_s,
            'spikelet_t_raw': spikelet_t_raw,
            'spikelet_dist': spikelet_dist,
            'spikelet_side': spikelet_side,
            'spikelet_order': spikelet_order,
            'spikelet_dist_to_stem': spikelet_dist,
            'spikelet_centers': spikelet_centers,
            'spikelet_guide_dir': spikelet_guide_dir,
            'stem_fit_points': stem_fit_points,
            'stem_length': stem_arc_length,
            'stem_direction': main_dir,
            'spline_x': spline_x,
            'spline_y': spline_y,
        }
