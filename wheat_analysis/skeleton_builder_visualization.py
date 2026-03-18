import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize_scalar
# 中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
# 先复制原SkeletonBuilder类（保持不变）
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
        4. 计算每个小穗到主茎的交点
        5. 归一化弧长参数 s∈[0,1]

        Returns:
            dict: {
                'stem_points': np.ndarray (M, 2),       # 主茎骨架采样点
                'stem_parameter': np.ndarray (M,),       # 对应的归一化弧长参数
                'spikelet_intersections': np.ndarray (N, 2),  # 小穗到主茎交点
                'spikelet_s': np.ndarray (N,),           # 小穗的归一化弧长位置 s∈[0,1]
                'spikelet_dist': np.ndarray (N,),        # 小穗中心到主茎的距离
                'spikelet_side': np.ndarray (N,),        # 小穗在主茎左(-1)或右(+1)侧
                'spikelet_order': np.ndarray (N,),       # 小穗沿主茎的排列序号
                'stem_length': float,                     # 主茎弧长(像素)
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
            # ✅ 修复：增大平滑系数，避免曲线过度贴合点
            smoothing = N * (total_length * 0.08) ** 2  # 从 0.02 改为 0.08

        # ✅ 修复：给首尾点更低权重，避免主茎穿过端点
        weights = np.ones(N)
        weights[0] = 0.45  # 首点权重降低
        weights[-1] = 0.45  # 尾点权重降低

        spline_x = UnivariateSpline(t, sorted_centers[:, 0], k=k, s=smoothing, w=weights)
        spline_y = UnivariateSpline(t, sorted_centers[:, 1], k=k, s=smoothing, w=weights)
        # ========== 4. 生成高密度骨架采样点 ==========
        num_samples = max(200, N * 10)
        t_fine = np.linspace(0, 1, num_samples)
        stem_x = spline_x(t_fine)
        stem_y = spline_y(t_fine)
        stem_points = np.column_stack([stem_x, stem_y])

        # ========== 5. 计算每个小穗到主茎的交点与弧长 ==========
        spikelet_intersections = np.zeros((N, 2))
        spikelet_s = np.zeros(N)
        spikelet_dist = np.zeros(N)
        spikelet_side = np.zeros(N)

        for i in range(N):
            cx, cy = centers[i]

            # 找到距离最近的骨架点作为初始估计
            dists = np.sqrt((stem_x - cx) ** 2 + (stem_y - cy) ** 2)
            idx_min = np.argmin(dists)
            t_init = t_fine[idx_min]

            # 精确优化
            def dist_func(tt):
                return (spline_x(tt) - cx) ** 2 + (spline_y(tt) - cy) ** 2

            result = minimize_scalar(
                dist_func,
                bounds=(max(0, t_init - 0.05), min(1, t_init + 0.05)),
                method='bounded'
            )
            t_opt = result.x

            px, py = float(spline_x(t_opt)), float(spline_y(t_opt))
            spikelet_intersections[i] = [px, py]
            spikelet_s[i] = t_opt
            spikelet_dist[i] = np.sqrt((cx - px) ** 2 + (cy - py) ** 2)

            # 判断左右侧：使用叉积
            # 主茎切线方向
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
        actual_stem_length = np.sum(stem_seg_lengths)

        return {
            'stem_points': stem_points,
            'stem_parameter': t_fine,
            'spikelet_intersections': spikelet_intersections,
            'spikelet_s': spikelet_s,
            'spikelet_dist': spikelet_dist,
            'spikelet_side': spikelet_side,
            'spikelet_order': spikelet_order,
            'spikelet_dist_to_stem': spikelet_dist,
            'stem_length': actual_stem_length,
            'stem_direction': main_dir,
            'spline_x': spline_x,
            'spline_y': spline_y,
            # 额外返回中间结果，用于可视化
            'mean': mean,
            'centered': centered,
            'main_dir': main_dir,
            'projections': projections,
            'sorted_centers': sorted_centers,
            't': t
        }

# ========== 测试可视化函数 ==========
def test_skeleton_builder_visualization():
    """
    测试SkeletonBuilder并可视化每一步结果：
    1. 原始小穗点 + PCA主方向
    2. 按投影排序后的小穗点
    3. 样条拟合的主茎骨架
    4. 小穗-主茎关联（交点、左右侧、距离）
    """
    # 1. 生成模拟数据（模拟麦穗小穗中心点）
    # 模拟弯曲的麦穗小穗分布
    np.random.seed(42)  # 固定随机种子，结果可复现
    # ✅ 新的模拟数据：左右两列交叉分布，沿主茎方向延伸
    np.random.seed(42)
    num_spikelets = 16  # 偶数，方便左右交替
    t_sim = np.linspace(0, 1, num_spikelets)
    base_x = 50 + 100 * t_sim  # 主茎基线x坐标
    base_y = 20 + 80 * t_sim  # 主茎基线y坐标

    # 左右交替偏移：奇数索引在左侧，偶数在右侧（或反之）
    offset = 15  # 左右偏移量，控制小穗到主茎的距离
    x_sim = []
    y_sim = []
    for i in range(num_spikelets):
        if i % 2 == 0:
            # 右侧小穗：x += offset
            x = base_x[i] + offset + np.random.normal(0, 2)
        else:
            # 左侧小穗：x -= offset
            x = base_x[i] - offset + np.random.normal(0, 2)
        y = base_y[i] + np.random.normal(0, 2)
        x_sim.append(x)
        y_sim.append(y)
    x_sim = np.array(x_sim)
    y_sim = np.array(y_sim)
    centers = np.column_stack([x_sim, y_sim])
    detection = {'centers': centers}

    # 2. 初始化并运行骨架构建
    builder = SkeletonBuilder(spline_smoothing=None)
    skeleton = builder.build(detection)

    # 3. 创建可视化子图（2行2列）
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('茎-穗骨架生成算法分步可视化', fontsize=16, fontweight='bold')

    # ========== 子图1：原始小穗点 + PCA主方向 ==========
    ax1 = axes[0, 0]
    ax1.scatter(centers[:, 0], centers[:, 1], c='blue', s=60, label='小穗中心点', alpha=0.8)
    # 绘制PCA主方向（从均值点向主方向延伸）
    mean = skeleton['mean']
    main_dir = -skeleton['main_dir']
    ax1.arrow(mean[0], mean[1], main_dir[0]*50, main_dir[1]*50,
              head_width=3, head_length=5, fc='red', ec='red', label='PCA主方向')
    ax1.scatter(mean[0], mean[1], c='red', s=100, marker='x', label='均值点')
    ax1.set_title('步骤1：原始小穗点 + PCA主方向', fontweight='bold')
    ax1.set_xlabel('X坐标 (像素)')
    ax1.set_ylabel('Y坐标 (像素)')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.axis('equal')

    # ========== 子图2：按投影排序后的小穗点 ==========
    ax2 = axes[0, 1]
    sorted_centers = skeleton['sorted_centers']
    ax2.scatter(sorted_centers[:, 0], sorted_centers[:, 1], c='orange', s=60, label='排序后小穗点')
    ax2.plot(sorted_centers[:, 0], sorted_centers[:, 1], color='orange', linestyle='--', alpha=0.7, label='排序后折线')
    ax2.set_title('步骤2：按主方向投影排序', fontweight='bold')
    ax2.set_xlabel('X坐标 (像素)')
    ax2.set_ylabel('Y坐标 (像素)')
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.axis('equal')

    # ========== 子图3：样条拟合的主茎骨架 ==========
    ax3 = axes[1, 0]
    stem_points = skeleton['stem_points']
    ax3.scatter(sorted_centers[:, 0], sorted_centers[:, 1], c='orange', s=60, label='排序后小穗点')
    ax3.plot(stem_points[:, 0], stem_points[:, 1], 'green', linewidth=3, label='样条拟合主茎')
    ax3.set_title('步骤3：样条拟合主茎骨架', fontweight='bold')
    ax3.set_xlabel('X坐标 (像素)')
    ax3.set_ylabel('Y坐标 (像素)')
    ax3.legend()
    ax3.grid(alpha=0.3)
    ax3.axis('equal')

    # ========== 子图4：小穗-主茎关联关系 ==========
    ax4 = axes[1, 1]
    spikelet_intersections = skeleton['spikelet_intersections']
    spikelet_side = skeleton['spikelet_side']
    # 绘制主茎
    ax4.plot(stem_points[:, 0], stem_points[:, 1], 'green', linewidth=3, label='主茎骨架')
    # 按左右侧区分颜色绘制小穗点
    left_idx = spikelet_side == -1
    right_idx = spikelet_side == 1
    ax4.scatter(centers[left_idx, 0], centers[left_idx, 1], c='purple', s=80, label='左侧小穗', alpha=0.8)
    ax4.scatter(centers[right_idx, 0], centers[right_idx, 1], c='cyan', s=80, label='右侧小穗', alpha=0.8)
    # 绘制小穗到主茎的连接线
    for i in range(len(centers)):
        ax4.plot([centers[i,0], spikelet_intersections[i,0]], [centers[i,1], spikelet_intersections[i,1]],
                 'gray', linestyle='--', alpha=0.5)
        ax4.scatter(spikelet_intersections[i,0], spikelet_intersections[i,1], c='red', s=30, marker='*')
    ax4.set_title('步骤4-5：小穗-主茎关联（交点/左右侧）', fontweight='bold')
    ax4.set_xlabel('X坐标 (像素)')
    ax4.set_ylabel('Y坐标 (像素)')
    ax4.legend()
    ax4.grid(alpha=0.3)
    ax4.axis('equal')

    # 调整布局并保存/显示
    plt.tight_layout()
    plt.savefig('skeleton_builder_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 输出关键数值结果
    print("===== 算法关键结果 =====")
    print(f"主茎弧长: {skeleton['stem_length']:.2f} 像素")
    print(f"主茎主方向向量: ({skeleton['main_dir'][0]:.3f}, {skeleton['main_dir'][1]:.3f})")
    print(f"小穗数量: {len(centers)}")
    print(f"左侧小穗数量: {sum(skeleton['spikelet_side'] == -1)}")
    print(f"右侧小穗数量: {sum(skeleton['spikelet_side'] == 1)}")
    print(f"小穗到主茎平均距离: {np.mean(skeleton['spikelet_dist']):.2f} 像素")

# 运行测试
if __name__ == "__main__":
    test_skeleton_builder_visualization()