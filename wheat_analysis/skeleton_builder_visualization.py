import numpy as np
import matplotlib.pyplot as plt

try:
    from wheat_analysis.skeleton import SkeletonBuilder
except ImportError:
    # 兼容直接在 wheat_analysis 目录下运行脚本
    from skeleton import SkeletonBuilder

# 中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']


def _make_obb_corners(center, angle_rad, long_len=24.0, short_len=8.0):
    """由中心点、方向角和长短轴长度生成矩形 OBB 四角点。"""
    cx, cy = center
    half_l = long_len * 0.5
    half_s = short_len * 0.5

    d = np.array([np.cos(angle_rad), np.sin(angle_rad)], dtype=float)
    n = np.array([-d[1], d[0]], dtype=float)

    p0 = np.array([cx, cy]) + half_l * d + half_s * n
    p1 = np.array([cx, cy]) + half_l * d - half_s * n
    p2 = np.array([cx, cy]) - half_l * d - half_s * n
    p3 = np.array([cx, cy]) - half_l * d + half_s * n
    return np.vstack([p0, p1, p2, p3])


def _create_demo_detection(num_spikelets=16, seed=42):
    """生成与新算法匹配的模拟输入：centers + xyxyxyxy。"""
    rng = np.random.default_rng(seed)

    t = np.linspace(0.0, 1.0, num_spikelets)
    stem_x = 80 + 110 * t + 8 * np.sin(2.2 * np.pi * t)
    stem_y = 35 + 185 * t

    # 左右交替分布在主茎两侧
    side_sign = np.where(np.arange(num_spikelets) % 2 == 0, 1.0, -1.0)
    lateral_offset = 14 + rng.normal(0, 1.5, size=num_spikelets)

    centers = np.column_stack(
        [
            stem_x + side_sign * lateral_offset + rng.normal(0, 1.6, size=num_spikelets),
            stem_y + rng.normal(0, 2.0, size=num_spikelets),
        ]
    )

    xyxyxyxy = np.zeros((num_spikelets, 4, 2), dtype=float)
    for i in range(num_spikelets):
        # 长轴接近竖直，加入小扰动
        angle = np.deg2rad(84 + rng.normal(0, 6))
        long_len = 22 + rng.normal(0, 2)
        short_len = 7 + rng.normal(0, 0.8)
        xyxyxyxy[i] = _make_obb_corners(centers[i], angle, long_len=long_len, short_len=short_len)

    return {
        'centers': centers,
        'xyxyxyxy': xyxyxyxy,
    }

# ========== 测试可视化函数 ==========
def test_skeleton_builder_visualization():
    """
    测试SkeletonBuilder并可视化每一步结果：
    1. 从 OBB 提取长轴端点（最高点/最低点）
    2. 以最低点做 PCA 并排序
    3. 样条拟合主茎骨架
    4. 小穗-主茎关联（左右侧、距离）
    """
    # 1. 生成与新算法一致的模拟输入（包含 OBB 角点）
    detection = _create_demo_detection(num_spikelets=16, seed=42)
    centers = detection['centers']
    obbs = detection['xyxyxyxy']

    # 2. 初始化并运行骨架构建
    builder = SkeletonBuilder(spline_smoothing=None)
    skeleton = builder.build(detection)

    highest_points = skeleton['spikelet_highest_points']
    lowest_points = skeleton['spikelet_lowest_points']
    stem_points = skeleton['stem_points']
    spikelet_side = skeleton['spikelet_side']

    # 可视化中复现一步 PCA（仅用于展示）
    mean_low = lowest_points.mean(axis=0)
    centered_low = lowest_points - mean_low
    cov = np.cov(centered_low.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    main_dir = eigvecs[:, np.argmax(eigvals)]
    if main_dir[1] < 0:
        main_dir = -main_dir
    projections = centered_low @ main_dir
    sorted_lowest = lowest_points[np.argsort(projections)]

    # 计算可视化用最近点（与 build 中最近点思想一致）
    nearest_on_stem = np.zeros_like(centers)
    for i in range(len(centers)):
        fit_x, fit_y = lowest_points[i]
        d2 = (stem_points[:, 0] - fit_x) ** 2 + (stem_points[:, 1] - fit_y) ** 2
        idx = int(np.argmin(d2))
        nearest_on_stem[i] = stem_points[idx]

    # 3. 创建可视化子图（2行2列）
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('新 SkeletonBuilder 算法分步可视化', fontsize=16, fontweight='bold')

    # ========== 子图1：OBB 与长轴端点提取 ==========
    ax1 = axes[0, 0]
    for i in range(len(obbs)):
        poly = np.vstack([obbs[i], obbs[i][0]])
        ax1.plot(poly[:, 0], poly[:, 1], color='gray', linewidth=1.2, alpha=0.6)
        ax1.plot(
            [highest_points[i, 0], lowest_points[i, 0]],
            [highest_points[i, 1], lowest_points[i, 1]],
            color='tab:blue',
            linewidth=1.8,
            alpha=0.9,
        )
    ax1.scatter(centers[:, 0], centers[:, 1], c='black', s=18, label='中心点', alpha=0.8)
    ax1.scatter(highest_points[:, 0], highest_points[:, 1], c='tab:green', s=28, label='长轴最高点')
    ax1.scatter(lowest_points[:, 0], lowest_points[:, 1], c='tab:red', s=28, label='长轴最低点')
    ax1.set_title('步骤1：由 OBB 提取长轴最高/最低点', fontweight='bold')
    ax1.set_xlabel('X坐标 (像素)')
    ax1.set_ylabel('Y坐标 (像素)')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.axis('equal')

    # ========== 子图2：最低点 PCA 与投影排序 ==========
    ax2 = axes[0, 1]
    ax2.scatter(lowest_points[:, 0], lowest_points[:, 1], c='tab:red', s=36, label='最低点')
    ax2.plot(sorted_lowest[:, 0], sorted_lowest[:, 1], '--', color='orange', linewidth=2, label='按投影排序后的折线')
    ax2.arrow(
        mean_low[0],
        mean_low[1],
        main_dir[0] * 60,
        main_dir[1] * 60,
        head_width=3,
        head_length=5,
        fc='purple',
        ec='purple',
        label='PCA 主方向',
    )
    ax2.scatter(mean_low[0], mean_low[1], c='purple', s=80, marker='x', label='最低点均值')
    ax2.set_title('步骤2：最低点 PCA + 投影排序', fontweight='bold')
    ax2.set_xlabel('X坐标 (像素)')
    ax2.set_ylabel('Y坐标 (像素)')
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.axis('equal')

    # ========== 子图3：样条拟合主茎骨架 ==========
    ax3 = axes[1, 0]
    ax3.scatter(sorted_lowest[:, 0], sorted_lowest[:, 1], c='orange', s=40, label='排序后最低点')
    ax3.plot(stem_points[:, 0], stem_points[:, 1], color='green', linewidth=3, label='样条拟合主茎')
    ax3.set_title('步骤3：样条拟合主茎骨架', fontweight='bold')
    ax3.set_xlabel('X坐标 (像素)')
    ax3.set_ylabel('Y坐标 (像素)')
    ax3.legend()
    ax3.grid(alpha=0.3)
    ax3.axis('equal')

    # ========== 子图4：小穗-主茎关联关系 ==========
    ax4 = axes[1, 1]
    ax4.plot(stem_points[:, 0], stem_points[:, 1], color='green', linewidth=3, label='主茎骨架')
    left_idx = spikelet_side < 0
    right_idx = spikelet_side >= 0
    ax4.scatter(centers[left_idx, 0], centers[left_idx, 1], c='tab:purple', s=70, label='左侧小穗', alpha=0.85)
    ax4.scatter(centers[right_idx, 0], centers[right_idx, 1], c='tab:cyan', s=70, label='右侧小穗', alpha=0.85)

    for i in range(len(centers)):
        ax4.plot(
            [centers[i, 0], nearest_on_stem[i, 0]],
            [centers[i, 1], nearest_on_stem[i, 1]],
            linestyle='--',
            color='gray',
            alpha=0.55,
        )
    ax4.scatter(nearest_on_stem[:, 0], nearest_on_stem[:, 1], c='red', s=26, marker='*', label='主茎最近点')
    ax4.set_title('步骤4-5：小穗-主茎关联（侧别/距离）', fontweight='bold')
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
    print(f"小穗数量: {len(centers)}")
    print(f"最低点平均纵坐标: {np.mean(lowest_points[:, 1]):.2f}")
    print(f"最高点平均纵坐标: {np.mean(highest_points[:, 1]):.2f}")
    print(f"左侧小穗数量: {sum(skeleton['spikelet_side'] == -1)}")
    print(f"右侧小穗数量: {sum(skeleton['spikelet_side'] == 1)}")
    print(f"小穗到主茎平均距离: {np.mean(skeleton['spikelet_dist']):.2f} 像素")

# 运行测试
if __name__ == "__main__":
    test_skeleton_builder_visualization()