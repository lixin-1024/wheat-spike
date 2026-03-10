# plot_paper_results.py
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path

# ========= 1) 全局科研风格 =========
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "SimSun", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 0.9,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
})

# 色盲友好 + 论文常用低饱和配色
COLOR_RAW = "#3B6FB6"      # 主曲线
COLOR_SMOOTH = "#D27D2D"   # 平滑曲线
GRID_COLOR = "#D9D9D9"

# ========= 2) 读取训练结果 =========
csv_path = Path(r"runs/obb/yolo11_1440_4/results.csv")
df = pd.read_csv(csv_path)
df.columns = [c.strip() for c in df.columns]

metrics = [
    "train/box_loss", "train/cls_loss", "train/dfl_loss", "train/angle_loss",
    "metrics/precision(B)", "metrics/recall(B)",
    "val/box_loss", "val/cls_loss", "val/dfl_loss", "val/angle_loss",
    "metrics/mAP50(B)", "metrics/mAP50-95(B)"
]

# 兼容部分版本字段缺失
metrics = [m for m in metrics if m in df.columns]

# ========= 3) 绘图 =========
n = len(metrics)
cols = 6
rows = (n + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.8 * rows), constrained_layout=True)
axes = axes.flatten()

x = df["epoch"] if "epoch" in df.columns else range(1, len(df) + 1)

for i, m in enumerate(metrics):
    ax = axes[i]
    y = df[m]
    y_smooth = y.rolling(window=7, min_periods=1).mean()

    ax.plot(x, y, color=COLOR_RAW, linewidth=1.8, marker="o", markersize=3.2, alpha=0.9, label="raw")
    ax.plot(x, y_smooth, color=COLOR_SMOOTH, linewidth=2.0, linestyle="--", alpha=0.95, label="smooth")

    ax.set_title(m, pad=6)
    ax.grid(True, color=GRID_COLOR, linewidth=0.7, alpha=0.8)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.tick_params(direction="in", length=4, width=0.8)

    for spine in ax.spines.values():
        spine.set_alpha(0.9)

    if i == 0:
        ax.legend(frameon=False, loc="best")

# 删除多余子图
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

out_png = csv_path.parent / "results_paper.png"
out_pdf = csv_path.parent / "results_paper.pdf"
fig.savefig(out_png, dpi=600)
fig.savefig(out_pdf)
plt.close(fig)

print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")