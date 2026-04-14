"""
可视化模块：绘制检测结果、骨架、表型标注等可视化图像。
"""
import cv2
import numpy as np
from pathlib import Path


class Visualizer:
    """小麦麦穗表型分析可视化"""

    # 配色方案
    COLOR_STEM = (0, 200, 0)           # 主茎线：绿色
    COLOR_OBB = (0, 255, 0)            # OBB框：绿色
    COLOR_CENTER = (0, 0, 255)         # 中心点：红色
    COLOR_TEXT = (255, 255, 255)       # 文字：白色
    COLOR_LEFT = (255, 100, 100)       # 左侧小穗
    COLOR_RIGHT = (111, 81, 255)      # 右侧小穗

    def _draw_order_badge(self, image: np.ndarray, center: tuple[int, int], order: int):
        """绘制实心红色圆圈序号（默认两位数显示）。"""
        cx, cy = center
        radius = 12
        badge_color = (0, 0, 255)
        label = f"{order:02d}"
        cv2.circle(image, (cx, cy), radius, badge_color, -1)
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        text_x = cx - text_w // 2
        text_y = cy + (text_h - baseline) // 2
        cv2.putText(image, label, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.COLOR_TEXT, 1)

    def _draw_conf_label(self, image: np.ndarray, center: np.ndarray, index: int, conf_value: float):
        """绘制形如 `index:conf` 的置信度标签。"""
        label = f"{index}:{conf_value:.2f}"

        # 置信度分级配色：高-深绿，中-橙，低-红
        if conf_value >= 0.8:
            bg_color = (0, 120, 0)
        elif conf_value >= 0.6:
            bg_color = (0, 165, 255)
        else:
            bg_color = (0, 0, 255)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # 以目标中心点为锚点，标签绘制在中心点右上方
        cx, cy = int(center[0]), int(center[1])
        x1 = max(0, cx - text_w // 2)  # 改为水平居中
        y2 = max(text_h + baseline + 6, cy - 6)
        y1 = y2 - text_h - baseline - 6
        x2 = min(image.shape[1] - 1, x1 + text_w + 8)

        if x2 - x1 < text_w + 4:
            x1 = max(0, x2 - text_w - 8)
        if x2 <= x1:
            return

        cv2.rectangle(image, (x1, y1), (x2, y2), bg_color, -1)
        cv2.putText(image, label, (x1 + 4, y2 - baseline - 3),
                    font, font_scale, self.COLOR_TEXT, thickness)

    def draw_detection(self, image: np.ndarray, detection: dict,
                       draw_obb: bool = True, draw_centers: bool = True,
                       draw_index: bool = False) -> np.ndarray:
        """绘制OBB检测结果"""
        vis = image.copy()
        xyxyxyxy = detection['xyxyxyxy']
        centers = detection['centers']
        conf = detection['conf']

        for i in range(len(xyxyxyxy)):
            corners = xyxyxyxy[i].astype(np.int32)

            if draw_obb:
                cv2.drawContours(vis, [corners], 0, self.COLOR_OBB, 2)
                self._draw_conf_label(vis, centers[i], i, float(conf[i]))

            if draw_centers:
                cx, cy = int(centers[i, 0]), int(centers[i, 1])
                cv2.circle(vis, (cx, cy), 5, self.COLOR_CENTER, -1)

            if draw_index:
                cx, cy = int(centers[i, 0]), int(centers[i, 1])
                self._draw_order_badge(vis, (cx + 12, cy - 12), i)

        return vis

    def draw_skeleton(self, image: np.ndarray, detection: dict,
                      skeleton: dict) -> np.ndarray:
        """绘制茎-穗骨架"""
        vis = image.copy()

        # 1. 绘制主茎线
        stem_pts = skeleton['stem_points'].astype(np.int32)
        for j in range(len(stem_pts) - 1):
            cv2.line(vis, tuple(stem_pts[j]), tuple(stem_pts[j + 1]),
                     self.COLOR_STEM, 3)

        # 2. 绘制小穗长轴（最高点-最低点）
        tops = skeleton['spikelet_highest_points']
        bottoms = skeleton['spikelet_lowest_points']
        sides = skeleton['spikelet_side']
        centers = detection['centers']

        # 构建排序标签: original_index → rank (1-based)
        spikelet_order = skeleton['spikelet_order']
        order_labels = np.empty_like(spikelet_order)
        order_labels[spikelet_order] = np.arange(len(spikelet_order))

        for i in range(len(tops)):
            tx, ty = int(tops[i, 0]), int(tops[i, 1])
            bx, by = int(bottoms[i, 0]), int(bottoms[i, 1])
            cx, cy = int(centers[i, 0]), int(centers[i, 1])

            color = self.COLOR_LEFT if sides[i] < 0 else self.COLOR_RIGHT
            cv2.line(vis, (tx, ty), (bx, by), color, 2)
            cv2.circle(vis, (tx, ty), 3, color, -1)
            cv2.circle(vis, (bx, by), 3, color, -1)
            cv2.circle(vis, (cx, cy), 4, self.COLOR_CENTER, -1)

            # 标注沿主茎排序序号
            self._draw_order_badge(vis, (cx + 14, cy - 14), int(order_labels[i] + 1))

        return vis
    
    def draw_full_analysis(self, image: np.ndarray, detection: dict,
                           skeleton: dict, spikelet_pheno: dict,
                           ear_pheno: dict) -> np.ndarray:
        """绘制完整的分析图（检测+骨架+表型标注）"""
        vis = image.copy()

        # 1. 绘制OBB框
        xyxyxyxy = detection['xyxyxyxy']
        conf = detection.get('conf')
        for i in range(len(xyxyxyxy)):
            corners = xyxyxyxy[i].astype(np.int32)
            cv2.drawContours(vis, [corners], 0, self.COLOR_OBB, 2)
            if conf is not None:
                self._draw_conf_label(vis, detection['centers'][i], i, float(conf[i]))

        # 2. 绘制骨架
        stem_pts = skeleton['stem_points'].astype(np.int32)
        for j in range(len(stem_pts) - 1):
            cv2.line(vis, tuple(stem_pts[j]), tuple(stem_pts[j + 1]),
                     self.COLOR_STEM, 3)

        # 3. 绘制小穗长轴和标注
        tops = skeleton['spikelet_highest_points']
        bottoms = skeleton['spikelet_lowest_points']
        sides = skeleton['spikelet_side']
        centers = detection['centers']
        lengths = spikelet_pheno['lengths']
        widths = spikelet_pheno['widths']
        angles = spikelet_pheno['attachment_angles_deg']

        # 构建排序标签
        spikelet_order = skeleton['spikelet_order']
        order_labels = np.empty_like(spikelet_order)
        order_labels[spikelet_order] = np.arange(len(spikelet_order))

        for i in range(len(tops)):
            tx, ty = int(tops[i, 0]), int(tops[i, 1])
            bx, by = int(bottoms[i, 0]), int(bottoms[i, 1])
            cx, cy = int(centers[i, 0]), int(centers[i, 1])

            color = self.COLOR_LEFT if sides[i] < 0 else self.COLOR_RIGHT
            cv2.line(vis, (tx, ty), (bx, by), color, 2)
            cv2.circle(vis, (tx, ty), 3, color, -1)
            cv2.circle(vis, (bx, by), 3, color, -1)
            cv2.circle(vis, (cx, cy), 4, self.COLOR_CENTER, -1)

            # 标注尺寸、着生角度和排序序号
            label = f"{order_labels[i] + 1}: L:{lengths[i]:.0f} W:{widths[i]:.0f} A:{angles[i]:.1f}"
            cv2.putText(vis, label, (cx + 10, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLOR_TEXT, 1)

        # 4. 在图像左上角绘制穗级表型摘要
        self._draw_ear_summary(vis, ear_pheno)

        return vis

    def _draw_ear_summary(self, image: np.ndarray, ear_pheno: dict):
        """在图像上绘制穗级表型摘要信息框"""
        lines = [
            f"Spikelets: {ear_pheno['spikelet_count']}",
            f"Spike Length: {ear_pheno['spike_length_px']:.1f} px",
            f"Mean L/W: {ear_pheno['mean_spikelet_length']:.1f}/{ear_pheno['mean_spikelet_width']:.1f}",
            f"Mean Attach Angle: {ear_pheno['mean_attachment_angle']:.1f} deg",
            f"Density: {ear_pheno['spikelet_density_px']:.4f}",
            f"Symmetry: {ear_pheno['symmetry_index']:.4f}",
            f"Centroid Offset: {ear_pheno['centroid_offset']:.4f}",
        ]

        if ear_pheno.get('spike_length_cm') is not None:
            lines.insert(2, f"Spike Length: {ear_pheno['spike_length_cm']:.2f} cm")

        x0, y0 = 20, 30
        line_h = 35
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2

        # 绘制半透明背景
        h = line_h * len(lines) + 20
        w = 400
        overlay = image.copy()
        cv2.rectangle(overlay, (x0 - 10, y0 - 25), (x0 + w, y0 + h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

        for i, line in enumerate(lines):
            cv2.putText(image, line, (x0, y0 + i * line_h),
                        font, font_scale, self.COLOR_TEXT, thickness)

    def save(self, image: np.ndarray, path: str):
        """保存图像"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), image)
