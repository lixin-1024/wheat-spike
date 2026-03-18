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
    COLOR_OBB = (0, 255, 255)          # OBB框：青色
    COLOR_CONNECT = (255, 165, 0)      # 连接线：橙色
    COLOR_CENTER = (0, 0, 255)         # 中心点：红色
    COLOR_INTERSECTION = (255, 0, 255) # 交点：紫色
    COLOR_TEXT = (255, 255, 255)       # 文字：白色
    COLOR_LEFT = (255, 100, 100)       # 左侧小穗
    COLOR_RIGHT = (100, 100, 255)      # 右侧小穗

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

            if draw_centers:
                cx, cy = int(centers[i, 0]), int(centers[i, 1])
                cv2.circle(vis, (cx, cy), 5, self.COLOR_CENTER, -1)

            if draw_index:
                cx, cy = int(centers[i, 0]), int(centers[i, 1])
                cv2.putText(vis, str(i), (cx + 8, cy - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_TEXT, 2)

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

        # 2. 绘制连接线（中心点引线与主茎交点）
        intersections = skeleton['spikelet_intersections']
        anchors = skeleton.get('spikelet_anchor_points', skeleton['spikelet_centers'])
        sides = skeleton['spikelet_side']
        centers = skeleton['spikelet_centers']

        # 构建排序标签: original_index → rank (1-based)
        spikelet_order = skeleton['spikelet_order']
        order_labels = np.empty_like(spikelet_order)
        order_labels[spikelet_order] = np.arange(len(spikelet_order))

        for i in range(len(intersections)):
            px, py = int(intersections[i, 0]), int(intersections[i, 1])
            sx, sy = int(anchors[i, 0]), int(anchors[i, 1])
            cx, cy = int(centers[i, 0]), int(centers[i, 1])

            color = self.COLOR_LEFT if sides[i] < 0 else self.COLOR_RIGHT
            cv2.line(vis, (px, py), (sx, sy), color, 2)
            cv2.circle(vis, (px, py), 4, self.COLOR_INTERSECTION, -1)
            cv2.circle(vis, (sx, sy), 3, color, -1)
            cv2.circle(vis, (cx, cy), 4, self.COLOR_CENTER, -1)

            # 标注沿主茎排序序号
            cv2.putText(vis, str(order_labels[i] + 1), (cx + 8, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_TEXT, 2)

        return vis
    
    def draw_full_analysis(self, image: np.ndarray, detection: dict,
                           skeleton: dict, spikelet_pheno: dict,
                           ear_pheno: dict) -> np.ndarray:
        """绘制完整的分析图（检测+骨架+表型标注）"""
        vis = image.copy()

        # 1. 绘制OBB框
        xyxyxyxy = detection['xyxyxyxy']
        for i in range(len(xyxyxyxy)):
            corners = xyxyxyxy[i].astype(np.int32)
            cv2.drawContours(vis, [corners], 0, self.COLOR_OBB, 2)

        # 2. 绘制骨架
        stem_pts = skeleton['stem_points'].astype(np.int32)
        for j in range(len(stem_pts) - 1):
            cv2.line(vis, tuple(stem_pts[j]), tuple(stem_pts[j + 1]),
                     self.COLOR_STEM, 3)

        # 3. 绘制连接线和标注
        intersections = skeleton['spikelet_intersections']
        anchors = skeleton.get('spikelet_anchor_points', skeleton['spikelet_centers'])
        sides = skeleton['spikelet_side']
        centers = skeleton['spikelet_centers']
        lengths = spikelet_pheno['lengths']
        widths = spikelet_pheno['widths']

        # 构建排序标签
        spikelet_order = skeleton['spikelet_order']
        order_labels = np.empty_like(spikelet_order)
        order_labels[spikelet_order] = np.arange(len(spikelet_order))

        for i in range(len(intersections)):
            px, py = int(intersections[i, 0]), int(intersections[i, 1])
            sx, sy = int(anchors[i, 0]), int(anchors[i, 1])
            cx, cy = int(centers[i, 0]), int(centers[i, 1])

            color = self.COLOR_LEFT if sides[i] < 0 else self.COLOR_RIGHT
            cv2.line(vis, (px, py), (sx, sy), color, 2)
            cv2.circle(vis, (px, py), 4, self.COLOR_INTERSECTION, -1)
            cv2.circle(vis, (sx, sy), 3, color, -1)
            cv2.circle(vis, (cx, cy), 4, self.COLOR_CENTER, -1)

            # 标注尺寸和排序序号
            label = f"{order_labels[i] + 1}: L:{lengths[i]:.0f} W:{widths[i]:.0f}"
            cv2.putText(vis, label, (cx + 10, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLOR_TEXT, 1)

        # 4. 在图像左上角绘制穗级表型摘要
        self._draw_ear_summary(vis, ear_pheno)

        return vis

    def _draw_ear_summary(self, image: np.ndarray, ear_pheno: dict):
        """在图像上绘制穗级表型摘要信息框"""
        lines = [
            f"Spikelets: {ear_pheno['spikelet_count']}",
            f"Stem Arc Length: {ear_pheno['stem_length']:.1f} px",
            f"ECI: {ear_pheno['ECI']:.4f}",
            f"SDU: {ear_pheno['SDU']:.4f}",
            f"SCO: {ear_pheno['SCO']:.4f}",
            f"SHI: {ear_pheno['SHI']:.6f}",
            f"Left/Right: {ear_pheno['left_count']}/{ear_pheno['right_count']}",
        ]

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
