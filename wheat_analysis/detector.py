"""
小穗检测模块：封装 YOLO OBB 模型推理，返回结构化检测结果。
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from pathlib import Path
from ultralytics import YOLO


class SpikeletDetector:
    """基于 YOLO11-OBB 的小穗检测器"""

    def __init__(self, model_path: str, imgsz: int = 1440, conf: float = 0.5):
        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.conf = conf

    def detect(self, image_path: str) -> dict:
        """
        对单张图片执行小穗检测。

        Returns:
            dict: {
                'image_path': str,
                'image_shape': (H, W),
                'count': int,                       # 小穗数量
                'xywhr': np.ndarray (N, 5),          # 中心x, 中心y, 宽, 高, 角度(rad)
                'xyxyxyxy': np.ndarray (N, 4, 2),    # 四个角点坐标
                'conf': np.ndarray (N,),             # 置信度
                'centers': np.ndarray (N, 2),        # 中心点 (x, y)
                'widths': np.ndarray (N,),           # 小穗宽度(短边)
                'heights': np.ndarray (N,),          # 小穗长度(长边)
                'angles': np.ndarray (N,),           # 旋转角度(弧度)
            }
        """
        results = self.model.predict(
            image_path, imgsz=self.imgsz, conf=self.conf, verbose=False
        )
        r = results[0]

        xywhr = r.obb.xywhr.cpu().numpy()
        xyxyxyxy = r.obb.xyxyxyxy.cpu().numpy()
        conf = r.obb.conf.cpu().numpy()

        # 确保 w < h (长边为 height)
        widths = np.minimum(xywhr[:, 2], xywhr[:, 3])
        heights = np.maximum(xywhr[:, 2], xywhr[:, 3])

        return {
            'image_path': str(image_path),
            'image_shape': r.orig_shape,        # (H, W)
            'count': len(xywhr),
            'xywhr': xywhr,
            'xyxyxyxy': xyxyxyxy,
            'conf': conf,
            'centers': xywhr[:, :2],            # (N, 2)
            'widths': widths,
            'heights': heights,
            'angles': xywhr[:, 4],              # 弧度
        }
