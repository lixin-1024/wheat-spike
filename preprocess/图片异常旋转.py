import cv2
import numpy as np
import os

# 配置路径
dir = r"D:\Desktop_1\毕设\wheatear_dataset\side"
# os.makedirs(output_dir, exist_ok=True)

# 遍历并处理所有jpg文件
for f in os.listdir(dir):
    if f.lower().endswith('.jpg'):
        # 读取图片（支持中文路径）
        img = cv2.imdecode(np.fromfile(os.path.join(dir, f), np.uint8), cv2.IMREAD_COLOR)
        # 保存图片（支持中文路径）
        cv2.imencode('.jpg', img)[1].tofile(os.path.join(dir, f))

print(f"\n所有JPG图片已处理完成")