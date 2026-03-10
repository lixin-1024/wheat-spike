import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 允许重复加载libiomp5md.dll

from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    model = YOLO('yolo11n-obb.pt')
    model.train(
        data='data.yaml',
        imgsz=1440,
        epochs=100,
        batch=4,
        device='0',
        workers=0,  # 不使用多线程加载数据，适合小数据集，避免过多内存占用
        name='yolo11_1440_4_SGD',

        patience=20,    # 早停机制，避免过拟合
        augment=True,   # 启用数据增强，提升模型泛化能力
        optimizer='SGD',
        lr0=0.001,      # 小数据集初始学习率降低至0.001，避免训练震荡
        cos_lr=True,   # 启用余弦退火学习率，后期缓慢降低，提升小数据集精度
    )

