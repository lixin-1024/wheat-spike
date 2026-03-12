"""
单张图片分析测试脚本
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from wheat_analysis.pipeline import WheatAnalysisPipeline


if __name__ == '__main__':
    # 初始化分析管线
    pipeline = WheatAnalysisPipeline(
        model_path='runs/obb/yolo11_1440_4/weights/best.pt',
        imgsz=1440,
        conf=0.5
    )

    # 分析单张图片
    result = pipeline.analyze_single(
        image_path='data/train/images/001.jpg',
        output_dir='results/test_single'
    )

    # 打印分析结果
    if result.get('ear_pheno'):
        print("\n===== 穗级表型参数 =====")
        for k, v in result['ear_pheno'].items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

        print("\n===== 小穗级表型参数(前5个) =====")
        spk = result['spikelet_pheno']
        for i in range(min(5, result['detection']['count'])):
            print(f"  小穗[{i}]: 长={spk['lengths'][i]:.1f}px, "
                  f"宽={spk['widths'][i]:.1f}px, "
                  f"长宽比={spk['aspect_ratios'][i]:.2f}, "
                  f"面积={spk['areas'][i]:.0f}px²")

        print(f"\n===== 穗型向量 (维度: {len(result['ear_vector'])}) =====")
        print(f"  {result['ear_vector']}")

        print(f"\n可视化结果已保存到 results/test_single/")
    else:
        print("分析失败:", result.get('error', '未知错误'))
