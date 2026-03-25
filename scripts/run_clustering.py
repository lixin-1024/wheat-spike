"""
批量分析并执行聚类的命令行脚本。

示例：
    conda run -n py312 python scripts/run_clustering.py ^
        --image-dir data/train/images ^
        --output-dir results/cluster_run ^
        --model-path runs/obb/yolo11_1440_4/weights/best.pt ^
        --clusters 3
"""
import argparse
from pathlib import Path

from wheat_analysis.pipeline import WheatAnalysisPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="小麦麦穗批量分析与聚类")
    parser.add_argument("--image-dir", required=True, help="待分析图片目录")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--model-path", required=True, help="YOLO OBB 模型路径")
    parser.add_argument("--imgsz", type=int, default=1440, help="模型输入尺寸")
    parser.add_argument("--conf", type=float, default=0.5, help="检测置信度阈值")
    parser.add_argument("--clusters", type=int, default=3, help="KMeans 聚类数")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = WheatAnalysisPipeline(
        model_path=args.model_path,
        imgsz=args.imgsz,
        conf=args.conf,
    )
    results, cluster_result = pipeline.analyze_and_cluster_batch(
        image_dir=args.image_dir,
        output_dir=str(output_dir),
        n_clusters=args.clusters,
    )

    valid = sum(1 for result in results if result.get('ear_pheno'))
    print(f"完成批量分析: {valid}/{len(results)} 张图片产生有效表型结果")
    if cluster_result is not None:
        print(f"聚类结果已输出到: {output_dir}")
    else:
        print("没有足够的有效样本用于聚类")


if __name__ == "__main__":
    main()
