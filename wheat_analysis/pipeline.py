"""
分析管线：串联检测→骨架生成→表型提取→可视化的完整流程。
"""
import csv
import cv2
import numpy as np
from pathlib import Path

from .detector import SpikeletDetector
from .skeleton import SkeletonBuilder
from .phenotype import PhenotypeExtractor
from .visualizer import Visualizer


class WheatAnalysisPipeline:
    """小麦麦穗表型分析完整管线"""

    def __init__(self, model_path: str, imgsz: int = 1440, conf: float = 0.5):
        self.detector = SpikeletDetector(model_path, imgsz, conf)
        self.skeleton_builder = SkeletonBuilder()
        self.phenotype_extractor = PhenotypeExtractor()
        self.visualizer = Visualizer()

    def analyze_single(self, image_path: str, output_dir: str = None) -> dict:
        """
        对单张图片执行完整分析。

        Returns:
            dict: {
                'detection': dict,
                'skeleton': dict,
                'spikelet_pheno': dict,
                'ear_pheno': dict,
                'ear_vector': np.ndarray,
                'vis_image': np.ndarray,
            }
        """
        # 1. 检测
        detection = self.detector.detect(image_path)

        if detection['count'] < 2:
            return {
                'detection': detection,
                'skeleton': None,
                'spikelet_pheno': None,
                'ear_pheno': None,
                'ear_vector': None,
                'vis_image': None,
                'error': '检测到的小穗数量不足(< 2)，无法构建骨架'
            }

        # 2. 骨架生成
        skeleton = self.skeleton_builder.build(detection)

        # 3. 表型提取
        spikelet_pheno = self.phenotype_extractor.extract_spikelet_phenotypes(detection)
        ear_pheno = self.phenotype_extractor.extract_ear_phenotypes(detection, skeleton)

        # 4. 穗型向量
        ear_vector = self.phenotype_extractor.compute_ear_vector_full(detection, skeleton)

        # 5. 可视化
        image = cv2.imread(str(image_path))
        vis_image = self.visualizer.draw_full_analysis(
            image, detection, skeleton, spikelet_pheno, ear_pheno
        )

        # 6. 保存结果
        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            stem = Path(image_path).stem

            # 保存可视化图
            self.visualizer.save(vis_image, str(out / f"{stem}_analysis.jpg"))

            # 保存骨架图
            skeleton_vis = self.visualizer.draw_skeleton(image, detection, skeleton)
            self.visualizer.save(skeleton_vis, str(out / f"{stem}_skeleton.jpg"))

            # 保存检测图
            detect_vis = self.visualizer.draw_detection(image, detection, draw_index=False)
            self.visualizer.save(detect_vis, str(out / f"{stem}_detection.jpg"))

        result = {
            'detection': detection,
            'skeleton': skeleton,
            'spikelet_pheno': spikelet_pheno,
            'ear_pheno': ear_pheno,
            'ear_vector': ear_vector,
            'vis_image': vis_image,
        }

        return result

    def analyze_batch(self, image_dir: str, output_dir: str,
                      extensions: tuple = ('.jpg', '.jpeg', '.png')) -> list:
        """
        对目录中的所有图片执行批量分析。

        Returns:
            list[dict]: 每张图片的分析结果
        """
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        image_paths = sorted([
            p for p in image_dir.iterdir()
            if p.suffix.lower() in extensions
        ])

        results = []
        csv_rows = []

        for idx, img_path in enumerate(image_paths):
            print(f"[{idx + 1}/{len(image_paths)}] 正在分析: {img_path.name}")
            result = self.analyze_single(str(img_path), str(output_dir))

            if result.get('ear_pheno'):
                ear = result['ear_pheno']
                csv_rows.append({
                    'image': img_path.name,
                    'spikelet_count': ear['spikelet_count'],
                    'effective_stem_length': f"{ear['effective_stem_length']:.2f}",
                    'mean_length': f"{ear['mean_spikelet_length']:.2f}",
                    'mean_width': f"{ear['mean_spikelet_width']:.2f}",
                    'mean_aspect_ratio': f"{ear['mean_aspect_ratio']:.2f}",
                    'ECI': f"{ear['ECI']:.6f}",
                    'SDU': f"{ear['SDU']:.6f}",
                    'SCO': f"{ear['SCO']:.6f}",
                    'SHI': f"{ear['SHI']:.8f}",
                    'left_count': ear['left_count'],
                    'right_count': ear['right_count'],
                })

            results.append(result)

        # 保存汇总CSV
        if csv_rows:
            csv_path = output_dir / "phenotype_results.csv"
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
                writer.writeheader()
                writer.writerows(csv_rows)
            print(f"表型结果已保存: {csv_path}")

        return results
