"""
分析管线模块。

按照调用场景拆分为两类：
- SingleImagePipeline：单张图片分析
- BatchImagePipeline：批量图片分析与聚类

同时保留 WheatAnalysisPipeline 兼容旧调用方式。
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from .calibration import ScaleCalibrator
from .clustering import PhenotypeClusterAnalyzer
from .detector import SpikeletDetector
from .phenotype import PhenotypeExtractor
from .skeleton import SkeletonBuilder
from .visualizer import Visualizer


class _BasePipeline:
    """共享的分析基础能力。"""

    def __init__(
        self,
        model_path: str,
        imgsz: int = 1440,
        conf: float = 0.5,
        detector=None,
        skeleton_builder=None,
        phenotype_extractor=None,
        visualizer=None,
        calibrator=None,
    ):
        self.detector = detector or SpikeletDetector(model_path, imgsz, conf)
        self.skeleton_builder = skeleton_builder or SkeletonBuilder()
        self.phenotype_extractor = phenotype_extractor or PhenotypeExtractor()
        self.visualizer = visualizer or Visualizer()
        self.calibrator = calibrator or ScaleCalibrator()

    def _empty_result(self, detection, calibration, error: str):
        return {
            'detection': detection,
            'calibration': calibration,
            'skeleton': None,
            'spikelet_pheno': None,
            'spikelet_records': None,
            'ear_pheno': None,
            'feature_names': None,
            'feature_vector': None,
            'vis_image': None,
            'error': error,
        }

    def _save_visual_outputs(self, output_dir: str, image_path: str, image, detection, skeleton, vis_image):
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(image_path).stem

        if image is None:
            return
        self.visualizer.save(image, str(out_dir / f"{stem}_original.jpg"))
        if vis_image is not None:
            self.visualizer.save(vis_image, str(out_dir / f"{stem}_analysis.jpg"))
        if skeleton is not None:
            skeleton_vis = self.visualizer.draw_skeleton(image, detection, skeleton)
            self.visualizer.save(skeleton_vis, str(out_dir / f"{stem}_skeleton.jpg"))
        detect_vis = self.visualizer.draw_detection(image, detection, draw_index=False)
        self.visualizer.save(detect_vis, str(out_dir / f"{stem}_detection.jpg"))

    def _safe_float(self, value):
        if value is None:
            return None
        if isinstance(value, (np.floating, float, int, np.integer)):
            return float(value)
        return value


class SingleImagePipeline(_BasePipeline):
    """单张图片分析管线。"""

    def analyze(self, image_path: str, output_dir: str | None = None) -> dict:
        """
        单张图片完整分析流程：
        1. 读取原图并做尺度标定
        2. 执行小穗检测
        3. 基于小穗基节点拟合主茎骨架
        4. 提取小穗级与穗级表型
        5. 组装前端交互所需的小穗记录
        6. 生成综合可视化图像
        """
        image = cv2.imread(str(image_path))
        calibration = self.calibrator.calibrate(image) if image is not None else {
            'calibration_ok': False,
            'px_per_cm': None,
            'mm_per_px': None,
            'disc_center': None,
            'disc_radius_px': None,
            'disc_diameter_px': None,
            'color_card_bbox': None,
        }

        detection = self.detector.detect(image_path)
        if detection['count'] < 2:
            return self._empty_result(detection, calibration, '检测到的小穗数量不足(<2)，无法构建骨架')

        skeleton = self.skeleton_builder.build(detection)
        spikelet_pheno = self.phenotype_extractor.extract_spikelet_phenotypes(detection, skeleton, calibration)
        spikelet_records = self.phenotype_extractor.build_spikelet_records(detection, skeleton, spikelet_pheno)
        ear_pheno = self.phenotype_extractor.extract_ear_phenotypes(detection, skeleton, spikelet_pheno, calibration)
        feature_names, feature_vector = self.phenotype_extractor.build_feature_vector(ear_pheno)

        vis_image = self.visualizer.draw_full_analysis(
            image, detection, skeleton, spikelet_pheno, ear_pheno
        ) if image is not None else None

        if output_dir:
            self._save_visual_outputs(output_dir, image_path, image, detection, skeleton, vis_image)

        return {
            'detection': detection,
            'calibration': calibration,
            'skeleton': skeleton,
            'spikelet_pheno': spikelet_pheno,
            'spikelet_records': spikelet_records,
            'ear_pheno': ear_pheno,
            'feature_names': feature_names,
            'feature_vector': feature_vector,
            'vis_image': vis_image,
        }

    def analyze_single(self, image_path: str, output_dir: str | None = None) -> dict:
        """兼容旧方法名。"""
        return self.analyze(image_path, output_dir)


class BatchImagePipeline(_BasePipeline):
    """批量图片分析与聚类管线。"""

    def __init__(self, *args, cluster_analyzer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.cluster_analyzer = cluster_analyzer or PhenotypeClusterAnalyzer()
        self.single_pipeline = SingleImagePipeline(
            model_path='unused.pt',
            detector=self.detector,
            skeleton_builder=self.skeleton_builder,
            phenotype_extractor=self.phenotype_extractor,
            visualizer=self.visualizer,
            calibrator=self.calibrator,
        )

    def analyze_paths(self, image_paths: list[str], output_dir: str | None = None) -> dict:
        """
        对一批已知路径的图片进行分析，并在样本数足够时执行聚类。
        """
        results = []
        phenotype_rows = []
        feature_rows = []
        samples = []

        output_path = Path(output_dir) if output_dir else None
        if output_path is not None:
            output_path.mkdir(parents=True, exist_ok=True)

        for idx, image_path in enumerate(image_paths):
            print(f"[{idx + 1}/{len(image_paths)}] 正在分析: {Path(image_path).name}")
            result = self.single_pipeline.analyze(image_path, str(output_path) if output_path else None)
            results.append(result)

            if result.get('ear_pheno') is None:
                continue

            image_name = Path(image_path).name
            phenotype_rows.append(self._build_phenotype_row(image_name, result['ear_pheno']))
            feature_rows.append(self._build_feature_row(image_name, result['feature_names'], result['feature_vector']))
            samples.append({
                'image': image_name,
                'feature_names': result['feature_names'],
                'features': result['feature_vector'],
            })

        cluster_result = None
        if output_path is not None and phenotype_rows:
            self._write_dict_csv(output_path / "phenotype_results.csv", phenotype_rows)
            self._write_dict_csv(output_path / "feature_vectors.csv", feature_rows)
        if len(samples) >= 2 and output_path is not None:
            cluster_result = self.cluster_analyzer.cluster(samples, str(output_path))

        return {
            'results': results,
            'cluster': cluster_result,
        }

    def analyze_dir(self, image_dir: str, output_dir: str, extensions: tuple = ('.jpg', '.jpeg', '.png')) -> dict:
        image_dir = Path(image_dir)
        image_paths = sorted([
            str(path) for path in image_dir.iterdir()
            if path.suffix.lower() in extensions
        ])
        return self.analyze_paths(image_paths, output_dir)

    def analyze_batch(self, image_dir: str, output_dir: str, extensions: tuple = ('.jpg', '.jpeg', '.png')) -> list:
        """兼容旧接口：仅返回结果列表。"""
        return self.analyze_dir(image_dir, output_dir, extensions)['results']

    def _build_phenotype_row(self, image_name: str, ear: dict) -> dict:
        return {
            'image': image_name,
            'calibration_ok': ear['calibration_ok'],
            'px_per_cm': self._safe_float(ear['px_per_cm']),
            'mm_per_px': self._safe_float(ear['mm_per_px']),
            'spikelet_count': ear['spikelet_count'],
            'mean_spikelet_length': self._safe_float(ear['mean_spikelet_length']),
            'mean_spikelet_length_mm': self._safe_float(ear['mean_spikelet_length_mm']),
            'mean_spikelet_width': self._safe_float(ear['mean_spikelet_width']),
            'mean_spikelet_width_mm': self._safe_float(ear['mean_spikelet_width_mm']),
            'mean_aspect_ratio': self._safe_float(ear['mean_aspect_ratio']),
            'mean_attachment_angle': self._safe_float(ear['mean_attachment_angle']),
            'spike_length_px': self._safe_float(ear['spike_length_px']),
            'spike_length_cm': self._safe_float(ear['spike_length_cm']),
            'spikelet_density_px': self._safe_float(ear['spikelet_density_px']),
            'spikelet_density_per_cm': self._safe_float(ear['spikelet_density_per_cm']),
            'asymmetry_index': self._safe_float(ear['asymmetry_index']),
            'centroid_offset': self._safe_float(ear['centroid_offset']),
        }

    def _build_feature_row(self, image_name: str, feature_names: list[str], feature_vector: np.ndarray) -> dict:
        row = {'image': image_name}
        for name, value in zip(feature_names, feature_vector):
            row[name] = self._safe_float(value)
        return row

    def _write_dict_csv(self, csv_path: Path, rows: list[dict]):
        with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"结果已保存: {csv_path}")


class WheatAnalysisPipeline(SingleImagePipeline):
    """
    兼容旧类名。

    内部仍然提供 analyze_single / analyze_batch，避免桌面端旧代码直接失效。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._batch_pipeline = BatchImagePipeline(
            model_path='unused.pt',
            detector=self.detector,
            skeleton_builder=self.skeleton_builder,
            phenotype_extractor=self.phenotype_extractor,
            visualizer=self.visualizer,
            calibrator=self.calibrator,
        )

    def analyze_batch(self, image_dir: str, output_dir: str, extensions: tuple = ('.jpg', '.jpeg', '.png')) -> list:
        return self._batch_pipeline.analyze_batch(image_dir, output_dir, extensions)
