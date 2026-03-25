"""
分析管线：串联检测 -> 标定 -> 骨架生成 -> 表型提取 -> 聚类分析。
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from .calibration import ScaleCalibrator
from .clustering import SpikeClusterAnalyzer
from .detector import SpikeletDetector
from .phenotype import PhenotypeExtractor
from .skeleton import SkeletonBuilder
from .visualizer import Visualizer


class WheatAnalysisPipeline:
    """小麦麦穗表型分析完整管线"""

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

    def analyze_single(self, image_path: str, output_dir: str = None) -> dict:
        """
        对单张图片执行完整分析。
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
            return {
                'detection': detection,
                'calibration': calibration,
                'skeleton': None,
                'spikelet_pheno': None,
                'ear_pheno': None,
                'ear_vector': None,
                'feature_names': None,
                'feature_vector': None,
                'vis_image': None,
                'error': '检测到的小穗数量不足(< 2)，无法构建骨架',
            }

        skeleton = self.skeleton_builder.build(detection)
        spikelet_pheno = self.phenotype_extractor.extract_spikelet_phenotypes(
            detection, skeleton, calibration
        )
        ear_pheno = self.phenotype_extractor.extract_ear_phenotypes(
            detection, skeleton, spikelet_pheno, calibration
        )
        ear_vector = self.phenotype_extractor.compute_ear_vector_full(detection, skeleton)
        feature_names, feature_vector = self.phenotype_extractor.build_feature_vector(ear_pheno, ear_vector)

        vis_image = None
        if image is not None:
            vis_image = self.visualizer.draw_full_analysis(
                image, detection, skeleton, spikelet_pheno, ear_pheno
            )

        if output_dir:
            self._save_single_outputs(output_dir, image_path, image, detection, skeleton, vis_image)

        return {
            'detection': detection,
            'calibration': calibration,
            'skeleton': skeleton,
            'spikelet_pheno': spikelet_pheno,
            'ear_pheno': ear_pheno,
            'ear_vector': ear_vector,
            'feature_names': feature_names,
            'feature_vector': feature_vector,
            'vis_image': vis_image,
        }

    def analyze_batch(self, image_dir: str, output_dir: str, extensions: tuple = ('.jpg', '.jpeg', '.png')) -> list:
        """
        对目录中的所有图片执行批量分析，并导出增强版 CSV。
        """
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        image_paths = sorted([path for path in image_dir.iterdir() if path.suffix.lower() in extensions])
        results = []
        phenotype_rows = []
        feature_rows = []

        for idx, image_path in enumerate(image_paths):
            print(f"[{idx + 1}/{len(image_paths)}] 正在分析: {image_path.name}")
            result = self.analyze_single(str(image_path), str(output_dir))
            results.append(result)

            if result.get('ear_pheno'):
                ear = result['ear_pheno']
                phenotype_rows.append(self._build_phenotype_row(image_path.name, ear))
                feature_rows.append(self._build_feature_row(image_path.name, result['feature_names'], result['feature_vector']))

        if phenotype_rows:
            self._write_dict_csv(output_dir / "phenotype_results.csv", phenotype_rows)
        if feature_rows:
            self._write_dict_csv(output_dir / "feature_vectors.csv", feature_rows)

        return results

    def cluster_batch_results(self, results: list[dict], output_dir: str, n_clusters: int = 3) -> dict | None:
        """
        对批量分析结果执行聚类并导出结果。
        """
        samples = []
        for result in results:
            if result.get('ear_pheno') and result.get('feature_vector') is not None:
                image_name = Path(result['detection']['image_path']).name
                samples.append({
                    'image': image_name,
                    'feature_names': result['feature_names'],
                    'features': result['feature_vector'],
                })

        if not samples:
            return None

        analyzer = SpikeClusterAnalyzer(n_clusters=n_clusters)
        return analyzer.cluster(samples, output_dir)

    def analyze_and_cluster_batch(
        self,
        image_dir: str,
        output_dir: str,
        n_clusters: int = 3,
        extensions: tuple = ('.jpg', '.jpeg', '.png'),
    ) -> tuple[list[dict], dict | None]:
        """
        批量分析并直接执行聚类。
        """
        results = self.analyze_batch(image_dir=image_dir, output_dir=output_dir, extensions=extensions)
        cluster_result = self.cluster_batch_results(results, output_dir, n_clusters=n_clusters)
        return results, cluster_result

    def _save_single_outputs(self, output_dir: str, image_path: str, image, detection, skeleton, vis_image):
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(image_path).stem

        if vis_image is not None:
            self.visualizer.save(vis_image, str(out_dir / f"{stem}_analysis.jpg"))

        if image is not None and skeleton is not None:
            skeleton_vis = self.visualizer.draw_skeleton(image, detection, skeleton)
            self.visualizer.save(skeleton_vis, str(out_dir / f"{stem}_skeleton.jpg"))
            detect_vis = self.visualizer.draw_detection(image, detection, draw_index=False)
            self.visualizer.save(detect_vis, str(out_dir / f"{stem}_detection.jpg"))

    def _build_phenotype_row(self, image_name: str, ear: dict) -> dict:
        row = {
            'image': image_name,
            'calibration_ok': ear['calibration_ok'],
            'px_per_cm': self._safe_float(ear['px_per_cm']),
            'mm_per_px': self._safe_float(ear['mm_per_px']),
            'spikelet_count': ear['spikelet_count'],
            'spike_length_px': self._safe_float(ear['spike_length_px']),
            'spike_length_cm': self._safe_float(ear['spike_length_cm']),
            'mean_spikelet_length': self._safe_float(ear['mean_spikelet_length']),
            'mean_spikelet_length_mm': self._safe_float(ear['mean_spikelet_length_mm']),
            'mean_spikelet_width': self._safe_float(ear['mean_spikelet_width']),
            'mean_spikelet_width_mm': self._safe_float(ear['mean_spikelet_width_mm']),
            'mean_aspect_ratio': self._safe_float(ear['mean_aspect_ratio']),
            'mean_attachment_angle': self._safe_float(ear['mean_attachment_angle']),
            'spikelet_density_px': self._safe_float(ear['spikelet_density_px']),
            'spikelet_density_per_cm': self._safe_float(ear['spikelet_density_per_cm']),
            'asymmetry_index': self._safe_float(ear['asymmetry_index']),
            'centroid_offset': self._safe_float(ear['centroid_offset']),
            'mean_dist_to_stem': self._safe_float(ear['mean_dist_to_stem']),
            'mean_dist_to_stem_mm': self._safe_float(ear['mean_dist_to_stem_mm']),
            'ECI': self._safe_float(ear['ECI']),
            'SDU': self._safe_float(ear['SDU']),
            'SCO': self._safe_float(ear['SCO']),
            'SHI': self._safe_float(ear['SHI']),
            'left_count': ear['left_count'],
            'right_count': ear['right_count'],
        }
        return row

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

    def _safe_float(self, value):
        if value is None:
            return None
        if isinstance(value, (np.floating, float, int, np.integer)):
            return float(value)
        return value
