import shutil
import unittest
from pathlib import Path

import cv2
import numpy as np

from wheat_analysis.calibration import ScaleCalibrator
from wheat_analysis.clustering import PhenotypeClusterAnalyzer
from wheat_analysis.phenotype import PhenotypeExtractor
from wheat_analysis.pipeline import BatchImagePipeline, SingleImagePipeline
from wheat_analysis.skeleton import SkeletonBuilder


def make_obb_corners(center, angle_rad, long_len=30.0, short_len=10.0):
    cx, cy = center
    direction = np.array([np.cos(angle_rad), np.sin(angle_rad)], dtype=float)
    normal = np.array([-direction[1], direction[0]], dtype=float)
    half_long = long_len / 2.0
    half_short = short_len / 2.0
    return np.array([
        [cx, cy] - half_long * direction - half_short * normal,
        [cx, cy] + half_long * direction - half_short * normal,
        [cx, cy] + half_long * direction + half_short * normal,
        [cx, cy] - half_long * direction + half_short * normal,
    ])


class DummyDetector:
    def __init__(self, detection):
        self.detection = detection

    def detect(self, image_path):
        result = dict(self.detection)
        result['image_path'] = str(image_path)
        return result


class TestScaleCalibrator(unittest.TestCase):
    def test_calibrate_with_white_disc(self):
        image = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.circle(image, (120, 160), 50, (255, 255, 255), -1)
        calibrator = ScaleCalibrator(disc_diameter_cm=5.0)

        result = calibrator.calibrate(image)

        self.assertTrue(result['calibration_ok'])
        self.assertAlmostEqual(result['disc_diameter_px'], 100.0, delta=8.0)
        self.assertAlmostEqual(result['px_per_cm'], 20.0, delta=1.6)


class TestPhenotypeExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = PhenotypeExtractor()

    def test_attachment_angle_uses_base_node_tangent(self):
        detection = {
            'count': 2,
            'heights': np.array([10, 10], dtype=float),
            'widths': np.array([2, 2], dtype=float),
            'angles': np.zeros(2, dtype=float),
        }
        skeleton = {
            'spikelet_highest_points': np.array([[0, 0], [10, 0]], dtype=float),
            'spikelet_lowest_points': np.array([[0, 10], [0, 0]], dtype=float),
            'spikelet_axis_dirs': np.array([[0, -1], [1, 0]], dtype=float),
            'spikelet_tangent': np.array([[0, 1], [0, 1]], dtype=float),
        }

        spikelet = self.extractor.extract_spikelet_phenotypes(detection, skeleton)

        self.assertAlmostEqual(spikelet['attachment_angles_deg'][0], 0.0, delta=1e-6)
        self.assertAlmostEqual(spikelet['attachment_angles_deg'][1], 90.0, delta=1e-6)

    def test_ear_phenotype_contains_only_target_metrics(self):
        detection = {
            'count': 4,
            'heights': np.array([10, 12, 16, 18], dtype=float),
            'widths': np.array([4, 4, 7, 8], dtype=float),
        }
        spikelet_pheno = {
            'lengths': np.array([10, 12, 16, 18], dtype=float),
            'widths': np.array([4, 4, 7, 8], dtype=float),
            'aspect_ratios': np.array([2.5, 3.0, 2.28, 2.25], dtype=float),
            'attachment_angles_deg': np.array([8, 10, 18, 22], dtype=float),
        }
        skeleton = {
            'stem_length': 120.0,
            'spikelet_side': np.array([-1, -1, 1, 1], dtype=float),
            'spikelet_s': np.array([0.2, 0.3, 0.7, 0.8], dtype=float),
        }

        ear = self.extractor.extract_ear_phenotypes(detection, skeleton, spikelet_pheno)

        expected_keys = {
            'spikelet_count',
            'mean_spikelet_length',
            'mean_spikelet_width',
            'mean_aspect_ratio',
            'mean_attachment_angle',
            'spike_length_px',
            'spikelet_density_px',
            'symmetry_index',
            'centroid_offset',
            'calibration_ok',
            'px_per_cm',
            'mm_per_px',
            'spike_length_cm',
            'spikelet_density_per_cm',
            'mean_spikelet_length_mm',
            'mean_spikelet_width_mm',
        }
        self.assertEqual(set(ear.keys()), expected_keys)


class TestSkeletonBuilder(unittest.TestCase):
    def test_build_outputs_base_tangent(self):
        centers = np.array([[100, 80], [102, 140], [105, 200], [108, 260]], dtype=float)
        xyxyxyxy = np.array([
            make_obb_corners(centers[0], np.deg2rad(85)),
            make_obb_corners(centers[1], np.deg2rad(87)),
            make_obb_corners(centers[2], np.deg2rad(89)),
            make_obb_corners(centers[3], np.deg2rad(92)),
        ])
        detection = {'centers': centers, 'xyxyxyxy': xyxyxyxy}

        skeleton = SkeletonBuilder().build(detection)

        self.assertIn('spikelet_tangent', skeleton)
        self.assertIn('spikelet_stem_points', skeleton)
        self.assertIn('abstract_stem_vector', skeleton)
        self.assertEqual(skeleton['spikelet_tangent'].shape, (4, 2))
        self.assertEqual(skeleton['spikelet_stem_points'].shape, (4, 2))
        self.assertEqual(len(skeleton['spikelet_s']), 4)
        self.assertAlmostEqual(
            skeleton['abstract_stem_length'],
            float(np.linalg.norm(skeleton['abstract_stem_end'] - skeleton['abstract_stem_start'])),
            delta=1e-6,
        )


class TestPipelinesAndClustering(unittest.TestCase):
    def setUp(self):
        centers = np.array([[100, 90], [108, 150], [115, 210], [122, 270]], dtype=float)
        xyxyxyxy = np.array([
            make_obb_corners(centers[0], np.deg2rad(82), long_len=34, short_len=10),
            make_obb_corners(centers[1], np.deg2rad(84), long_len=32, short_len=11),
            make_obb_corners(centers[2], np.deg2rad(87), long_len=30, short_len=9),
            make_obb_corners(centers[3], np.deg2rad(90), long_len=28, short_len=8),
        ])
        heights = np.array([34, 32, 30, 28], dtype=float)
        widths = np.array([10, 11, 9, 8], dtype=float)
        self.detection = {
            'count': 4,
            'xyxyxyxy': xyxyxyxy,
            'xywhr': np.column_stack([centers, widths, heights, np.deg2rad([82, 84, 87, 90])]),
            'conf': np.array([0.9, 0.91, 0.93, 0.95], dtype=float),
            'centers': centers,
            'widths': widths,
            'heights': heights,
            'angles': np.deg2rad([82, 84, 87, 90]),
            'image_shape': (400, 400),
        }

    def test_single_and_batch_pipeline(self):
        base_dir = Path("results") / "test_pipeline_v2"
        shutil.rmtree(base_dir, ignore_errors=True)
        image_dir = base_dir / "images"
        output_dir = base_dir / "outputs"
        image_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        image = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.circle(image, (70, 70), 45, (255, 255, 255), -1)
        for idx in range(4):
            cv2.imwrite(str(image_dir / f"sample_{idx}.png"), image)

        single = SingleImagePipeline(model_path='unused.pt', detector=DummyDetector(self.detection))
        single_result = single.analyze(str(image_dir / "sample_0.png"), str(output_dir))
        self.assertIn('spikelet_records', single_result)
        self.assertIn('attachment_angles_deg', single_result['spikelet_pheno'])
        self.assertIn('symmetry_index', single_result['ear_pheno'])

        batch = BatchImagePipeline(
            model_path='unused.pt',
            detector=DummyDetector(self.detection),
        )
        analysis = batch.analyze_dir(str(image_dir), str(output_dir))

        self.assertEqual(len(analysis['results']), 4)
        self.assertTrue((output_dir / "phenotype_results.csv").exists())
        self.assertTrue((output_dir / "feature_vectors.csv").exists())
        self.assertTrue((output_dir / "phenotype_workbook.xlsx").exists())
        self.assertIsNotNone(analysis['cluster'])
        self.assertTrue((output_dir / "cluster_embedding.png").exists())
        self.assertTrue((output_dir / "sample_similarity_heatmap.png").exists())
        self.assertTrue((output_dir / "cluster_dendrogram.png").exists())

    def test_cluster_analyzer(self):
        analyzer = PhenotypeClusterAnalyzer(n_clusters=2)
        output_dir = Path("results") / "test_cluster_v2"
        shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        samples = [
            {'image': 'a.png', 'feature_names': ['f1', 'f2', 'f3'], 'features': np.array([0.1, 0.2, 0.1])},
            {'image': 'b.png', 'feature_names': ['f1', 'f2', 'f3'], 'features': np.array([0.0, 0.1, 0.2])},
            {'image': 'c.png', 'feature_names': ['f1', 'f2', 'f3'], 'features': np.array([3.0, 3.2, 2.9])},
            {'image': 'd.png', 'feature_names': ['f1', 'f2', 'f3'], 'features': np.array([2.8, 3.1, 3.3])},
        ]

        result = analyzer.cluster(samples, str(output_dir))

        self.assertEqual(len(result['labels']), 4)
        self.assertTrue((output_dir / "clustering_results.csv").exists())
        self.assertTrue((output_dir / "cluster_centers.csv").exists())


if __name__ == '__main__':
    unittest.main()
