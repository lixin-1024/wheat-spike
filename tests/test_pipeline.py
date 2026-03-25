import unittest
from pathlib import Path
import shutil

import cv2
import numpy as np

from wheat_analysis.calibration import ScaleCalibrator
from wheat_analysis.clustering import SpikeClusterAnalyzer
from wheat_analysis.phenotype import PhenotypeExtractor
from wheat_analysis.pipeline import WheatAnalysisPipeline
from wheat_analysis.skeleton import SkeletonBuilder


def make_obb_corners(center, angle_rad, long_len=30.0, short_len=10.0):
    cx, cy = center
    direction = np.array([np.cos(angle_rad), np.sin(angle_rad)], dtype=float)
    normal = np.array([-direction[1], direction[0]], dtype=float)
    half_long = long_len / 2.0
    half_short = short_len / 2.0
    corners = np.array([
        [cx, cy] - half_long * direction - half_short * normal,
        [cx, cy] + half_long * direction - half_short * normal,
        [cx, cy] + half_long * direction + half_short * normal,
        [cx, cy] - half_long * direction + half_short * normal,
    ])
    return corners


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
        self.assertAlmostEqual(result['mm_per_px'], 0.5, delta=0.05)

    def test_calibrate_without_disc(self):
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        calibrator = ScaleCalibrator()
        result = calibrator.calibrate(image)

        self.assertFalse(result['calibration_ok'])
        self.assertIsNone(result['px_per_cm'])


class TestPhenotypeExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = PhenotypeExtractor()

    def test_attachment_angle_parallel_perpendicular_and_mirror(self):
        detection = {
            'count': 3,
            'heights': np.array([10, 10, 10], dtype=float),
            'widths': np.array([2, 2, 2], dtype=float),
            'angles': np.array([0.0, 0.0, 0.0], dtype=float),
        }
        skeleton = {
            'spikelet_highest_points': np.array([[0, 0], [0, 0], [0, 10]], dtype=float),
            'spikelet_lowest_points': np.array([[0, 10], [10, 0], [0, 0]], dtype=float),
            'spikelet_tangent': np.array([[0, 1], [0, 1], [0, -1]], dtype=float),
            'spikelet_dist': np.array([1, 1, 1], dtype=float),
        }

        spikelet = self.extractor.extract_spikelet_phenotypes(detection, skeleton=skeleton)

        self.assertAlmostEqual(spikelet['attachment_angles_deg'][0], 0.0, delta=1e-6)
        self.assertAlmostEqual(spikelet['attachment_angles_deg'][1], 90.0, delta=1e-6)
        self.assertAlmostEqual(spikelet['attachment_angles_deg'][2], 0.0, delta=1e-6)

    def test_asymmetry_index(self):
        detection = {
            'count': 4,
            'heights': np.array([10, 10, 20, 20], dtype=float),
            'widths': np.array([4, 4, 8, 8], dtype=float),
            'angles': np.zeros(4, dtype=float),
        }
        skeleton = {
            'spikelet_highest_points': np.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=float),
            'spikelet_lowest_points': np.array([[0, 10], [0, 10], [0, 20], [0, 20]], dtype=float),
            'spikelet_tangent': np.tile(np.array([[0, 1]], dtype=float), (4, 1)),
            'spikelet_s': np.array([0.2, 0.4, 0.6, 0.8], dtype=float),
            'spikelet_dist': np.array([3, 3, 6, 6], dtype=float),
            'spikelet_side': np.array([-1, -1, 1, 1], dtype=float),
            'stem_length': 100.0,
        }

        spikelet = self.extractor.extract_spikelet_phenotypes(detection, skeleton=skeleton)
        ear = self.extractor.extract_ear_phenotypes(detection, skeleton, spikelet_pheno=spikelet)

        self.assertGreater(ear['asymmetry_index'], 0.3)

    def test_physical_scale_fields(self):
        detection = {
            'count': 2,
            'heights': np.array([20, 30], dtype=float),
            'widths': np.array([5, 6], dtype=float),
            'angles': np.zeros(2, dtype=float),
        }
        skeleton = {
            'spikelet_highest_points': np.array([[0, 0], [0, 0]], dtype=float),
            'spikelet_lowest_points': np.array([[0, 20], [0, 30]], dtype=float),
            'spikelet_tangent': np.tile(np.array([[0, 1]], dtype=float), (2, 1)),
            'spikelet_s': np.array([0.3, 0.7], dtype=float),
            'spikelet_dist': np.array([10, 20], dtype=float),
            'spikelet_side': np.array([-1, 1], dtype=float),
            'stem_length': 200.0,
        }
        calibration = {'calibration_ok': True, 'px_per_cm': 100.0, 'mm_per_px': 0.1}

        spikelet = self.extractor.extract_spikelet_phenotypes(detection, skeleton=skeleton, calibration=calibration)
        ear = self.extractor.extract_ear_phenotypes(detection, skeleton, spikelet_pheno=spikelet, calibration=calibration)

        np.testing.assert_allclose(spikelet['lengths_mm'], np.array([2.0, 3.0]))
        self.assertAlmostEqual(ear['spike_length_cm'], 2.0, delta=1e-6)
        self.assertAlmostEqual(ear['spikelet_density_per_cm'], 1.0, delta=1e-6)


class TestSkeletonBuilder(unittest.TestCase):
    def test_build_outputs_tangent(self):
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
        self.assertEqual(skeleton['spikelet_tangent'].shape, (4, 2))
        self.assertEqual(skeleton['spikelet_anchor_points'].shape, (4, 2))


class TestPipelineAndClustering(unittest.TestCase):
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

    def test_analyze_single_and_batch_cluster(self):
        tmp_path = Path("results") / "test_pipeline_artifacts"
        image_dir = tmp_path / "images"
        output_dir = tmp_path / "results"
        shutil.rmtree(tmp_path, ignore_errors=True)
        image_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        image = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.circle(image, (70, 70), 45, (255, 255, 255), -1)
        for idx in range(3):
            img_path = image_dir / f"sample_{idx}.png"
            cv2.imwrite(str(img_path), image)

        pipeline = WheatAnalysisPipeline(
            model_path='unused.pt',
            detector=DummyDetector(self.detection),
        )

        result = pipeline.analyze_single(str(image_dir / "sample_0.png"), str(output_dir))
        self.assertIn('calibration', result)
        self.assertIn('feature_vector', result)
        self.assertIn('attachment_angles_deg', result['spikelet_pheno'])
        self.assertIn('asymmetry_index', result['ear_pheno'])

        results = pipeline.analyze_batch(str(image_dir), str(output_dir))
        self.assertTrue((output_dir / "phenotype_results.csv").exists())
        self.assertTrue((output_dir / "feature_vectors.csv").exists())
        self.assertEqual(len(results), 3)

        cluster_result = pipeline.cluster_batch_results(results, str(output_dir), n_clusters=2)
        self.assertIsNotNone(cluster_result)
        self.assertTrue((output_dir / "clustering_results.csv").exists())
        self.assertTrue((output_dir / "cluster_centers.csv").exists())
        self.assertTrue((output_dir / "clustering_pca.png").exists())

    def test_cluster_analyzer_direct(self):
        analyzer = SpikeClusterAnalyzer(n_clusters=2)
        output_dir = Path("results") / "test_cluster_artifacts"
        shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        samples = [
            {'image': 'a.png', 'feature_names': ['f1', 'f2'], 'features': np.array([0.0, 0.1])},
            {'image': 'b.png', 'feature_names': ['f1', 'f2'], 'features': np.array([0.2, 0.0])},
            {'image': 'c.png', 'feature_names': ['f1', 'f2'], 'features': np.array([4.0, 4.1])},
        ]
        result = analyzer.cluster(samples, str(output_dir))
        self.assertEqual(result['cluster_centers'].shape[1], 2)


if __name__ == '__main__':
    unittest.main()
