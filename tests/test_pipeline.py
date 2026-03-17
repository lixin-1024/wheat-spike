import unittest
import numpy as np
import cv2
from pathlib import Path
from wheat_analysis.pipeline import WheatAnalysisPipeline
from wheat_analysis.detector import SpikeletDetector
from wheat_analysis.skeleton import SkeletonBuilder
from wheat_analysis.phenotype import PhenotypeExtractor


class TestWheatAnalysisPipeline(unittest.TestCase):
    """测试分析管线"""

    def setUp(self):
        """测试前准备"""
        self.pipeline = WheatAnalysisPipeline(
            model_path='runs/obb/yolo11_1440_4/weights/best.pt',
            imgsz=1440,
            conf=0.5
        )
        self.test_image = 'data/train/images/001.jpg'

    def test_analyze_single(self):
        """测试单张图片分析"""
        result = self.pipeline.analyze_single(self.test_image)

        # 检查结果结构
        self.assertIn('detection', result)
        self.assertIn('skeleton', result)
        self.assertIn('spikelet_pheno', result)
        self.assertIn('ear_pheno', result)
        self.assertIn('ear_vector', result)
        self.assertIn('vis_image', result)

        # 检查检测结果
        detection = result['detection']
        self.assertGreater(detection['count'], 0)
        self.assertIn('xywhr', detection)
        self.assertIn('centers', detection)

        # 检查骨架
        skeleton = result['skeleton']
        self.assertIn('stem_points', skeleton)
        self.assertIn('spikelet_proj', skeleton)

        # 检查表型参数
        ear_pheno = result['ear_pheno']
        self.assertGreater(ear_pheno['spikelet_count'], 0)
        self.assertIn('ECI', ear_pheno)
        self.assertIn('SDU', ear_pheno)
        self.assertIn('SCO', ear_pheno)
        self.assertIn('SHI', ear_pheno)

    def test_batch_analysis(self):
        """测试批量分析"""
        result = self.pipeline.analyze_batch(
            image_dir='data/train/images',
            output_dir='test_results'
        )

        self.assertGreater(len(result), 0)
        self.assertTrue(Path('test_results/phenotype_results.csv').exists())


class TestSpikeletDetector(unittest.TestCase):
    """测试小穗检测器"""

    def setUp(self):
        self.detector = SpikeletDetector(
            model_path='runs/obb/yolo11_1440_4/weights/best.pt',
            imgsz=1440,
            conf=0.5
        )

    def test_detect(self):
        """测试检测功能"""
        result = self.detector.detect('data/train/images/001.jpg')

        self.assertGreater(result['count'], 0)
        self.assertIn('xywhr', result)
        self.assertIn('centers', result)
        self.assertIn('widths', result)
        self.assertIn('heights', result)


class TestSkeletonBuilder(unittest.TestCase):
    """测试骨架生成器"""

    def setUp(self):
        self.skeleton_builder = SkeletonBuilder()

    def test_build(self):
        """测试骨架构建"""
        # 创建模拟检测数据
        detection = {
            'centers': np.array([
                [100, 100],
                [150, 150],
                [200, 200],
                [250, 250]
            ])
        }

        skeleton = self.skeleton_builder.build(detection)

        self.assertIn('stem_points', skeleton)
        self.assertIn('spikelet_proj', skeleton)
        self.assertIn('spikelet_s', skeleton)
        self.assertIn('stem_length', skeleton)


class TestPhenotypeExtractor(unittest.TestCase):
    """测试表型提取器"""

    def setUp(self):
        self.extractor = PhenotypeExtractor()

    def test_extract_spikelet_phenotypes(self):
        """测试小穗级表型提取"""
        detection = {
            'heights': np.array([100, 120, 110]),
            'widths': np.array([50, 60, 55]),
            'angles': np.array([0.1, 0.2, 0.15])
        }

        pheno = self.extractor.extract_spikelet_phenotypes(detection)

        self.assertIn('lengths', pheno)
        self.assertIn('widths', pheno)
        self.assertIn('aspect_ratios', pheno)
        self.assertIn('areas', pheno)
        self.assertIn('angles_deg', pheno)

    def test_extract_ear_phenotypes(self):
        """测试穗级表型提取"""
        detection = {
            'count': 5,
            'heights': np.array([100, 120, 110, 130, 115]),
            'widths': np.array([50, 60, 55, 65, 58]),
            'centers': np.array([
                [100, 100],
                [150, 150],
                [200, 200],
                [250, 250],
                [300, 300]
            ])
        }

        skeleton = {
            'spikelet_s': np.array([0.1, 0.3, 0.5, 0.7, 0.9]),
            'spikelet_dist': np.array([10, 15, 12, 18, 14]),
            'spikelet_side': np.array([1, -1, 1, -1, 1])
        }

        pheno = self.extractor.extract_ear_phenotypes(detection, skeleton)

        self.assertIn('spikelet_count', pheno)
        self.assertIn('stem_length', pheno)
        self.assertIn('mean_spikelet_length', pheno)
        self.assertIn('ECI', pheno)
        self.assertIn('SDU', pheno)
        self.assertIn('SCO', pheno)
        self.assertIn('SHI', pheno)


if __name__ == '__main__':
    unittest.main()