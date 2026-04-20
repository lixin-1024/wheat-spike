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
from openpyxl import Workbook

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

    def _save_visual_outputs(self, output_dir: str, image_path: str, image, detection, skeleton, vis_image, corrected_image=None, calibration=None):
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(image_path).stem

        if image is None:
            return
        self.visualizer.save(image, str(out_dir / f"{stem}_original.jpg"))
        if corrected_image is not None:
            self.visualizer.save(corrected_image, str(out_dir / f"{stem}_corrected.jpg"))
        else:
            self.visualizer.save(image, str(out_dir / f"{stem}_corrected.jpg"))
        calibration_vis = self.calibrator.draw_calibration_overlay(image, calibration)
        if calibration_vis is not None:
            self.visualizer.save(calibration_vis, str(out_dir / f"{stem}_calibration.jpg"))
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

    def _safe_int(self, value):
        if value is None:
            return None
        if isinstance(value, (np.integer, int)):
            return int(value)
        if isinstance(value, (np.floating, float)):
            return int(round(float(value)))
        return value

    def _px_to_cm(self, value, calibration: dict | None):
        if value is None or calibration is None or not calibration.get('calibration_ok'):
            return None
        px_per_cm = calibration.get('px_per_cm')
        if px_per_cm in (None, 0):
            return None
        return float(value) / float(px_per_cm)

    def _point_export_fields(self, prefix: str, point, calibration: dict | None) -> dict:
        if point is None:
            return {
                f'{prefix}_x_px': None,
                f'{prefix}_y_px': None,
                f'{prefix}_x_cm': None,
                f'{prefix}_y_cm': None,
            }

        point = np.asarray(point, dtype=float)
        return {
            f'{prefix}_x_px': float(point[0]),
            f'{prefix}_y_px': float(point[1]),
            f'{prefix}_x_cm': self._px_to_cm(point[0], calibration),
            f'{prefix}_y_cm': self._px_to_cm(point[1], calibration),
        }

    def _vector_export_fields(self, skeleton: dict | None, calibration: dict | None) -> dict:
        if not skeleton:
            return {
                'abstract_vector_dx_px': None,
                'abstract_vector_dy_px': None,
                'abstract_vector_length_px': None,
                'abstract_vector_dx_cm': None,
                'abstract_vector_dy_cm': None,
                'abstract_vector_length_cm': None,
                'abstract_vector_angle_deg': None,
            }

        vector = np.asarray(skeleton.get('abstract_stem_vector'), dtype=float)
        return {
            'abstract_vector_dx_px': float(vector[0]),
            'abstract_vector_dy_px': float(vector[1]),
            'abstract_vector_length_px': self._safe_float(skeleton.get('abstract_stem_length')),
            'abstract_vector_dx_cm': self._px_to_cm(vector[0], calibration),
            'abstract_vector_dy_cm': self._px_to_cm(vector[1], calibration),
            'abstract_vector_length_cm': self._px_to_cm(skeleton.get('abstract_stem_length'), calibration),
            'abstract_vector_angle_deg': self._safe_float(skeleton.get('abstract_stem_angle_deg')),
        }


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
            'color_card_corners': None,
            'color_card_size': None,
            'color_calibration_ok': False,
            'color_card_orientation': None,
            'color_matrix': None,
            'color_patch_error': None,
            'color_patch_means': None,
        }
        corrected_image = self.calibrator.apply_color_correction(image, calibration) if image is not None else None

        detection = self.detector.detect(image_path)
        if detection['count'] < 2:
            return self._empty_result(detection, calibration, '检测到的小穗数量不足(<2)，无法构建骨架')

        skeleton = self.skeleton_builder.build(detection)
        spikelet_pheno = self.phenotype_extractor.extract_spikelet_phenotypes(detection, skeleton, calibration)
        spikelet_records = self.phenotype_extractor.build_spikelet_records(detection, skeleton, spikelet_pheno)
        ear_pheno = self.phenotype_extractor.extract_ear_phenotypes(
            detection,
            skeleton,
            spikelet_pheno,
            calibration,
            corrected_image if corrected_image is not None else image,
        )
        feature_names, feature_vector = self.phenotype_extractor.build_feature_vector(ear_pheno)

        vis_image = self.visualizer.draw_full_analysis(
            image, detection, skeleton, spikelet_pheno, ear_pheno
        ) if image is not None else None

        if output_dir:
            self._save_visual_outputs(
                output_dir,
                image_path,
                image,
                detection,
                skeleton,
                vis_image,
                corrected_image=corrected_image,
                calibration=calibration,
            )

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

    SUMMARY_LABELS = {
        'image': '图像名称',
        'calibration_ok': '标定是否成功',
        'px_per_cm': '每厘米像素数',
        'mm_per_px': '每像素毫米数',
        'spikelet_count': '小穗数量',
        'mean_spikelet_length': '平均小穗长度(px)',
        'mean_spikelet_length_mm': '平均小穗长度(mm)',
        'mean_spikelet_width': '平均小穗宽度(px)',
        'mean_spikelet_width_mm': '平均小穗宽度(mm)',
        'mean_aspect_ratio': '平均小穗长宽比',
        'mean_attachment_angle': '平均着生角(deg)',
        'spike_length_px': '穗长(px)',
        'spike_length_cm': '穗长(cm)',
        'spikelet_density_px': '小穗密度(每像素)',
        'spikelet_density_per_cm': '小穗密度(每厘米)',
        'symmetry_index': '对称度指数',
        'centroid_offset': '重心偏移度',
        'color_calibration_ok': '色彩校正是否成功',
        'mean_hue_deg': '平均色相(deg)',
        'mean_saturation': '平均饱和度',
        'std_hue': '色相标准差(deg)',
        'abstract_vector_dx_px': '抽象骨架向量X分量(px)',
        'abstract_vector_dy_px': '抽象骨架向量Y分量(px)',
        'abstract_vector_length_px': '抽象骨架长度(px)',
        'abstract_vector_dx_cm': '抽象骨架向量X分量(cm)',
        'abstract_vector_dy_cm': '抽象骨架向量Y分量(cm)',
        'abstract_vector_length_cm': '抽象骨架长度(cm)',
        'abstract_vector_angle_deg': '抽象骨架方向角(deg)',
        'abstract_start_x_px': '抽象骨架起点X坐标(px)',
        'abstract_start_y_px': '抽象骨架起点Y坐标(px)',
        'abstract_start_x_cm': '抽象骨架起点X坐标(cm)',
        'abstract_start_y_cm': '抽象骨架起点Y坐标(cm)',
        'abstract_end_x_px': '抽象骨架终点X坐标(px)',
        'abstract_end_y_px': '抽象骨架终点Y坐标(px)',
        'abstract_end_x_cm': '抽象骨架终点X坐标(cm)',
        'abstract_end_y_cm': '抽象骨架终点Y坐标(cm)',
    }

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

    def analyze_paths(self, image_paths: list[str], output_dir: str | None = None, progress_callback=None) -> dict:
        """
        对一批已知路径的图片进行分析，并在样本数足够时执行聚类。
        """
        results = []
        phenotype_rows = []
        feature_rows = []
        samples = []
        workbook_samples = []

        output_path = Path(output_dir) if output_dir else None
        if output_path is not None:
            output_path.mkdir(parents=True, exist_ok=True)

        total = len(image_paths)

        for idx, image_path in enumerate(image_paths):
            if progress_callback:
                progress_callback({
                    'stage': 'analyzing',
                    'current': idx,
                    'total': total,
                    'current_file': Path(image_path).name,
                })
            print(f"[{idx + 1}/{len(image_paths)}] 正在分析: {Path(image_path).name}")
            result = self.single_pipeline.analyze(image_path, str(output_path) if output_path else None)
            results.append(result)

            if result.get('ear_pheno') is None:
                continue

            image_name = Path(image_path).name
            phenotype_rows.append(self._build_phenotype_row(image_name, result['ear_pheno']))
            feature_rows.append(self._build_feature_row(image_name, result['feature_names'], result['feature_vector']))
            workbook_samples.append({
                'image': image_name,
                'result': result,
            })
            samples.append({
                'image': image_name,
                'feature_names': result['feature_names'],
                'features': result['feature_vector'],
                'ear_pheno': result['ear_pheno'],
                'image_path': image_path,
            })

        cluster_result = None
        if output_path is not None and phenotype_rows:
            self._write_dict_csv(output_path / "phenotype_results.csv", phenotype_rows)
            self._write_dict_csv(output_path / "feature_vectors.csv", feature_rows)
            self._write_batch_workbook(output_path / "phenotype_workbook.xlsx", workbook_samples)
        if len(samples) >= 2 and output_path is not None:
            if progress_callback:
                progress_callback({
                    'stage': 'clustering',
                    'current': total,
                    'total': total,
                    'current_file': None,
                })
            cluster_result = self.cluster_analyzer.cluster(samples, str(output_path))

        if progress_callback:
            progress_callback({
                'stage': 'completed',
                'current': total,
                'total': total,
                'current_file': None,
            })

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
            'mean_hue_deg': self._safe_float(ear.get('mean_hue_deg')),
            'mean_saturation': self._safe_float(ear.get('mean_saturation')),
            'std_hue': self._safe_float(ear.get('std_hue')),
            'spike_length_px': self._safe_float(ear['spike_length_px']),
            'spike_length_cm': self._safe_float(ear['spike_length_cm']),
            'spikelet_density_px': self._safe_float(ear['spikelet_density_px']),
            'spikelet_density_per_cm': self._safe_float(ear['spikelet_density_per_cm']),
            'symmetry_index': self._safe_float(ear['symmetry_index']),
            'centroid_offset': self._safe_float(ear['centroid_offset']),
            'color_calibration_ok': bool(ear.get('color_calibration_ok')),
        }

    def _build_feature_row(self, image_name: str, feature_names: list[str], feature_vector: np.ndarray) -> dict:
        row = {'image': image_name}
        for name, value in zip(feature_names, feature_vector):
            row[name] = self._safe_float(value)
        return row

    def _build_workbook_summary_row(self, image_name: str, result: dict) -> dict:
        ear = result['ear_pheno']
        calibration = result.get('calibration')
        skeleton = result.get('skeleton')
        row = self._build_phenotype_row(image_name, ear)
        row.update(self._vector_export_fields(skeleton, calibration))
        row.update(self._point_export_fields('abstract_start', skeleton.get('abstract_stem_start') if skeleton else None, calibration))
        row.update(self._point_export_fields('abstract_end', skeleton.get('abstract_stem_end') if skeleton else None, calibration))
        return row

    def _translate_summary_row(self, row: dict) -> dict:
        return {self.SUMMARY_LABELS.get(key, key): value for key, value in row.items()}

    def _write_dict_csv(self, csv_path: Path, rows: list[dict]):
        with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"结果已保存: {csv_path}")

    def _safe_sheet_title(self, base_name: str, used_titles: set[str], index: int) -> str:
        invalid_chars = set(r'[]:*?/\\')
        clean = ''.join('_' if ch in invalid_chars else ch for ch in base_name).strip() or f'sample_{index}'
        clean = clean[:31]
        candidate = clean
        suffix = 1
        while candidate in used_titles:
            suffix_text = f'_{suffix}'
            candidate = f'{clean[:31 - len(suffix_text)]}{suffix_text}'
            suffix += 1
        used_titles.add(candidate)
        return candidate

    def _append_table_rows(self, worksheet, rows: list[list], spacer: bool = True):
        for row in rows:
            worksheet.append(row)
        if spacer:
            worksheet.append([])

    def _write_batch_workbook(self, workbook_path: Path, samples: list[dict]):
        if not samples:
            return

        workbook = Workbook()
        summary_ws = workbook.active
        summary_ws.title = '总表'

        summary_rows = [self._translate_summary_row(self._build_workbook_summary_row(sample['image'], sample['result'])) for sample in samples]
        summary_ws.append(list(summary_rows[0].keys()))
        for row in summary_rows:
            summary_ws.append(list(row.values()))

        used_titles = {summary_ws.title}
        for index, sample in enumerate(samples, start=1):
            image_name = sample['image']
            result = sample['result']
            calibration = result.get('calibration')
            skeleton = result.get('skeleton')
            spikelet_pheno = result.get('spikelet_pheno')
            spikelet_records = result.get('spikelet_records') or []
            sheet = workbook.create_sheet(self._safe_sheet_title(Path(image_name).stem, used_titles, index))

            self._append_table_rows(sheet, [
                ['图片', image_name],
                ['标定是否成功', result['ear_pheno'].get('calibration_ok')],
                ['每厘米像素数', self._safe_float(result['ear_pheno'].get('px_per_cm'))],
                ['每像素毫米数', self._safe_float(result['ear_pheno'].get('mm_per_px'))],
            ])

            global_rows = [['全局表型特征', '值']]
            for key, value in self._build_phenotype_row(image_name, result['ear_pheno']).items():
                if key == 'image':
                    continue
                global_rows.append([self.SUMMARY_LABELS.get(key, key), value])
            self._append_table_rows(sheet, global_rows)

            local_rows = [[
                '小穗序号', '着生顺序', '侧别',
                '长度(px)', '宽度(px)', '长宽比', '着生角(deg)',
                '基点X坐标(cm)', '基点Y坐标(cm)', '顶点X坐标(cm)', '顶点Y坐标(cm)',
                '茎骨架对应点X坐标(cm)', '茎骨架对应点Y坐标(cm)',
            ]]
            stem_points = np.asarray(skeleton.get('spikelet_stem_points'), dtype=float) if skeleton else np.zeros((0, 2))
            lowest_points = np.asarray(skeleton.get('spikelet_lowest_points'), dtype=float) if skeleton else np.zeros((0, 2))
            highest_points = np.asarray(skeleton.get('spikelet_highest_points'), dtype=float) if skeleton else np.zeros((0, 2))
            for idx, record in enumerate(spikelet_records):
                local_rows.append([
                    idx + 1,
                    self._safe_int(record.get('order')),
                    record.get('side'),
                    self._safe_float(spikelet_pheno['lengths'][idx]) if spikelet_pheno else None,
                    self._safe_float(spikelet_pheno['widths'][idx]) if spikelet_pheno else None,
                    self._safe_float(spikelet_pheno['aspect_ratios'][idx]) if spikelet_pheno else None,
                    self._safe_float(spikelet_pheno['attachment_angles_deg'][idx]) if spikelet_pheno else None,
                    self._px_to_cm(lowest_points[idx][0], calibration) if len(lowest_points) > idx else None,
                    self._px_to_cm(lowest_points[idx][1], calibration) if len(lowest_points) > idx else None,
                    self._px_to_cm(highest_points[idx][0], calibration) if len(highest_points) > idx else None,
                    self._px_to_cm(highest_points[idx][1], calibration) if len(highest_points) > idx else None,
                    self._px_to_cm(stem_points[idx][0], calibration) if len(stem_points) > idx else None,
                    self._px_to_cm(stem_points[idx][1], calibration) if len(stem_points) > idx else None,
                ])
            self._append_table_rows(sheet, local_rows)

            keypoint_rows = [[
                '点类别', '名称', '所属小穗序号',
                'X坐标(px)', 'Y坐标(px)', 'X坐标(cm)', 'Y坐标(cm)',
            ]]
            for idx, record in enumerate(spikelet_records):
                order = self._safe_int(record.get('order'))
                for point_type, point_name, points in [
                    ('spikelet_base', '小穗基点', lowest_points),
                    ('spikelet_apex', '小穗顶点', highest_points),
                    ('stem_match', '茎骨架对应点', stem_points),
                ]:
                    if len(points) <= idx:
                        continue
                    point = points[idx]
                    keypoint_rows.append([
                        point_type,
                        f'{point_name}#{order}',
                        order,
                        self._safe_float(point[0]),
                        self._safe_float(point[1]),
                        self._px_to_cm(point[0], calibration),
                        self._px_to_cm(point[1], calibration),
                    ])
            for point_type, point_name, point in [
                ('stem_endpoint', '茎骨架起点', skeleton.get('abstract_stem_start') if skeleton else None),
                ('stem_endpoint', '茎骨架终点', skeleton.get('abstract_stem_end') if skeleton else None),
            ]:
                if point is None:
                    continue
                point = np.asarray(point, dtype=float)
                keypoint_rows.append([
                    point_type,
                    point_name,
                    None,
                    self._safe_float(point[0]),
                    self._safe_float(point[1]),
                    self._px_to_cm(point[0], calibration),
                    self._px_to_cm(point[1], calibration),
                ])
            self._append_table_rows(sheet, keypoint_rows)

            vector_rows = [['抽象骨架信息', '值']]
            vector_fields = self._vector_export_fields(skeleton, calibration)
            vector_fields.update(self._point_export_fields('abstract_start', skeleton.get('abstract_stem_start') if skeleton else None, calibration))
            vector_fields.update(self._point_export_fields('abstract_end', skeleton.get('abstract_stem_end') if skeleton else None, calibration))
            for key, value in vector_fields.items():
                vector_rows.append([self.SUMMARY_LABELS.get(key, key), value])
            self._append_table_rows(sheet, vector_rows, spacer=False)

        workbook.save(workbook_path)
        print(f"结果已保存: {workbook_path}")


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
