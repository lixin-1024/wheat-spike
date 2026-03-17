import sys
import os
import cv2
import numpy as np
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                           QHBoxLayout, QPushButton, QLabel, QFileDialog,
                           QComboBox, QProgressBar, QTextEdit, QTabWidget,
                           QSplitter, QFrame, QGridLayout, QGroupBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QFont

from wheat_analysis.pipeline import WheatAnalysisPipeline


class AnalysisThread(QThread):
    """分析线程，避免阻塞UI"""
    progress = pyqtSignal(int)
    result_ready = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, pipeline, image_path):
        super().__init__()
        self.pipeline = pipeline
        self.image_path = image_path

    def run(self):
        try:
            result = self.pipeline.analyze_single(self.image_path)
            self.result_ready.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class WheatAnalysisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("小麦麦穗表型分析系统")
        self.setGeometry(100, 100, 1200, 800)

        # 初始化分析管线
        self.pipeline = WheatAnalysisPipeline(
            model_path='runs/obb/yolo11_1440_4/weights/best.pt',
            imgsz=1440,
            conf=0.5
        )

        self.current_image_path = None
        self.analysis_result = None

        self.init_ui()

    def init_ui(self):
        # 主窗口布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # 左侧控制面板
        control_panel = self.create_control_panel()
        splitter.addWidget(control_panel)

        # 右侧结果显示区域
        result_panel = self.create_result_panel()
        splitter.addWidget(result_panel)

        splitter.setSizes([300, 900])

    def create_control_panel(self):
        """创建左侧控制面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # 标题
        title = QLabel("小麦麦穗表型分析系统")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # 图片选择按钮
        self.load_btn = QPushButton("选择图片")
        self.load_btn.clicked.connect(self.load_image)
        layout.addWidget(self.load_btn)

        # 分析按钮
        self.analyze_btn = QPushButton("开始分析")
        self.analyze_btn.clicked.connect(self.start_analysis)
        self.analyze_btn.setEnabled(False)
        layout.addWidget(self.analyze_btn)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # 分析状态
        self.status_label = QLabel("就绪")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        # 模型信息
        model_info = QGroupBox("模型信息")
        model_layout = QVBoxLayout(model_info)
        model_info.setFont(QFont("Arial", 10))

        self.model_path_label = QLabel("模型路径: runs/obb/yolo11_1440_4/weights/best.pt")
        model_layout.addWidget(self.model_path_label)

        self.imgsz_label = QLabel("输入尺寸: 1440")
        model_layout.addWidget(self.imgsz_label)

        self.conf_label = QLabel("置信度阈值: 0.5")
        model_layout.addWidget(self.conf_label)

        layout.addWidget(model_info)

        # 分析参数
        param_group = QGroupBox("分析参数")
        param_layout = QVBoxLayout(param_group)
        param_group.setFont(QFont("Arial", 10))

        # 置信度阈值
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("置信度阈值:"))
        self.conf_input = QComboBox()
        self.conf_input.addItems(["0.3", "0.5", "0.7", "0.9"])
        self.conf_input.setCurrentText("0.5")
        self.conf_input.currentTextChanged.connect(self.update_conf)
        conf_layout.addWidget(self.conf_input)
        param_layout.addLayout(conf_layout)

        layout.addWidget(param_group)

        # 结果导出
        export_group = QGroupBox("结果导出")
        export_layout = QVBoxLayout(export_group)
        export_group.setFont(QFont("Arial", 10))

        self.save_btn = QPushButton("保存分析结果")
        self.save_btn.clicked.connect(self.save_results)
        self.save_btn.setEnabled(False)
        export_layout.addWidget(self.save_btn)

        layout.addWidget(export_group)

        # 添加弹性空间
        layout.addStretch()

        return panel

    def create_result_panel(self):
        """创建右侧结果显示区域"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # 标签页
        tab_widget = QTabWidget()
        layout.addWidget(tab_widget)

        # 原始图片标签页
        self.original_tab = QWidget()
        original_layout = QVBoxLayout(self.original_tab)
        self.original_label = QLabel("请选择图片")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setStyleSheet("border: 1px solid gray;")
        original_layout.addWidget(self.original_label)
        tab_widget.addTab(self.original_tab, "原始图片")

        # 分析结果标签页
        self.analysis_tab = QWidget()
        analysis_layout = QVBoxLayout(self.analysis_tab)
        self.analysis_label = QLabel("分析结果将显示在这里")
        self.analysis_label.setAlignment(Qt.AlignCenter)
        self.analysis_label.setStyleSheet("border: 1px solid gray;")
        analysis_layout.addWidget(self.analysis_label)
        tab_widget.addTab(self.analysis_tab, "分析结果")

        # 表型参数标签页
        self.phenotype_tab = QWidget()
        phenotype_layout = QVBoxLayout(self.phenotype_tab)
        self.phenotype_text = QTextEdit()
        self.phenotype_text.setReadOnly(True)
        phenotype_layout.addWidget(self.phenotype_text)
        tab_widget.addTab(self.phenotype_tab, "表型参数")

        return panel

    def load_image(self):
        """加载图片"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "Images (*.jpg *.jpeg *.png *.bmp)"
        )

        if file_path:
            self.current_image_path = file_path
            self.load_btn.setText(f"已选择: {Path(file_path).name}")
            self.analyze_btn.setEnabled(True)

            # 显示原始图片
            pixmap = QPixmap(file_path)
            pixmap = pixmap.scaled(600, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.original_label.setPixmap(pixmap)

            # 清空分析结果
            self.analysis_label.clear()
            self.phenotype_text.clear()
            self.save_btn.setEnabled(False)

    def update_conf(self, value):
        """更新置信度阈值"""
        try:
            conf = float(value)
            self.pipeline.detector.conf = conf
            self.conf_label.setText(f"置信度阈值: {conf}")
        except ValueError:
            pass

    def start_analysis(self):
        """开始分析"""
        if not self.current_image_path:
            return

        self.status_label.setText("分析中...")
        self.progress_bar.setValue(0)
        self.analyze_btn.setEnabled(False)

        # 创建分析线程
        self.analysis_thread = AnalysisThread(self.pipeline, self.current_image_path)
        self.analysis_thread.progress.connect(self.update_progress)
        self.analysis_thread.result_ready.connect(self.handle_result)
        self.analysis_thread.error.connect(self.handle_error)
        self.analysis_thread.start()

    def update_progress(self, value):
        """更新进度条"""
        self.progress_bar.setValue(value)

    def handle_result(self, result):
        """处理分析结果"""
        self.analysis_result = result

        # 显示分析结果图片
        if result.get('vis_image') is not None:
            height, width = result['vis_image'].shape[:2]
            q_img = QImage(result['vis_image'].data, width, height,
                         width * 3, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            pixmap = pixmap.scaled(600, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.analysis_label.setPixmap(pixmap)

        # 显示表型参数
        self.display_phenotype(result)

        # 更新状态
        self.status_label.setText("分析完成")
        self.progress_bar.setValue(100)
        self.analyze_btn.setEnabled(True)
        self.save_btn.setEnabled(True)

    def handle_error(self, error_msg):
        """处理错误"""
        self.status_label.setText(f"分析失败: {error_msg}")
        self.progress_bar.setValue(0)
        self.analyze_btn.setEnabled(True)

    def display_phenotype(self, result):
        """显示表型参数"""
        self.phenotype_text.clear()

        if not result.get('ear_pheno'):
            self.phenotype_text.append("分析失败，无法提取表型参数")
            return

        # 穗级表型参数
        self.phenotype_text.append("===== 穗级表型参数 =====")
        ear_pheno = result['ear_pheno']
        self.phenotype_text.append(f"小穗数量: {ear_pheno['spikelet_count']}")
        self.phenotype_text.append(f"主茎有效穗段长度: {ear_pheno['effective_stem_length']:.2f} px")
        self.phenotype_text.append(f"平均小穗长度: {ear_pheno['mean_spikelet_length']:.2f} px")
        self.phenotype_text.append(f"平均小穗宽度: {ear_pheno['mean_spikelet_width']:.2f} px")
        self.phenotype_text.append(f"平均长宽比: {ear_pheno['mean_aspect_ratio']:.2f}")
        self.phenotype_text.append(f"穗型紧密度指数(ECI): {ear_pheno['ECI']:.6f}")
        self.phenotype_text.append(f"小穗分布均匀度(SDU): {ear_pheno['SDU']:.6f}")
        self.phenotype_text.append(f"穗型重心偏移度(SCO): {ear_pheno['SCO']:.6f}")
        self.phenotype_text.append(f"穗型异质性指数(SHI): {ear_pheno['SHI']:.8f}")
        self.phenotype_text.append(f"左侧小穗数: {ear_pheno['left_count']}")
        self.phenotype_text.append(f"右侧小穗数: {ear_pheno['right_count']}")
        self.phenotype_text.append(f"平均到主茎距离: {ear_pheno['mean_dist_to_stem']:.2f} px")

        # 小穗级表型参数
        self.phenotype_text.append("\n===== 小穗级表型参数 =====")
        spikelet_pheno = result['spikelet_pheno']
        count = min(10, result['detection']['count'])  # 最多显示10个小穗

        for i in range(count):
            self.phenotype_text.append(
                f"小穗[{i}]: 长={spikelet_pheno['lengths'][i]:.1f}px, "
                f"宽={spikelet_pheno['widths'][i]:.1f}px, "
                f"长宽比={spikelet_pheno['aspect_ratios'][i]:.2f}"
            )

        # 穗型向量
        self.phenotype_text.append("\n===== 穗型向量 =====")
        ear_vector = result['ear_vector']
        self.phenotype_text.append(f"维度: {len(ear_vector)}")
        self.phenotype_text.append(f"向量值: {ear_vector}")

    def save_results(self):
        """保存分析结果"""
        if not self.analysis_result or not self.current_image_path:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存分析结果", "", "Images (*.jpg);;CSV Files (*.csv);;All Files (*)"
        )

        if file_path:
            if file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                # 保存分析图片
                cv2.imwrite(file_path, self.analysis_result['vis_image'])
            elif file_path.lower().endswith('.csv'):
                # 保存表型参数为CSV
                import csv
                ear_pheno = self.analysis_result['ear_pheno']
                spikelet_pheno = self.analysis_result['spikelet_pheno']

                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)

                    # 写入穗级参数
                    writer.writerow(['穗级表型参数'])
                    writer.writerow(['参数', '值'])
                    writer.writerow(['小穗数量', ear_pheno['spikelet_count']])
                    writer.writerow(['主茎长度', f"{ear_pheno['stem_length']:.2f}"])
                    writer.writerow(['有效穗段长度', f"{ear_pheno['effective_stem_length']:.2f}"])
                    writer.writerow(['平均小穗长度', f"{ear_pheno['mean_spikelet_length']:.2f}"])
                    writer.writerow(['平均小穗宽度', f"{ear_pheno['mean_spikelet_width']:.2f}"])
                    writer.writerow(['平均小穗面积', f"{ear_pheno['mean_spikelet_area']:.2f}"])
                    writer.writerow(['穗型紧密度指数(ECI)', f"{ear_pheno['ECI']:.6f}"])
                    writer.writerow(['小穗分布均匀度(SDU)', f"{ear_pheno['SDU']:.6f}"])
                    writer.writerow(['穗型重心偏移度(SCO)', f"{ear_pheno['SCO']:.6f}"])
                    writer.writerow(['穗型异质性指数(SHI)', f"{ear_pheno['SHI']:.8f}"])
                    writer.writerow(['左侧小穗数', ear_pheno['left_count']])
                    writer.writerow(['右侧小穗数', ear_pheno['right_count']])
                    writer.writerow(['平均到主茎距离', f"{ear_pheno['mean_dist_to_stem']:.2f}"])

                    # 写入小穗级参数
                    writer.writerow([])
                    writer.writerow(['小穗级表型参数'])
                    writer.writerow(['小穗序号', '长度(px)', '宽度(px)', '长宽比', '面积(px²)'])

                    for i in range(min(10, len(spikelet_pheno['lengths']))):
                        writer.writerow([
                            i,
                            f"{spikelet_pheno['lengths'][i]:.1f}",
                            f"{spikelet_pheno['widths'][i]:.1f}",
                            f"{spikelet_pheno['aspect_ratios'][i]:.2f}",
                            f"{spikelet_pheno['areas'][i]:.0f}"
                        ])

            self.status_label.setText(f"结果已保存到: {file_path}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = WheatAnalysisApp()
    window.show()
    sys.exit(app.exec_())