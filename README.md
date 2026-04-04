# 小麦麦穗表型分析系统（毕设）

## 项目简介

本项目是一个面向小麦麦穗表型分析的毕业设计系统，围绕“检测 - 骨架 - 表型 - 聚类 - 可视化”构建完整流程，支持：

- 基于 YOLO OBB 的小穗检测
- 基于检测结果的主茎/分枝骨架构建
- 小穗级与穗级表型参数提取
- 批量样本聚类分析与可视化
- Web 端交互式分析与结果导出

## 当前状态说明

- Web 端为当前唯一维护入口。
- 桌面端已废弃，不再作为运行与交付目标。

## 系统能力概览

### 1) 单图分析能力

输入一张麦穗图像，输出：

- 小穗检测结果（OBB）
- 骨架结构（主茎、分枝、顺序、侧别）
- 小穗级指标（长度、宽度、长宽比、着生角等）
- 穗级指标（穗长、小穗数、着生密度、对称度、重心偏移等）
- 可视化图（原图、检测图、骨架图、综合分析图）

### 2) 批量分析与聚类能力

输入多张图像，输出：

- 批量表型汇总 CSV
- 特征向量 CSV
- 聚类标签、聚类中心
- 聚类散点图、层次树状图
- Web 端筛选、排序、重聚类、对比分析与导出

### 3) 任务与服务能力

- 批量任务异步执行（后台线程）
- 状态轮询（queued/running/completed/error）
- 结果过期回收（TTL）
- 大文件上传保护（413）
- JSON 化错误返回与前端恢复机制

## 项目结构

```text
毕设/
├─ README.md
├─ .env
├─ apps/
│  └─ web/
│     ├─ app.py                 # Flask Web 服务
│     ├─ README.md              # Web 端详细文档
│     ├─ templates/             # HTML 模板
│     ├─ static/                # JS/CSS
│     ├─ uploads/               # 上传文件（运行时）
│     └─ results/               # 分析输出（运行时）
├─ wheat_analysis/
│  ├─ detector.py               # YOLO OBB 检测封装
│  ├─ skeleton.py               # 骨架构建
│  ├─ calibration.py            # 尺度标定
│  ├─ phenotype.py              # 表型提取
│  ├─ clustering.py             # 聚类分析
│  ├─ visualizer.py             # 可视化绘制
│  └─ pipeline.py               # 单图/批量分析管线
├─ training/
│  ├─ train.py                  # 模型训练脚本
│  ├─ data.yaml                 # 数据集配置
│  ├─ preprocess/               # 标注格式转换与预处理
│  └─ *.pt                      # 训练初始权重
├─ scripts/
│  ├─ run_clustering.py         # 批量分析+聚类命令行脚本
│  └─ plot_paper_results.py     # 训练曲线论文图生成
├─ tests/
│  ├─ test_analysis.py
│  ├─ test_pipeline.py
│  └─ test_performance.py
├─ data/
│  ├─ train/
│  ├─ val/
│  └─ test/
└─ runs/                        # 训练输出
```

## 技术栈

- Python 3.10+
- Flask / flask-cors / werkzeug
- ultralytics（YOLO）
- OpenCV / NumPy / SciPy / scikit-learn / matplotlib

## 环境准备

### 1) 创建并激活环境

建议使用你当前的 conda 环境（如 py312）。

### 2) 安装依赖

如果仓库尚未提供 requirements.txt，可手动安装：

```bash
pip install flask flask-cors werkzeug ultralytics opencv-python numpy scipy scikit-learn matplotlib pandas psutil
```

### 3) 设置 Python 路径

项目根目录的 .env 已包含：

```env
PYTHONPATH=.
```

建议从项目根目录执行所有命令。

## 快速开始（Web 端）

### 1) 启动服务

在项目根目录执行：

```bash
python apps/web/app.py
```

默认地址：

- http://127.0.0.1:5000

### 2) 可选环境变量

- MAX_CONTENT_LENGTH_MB（默认 128）
- BATCH_JOB_TTL_SECONDS（默认 7200）

PowerShell 示例：

```powershell
$env:MAX_CONTENT_LENGTH_MB="256"
$env:BATCH_JOB_TTL_SECONDS="10800"
python apps/web/app.py
```

### 3) Web API（核心）

- GET /
- POST /api/analyze-single
- POST /api/analyze-batch
- GET /api/batch-status/<run_id>
- GET /api/batch-result/<run_id>
- POST /api/recluster
- GET /api/export-cluster/<run_id>/<cluster_id>
- GET /results/<run_id>/<filename>

更多前端交互与接口说明见 Web 文档。

## 训练流程

### 1) 数据组织

training/data.yaml 当前配置：

- path: ./data
- train: train/images
- val: val/images
- nc: 1
- names: wheatear

对应目录：

- data/train/images
- data/train/labels
- data/val/images
- data/val/labels

### 2) 启动训练

```bash
cd training
python train.py
```

脚本默认以 YOLO OBB 方式训练，并输出到 runs/obb 下。

### 3) 训练结果可视化（论文图）

```bash
python scripts/plot_paper_results.py
```

默认读取：runs/obb/yolo11_1440_4/results.csv，生成高分辨率 PNG/PDF 图。

## 数据预处理脚本

training/preprocess 下包含：

- roxml_to_dota.py：roLabelImg XML 转 DOTA 标注
- dota_to_yolo_obb.py：DOTA 转 YOLO OBB
- 图片异常旋转.py：图像批量重编码修复脚本

注意：这些脚本中部分路径为本地硬编码，使用前请先按你的环境修改路径。

## 命令行批量聚类

可使用脚本直接进行批量分析与聚类：

```bash
python scripts/run_clustering.py --image-dir data/train/images --output-dir results/cluster_run --model-path runs/obb/yolo11_1440_4/weights/best.pt --clusters 3
```

## 测试与验证

### 1) 单元/集成测试

```bash
python -m unittest tests.test_pipeline
python -m unittest tests.test_analysis
```

### 2) 性能测试

```bash
python tests/test_performance.py
```

## 主要输出产物

Web 或脚本运行后常见输出：

- phenotype_results.csv
- feature_vectors.csv
- clustering_results.csv
- cluster_centers.csv
- cluster_embedding.png
- cluster_dendrogram.png
- *_original.jpg / *_analysis.jpg / *_skeleton.jpg / *_detection.jpg

## 常见问题

### 1) 启动 Web 失败

请依次检查：

- 当前目录是否为项目根目录
- runs/obb/yolo11_1440_4/weights/best.pt 是否存在
- 依赖是否安装完整
- PYTHONPATH 是否包含项目根目录

### 2) 上传时报 413

- 增大 MAX_CONTENT_LENGTH_MB
- 重启服务后重试

### 3) 批量任务结果丢失（410）

- 任务超过 BATCH_JOB_TTL_SECONDS 被清理
- 重新上传并发起批量任务

## 致谢与备注

本项目用于毕业设计研究与系统实现。当前交付聚焦 Web 端闭环，后续可进一步扩展为生产级任务队列、持久化任务状态与模型在线更新机制。
