# Wheat Phenolab Web 端 README

## 1. 项目简介

本目录是小麦麦穗表型智能分析平台的 Web 端实现，技术栈为 Flask + 原生前端（HTML/CSS/JavaScript）。

系统面向小麦穗部表型分析场景，围绕单样本精细分析和多样本聚类分析两条主线，提供从上传、推理、可视化到导出的完整闭环。

页面入口为 /，后端接口统一在 /api 下。

## 2. 系统实现功能详解

本章节用于详尽说明系统已实现功能，可直接用于论文或答辩中的“系统实现”部分。

### 2.1 前端交互与任务入口功能

1. 分析模式切换
- 支持个体分析模式和聚类分析模式一键切换。
- 模式切换时自动清理上一次选择文件、结果面板与状态提示，避免跨模式数据污染。

2. 文件上传与管理
- 支持点击选择文件和拖拽上传两种方式。
- 单图模式自动限制为 1 张图片，批量模式支持多图。
- 提供文件缩略图、文件名、文件大小展示。
- 支持单个文件预览、移除文件、同步 input 文件列表。

3. 状态卡片与流程引导
- 根据当前状态展示 Idle、Ready、Running、Success、Error。
- 就绪状态可在状态卡片内直接触发“开始分析”。
- 批量分析时显示阶段文案、百分比进度、当前文件与已处理数量。

4. 会话级任务恢复
- 前端将批量任务 run_id、筛选条件、排序条件等信息写入 sessionStorage。
- 页面刷新后可自动恢复上次批量任务并继续轮询状态。
- 若任务过期或不存在，前端自动清理恢复信息并提示用户重新上传。

### 2.2 单图分析功能

1. 单图完整分析链路
- 上传单张图片后调用后端单图分析接口。
- 后端完成小穗检测、骨架构建、表型提取、特征向量计算并返回结构化结果。

2. 双视图可视化
- 小穗检测视图：在原图上叠加小穗多边形框。
- 骨架提取视图：显示主茎与分枝骨架线，支持动态高亮。

3. 小穗交互详情
- 鼠标悬停小穗区域显示 tooltip，包含：
  - 小穗序号
  - 长度
  - 宽度
  - 长宽比
  - 着生角度
  - 左右侧别

4. 画布缩放与漫游
- 支持按钮缩放、滚轮缩放、拖拽平移、重置缩放。
- 自动计算适配缩放基线，显示实时缩放百分比。

5. 骨架悬停动效增强
- 鼠标悬停骨架路径时，主茎与对应分枝联动高亮。
- 通过 canvas 光效轨迹增强路径可读性，提升答辩演示观感。

6. 单图指标看板
- 以卡片形式展示穗级核心指标，包括：
  - 平均小穗长度
  - 平均小穗宽度
  - 平均小穗长宽比
  - 平均着生角
  - 穗长
  - 小穗数
  - 着生密度
  - 对称度指数
  - 重心偏移度

### 2.3 批量分析与聚类功能

1. 异步批量任务机制
- 上传多图后创建独立 run_id 任务。
- 后端使用后台线程执行分析，前端按固定周期轮询任务状态。
- 阶段覆盖 queued、analyzing、clustering、completed、error。

2. 聚类结果概览
- 展示聚类轮廓系数（Silhouette）作为聚类质量参考。
- 渲染每个簇的卡片，展示样本数、代表图及关键指标值。

3. 聚类散点图
- 将降维后的样本坐标绘制为交互散点图。
- 每个点显示文件名，并按簇着色。
- 支持 hover 高亮与 click 选中，联动其他视图状态。

4. 层次聚类树状图
- 展示聚类层级结构与合并关系。
- 支持节点悬停提示、激活链路高亮。

5. 排序与筛选能力
- 支持按多种指标对簇卡片排序。
- 支持指标阈值筛选，并可仅显示命中簇。
- 指标分类包含小穗级特征与穗级特征两大类。

6. 动态重聚类
- 通过滑块调整聚类簇数，触发后端重聚类接口。
- 重聚类成功后前端自动刷新散点图、树状图、簇卡片与对比数据。

7. 聚类对比分析
- 支持将多个簇加入对比集合。
- 对比区展示：
  - 小穗级雷达图
  - 穗级指标雷达图
  - 多指标柱状对比图
- 支持一键清空对比与单簇移除。

8. 聚类数据导出
- 支持按簇导出汇总 CSV。
- 支持下载批量分析结果文件（表型汇总、特征向量等）。

### 2.4 后端分析服务功能

1. 模型与分析管线接入
- 固定加载 YOLO OBB 模型进行小穗检测。
- 复用项目中的 SingleImagePipeline 与 BatchImagePipeline 完成算法流程。

2. 结果组织与序列化
- 将检测结果、骨架信息、表型指标、特征向量统一组装为前端可消费 JSON。
- 对 numpy 类型进行 JSON 安全转换，避免序列化失败。
- 为前端生成规范化结果资源路径。

3. 批量任务管理
- 内存字典维护任务状态，线程锁保障并发安全。
- 支持任务过期回收，避免内存长期堆积。
- 提供标准错误码区分任务不存在、过期、未完成、失败等场景。

4. 重聚类服务
- 基于已完成批量结果中的特征样本，按新簇数进行再次聚类。
- 返回更新后的 cluster 对象，前端无须重复上传图片。

5. 导出服务
- 生成并返回指定簇的 CSV 汇总内容。
- 提供结果文件静态访问路由，支持前端直接展示分析图。

### 2.5 健壮性与异常处理功能

1. 上传安全校验
- 校验文件是否存在、文件名是否为空、扩展名是否在白名单内。
- 白名单格式：png、jpg、jpeg、bmp。

2. 大文件保护
- 通过 MAX_CONTENT_LENGTH 限制请求体大小。
- 对 413 场景返回 JSON 错误信息，避免前端解析 HTML 异常。

3. 运行错误兜底
- 单图、批量、重聚类、导出接口均有异常捕获与错误回传。
- 批量线程异常会落地到任务状态并在前端可见。

4. 跨域支持
- 启用 CORS，便于后续前后端分离部署。

## 3. 目录结构

```text
apps/web/
├─ app.py                 # Flask 服务入口
├─ templates/
│  └─ index.html          # 前端页面模板
├─ static/
│  ├─ app.js              # 前端交互逻辑
│  └─ styles.css          # 页面样式
├─ uploads/               # 上传文件目录（运行时生成）
└─ results/               # 分析结果目录（运行时生成）
```

## 4. 运行环境

建议环境：

- Python 3.10+
- Windows / Linux / macOS
- 已安装并可用的模型权重文件

后端依赖来自项目代码 import，通常包括：

- flask
- flask-cors
- werkzeug
- opencv-python
- numpy
- ultralytics
- scipy
- scikit-learn
- matplotlib

如果项目根目录已有统一依赖文件，建议优先使用统一安装方式。

## 5. 启动方式

### 5.1 推荐从项目根目录启动

```bash
python apps/web/app.py
```

默认监听地址：

- http://127.0.0.1:5000

### 5.2 可配置环境变量

- MAX_CONTENT_LENGTH_MB
  - 含义：单次请求最大上传体积（MB）
  - 默认：128
- BATCH_JOB_TTL_SECONDS
  - 含义：批量任务完成后在内存中保留状态与结果的时间（秒）
  - 默认：7200

PowerShell 示例：

```powershell
$env:MAX_CONTENT_LENGTH_MB="256"
$env:BATCH_JOB_TTL_SECONDS="10800"
python apps/web/app.py
```

## 6. 核心接口

### 6.1 页面与资源

- GET /
- GET /results/<run_id>/<filename>

### 6.2 单图分析

- POST /api/analyze-single
- Content-Type: multipart/form-data
- 字段：file

### 6.3 批量分析

1. 发起任务
- POST /api/analyze-batch
- Content-Type: multipart/form-data
- 字段：files

2. 查询状态
- GET /api/batch-status/<run_id>

3. 获取结果
- GET /api/batch-result/<run_id>

### 6.4 重聚类

- POST /api/recluster
- Content-Type: application/json
- 字段：run_id、n_clusters

### 6.5 导出簇结果

- GET /api/export-cluster/<run_id>/<cluster_id>

## 7. 结果文件说明

批量任务结果目录：

```text
apps/web/results/batch_xxxxxxxx/
```

常见产物：

- phenotype_results.csv：表型汇总
- feature_vectors.csv：特征向量
- clustering_results.csv：聚类标签
- cluster_centers.csv：簇中心
- cluster_dendrogram.png：树状图
- sample_original.jpg / sample_analysis.jpg / sample_skeleton.jpg / sample_detection.jpg

## 8. 常见问题

### 8.1 上传返回 413

原因：超过 MAX_CONTENT_LENGTH_MB。

处理：增大环境变量并重启服务。

### 8.2 批量结果返回 410

原因：任务超过 BATCH_JOB_TTL_SECONDS 已过期。

处理：重新上传并发起任务。

### 8.3 启动报模型或依赖错误

请检查：

- runs/obb/yolo11_1440_4/weights/best.pt 是否存在
- 是否在项目根目录运行，确保 wheat_analysis 可导入
- 当前 Python 环境依赖是否完整

## 9. 后续优化建议

1. 生产环境建议关闭 debug，并接入生产级 WSGI 服务与反向代理。
2. 批量任务当前基于内存字典，可升级为数据库或消息队列以支持高并发和持久化。
3. uploads 与 results 建议引入定时清理与对象存储策略。
