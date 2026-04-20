// ==================== i18n 翻译模块 ====================
const I18N = {
    lang: localStorage.getItem('wheat.lang') || 'zh',
    translations: {
        zh: {
            // 页面标题
            pageTitle: 'Wheat Phenolab',
            heroTitle: '小麦麦穗表型智能分析平台',
            heroSubtitle: '—— 让每一株麦穗，都有可计算的答案',
            // 模式切换
            modeSingle: '单株分析',
            modeSingleSub: '精细画布 骨架提取',
            modeBatch: '聚类分析',
            modeBatchSub: '批量分析 聚类洞察',
            // 上传区
            dropzoneTitle: '拖拽或选择小麦图片开始分析',
            dropzoneHint: '批量模式可一次上传多张图片',
            dropzoneHintSingle: '单张模式下请选择 1 张图片',
            fileListEmpty: '暂无文件，拖拽或点击上方按钮选择图片',
            pickBtn: '选择图片',
            // 状态卡片
            statusIdle: '等待上传',
            statusReady: '就绪',
            statusRunning: '分析中',
            statusSuccess: '分析完成',
            statusError: '分析失败',
            runBtn: '开始分析',
            exportBtn: '导出表型分析结果',
            // 批量状态
            batchQueued: '等待进入分析队列',
            batchAnalyzing: '正在分析 {file}',
            batchClustering: '正在执行聚类与树状图计算',
            batchCompleted: '批量分析与聚类完成',
            batchError: '批量分析失败',
            batchProcessing: '批量任务处理中',
            batchProgressLabel: '批量分析中',
            // 单株分析面板
            panelSingle: '单株分析',
            viewerTabSpikelet: '小穗检测',
            viewerTabSkeleton: '骨架提取',
            resetZoom: '重置',
            // 指标名称
            metricSpikeletLength: '平均小穗长度',
            metricSpikeletWidth: '平均小穗宽度',
            metricSpikeletAspectRatio: '平均小穗长宽比',
            metricAttachmentAngle: '平均着生角',
            metricSpikeLength: '穗长',
            metricSpikeletCount: '小穗数',
            metricSpikeletDensity: '着生密度',
            metricSymmetry: '对称度指数',
            metricCentroidOffset: '重心偏移度',
            metricMeanHue: '平均色相',
            metricMeanSaturation: '平均饱和度',
            metricStdHue: '色相标准差',
            // 小穗侧别
            sideLeft: '左侧',
            sideRight: '右侧',
            // 骨架 tooltip
            skeletonStemStart: '茎骨架起点',
            skeletonStemEnd: '茎骨架终点',
            skeletonStemFit: '茎骨架拟合曲线',
            skeletonAbstract: '抽象骨架',
            skeletonAbstractTip: '沿小穗基点拟合得到的主茎骨架曲线',
            skeletonAbstractTip2: '去掉小穗末端毛刺后的抽象骨架',
            tooltipSpikeletTitle: '小穗 #{order}',
            tooltipLength: '长度',
            tooltipWidth: '宽度',
            tooltipAspectRatio: '长宽比',
            tooltipAttachmentAngle: '着生角度',
            tooltipSide: '侧别',
            tooltipCoordinate: '坐标(O-左上): {value}',
            tooltipVector: '方向向量(O-左上): {value}',
            tooltipMagnitude: '大小: {value}',
            tooltipDirectionAngle: '方向角(相对+x轴, 图像坐标系): {value}',
            tooltipStartPoint: '起点: {value}',
            tooltipEndPoint: '终点: {value}',
            tooltipSpikeletBasePointTitle: '小穗基点 #{order}',
            tooltipSpikeletTipPointTitle: '小穗顶点 #{order}',
            tooltipStemMatchedPointTitle: '茎骨架对应点 #{order}',
            tooltipSpikeletSkeletonTitle: '小穗骨架 #{order}',
            tooltipBasePoint: '基点: {value}',
            tooltipTipPoint: '顶点: {value}',
            // 聚类分析面板
            panelBatch: '聚类分析',
            clusterCountLabel: '聚类簇数',
            clusterScatter: '聚类散点图',
            clusterDendrogram: '层次聚类树状图',
            silhouetteScore: 'Silhouette',
            clusterFilter: '聚类结果筛选',
            sortBy: '排序',
            filterBy: '筛选',
            selectMetric: '请选择指标',
            selectGroup: '未选择分组',
            thresholdPlaceholder: '阈值，例如 5.0',
            showMatchedOnly: '仅显示命中类',
            // 簇卡片
            sampleCount: '样本数',
            addToCompare: '加入对比',
            cancelCompare: '取消对比',
            exportCluster: '导出',
            clickToViewCluster: '点击查看该簇详情',
            // 簇间比较
            clusterCompare: '簇间比较',
            clearCompare: '清空对比',
            spikeletFeatures: '小穗级特征',
            earFeatures: '穗级特征',
            selectedClusters: '已选择 {count} 类簇',
            compareNeedTwo: '至少选择 2 个类簇后展示多类对比图表。',
            radarChartSpikelet: '雷达图 • 小穗级特征',
            radarChartEar: '雷达图 • 穗级特征',
            barChart: '柱状图',
            clusterLabel: '第 {n} 类',
            sampleCountText: '{count} 个样本',
            clusterDetailTitleWithId: '第 {n} 类详情',
            clusterSamplesCount: '样本数：{count}',
            treeNodeCoverSamples: '树节点覆盖 {count} 个样本',
            extraSamples: ' +{count} 个样本',
            // 指标分组
            groupSpikelet: '小穗级特征',
            groupEar: '穗级特征',
            // 模态框
            previewTitle: '图片预览',
            clusterDetailTitle: '聚类详情',
            closeModal: '关闭',
            // 文件操作
            removeFile: '移除文件',
            previewBtn: '预览',
            // 错误信息
            errorAnalysisFailed: '分析失败',
            errorBatchStatusFailed: '批量状态获取失败',
            errorBatchFailed: '批量分析失败',
            errorBatchResultFailed: '批量结果获取失败',
            errorReclusterFailed: '重聚类失败',
            errorTaskUnavailable: '任务状态不可用',
            errorLastBatchFailed: '上次批量任务失败：请重新上传后重试',
            // 恢复提示
            sessionRestoreTip: '检测到上次批量任务，是否恢复？',
            sessionExpired: '批量结果已过期，请重新上传并发起分析',
            // 其他
            silelhoutteNA: 'N/A',
            currentSample: '当前样本',
            // setStatus 动态文本
            analyzingSingle: '单张图片分析中...',
            analyzingBatch: '批量分析与聚类中...',
            singleComplete: '单张分析完成',
            batchStarted: '批量任务已启动，正在分析中...',
            analysisFailed: '分析失败：{error}',
            batchFailed: '批量分析失败：{error}',
            analysisCompleteExport: '分析完成，点击导出表型分析结果',
            adjustingClusters: '正在调整聚类类别到 {count} 类...',
            clustersUpdated: '已更新为 {count} 簇聚类',
            reclusterFailed: '重聚类失败：{error}',
            selectOneFile: '已选择 1 张图片，点击开始分析',
            selectMultiFiles: '已选择 {count} 张图片，点击开始分析',
            exportFailedNoRunId: '导出失败：缺少批量任务 run_id，请重新执行一次批量分析。',
            restoringBatch: '已恢复上次批量任务，正在同步状态...',
            taskExpired: '上次任务已过期或不存在，请重新上传后开始分析。',
            lastBatchFailed: '上次批量任务失败：{error}',
            restoreFailed: '任务恢复失败：{error}',
        },
        en: {
            // 页面标题
            pageTitle: 'Wheat Phenolab',
            heroTitle: 'Wheat Spike Phenotype Analysis Platform',
            heroSubtitle: '—— Every wheat spike deserves a computable answer',
            // 模式切换
            modeSingle: 'Single Analysis',
            modeSingleSub: 'Fine Canvas · Skeleton Extraction',
            modeBatch: 'Cluster Analysis',
            modeBatchSub: 'Batch Processing · Cluster Insights',
            // 上传区
            dropzoneTitle: 'Drag & drop or select wheat images to start',
            dropzoneHint: 'batch mode supports multiple images',
            dropzoneHintSingle: 'Please select 1 image in single mode',
            fileListEmpty: 'No files yet. Drag & drop or click above to select images',
            pickBtn: 'Select Images',
            // 状态卡片
            statusIdle: 'Waiting for upload',
            statusReady: 'Ready',
            statusRunning: 'Analyzing',
            statusSuccess: 'Analysis Complete',
            statusError: 'Analysis Failed',
            runBtn: 'Start Analysis',
            exportBtn: 'Export Phenotype Results',
            // 批量状态
            batchQueued: 'Waiting in analysis queue',
            batchAnalyzing: 'Analyzing {file}',
            batchClustering: 'Performing clustering and dendrogram calculation',
            batchCompleted: 'Batch analysis and clustering complete',
            batchError: 'Batch analysis failed',
            batchProcessing: 'Batch task processing',
            batchProgressLabel: 'Batch analyzing',
            // 单株分析面板
            panelSingle: 'Single Analysis',
            viewerTabSpikelet: 'Spikelet Detection',
            viewerTabSkeleton: 'Skeleton Extraction',
            resetZoom: 'Reset',
            // 指标名称
            metricSpikeletLength: 'Avg. Spikelet Length',
            metricSpikeletWidth: 'Avg. Spikelet Width',
            metricSpikeletAspectRatio: 'Avg. Spikelet L/W Ratio',
            metricAttachmentAngle: 'Avg. Attachment Angle',
            metricSpikeLength: 'Spike Length',
            metricSpikeletCount: 'Spikelet Count',
            metricSpikeletDensity: 'Spikelet Density',
            metricSymmetry: 'Symmetry Index',
            metricCentroidOffset: 'Centroid Offset',
            metricMeanHue: 'Avg. Hue',
            metricMeanSaturation: 'Avg. Saturation',
            metricStdHue: 'Hue Std. Dev.',
            // 小穗侧别
            sideLeft: 'Left',
            sideRight: 'Right',
            // 骨架 tooltip
            skeletonStemStart: 'Stem Skeleton Start',
            skeletonStemEnd: 'Stem Skeleton End',
            skeletonStemFit: 'Stem Skeleton Fit Curve',
            skeletonAbstract: 'Abstract Skeleton',
            skeletonAbstractTip: 'Main stem skeleton curve fitted along spikelet base points',
            skeletonAbstractTip2: 'Abstract skeleton after removing spikelet tip burrs',
            tooltipSpikeletTitle: 'Spikelet #{order}',
            tooltipLength: 'Length',
            tooltipWidth: 'Width',
            tooltipAspectRatio: 'Aspect Ratio',
            tooltipAttachmentAngle: 'Attachment Angle',
            tooltipSide: 'Side',
            tooltipCoordinate: 'Coordinate (origin top-left): {value}',
            tooltipVector: 'Vector (origin top-left): {value}',
            tooltipMagnitude: 'Magnitude: {value}',
            tooltipDirectionAngle: 'Direction Angle (relative to +x, image coordinates): {value}',
            tooltipStartPoint: 'Start Point: {value}',
            tooltipEndPoint: 'End Point: {value}',
            tooltipSpikeletBasePointTitle: 'Spikelet Base Point #{order}',
            tooltipSpikeletTipPointTitle: 'Spikelet Tip Point #{order}',
            tooltipStemMatchedPointTitle: 'Stem Matched Point #{order}',
            tooltipSpikeletSkeletonTitle: 'Spikelet Skeleton #{order}',
            tooltipBasePoint: 'Base Point: {value}',
            tooltipTipPoint: 'Tip Point: {value}',
            // 聚类分析面板
            panelBatch: 'Cluster Analysis',
            clusterCountLabel: 'Cluster Count',
            clusterScatter: 'Cluster Scatter Plot',
            clusterDendrogram: 'Hierarchical Clustering Dendrogram',
            silhouetteScore: 'Silhouette',
            clusterFilter: 'Cluster Results Filter',
            sortBy: 'Sort',
            filterBy: 'Filter',
            selectMetric: 'Select metric',
            selectGroup: 'No group selected',
            thresholdPlaceholder: 'Threshold, e.g. 5.0',
            showMatchedOnly: 'Show matched only',
            // 簇卡片
            sampleCount: 'Samples',
            addToCompare: 'Add to Compare',
            cancelCompare: 'Remove',
            exportCluster: 'Export',
            clickToViewCluster: 'Click to view cluster details',
            // 簇间比较
            clusterCompare: 'Cluster Comparison',
            clearCompare: 'Clear All',
            spikeletFeatures: 'Spikelet Features',
            earFeatures: 'Spike Features',
            selectedClusters: '{count} clusters selected',
            compareNeedTwo: 'Select at least 2 clusters to show comparison charts.',
            radarChartSpikelet: 'Radar Chart · Spikelet Features',
            radarChartEar: 'Radar Chart · Spike Features',
            barChart: 'Bar Chart',
            clusterLabel: 'Cluster {n}',
            sampleCountText: '{count} samples',
            clusterDetailTitleWithId: 'Cluster {n} Details',
            clusterSamplesCount: 'Samples: {count}',
            treeNodeCoverSamples: 'Tree node covers {count} samples',
            extraSamples: ' +{count} samples',
            // 指标分组
            groupSpikelet: 'Spikelet Features',
            groupEar: 'Spike Features',
            // 模态框
            previewTitle: 'Image Preview',
            clusterDetailTitle: 'Cluster Details',
            closeModal: 'Close',
            // 文件操作
            removeFile: 'Remove file',
            previewBtn: 'Preview',
            // 错误信息
            errorAnalysisFailed: 'Analysis failed',
            errorBatchStatusFailed: 'Failed to get batch status',
            errorBatchFailed: 'Batch analysis failed',
            errorBatchResultFailed: 'Failed to get batch results',
            errorReclusterFailed: 'Reclustering failed',
            errorTaskUnavailable: 'Task status unavailable',
            errorLastBatchFailed: 'Previous batch task failed: please re-upload and retry',
            // 恢复提示
            sessionRestoreTip: 'Previous batch task detected. Restore?',
            sessionExpired: 'Batch results expired. Please re-upload and start analysis.',
            // 其他
            silelhoutteNA: 'N/A',
            currentSample: 'current sample',
            // setStatus 动态文本
            analyzingSingle: 'Analyzing single image...',
            analyzingBatch: 'Performing batch analysis and clustering...',
            singleComplete: 'Single image analysis complete',
            batchStarted: 'Batch task started, analyzing...',
            analysisFailed: 'Analysis failed: {error}',
            batchFailed: 'Batch analysis failed: {error}',
            analysisCompleteExport: 'Analysis complete. Click Export Phenotype Results',
            adjustingClusters: 'Adjusting to {count} clusters...',
            clustersUpdated: 'Updated to {count} clusters',
            reclusterFailed: 'Reclustering failed: {error}',
            selectOneFile: '1 image selected. Click Start Analysis',
            selectMultiFiles: '{count} images selected. Click Start Analysis',
            exportFailedNoRunId: 'Export failed: No batch task run_id. Please run batch analysis again.',
            restoringBatch: 'Restoring last batch task, syncing status...',
            taskExpired: 'Previous task expired or not found. Please re-upload and start.',
            lastBatchFailed: 'Previous batch task failed: {error}',
            restoreFailed: 'Task restoration failed: {error}',
        }
    },
    t(key, params = {}) {
        let text = this.translations[this.lang][key] || key;
        Object.keys(params).forEach(k => {
            text = text.replace(`{${k}}`, params[k]);
        });
        return text;
    },
    setLang(lang) {
        this.lang = lang;
        localStorage.setItem('wheat.lang', lang);
        document.documentElement.lang = lang;
        this.updatePage();
        this.updateLangSwitcherUI();
    },
    toggleLang() {
        this.setLang(this.lang === 'zh' ? 'en' : 'zh');
    },
    updateLangSwitcherUI() {
        const toggle = document.getElementById('langToggle');
        const labels = document.querySelectorAll('.lang-toggle__label');
        if (toggle && labels.length) {
            const isZh = this.lang === 'zh';
            toggle.checked = !isZh;
            labels[0].classList.toggle('active', isZh);
            labels[1].classList.toggle('active', !isZh);
        }
    },
    updatePage() {
        // 更新所有带 data-i18n 属性的元素
        document.querySelectorAll('[data-i18n]').forEach(el => {
            const key = el.getAttribute('data-i18n');
            if (el.tagName === 'INPUT' && el.placeholder) {
                el.placeholder = this.t(key);
            } else {
                el.textContent = this.t(key);
            }
        });
        // 更新 title
        document.title = this.t('pageTitle');
        // 触发自定义事件，让业务逻辑可以响应语言切换
        window.dispatchEvent(new CustomEvent('langChange', { detail: { lang: this.lang } }));
    },
    init() {
        document.documentElement.lang = this.lang;
        // 绑定语言切换器事件
        const langToggle = document.getElementById('langToggle');
        if (langToggle) {
            langToggle.addEventListener('change', () => this.toggleLang());
        }
        this.updateLangSwitcherUI();
    }
};

const state = {
    mode: 'single',
    files: [],
    previewUrls: [],
    singleResult: null,
    batchResult: null,
    batchRunId: null,
    batchStatusPoller: null,
    statusText: I18N.t('statusIdle'),
    statusType: 'idle',
    isAnalyzing: false,
    currentView: 'spikelet',
    batchSortMetric: 'mean_spike_length_cm',
    batchFilterMetric: 'mean_spike_length_cm',
    batchFilterValue: '',
    batchHideUnmatched: false,
    batchProgress: null,
    comparisonClusterIds: [],
    hoveredSampleName: null,
    selectedSampleName: null,
    hoveredClusterId: null,
    selectedClusterId: null,
    hoveredDendrogramNodeId: null,
    scale: 1,
    fitScale: null,
    translateX: 0,
    translateY: 0,
    zoomNeedsSync: true,
    dragging: false,
    dragStartX: 0,
    dragStartY: 0,
    skeletonOverlay: null,
    skeletonFxPointer: null,
    skeletonFxTargets: [],
    skeletonFxAnimationFrame: null,
    clusterMetricOptions: [],
    batchStatusSignature: null,
    batchRenderToken: 0,
    clusterLabelMap: {},
    clusterResultMap: {},
};

const refs = {};
let panelTransitionToken = 0;
const BATCH_SESSION_KEY = 'wheat.batch.session.v1';

document.addEventListener('DOMContentLoaded', () => {
    I18N.init();
    bindRefs();
    bindEvents();
    renderMode({ instant: true });
    renderFiles();
    void restoreBatchSession();
});

function bindRefs() {
    refs.modeButtons = document.querySelectorAll('.mode-switch__btn');
    refs.dropzone = document.getElementById('dropzone');
    refs.fileInput = document.getElementById('fileInput');
    refs.pickBtn = document.getElementById('pickBtn');
    refs.dropHint = document.getElementById('dropHint');
    refs.fileList = document.getElementById('fileList');
    refs.statusCard = document.getElementById('statusCard');
    refs.workspace = document.querySelector('.workspace');
    refs.singlePanel = document.getElementById('singlePanel');
    refs.batchPanel = document.getElementById('batchPanel');
    refs.viewerTabs = document.querySelectorAll('.viewer-tabs__btn');
    refs.viewer = document.getElementById('viewer');
    refs.viewerStage = document.getElementById('viewerStage');
    refs.viewerImage = document.getElementById('viewerImage');
    refs.skeletonLayer = document.getElementById('skeletonLayer');
    refs.skeletonSvg = document.getElementById('skeletonSvg');
    refs.skeletonFxCanvas = document.getElementById('skeletonFxCanvas');
    refs.overlaySvg = document.getElementById('overlaySvg');
    refs.tooltip = document.getElementById('tooltip');
    refs.singleMetrics = document.getElementById('singleMetrics');
    refs.zoomInBtn = document.getElementById('zoomInBtn');
    refs.zoomOutBtn = document.getElementById('zoomOutBtn');
    refs.resetZoomBtn = document.getElementById('resetZoomBtn');
    refs.zoomBadge = document.getElementById('zoomBadge');
    refs.clusterMap = document.getElementById('clusterMap');
    refs.clusterDendrogram = document.getElementById('clusterDendrogram');
    refs.clusterCards = document.getElementById('clusterCards');
    refs.clusterScore = document.getElementById('clusterScore');
    refs.clusterCountControl = document.getElementById('clusterCountControl');
    refs.clusterCountInput = document.getElementById('clusterCountInput');
    refs.clusterCountValue = document.getElementById('clusterCountValue');
    refs.clusterSortTrigger = document.getElementById('clusterSortTrigger');
    refs.clusterSortMenu = document.getElementById('clusterSortMenu');
    refs.clusterFilterTrigger = document.getElementById('clusterFilterTrigger');
    refs.clusterFilterMenu = document.getElementById('clusterFilterMenu');
    refs.clusterFilterValue = document.getElementById('clusterFilterValue');
    refs.clusterHideUnmatched = document.getElementById('clusterHideUnmatched');
    refs.previewModal = document.getElementById('previewModal');
    refs.previewBackdrop = document.getElementById('previewBackdrop');
    refs.previewClose = document.getElementById('previewClose');
    refs.previewImage = document.getElementById('previewImage');
    refs.previewTitle = document.getElementById('previewTitle');
    refs.clusterModal = document.getElementById('clusterModal');
    refs.clusterModalBackdrop = document.getElementById('clusterModalBackdrop');
    refs.clusterModalClose = document.getElementById('clusterModalClose');
    refs.clusterModalTitle = document.getElementById('clusterModalTitle');
    refs.clusterModalSummary = document.getElementById('clusterModalSummary');
    refs.clusterModalGrid = document.getElementById('clusterModalGrid');
    refs.clusterHoverCard = document.getElementById('clusterHoverCard');
    refs.clusterCompare = document.getElementById('clusterCompare');
    refs.clusterCompareSummary = document.getElementById('clusterCompareSummary');
    refs.clusterCompareChart = document.getElementById('clusterCompareChart');
    refs.clusterCompareGallery = document.getElementById('clusterCompareGallery');
}

function bindEvents() {
    refs.modeButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            if (state.mode === btn.dataset.mode) {
                return;
            }
            state.mode = btn.dataset.mode;
            state.currentView = 'spikelet';
            clearPreviewUrls();
            state.files = [];
            state.singleResult = null;
            state.batchResult = null;
            refs.fileInput.value = '';
            closePreview();
            resetPanels();
            renderMode();
            renderFiles();
        });
    });

    refs.pickBtn.addEventListener('click', () => refs.fileInput.click());
    refs.fileInput.addEventListener('change', (event) => {
        let files = Array.from(event.target.files || []).filter(file => file.type.startsWith('image/'));
        if (state.mode === 'single' && files.length > 1) {
            files = files.slice(0, 1);
        }
        setFiles(files);
        renderFiles();
    });

    refs.statusCard.addEventListener('click', (event) => {
        const runBtn = event.target.closest('#runBtn');
        if (runBtn && !runBtn.disabled) {
            runAnalysis();
            return;
        }
        const exportBtn = event.target.closest('#exportResultBtn');
        if (exportBtn && !exportBtn.disabled) {
            exportPhenotypeWorkbook();
        }
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        refs.dropzone.addEventListener(eventName, (event) => {
            event.preventDefault();
            refs.dropzone.classList.add('dragover');
        });
    });
    ['dragleave', 'drop'].forEach(eventName => {
        refs.dropzone.addEventListener(eventName, (event) => {
            event.preventDefault();
            refs.dropzone.classList.remove('dragover');
        });
    });
    refs.dropzone.addEventListener('drop', (event) => {
        let files = Array.from(event.dataTransfer.files || []).filter(file => file.type.startsWith('image/'));
        if (state.mode === 'single') {
            files = files.slice(0, 1);
        }
        setFiles(files);
        renderFiles();
    });

    refs.fileList.addEventListener('click', (event) => {
        const actionTarget = event.target.closest('[data-action]');
        if (!actionTarget) {
            return;
        }

        const index = Number.parseInt(actionTarget.dataset.index, 10);
        if (!Number.isInteger(index) || index < 0 || index >= state.files.length) {
            return;
        }

        if (actionTarget.dataset.action === 'remove') {
            removeFileAt(index);
        }
        if (actionTarget.dataset.action === 'preview') {
            openPreview(index);
        }
    });

    refs.previewClose.addEventListener('click', closePreview);
    refs.previewBackdrop.addEventListener('click', closePreview);
    refs.clusterModalClose.addEventListener('click', closeClusterModal);
    refs.clusterModalBackdrop.addEventListener('click', closeClusterModal);
    window.addEventListener('keydown', (event) => {
        if (event.key === 'Escape' && !refs.previewModal.classList.contains('hidden')) {
            closePreview();
        }
        if (event.key === 'Escape' && !refs.clusterModal.classList.contains('hidden')) {
            closeClusterModal();
        }
    });

    refs.viewerTabs.forEach(btn => {
        btn.addEventListener('click', () => {
            state.currentView = btn.dataset.view;
            renderViewerTabs();
            renderSingleView();
        });
    });

    refs.zoomInBtn.addEventListener('click', () => stepZoom(1.15, getViewerCenterAnchor()));
    refs.zoomOutBtn.addEventListener('click', () => stepZoom(1 / 1.15, getViewerCenterAnchor()));
    refs.resetZoomBtn.addEventListener('click', resetZoom);
    refs.viewer.addEventListener('wheel', handleWheelZoom, { passive: false });
    refs.viewer.addEventListener('pointerdown', handlePointerDown);
    refs.viewer.addEventListener('pointermove', updateSkeletonHoverFx);
    refs.viewer.addEventListener('pointerleave', clearSkeletonHoverFx);
    window.addEventListener('pointermove', handlePointerMove);
    window.addEventListener('pointerup', handlePointerUp);

    refs.clusterCountInput.addEventListener('input', handleClusterCountInput);
    refs.clusterCountInput.addEventListener('change', handleClusterCountChange);
    refs.clusterSortTrigger.addEventListener('click', (event) => {
        event.stopPropagation();
        toggleMetricCascadeMenu('sort');
    });
    refs.clusterFilterTrigger.addEventListener('click', (event) => {
        event.stopPropagation();
        toggleMetricCascadeMenu('filter');
    });
    refs.clusterSortMenu.addEventListener('click', (event) => {
        if (!handleMetricCascadeSelect(event, 'sort')) {
            return;
        }
        renderBatch(state.batchResult);
    });
    refs.clusterFilterMenu.addEventListener('click', (event) => {
        if (!handleMetricCascadeSelect(event, 'filter')) {
            return;
        }
        renderBatch(state.batchResult);
    });
    document.addEventListener('click', () => closeMetricCascadeMenus());
    refs.clusterFilterValue.addEventListener('input', () => {
        state.batchFilterValue = refs.clusterFilterValue.value;
        renderBatch(state.batchResult);
    });
    refs.clusterHideUnmatched.addEventListener('change', () => {
        state.batchHideUnmatched = refs.clusterHideUnmatched.checked;
        renderBatch(state.batchResult);
    });

    refs.clusterCards.addEventListener('mouseover', handleClusterCardMouseOver);
    refs.clusterCards.addEventListener('mouseout', handleClusterCardMouseOut);
    refs.clusterCards.addEventListener('click', handleClusterCardClick);

    refs.clusterMap.addEventListener('mouseover', handleClusterMapMouseOver);
    refs.clusterMap.addEventListener('mousemove', handleClusterMapMouseMove);
    refs.clusterMap.addEventListener('mouseout', handleClusterMapMouseOut);
    refs.clusterMap.addEventListener('click', handleClusterMapClick);

    refs.clusterDendrogram.addEventListener('mouseover', handleDendrogramMouseOver);
    refs.clusterDendrogram.addEventListener('mousemove', handleDendrogramMouseMove);
    refs.clusterDendrogram.addEventListener('mouseout', handleDendrogramMouseOut);

    refs.clusterCompareSummary.addEventListener('click', handleCompareSummaryClick);

    window.addEventListener('langChange', handleLanguageChange);
}

function handleLanguageChange() {
    renderMode({ instant: true });
    renderFiles();
    renderStatusCard();
    renderViewerTabs();
    if (state.singleResult) {
        renderSingleMetrics(state.singleResult);
    }
    if (state.batchResult) {
        renderBatch(state.batchResult);
    }
    if (!refs.clusterModal.classList.contains('hidden') && Number.isInteger(state.selectedClusterId)) {
        openClusterModal(state.selectedClusterId);
    }
}

function renderMode(options = {}) {
    const { instant = false } = options;
    refs.modeButtons.forEach(btn => btn.classList.toggle('active', btn.dataset.mode === state.mode));
    refs.dropHint.textContent = state.mode === 'single'
        ? I18N.t('dropzoneHintSingle')
        : I18N.t('dropzoneHint');
    refs.fileInput.multiple = state.mode === 'batch';
    refs.workspace.classList.toggle('workspace--single', state.mode === 'single');
    refs.workspace.classList.toggle('workspace--batch', state.mode === 'batch');
    switchWorkspacePanel(state.mode, instant);
}

function switchWorkspacePanel(mode, instant = false) {
    const enteringPanel = mode === 'single' ? refs.singlePanel : refs.batchPanel;
    const leavingPanel = mode === 'single' ? refs.batchPanel : refs.singlePanel;

    if (instant) {
        enteringPanel.classList.remove('hidden', 'panel-enter', 'panel-enter-active', 'panel-leave', 'panel-leave-active');
        leavingPanel.classList.add('hidden');
        leavingPanel.classList.remove('panel-enter', 'panel-enter-active', 'panel-leave', 'panel-leave-active');
        return;
    }

    const transitionToken = ++panelTransitionToken;
    enteringPanel.classList.remove('hidden', 'panel-leave', 'panel-leave-active', 'panel-enter', 'panel-enter-active');
    leavingPanel.classList.remove('panel-enter', 'panel-enter-active', 'panel-leave', 'panel-leave-active');
    leavingPanel.classList.add('hidden');
    enteringPanel.classList.add('panel-enter');

    requestAnimationFrame(() => {
        if (transitionToken !== panelTransitionToken) {
            return;
        }
        enteringPanel.classList.add('panel-enter-active');
    });

    window.setTimeout(() => {
        if (transitionToken !== panelTransitionToken) {
            return;
        }
        enteringPanel.classList.remove('panel-enter', 'panel-enter-active');
    }, 420);
}

function renderFiles() {
    refs.fileList.innerHTML = state.files.length
        ? state.files.map((file, index) => `
            <div class="file-item">
                <div class="file-item__thumb-wrap">
                    <img class="file-item__thumb" src="${state.previewUrls[index]}" alt="${escapeHtml(file.name)}">
                </div>
                <div class="file-item__meta">
                    <div class="file-item__name" title="${escapeHtml(file.name)}">${escapeHtml(file.name)}</div>
                    <div class="file-item__size">${formatFileSize(file.size)}</div>
                </div>
                <div class="file-item__actions">
                    <button class="file-action-btn" data-action="preview" data-index="${index}">${I18N.t('previewBtn')}</button>
                    <button class="file-action-btn file-action-btn--danger" data-action="remove" data-index="${index}" aria-label="${I18N.t('removeFile')}">×</button>
                </div>
            </div>
        `).join('')
        : `<div class="file-list__empty">${I18N.t('fileListEmpty')}</div>`;
    if (!state.isAnalyzing) {
        setStatusBySelection();
    }
    syncClusterCountControlState();
}

function setStatus(text, type = 'idle') {
    state.statusText = text;
    state.statusType = type;
    renderStatusCard();
}

function renderStatusCard() {
    const modeTag = state.mode === 'single' ? 'SINGLE' : 'BATCH';
    const levelTag = state.statusType.toUpperCase();
    const timeTag = formatStatusTime(new Date());
    const showTime = !(state.statusType === 'idle' && state.statusText === I18N.t('statusIdle'));
    const showInlineRunBtn = state.statusType === 'ready' && !state.isAnalyzing && state.files.length > 0;
    const showInlineExportBtn = state.mode === 'batch'
        && state.statusType === 'success'
        && !state.isAnalyzing
        && Boolean(state.batchResult?.downloads?.phenotypes_xlsx);
    const showBatchProgress = state.mode === 'batch' && state.statusType === 'running' && state.batchProgress && state.batchProgress.total > 0;
    let statusTextHtml = escapeHtml(state.statusText);

    if (showInlineRunBtn) {
        const runBtnLabel = I18N.t('runBtn');
        const inlineButtonHtml = `
            <button id="runBtn" class="status-run-btn status-run-btn--inline" aria-label="${runBtnLabel}">
                <span class="status-run-btn__shine" aria-hidden="true"></span>
                <span class="status-run-btn__text">${runBtnLabel}</span>
            </button>
        `.trim();
        if (statusTextHtml.includes(runBtnLabel)) {
            statusTextHtml = statusTextHtml.replace(runBtnLabel, inlineButtonHtml);
        } else {
            statusTextHtml = `${statusTextHtml} ${inlineButtonHtml}`;
        }
    }

    if (showInlineExportBtn) {
        const exportBtnLabel = I18N.t('exportBtn');
        const inlineButtonHtml = `
            <button id="exportResultBtn" class="status-run-btn status-run-btn--inline status-run-btn--export" aria-label="${exportBtnLabel}">
                <span class="status-run-btn__shine" aria-hidden="true"></span>
                <span class="status-run-btn__text">${exportBtnLabel}</span>
            </button>
        `.trim();
        if (statusTextHtml.includes(exportBtnLabel)) {
            statusTextHtml = statusTextHtml.replace(exportBtnLabel, inlineButtonHtml);
        } else {
            statusTextHtml = `${statusTextHtml} ${inlineButtonHtml}`;
        }
    }

    refs.statusCard.innerHTML = `
        <div class="status-card__content">
            <div class="status-card__meta">
                <span class="status-card__mode">${modeTag}</span>
                <span class="status-card__level">${levelTag}</span>
                ${showTime ? `<span class="status-card__time">${timeTag}</span>` : ''}
            </div>
            <div class="status-card__text">${statusTextHtml}</div>
            ${showBatchProgress ? `
                <div class="status-card__progress">
                    <div class="status-card__progress-label">${escapeHtml(state.batchProgress.label || I18N.t('batchProgressLabel'))}</div>
                    <div class="status-card__progress-bar"><span style="width:${Math.max(0, Math.min(state.batchProgress.percent || 0, 100))}%"></span></div>
                    <div class="status-card__progress-meta">${state.batchProgress.current || 0} / ${state.batchProgress.total || 0}</div>
                </div>
            ` : ''}
        </div>
    `;
    refs.statusCard.className = `status-card status-card--${state.statusType}`;
}

function resetPanels() {
    refs.singleMetrics.innerHTML = '';
    refs.clusterCards.innerHTML = '';
    refs.clusterMap.innerHTML = '';
    refs.clusterDendrogram.innerHTML = '';
    refs.clusterScore.textContent = 'Silhouette: N/A';
    refs.clusterCompare.classList.add('hidden');
    refs.clusterCompareSummary.innerHTML = '';
    refs.clusterCompareChart.innerHTML = '';
    refs.clusterCompareGallery.innerHTML = '';
    state.batchRunId = null;
    state.batchProgress = null;
    state.comparisonClusterIds = [];
    state.hoveredSampleName = null;
    state.selectedSampleName = null;
    state.hoveredClusterId = null;
    state.selectedClusterId = null;
    state.hoveredDendrogramNodeId = null;
    state.batchStatusSignature = null;
    state.clusterLabelMap = {};
    state.clusterResultMap = {};
    syncWorkbookExportButton();
    stopBatchPolling();
    clearBatchSession();
    refs.viewerImage.removeAttribute('src');
    refs.viewerImage.classList.add('hidden');
    state.fitScale = null;
    state.zoomNeedsSync = true;
    renderZoomBadge();
    refs.skeletonLayer.classList.add('hidden');
    state.skeletonOverlay = null;
    refs.skeletonSvg.innerHTML = '';
    clearSkeletonHoverFx();
    refs.overlaySvg.innerHTML = '';
    refs.tooltip.classList.add('hidden');
    hideClusterHoverCard();
    closeClusterModal();
}

async function runAnalysis() {
    if (!state.files.length) {
        return;
    }

    state.isAnalyzing = true;
    setStatus(state.mode === 'single' ? I18N.t('analyzingSingle') : I18N.t('analyzingBatch'), 'running');

    const formData = new FormData();
    const endpoint = state.mode === 'single' ? '/api/analyze-single' : '/api/analyze-batch';
    if (state.mode === 'single') {
        formData.append('file', state.files[0]);
    } else {
        state.files.forEach(file => formData.append('files', file));
    }

    try {
        const response = await fetch(endpoint, { method: 'POST', body: formData });
        const payload = await response.json();
        if (!response.ok || payload.error) {
            throw new Error(payload.error || '分析失败');
        }

        if (state.mode === 'single') {
            state.singleResult = payload;
            state.currentView = 'spikelet';
            renderViewerTabs();
            renderSingleView();
            renderSingleMetrics(payload);
            setStatus(I18N.t('singleComplete'), 'success');
        } else {
            state.batchRunId = payload.run_id;
            updateBatchProgress({ stage: 'queued', current: 0, total: state.files.length, percent: 0 });
            startBatchPolling(payload.run_id);
            setStatus(I18N.t('batchStarted'), 'running');
            saveBatchSession({ state: 'queued' });
        }
    } catch (error) {
        setStatus(I18N.t('analysisFailed', { error: error.message }), 'error');
    } finally {
        state.isAnalyzing = false;
        renderStatusCard();
    }
}

function startBatchPolling(runId) {
    stopBatchPolling();
    state.batchRunId = runId;
    saveBatchSession({ state: 'running' });
    let inFlight = false;

    const poll = async () => {
        if (inFlight) {
            return;
        }
        inFlight = true;
        try {
            const response = await fetch(`/api/batch-status/${runId}`);
            const status = await response.json();
            if (!response.ok || status.error) {
                throw new Error(status.error || '批量状态获取失败');
            }
            if (state.batchRunId !== runId) {
                return;
            }
            updateBatchProgress(status);
            if (status.state === 'completed') {
                stopBatchPolling();
                await fetchBatchResult(runId);
                return;
            }
            if (status.state === 'error') {
                stopBatchPolling();
                throw new Error(status.error || '批量分析失败');
            }
        } catch (error) {
            stopBatchPolling();
            state.isAnalyzing = false;
            state.batchProgress = null;
            setStatus(I18N.t('batchFailed', { error: error.message }), 'error');
            renderStatusCard();
            saveBatchSession({ state: 'error' });
        } finally {
            inFlight = false;
        }
    };

    poll();
    state.batchStatusPoller = window.setInterval(poll, 1200);
}

function stopBatchPolling() {
    if (!state.batchStatusPoller) {
        return;
    }
    window.clearInterval(state.batchStatusPoller);
    state.batchStatusPoller = null;
}

async function fetchBatchResult(runId) {
    const response = await fetch(`/api/batch-result/${runId}`);
    const payload = await response.json();
    if (!response.ok || payload.error) {
        if (payload?.code === 'job_expired' || payload?.code === 'job_not_found') {
            clearBatchSession();
        }
        throw new Error(payload.error || '批量结果获取失败');
    }
    state.batchRunId = payload.run_id || runId;
    state.batchResult = payload;
    state.isAnalyzing = false;
    state.batchProgress = null;
    state.batchStatusSignature = null;
    initializeBatchControls(payload.cluster);
    syncWorkbookExportButton();
    renderBatch(payload);
    setStatus(I18N.t('analysisCompleteExport'), 'success');
    renderStatusCard();
    saveBatchSession({ state: 'completed' });
}

function syncWorkbookExportButton() {
    renderStatusCard();
}

function exportPhenotypeWorkbook() {
    const runId = state.batchRunId || state.batchResult?.run_id;
    const downloadUrl = state.batchResult?.downloads?.phenotypes_xlsx;
    if (!runId || !downloadUrl) {
        return;
    }
    window.open(downloadUrl, '_blank', 'noopener');
}

function updateBatchProgress(status) {
    const stageMap = {
        queued: I18N.t('batchQueued'),
        analyzing: I18N.t('batchAnalyzing', { file: status.current_file || I18N.t('currentSample') }),
        clustering: I18N.t('batchClustering'),
        completed: I18N.t('batchCompleted'),
        error: I18N.t('batchError'),
    };
    const nextProgress = {
        label: stageMap[status.stage] || I18N.t('batchProcessing'),
        percent: Math.max(0, Math.min(status.percent || 0, 100)),
        current: status.current || 0,
        total: status.total || 0,
    };
    const signature = [status.state || '', status.stage || '', nextProgress.current, nextProgress.total, Math.round(nextProgress.percent || 0)].join('|');
    if (signature === state.batchStatusSignature) {
        return;
    }
    state.batchStatusSignature = signature;
    state.batchProgress = nextProgress;
    renderStatusCard();
    saveBatchSession({ state: status.state || 'running' });
}

function initializeBatchControls(cluster) {
    const metrics = getClusterMetricOptions(cluster);
    state.clusterMetricOptions = metrics;

    const fallbackMetric = metrics.find(metric => metric.key === 'mean_spike_length_cm')?.key || metrics[0]?.key || '';
    state.batchSortMetric = metrics.some(metric => metric.key === state.batchSortMetric) ? state.batchSortMetric : fallbackMetric;
    state.batchFilterMetric = metrics.some(metric => metric.key === state.batchFilterMetric) ? state.batchFilterMetric : fallbackMetric;
    renderMetricCascadeControls();

    const options = cluster?.cluster_options || { min: 2, max: 8, current: 3 };
    const minValue = Math.max(2, Number(options.min) || 2);
    const sampleCount = Number(cluster?.image_names?.length || cluster?.embedding?.length || 0);
    const backendMax = Math.max(minValue, Number(options.max) || 8);
    const safeMax = sampleCount > minValue ? Math.min(backendMax, sampleCount - 1) : minValue;
    const safeCurrent = Math.min(Math.max(Number(options.current) || minValue, minValue), safeMax);

    refs.clusterCountInput.min = String(minValue);
    refs.clusterCountInput.max = String(safeMax);
    refs.clusterCountInput.value = String(safeCurrent);
    refs.clusterCountValue.textContent = String(safeCurrent);
    const validClusterIds = new Set((cluster?.clusters || []).map(item => item.cluster_id));
    state.comparisonClusterIds = state.comparisonClusterIds.filter(id => validClusterIds.has(id));
    syncClusterCountControlState();
}

function handleClusterCountInput() {
    refs.clusterCountValue.textContent = String(refs.clusterCountInput.value);
}

async function handleClusterCountChange() {
    const minValue = Number.parseInt(refs.clusterCountInput.min, 10);
    const maxValue = Number.parseInt(refs.clusterCountInput.max, 10);
    const rawCount = Number.parseInt(refs.clusterCountInput.value, 10);
    const nextCount = Math.min(Math.max(rawCount, minValue), maxValue);
    refs.clusterCountInput.value = String(nextCount);
    refs.clusterCountValue.textContent = String(nextCount);
    if (!state.batchRunId || !Number.isInteger(nextCount)) {
        return;
    }

    try {
        refs.clusterCountInput.dataset.busy = '1';
        syncClusterCountControlState();
        setStatus(I18N.t('adjustingClusters', { count: nextCount }), 'running');
        const response = await fetch('/api/recluster', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ run_id: state.batchRunId, n_clusters: nextCount }),
        });
        const payload = await response.json();
        if (!response.ok || payload.error) {
            throw new Error(payload.error || '重聚类失败');
        }
        state.batchResult.cluster = payload.cluster;
        initializeBatchControls(state.batchResult.cluster);
        refs.clusterCountInput.value = nextCount;
        refs.clusterCountValue.textContent = nextCount;
        renderBatch(state.batchResult);
        setStatus(I18N.t('clustersUpdated', { count: nextCount }), 'success');
    } catch (error) {
        state.batchProgress = null;
        setStatus(I18N.t('reclusterFailed', { error: error.message }), 'error');
    } finally {
        refs.clusterCountInput.dataset.busy = '0';
        syncClusterCountControlState();
        renderStatusCard();
    }
}

function syncClusterCountControlState() {
    if (!refs.clusterCountInput) {
        return;
    }
    const hasUploadedFiles = state.files.length > 0;
    const isBusy = refs.clusterCountInput.dataset.busy === '1';
    const disabled = !hasUploadedFiles || isBusy;
    refs.clusterCountInput.disabled = disabled;
    refs.clusterCountControl?.classList.toggle('is-disabled', disabled);
}

function setFiles(files) {
    clearPreviewUrls();
    state.files = files;
    state.previewUrls = state.files.map(file => URL.createObjectURL(file));
    syncInputFiles();
}

function clearPreviewUrls() {
    state.previewUrls.forEach(url => URL.revokeObjectURL(url));
    state.previewUrls = [];
}

function syncInputFiles() {
    const dataTransfer = new DataTransfer();
    state.files.forEach(file => dataTransfer.items.add(file));
    refs.fileInput.files = dataTransfer.files;
}

function removeFileAt(index) {
    const removedPreviewUrl = state.previewUrls[index];
    URL.revokeObjectURL(removedPreviewUrl);
    state.previewUrls.splice(index, 1);
    state.files.splice(index, 1);
    syncInputFiles();
    renderFiles();

    const removedPreviewSrc = refs.previewImage.getAttribute('src');
    if (removedPreviewSrc && removedPreviewSrc === removedPreviewUrl) {
        closePreview();
    }
}

function openPreview(index) {
    openImagePreview(state.files[index].name, state.previewUrls[index]);
}

function openImagePreview(title, imageUrl) {
    if (!imageUrl) {
        return;
    }
    refs.previewTitle.textContent = title || I18N.t('previewTitle');
    refs.previewImage.src = imageUrl;
    refs.previewModal.classList.remove('hidden');
    document.body.classList.add('modal-open');
}

function closePreview() {
    refs.previewModal.classList.add('hidden');
    refs.previewImage.removeAttribute('src');
    if (!refs.clusterModal || refs.clusterModal.classList.contains('hidden')) {
        document.body.classList.remove('modal-open');
    }
}

function setStatusBySelection() {
    if (!state.files.length) {
        setStatus(I18N.t('statusIdle'), 'idle');
        return;
    }
    if (state.mode === 'single') {
        setStatus(I18N.t('selectOneFile'), 'ready');
        return;
    }
    setStatus(I18N.t('selectMultiFiles', { count: state.files.length }), 'ready');
}

function formatFileSize(bytes) {
    if (!Number.isFinite(bytes) || bytes <= 0) {
        return '0 B';
    }
    const units = ['B', 'KB', 'MB', 'GB'];
    const unitIndex = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
    const value = bytes / (1024 ** unitIndex);
    return `${value.toFixed(unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

function escapeHtml(text) {
    return String(text)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function formatStatusTime(date) {
    const hh = String(date.getHours()).padStart(2, '0');
    const mm = String(date.getMinutes()).padStart(2, '0');
    const ss = String(date.getSeconds()).padStart(2, '0');
    return `${hh}:${mm}:${ss}`;
}

function renderViewerTabs() {
    refs.viewerTabs.forEach(btn => btn.classList.toggle('active', btn.dataset.view === state.currentView));
}

function renderSingleView() {
    const result = state.singleResult;
    if (!result || !result.images) {
        refs.viewerImage.removeAttribute('src');
        refs.viewerImage.classList.add('hidden');
        state.zoomNeedsSync = true;
        renderZoomBadge();
        return;
    }

    const imageByView = {
        spikelet: result.images.original,
        skeleton_extract: result.images.original,
    };
    const baseImageUrl = imageByView[state.currentView] || result.images.original;
    const skeletonOverlay = result.skeleton_overlay || null;
    if (!baseImageUrl) {
        return;
    }

    refs.viewerImage.onload = () => {
        refs.viewerImage.classList.remove('hidden');
        refs.overlaySvg.setAttribute('viewBox', `0 0 ${refs.viewerImage.naturalWidth} ${refs.viewerImage.naturalHeight}`);
        refs.overlaySvg.setAttribute('width', refs.viewerImage.naturalWidth);
        refs.overlaySvg.setAttribute('height', refs.viewerImage.naturalHeight);
        refs.skeletonLayer.style.width = `${refs.viewerImage.naturalWidth}px`;
        refs.skeletonLayer.style.height = `${refs.viewerImage.naturalHeight}px`;
        refs.skeletonLayer.classList.toggle('hidden', state.currentView !== 'skeleton_extract');
        refs.overlaySvg.style.display = state.currentView === 'spikelet' ? 'block' : 'none';
        refs.skeletonLayer.classList.toggle('skeleton-layer--empty', !skeletonOverlay);
        refs.skeletonSvg.setAttribute('viewBox', `0 0 ${refs.viewerImage.naturalWidth} ${refs.viewerImage.naturalHeight}`);
        refs.skeletonSvg.setAttribute('width', refs.viewerImage.naturalWidth);
        refs.skeletonSvg.setAttribute('height', refs.viewerImage.naturalHeight);
        refs.skeletonFxCanvas.width = refs.viewerImage.naturalWidth;
        refs.skeletonFxCanvas.height = refs.viewerImage.naturalHeight;
        state.skeletonOverlay = skeletonOverlay;
        renderSkeletonOverlay();
        clearSkeletonHoverFx();
        resetZoom();
        renderOverlay();
    };
    state.zoomNeedsSync = true;
    refs.viewerImage.classList.remove('hidden');
    refs.viewerImage.src = baseImageUrl;
}

function renderSkeletonOverlay() {
    refs.skeletonSvg.innerHTML = '';

    if (!state.skeletonOverlay || !state.skeletonOverlay.stem_points?.length) {
        return;
    }

    const stemPath = buildSvgPath(state.skeletonOverlay.stem_points);
    const abstractStem = state.skeletonOverlay.abstract_stem || null;
    const stemGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    stemGroup.setAttribute('class', 'skeleton-group');
    stemGroup.innerHTML = `
        <path class="skeleton-path skeleton-path--stem" data-path-id="stem" d="${stemPath}"></path>
        <path class="skeleton-path skeleton-path--stem skeleton-path--glow" data-path-id="stem" d="${stemPath}"></path>
        <path class="skeleton-hit skeleton-hit--stem" data-path-id="stem" d="${stemPath}"></path>
    `;
    refs.skeletonSvg.appendChild(stemGroup);

    if (abstractStem?.start_point && abstractStem?.end_point) {
        const abstractPath = buildSvgPath([abstractStem.start_point, abstractStem.end_point]);
        const abstractGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        abstractGroup.setAttribute('class', 'skeleton-group');
        abstractGroup.innerHTML = `
            <path class="skeleton-path skeleton-path--abstract" data-path-id="abstract" d="${abstractPath}"></path>
            <path class="skeleton-path skeleton-path--abstract skeleton-path--glow" data-path-id="abstract" d="${abstractPath}"></path>
            <path class="skeleton-hit skeleton-hit--stem" data-path-id="abstract" d="${abstractPath}"></path>
            <circle class="skeleton-node skeleton-node--stem-endpoint" data-node-id="stem-start" cx="${abstractStem.start_point[0]}" cy="${abstractStem.start_point[1]}" r="6.2"></circle>
            <circle class="skeleton-node skeleton-node--stem-endpoint" data-node-id="stem-end" cx="${abstractStem.end_point[0]}" cy="${abstractStem.end_point[1]}" r="6.2"></circle>
            <circle class="skeleton-node-hit" data-node-id="stem-start" cx="${abstractStem.start_point[0]}" cy="${abstractStem.start_point[1]}" r="12"></circle>
            <circle class="skeleton-node-hit" data-node-id="stem-end" cx="${abstractStem.end_point[0]}" cy="${abstractStem.end_point[1]}" r="12"></circle>
        `;
        refs.skeletonSvg.appendChild(abstractGroup);
    }

    (state.skeletonOverlay.spikelets || []).forEach(spikelet => {
        const branchPath = buildSvgPath([spikelet.highest_point, spikelet.lowest_point]);
        const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        group.setAttribute('class', 'skeleton-group');
        group.dataset.pathId = `spikelet-${spikelet.index}`;
        group.dataset.side = spikelet.side;
        group.innerHTML = `
            <path class="skeleton-path skeleton-path--branch ${spikelet.side}" data-path-id="spikelet-${spikelet.index}" d="${branchPath}"></path>
            <path class="skeleton-path skeleton-path--branch skeleton-path--glow ${spikelet.side}" data-path-id="spikelet-${spikelet.index}" d="${branchPath}"></path>
            <circle class="skeleton-node ${spikelet.side}" data-node-id="spikelet-base-${spikelet.index}" cx="${spikelet.lowest_point[0]}" cy="${spikelet.lowest_point[1]}" r="5.6"></circle>
            <circle class="skeleton-node ${spikelet.side}" data-node-id="spikelet-tip-${spikelet.index}" cx="${spikelet.highest_point[0]}" cy="${spikelet.highest_point[1]}" r="5"></circle>
            ${spikelet.stem_point ? `<circle class="skeleton-node skeleton-node--stem-match" data-node-id="spikelet-stem-${spikelet.index}" cx="${spikelet.stem_point[0]}" cy="${spikelet.stem_point[1]}" r="5.3"></circle>` : ''}
            <circle class="skeleton-node-hit" data-node-id="spikelet-base-${spikelet.index}" cx="${spikelet.lowest_point[0]}" cy="${spikelet.lowest_point[1]}" r="11"></circle>
            <circle class="skeleton-node-hit" data-node-id="spikelet-tip-${spikelet.index}" cx="${spikelet.highest_point[0]}" cy="${spikelet.highest_point[1]}" r="11"></circle>
            ${spikelet.stem_point ? `<circle class="skeleton-node-hit" data-node-id="spikelet-stem-${spikelet.index}" cx="${spikelet.stem_point[0]}" cy="${spikelet.stem_point[1]}" r="11"></circle>` : ''}
            <path class="skeleton-hit skeleton-hit--branch" data-path-id="spikelet-${spikelet.index}" d="${branchPath}"></path>
        `;
        refs.skeletonSvg.appendChild(group);
    });

    bindSkeletonOverlayEvents();
}

function buildSvgPath(points) {
    return points.map((point, index) => `${index === 0 ? 'M' : 'L'} ${point[0]} ${point[1]}`).join(' ');
}

function bindSkeletonOverlayEvents() {
    refs.skeletonSvg.querySelectorAll('.skeleton-hit').forEach(hitPath => {
        hitPath.addEventListener('pointerenter', handleSkeletonPathHover);
        hitPath.addEventListener('pointermove', handleSkeletonPathHover);
        hitPath.addEventListener('pointerleave', clearSkeletonHoverFx);
    });
    refs.skeletonSvg.querySelectorAll('.skeleton-node-hit').forEach(node => {
        node.addEventListener('pointerenter', handleSkeletonNodeHover);
        node.addEventListener('pointermove', handleSkeletonNodeHover);
        node.addEventListener('pointerleave', clearSkeletonHoverFx);
    });
}

function handleSkeletonPathHover(event) {
    if (state.currentView !== 'skeleton_extract' || refs.skeletonLayer.classList.contains('hidden')) {
        return;
    }

    const pathId = event.currentTarget.dataset.pathId;
    if (!pathId) {
        return;
    }

    const targetIds = [pathId];
    setActiveSkeletonElements(targetIds);

    const rect = refs.viewer.getBoundingClientRect();
    state.skeletonFxPointer = {
        x: (event.clientX - rect.left - state.translateX) / state.scale,
        y: (event.clientY - rect.top - state.translateY) / state.scale,
    };
    showSkeletonPathTooltip(event, pathId);
    ensureSkeletonFxLoop();
}

function handleSkeletonNodeHover(event) {
    if (state.currentView !== 'skeleton_extract' || refs.skeletonLayer.classList.contains('hidden')) {
        return;
    }

    const nodeId = event.currentTarget.dataset.nodeId;
    const nodeMeta = getSkeletonNodeMeta(nodeId);
    if (!nodeMeta) {
        return;
    }

    const rect = refs.viewer.getBoundingClientRect();
    state.skeletonFxPointer = {
        x: (event.clientX - rect.left - state.translateX) / state.scale,
        y: (event.clientY - rect.top - state.translateY) / state.scale,
    };
    setActiveSkeletonElements(nodeMeta.pathIds, nodeMeta.nodeIds);
    showTooltipHtml(event, nodeMeta.title, nodeMeta.lines);
    ensureSkeletonFxLoop();
}

function setActiveSkeletonElements(pathIds, nodeIds = []) {
    state.skeletonFxTargets = pathIds
        .map(pathId => refs.skeletonSvg.querySelector(`.skeleton-path[data-path-id="${pathId}"]:not(.skeleton-path--glow)`))
        .filter(Boolean);

    refs.skeletonSvg.querySelectorAll('.skeleton-path, .skeleton-node').forEach(element => {
        element.classList.remove('is-active');
    });

    pathIds.forEach(pathId => {
        refs.skeletonSvg.querySelectorAll(`.skeleton-path[data-path-id="${pathId}"], .skeleton-node[data-node-id="${pathId}"]`).forEach(element => {
            element.classList.add('is-active');
        });
    });

    nodeIds.forEach(nodeId => {
        refs.skeletonSvg.querySelectorAll(`.skeleton-node[data-node-id="${nodeId}"]`).forEach(element => {
            element.classList.add('is-active');
        });
    });
}

function renderOverlay() {
    refs.overlaySvg.innerHTML = '';
    refs.tooltip.classList.add('hidden');

    const result = state.singleResult;
    if (!result || !result.spikelet_records || state.currentView !== 'spikelet') {
        return;
    }

    result.spikelet_records.forEach(record => {
        const polygon = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
        polygon.setAttribute('points', record.corners.map(point => point.join(',')).join(' '));
        polygon.setAttribute('class', `overlay-polygon ${record.side}`);
        polygon.addEventListener('mouseenter', (event) => showTooltip(event, record));
        polygon.addEventListener('mousemove', (event) => moveTooltip(event));
        polygon.addEventListener('mouseleave', hideTooltip);
        refs.overlaySvg.appendChild(polygon);
    });
}

function showTooltip(event, record) {
    showTooltipHtml(event, I18N.t('tooltipSpikeletTitle', { order: record.order }), [
        `${I18N.t('tooltipLength')}: ${record.length.toFixed(2)} px`,
        `${I18N.t('tooltipWidth')}: ${record.width.toFixed(2)} px`,
        `${I18N.t('tooltipAspectRatio')}: ${record.aspect_ratio.toFixed(3)}`,
        `${I18N.t('tooltipAttachmentAngle')}: ${record.attachment_angle.toFixed(2)}°`,
        `${I18N.t('tooltipSide')}: ${record.side === 'left' ? I18N.t('sideLeft') : I18N.t('sideRight')}`,
    ]);
}

function showTooltipHtml(event, title, lines) {
    refs.tooltip.innerHTML = `
        <h4>${escapeHtml(title)}</h4>
        ${lines.map(line => renderTooltipLine(line)).join('')}
    `;
    refs.tooltip.classList.remove('hidden');
    moveTooltip(event);
}

function renderTooltipLine(rawLine) {
    const text = String(rawLine ?? '');
    const cnIndex = text.indexOf('：');
    const enIndex = text.indexOf(':');
    let separatorIndex = -1;

    if (cnIndex >= 0 && enIndex >= 0) {
        separatorIndex = Math.min(cnIndex, enIndex);
    } else {
        separatorIndex = Math.max(cnIndex, enIndex);
    }

    if (separatorIndex < 0) {
        return `<p class="spikelet-tooltip__line">${escapeHtml(text)}</p>`;
    }

    const label = text.slice(0, separatorIndex).trim();
    const value = text.slice(separatorIndex + 1).trim();
    return `
        <div class="spikelet-tooltip__kv">
            <span class="spikelet-tooltip__kv-label">${escapeHtml(label)}</span>
            <strong class="spikelet-tooltip__kv-value">${escapeHtml(value)}</strong>
        </div>
    `;
}

function moveTooltip(event) {
    const rect = refs.viewer.getBoundingClientRect();
    refs.tooltip.style.left = `${event.clientX - rect.left + 18}px`;
    refs.tooltip.style.top = `${event.clientY - rect.top + 18}px`;
}

function hideTooltip() {
    refs.tooltip.classList.add('hidden');
}

function getPxPerCm() {
    const pxPerCm = Number(state.singleResult?.calibration?.px_per_cm);
    return Number.isFinite(pxPerCm) && pxPerCm > 0 ? pxPerCm : null;
}

function pxToCm(value) {
    const pxPerCm = getPxPerCm();
    return pxPerCm ? Number(value) / pxPerCm : null;
}

function formatCanvasValue(valuePx) {
    const cmValue = pxToCm(valuePx);
    if (cmValue === null) {
        return `${Number(valuePx).toFixed(1)} px`;
    }
    return `${cmValue.toFixed(2)} cm`;
}

function formatCanvasPoint(point) {
    if (!point) {
        return 'N/A';
    }
    return `(${formatCanvasValue(point[0])}, ${formatCanvasValue(point[1])})`;
}

function getSkeletonNodeMeta(nodeId) {
    if (!nodeId || !state.skeletonOverlay) {
        return null;
    }

    const abstractStem = state.skeletonOverlay.abstract_stem || {};
    if (nodeId === 'stem-start') {
        return {
            title: I18N.t('skeletonStemStart'),
            lines: [I18N.t('tooltipCoordinate', { value: formatCanvasPoint(abstractStem.start_point) })],
            pathIds: ['abstract'],
            nodeIds: ['stem-start'],
        };
    }
    if (nodeId === 'stem-end') {
        return {
            title: I18N.t('skeletonStemEnd'),
            lines: [I18N.t('tooltipCoordinate', { value: formatCanvasPoint(abstractStem.end_point) })],
            pathIds: ['abstract'],
            nodeIds: ['stem-end'],
        };
    }

    const match = nodeId.match(/^spikelet-(base|tip|stem)-(\d+)$/);
    if (!match) {
        return null;
    }

    const [, pointType, spikeletIndexText] = match;
    const spikeletIndex = Number.parseInt(spikeletIndexText, 10);
    const spikelet = (state.skeletonOverlay.spikelets || []).find(item => item.index === spikeletIndex);
    if (!spikelet) {
        return null;
    }

    const pointMap = {
        base: {
            title: I18N.t('tooltipSpikeletBasePointTitle', { order: spikelet.order }),
            point: spikelet.lowest_point,
            pathIds: [`spikelet-${spikelet.index}`],
        },
        tip: {
            title: I18N.t('tooltipSpikeletTipPointTitle', { order: spikelet.order }),
            point: spikelet.highest_point,
            pathIds: [`spikelet-${spikelet.index}`],
        },
        stem: {
            title: I18N.t('tooltipStemMatchedPointTitle', { order: spikelet.order }),
            point: spikelet.stem_point,
            pathIds: [`spikelet-${spikelet.index}`],
        },
    };
    const current = pointMap[pointType];
    if (!current?.point) {
        return null;
    }

    return {
        title: current.title,
        lines: [
            I18N.t('tooltipCoordinate', { value: formatCanvasPoint(current.point) }),
            `${I18N.t('tooltipSide')}: ${spikelet.side === 'left' ? I18N.t('sideLeft') : I18N.t('sideRight')}`,
        ],
        pathIds: current.pathIds,
        nodeIds: [nodeId],
    };
}

function showSkeletonPathTooltip(event, pathId) {
    if (pathId === 'stem') {
        showTooltipHtml(event, I18N.t('skeletonStemFit'), [I18N.t('skeletonAbstractTip')]);
        return;
    }
    if (pathId === 'abstract') {
        const abstractStem = state.skeletonOverlay?.abstract_stem;
        if (!abstractStem) {
            return;
        }
        showTooltipHtml(event, I18N.t('skeletonAbstract'), [
            I18N.t('tooltipVector', { value: `(${formatCanvasValue(abstractStem.vector?.[0] || 0)}, ${formatCanvasValue(abstractStem.vector?.[1] || 0)})` }),
            I18N.t('tooltipMagnitude', { value: formatCanvasValue(abstractStem.length_px || 0) }),
            I18N.t('tooltipDirectionAngle', { value: `${Number(abstractStem.angle_deg || 0).toFixed(2)}°` }),
            I18N.t('tooltipStartPoint', { value: formatCanvasPoint(abstractStem.start_point) }),
            I18N.t('tooltipEndPoint', { value: formatCanvasPoint(abstractStem.end_point) }),
        ]);
        return;
    }

    const spikeletIndex = Number.parseInt(pathId.replace('spikelet-', ''), 10);
    const spikelet = (state.skeletonOverlay?.spikelets || []).find(item => item.index === spikeletIndex);
    if (!spikelet) {
        return;
    }
    showTooltipHtml(event, I18N.t('tooltipSpikeletSkeletonTitle', { order: spikelet.order }), [
        `${I18N.t('tooltipSide')}: ${spikelet.side === 'left' ? I18N.t('sideLeft') : I18N.t('sideRight')}`,
        I18N.t('tooltipBasePoint', { value: formatCanvasPoint(spikelet.lowest_point) }),
        I18N.t('tooltipTipPoint', { value: formatCanvasPoint(spikelet.highest_point) }),
    ]);
}

function renderSingleMetrics(result) {
    const ear = result.ear_pheno;
    const spikeletMetrics = [
        [I18N.t('metricSpikeletLength'), formatMetric(ear.mean_spikelet_length_mm, 'mm', ear.mean_spikelet_length, 'px')],
        [I18N.t('metricSpikeletWidth'), formatMetric(ear.mean_spikelet_width_mm, 'mm', ear.mean_spikelet_width, 'px')],
        [I18N.t('metricSpikeletAspectRatio'), ear.mean_aspect_ratio.toFixed(3)],
        [I18N.t('metricAttachmentAngle'), `${ear.mean_attachment_angle.toFixed(2)}°`],
        [I18N.t('metricMeanHue'), `${Number(ear.mean_hue_deg ?? 0).toFixed(2)}°`],
        [I18N.t('metricMeanSaturation'), Number(ear.mean_saturation ?? 0).toFixed(2)],
        [I18N.t('metricStdHue'), Number(ear.std_hue ?? 0).toFixed(2)],
    ];
    const earMetrics = [
        [I18N.t('metricSpikeLength'), formatMetric(ear.spike_length_cm, 'cm', ear.spike_length_px, 'px')],
        [I18N.t('metricSpikeletCount'), `${ear.spikelet_count}`],
        [I18N.t('metricSpikeletDensity'), formatMetric(ear.spikelet_density_per_cm, '/cm', ear.spikelet_density_px, '/px')],
        [I18N.t('metricSymmetry'), ear.symmetry_index.toFixed(4)],
        [I18N.t('metricCentroidOffset'), ear.centroid_offset.toFixed(4)],
    ];

    refs.singleMetrics.innerHTML = `
        <div class="single-metrics-layout">
            ${renderSingleMetricGroup('spikelet', I18N.t('groupSpikelet'), 'SPIKELET', spikeletMetrics)}
            ${renderSingleMetricGroup('ear', I18N.t('groupEar'), 'SPIKE', earMetrics)}
        </div>
    `;
}

function renderSingleMetricGroup(groupClass, title, eyebrow, metrics) {
    return `
        <section class="single-metrics-group single-metrics-group--${groupClass}">
            <header class="single-metrics-group__header">
                <div>
                    <p class="single-metrics-group__eyebrow">${eyebrow}</p>
                    <h4 class="single-metrics-group__title">${title}</h4>
                </div>
                <span class="single-metrics-group__count">${metrics.length}</span>
            </header>
            <div class="single-metrics-group__grid">
                ${metrics.map(([label, value]) => `
                    <article class="single-metric-card">
                        <div class="single-metric-card__label">${label}</div>
                        <div class="single-metric-card__value">${value}</div>
                    </article>
                `).join('')}
            </div>
        </section>
    `;
}

function formatMetric(preferredValue, preferredUnit, fallbackValue, fallbackUnit) {
    if (preferredValue !== null && preferredValue !== undefined) {
        return `${preferredValue.toFixed(3)} ${preferredUnit}`;
    }
    return `${fallbackValue.toFixed(3)} ${fallbackUnit}`;
}

function stepZoom(factor, anchor) {
    if (!ensureZoomBaseline()) {
        return;
    }
    updateZoom(state.scale * factor, anchor);
}

function updateZoom(nextScale, anchor = getViewerCenterAnchor()) {
    const minScale = state.fitScale ?? 0.1;
    const previousScale = Math.max(state.scale || minScale, 1e-6);
    const targetScale = Math.min(Math.max(nextScale, minScale), 8);
    const normalizedAnchor = normalizeAnchor(anchor);
    const scaleRatio = targetScale / previousScale;

    state.translateX = normalizedAnchor.x - (normalizedAnchor.x - state.translateX) * scaleRatio;
    state.translateY = normalizedAnchor.y - (normalizedAnchor.y - state.translateY) * scaleRatio;
    state.scale = targetScale;
    state.zoomNeedsSync = false;
    applyTransform();
}

function normalizeAnchor(anchor) {
    const viewerWidth = refs.viewer.clientWidth || 0;
    const viewerHeight = refs.viewer.clientHeight || 0;
    if (!anchor) {
        return getViewerCenterAnchor();
    }
    return {
        x: Math.min(Math.max(anchor.x, 0), viewerWidth),
        y: Math.min(Math.max(anchor.y, 0), viewerHeight),
    };
}

function getViewerCenterAnchor() {
    return {
        x: (refs.viewer.clientWidth || 0) / 2,
        y: (refs.viewer.clientHeight || 0) / 2,
    };
}

function ensureZoomBaseline() {
    if (!refs.viewerImage.src) {
        return false;
    }
    if (state.zoomNeedsSync) {
        resetZoom();
    }
    return true;
}

function resetZoom() {
    const viewerWidth = refs.viewer.clientWidth;
    const viewerHeight = refs.viewer.clientHeight;
    const imageWidth = refs.viewerImage.naturalWidth;
    const imageHeight = refs.viewerImage.naturalHeight;

    if (!viewerWidth || !viewerHeight || !imageWidth || !imageHeight) {
        state.scale = 1;
        state.fitScale = null;
        state.translateX = 0;
        state.translateY = 0;
        state.zoomNeedsSync = true;
        applyTransform();
        return;
    }

    const fitScale = Math.min(viewerWidth / imageWidth, viewerHeight / imageHeight);
    state.fitScale = Math.min(Math.max(fitScale * 0.96, 0.1), 8);
    state.scale = state.fitScale;
    state.translateX = (viewerWidth - imageWidth * state.scale) / 2;
    state.translateY = (viewerHeight - imageHeight * state.scale) / 2;
    state.zoomNeedsSync = false;
    applyTransform();
}

function applyTransform() {
    refs.viewerStage.style.transform = `translate(${state.translateX}px, ${state.translateY}px) scale(${state.scale})`;
    renderZoomBadge();
}

function renderZoomBadge() {
    if (!refs.zoomBadge) {
        return;
    }

    const hasImage = Boolean(refs.viewerImage.getAttribute('src'));
    if (!hasImage || !Number.isFinite(state.scale) || state.scale <= 0) {
        refs.zoomBadge.classList.add('hidden');
        return;
    }

    const baseline = state.fitScale && state.fitScale > 0 ? state.fitScale : state.scale;
    const percent = Math.max(1, Math.round((state.scale / baseline) * 100));
    refs.zoomBadge.textContent = `${percent}%`;
    refs.zoomBadge.classList.remove('hidden');
}

function handleWheelZoom(event) {
    event.preventDefault();
    if (!ensureZoomBaseline()) {
        return;
    }
    const factor = event.deltaY < 0 ? 1.08 : 0.92;
    const rect = refs.viewer.getBoundingClientRect();
    stepZoom(factor, {
        x: event.clientX - rect.left,
        y: event.clientY - rect.top,
    });
}

function handlePointerDown(event) {
    if (!refs.viewerImage.src) {
        return;
    }
    state.dragging = true;
    state.dragStartX = event.clientX - state.translateX;
    state.dragStartY = event.clientY - state.translateY;
    refs.viewerStage.style.cursor = 'grabbing';
}

function handlePointerMove(event) {
    if (!state.dragging) {
        return;
    }
    state.translateX = event.clientX - state.dragStartX;
    state.translateY = event.clientY - state.dragStartY;
    applyTransform();
}

function handlePointerUp() {
    state.dragging = false;
    refs.viewerStage.style.cursor = 'grab';
}

function updateSkeletonHoverFx(event) {
    if (state.currentView !== 'skeleton_extract' || refs.skeletonLayer.classList.contains('hidden') || !state.skeletonFxTargets.length) {
        return;
    }

    const rect = refs.viewer.getBoundingClientRect();
    state.skeletonFxPointer = {
        x: (event.clientX - rect.left - state.translateX) / state.scale,
        y: (event.clientY - rect.top - state.translateY) / state.scale,
    };
}

function clearSkeletonHoverFx() {
    const ctx = refs.skeletonFxCanvas.getContext('2d');
    ctx.clearRect(0, 0, refs.skeletonFxCanvas.width, refs.skeletonFxCanvas.height);
    refs.skeletonSvg.querySelectorAll('.skeleton-path, .skeleton-node').forEach(element => {
        element.classList.remove('is-active');
    });
    hideTooltip();
    state.skeletonFxPointer = null;
    state.skeletonFxTargets = [];
    if (state.skeletonFxAnimationFrame) {
        cancelAnimationFrame(state.skeletonFxAnimationFrame);
        state.skeletonFxAnimationFrame = null;
    }
}

function ensureSkeletonFxLoop() {
    if (state.skeletonFxAnimationFrame) {
        return;
    }

    const frame = (timestamp) => {
        state.skeletonFxAnimationFrame = null;
        if (!state.skeletonFxTargets.length || !state.skeletonFxPointer || state.currentView !== 'skeleton_extract') {
            return;
        }
        drawSkeletonHoverFx(timestamp);
        state.skeletonFxAnimationFrame = requestAnimationFrame(frame);
    };

    state.skeletonFxAnimationFrame = requestAnimationFrame(frame);
}

function drawSkeletonHoverFx(timestamp) {
    const canvas = refs.skeletonFxCanvas;
    const ctx = canvas.getContext('2d');
    const time = timestamp * 0.001;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.globalCompositeOperation = 'screen';

    state.skeletonFxTargets.forEach((pathElement, index) => {
        const totalLength = pathElement.getTotalLength();
        const isStem = pathElement.dataset.pathId === 'stem';
        const pulseCount = isStem ? 3 : 2;
        const phaseShift = index * 0.6;
        const speed = isStem ? 84 : 62;

        for (let pulseIndex = 0; pulseIndex < pulseCount; pulseIndex += 1) {
            const travel = ((time * speed) + (pulseIndex * totalLength / pulseCount) + phaseShift * 36) % totalLength;
            const point = pathElement.getPointAtLength(travel);
            // Avoid wrap-around sampling near path start; otherwise tail can jump to path end and draw a false bridge.
            const tailPos = Math.max(0, travel - 30);
            const shoulderPos = Math.max(0, travel - 16);
            const tail = pathElement.getPointAtLength(tailPos);
            const shoulder = pathElement.getPointAtLength(shoulderPos);
            const radius = isStem ? 6.8 : 5.2;

            const streak = ctx.createLinearGradient(tail.x, tail.y, point.x, point.y);
            streak.addColorStop(0, 'rgba(118, 231, 255, 0)');
            streak.addColorStop(0.52, isStem ? 'rgba(134, 250, 182, 0.16)' : 'rgba(132, 233, 255, 0.16)');
            streak.addColorStop(1, isStem ? 'rgba(204, 255, 224, 0.7)' : 'rgba(214, 250, 255, 0.68)');

            ctx.strokeStyle = streak;
            ctx.lineCap = 'round';
            ctx.lineWidth = isStem ? 2.1 : 1.5;
            ctx.beginPath();
            ctx.moveTo(tail.x, tail.y);
            ctx.quadraticCurveTo(shoulder.x, shoulder.y, point.x, point.y);
            ctx.stroke();

            const spark = ctx.createRadialGradient(point.x, point.y, 0, point.x, point.y, radius * 2.2);
            spark.addColorStop(0, isStem ? 'rgba(242, 255, 249, 0.98)' : 'rgba(246, 253, 255, 0.95)');
            spark.addColorStop(0.32, isStem ? 'rgba(133, 255, 182, 0.38)' : 'rgba(132, 236, 255, 0.36)');
            spark.addColorStop(1, 'rgba(118, 231, 255, 0)');
            ctx.fillStyle = spark;
            ctx.beginPath();
            ctx.arc(point.x, point.y, radius * 2.2, 0, Math.PI * 2);
            ctx.fill();
        }
    });

}

function renderBatch(payload) {
    if (!payload) {
        return;
    }
    if (!state.batchRunId && payload.run_id) {
        state.batchRunId = payload.run_id;
    }
    const cluster = payload?.cluster;
    const results = payload?.results || [];

    if (!cluster || !cluster.embedding || !cluster.embedding.length) {
        refs.clusterMap.innerHTML = '<p class="cluster-detail__placeholder">有效样本不足，暂未生成聚类结果。</p>';
        refs.clusterDendrogram.innerHTML = '<p class="cluster-detail__placeholder">暂无树状图数据。</p>';
        refs.clusterCards.innerHTML = '';
        refs.clusterScore.textContent = 'Silhouette: N/A';
        return;
    }

    const token = ++state.batchRenderToken;
    refs.clusterScore.textContent = `Silhouette: ${cluster.silhouette_score ? cluster.silhouette_score.toFixed(3) : 'N/A'}`;
    renderClusterCards(cluster);

    window.requestAnimationFrame(() => {
        if (token !== state.batchRenderToken) {
            return;
        }
        renderClusterMap(results, cluster);
        window.requestAnimationFrame(() => {
            if (token !== state.batchRenderToken) {
                return;
            }
            renderDendrogram(cluster);
            renderClusterCompare(cluster);
            syncBatchDetail(cluster, results);
        });
    });
}

function renderClusterCards(cluster) {
    const cards = getVisibleClusters(cluster);
    const clusterMap = new Map((cluster?.clusters || []).map(item => [item.cluster_id, item]));
    refs.clusterCards.innerHTML = cards.map(item => {
        const active = state.selectedClusterId === item.cluster_id || state.hoveredClusterId === item.cluster_id;
        const inComparison = state.comparisonClusterIds.includes(item.cluster_id);
        const clusterLabel = I18N.lang === 'zh' ? `第 ${item.cluster_id + 1} 类` : `Cluster ${item.cluster_id + 1}`;
        const sampleCountText = I18N.lang === 'zh' ? `${item.sample_count} 个样本` : `${item.sample_count} samples`;
        return `
            <article class="cluster-card ${active ? 'is-active' : ''} ${item.__matched ? '' : 'is-muted'}" data-cluster-id="${item.cluster_id}">
                <div class="cluster-card__eyebrow">Cluster ${item.cluster_id + 1}</div>
                <div class="cluster-card__header">
                    <div>
                        <h3>${clusterLabel}</h3>
                        <p>${sampleCountText}</p>
                    </div>
                    <div class="cluster-card__badge">${formatClusterMetric(item.aggregate_metrics[state.batchSortMetric])}</div>
                </div>
                <div class="cluster-card__cover">
                    ${item.representative_image ? `<img src="${item.representative_image}" alt="cluster ${item.cluster_id} representative">` : '<div class="cluster-card__cover-empty">No Image</div>'}
                </div>
                <div class="cluster-card__hint" data-i18n="clickToViewCluster">${I18N.t('clickToViewCluster')}</div>
                <div class="cluster-card__actions">
                    <button class="ghost-btn cluster-card__action cluster-card__action--compare ${inComparison ? 'is-active' : ''}" data-action="compare" data-cluster-id="${item.cluster_id}">
                        ${inComparison ? I18N.t('cancelCompare') : I18N.t('addToCompare')}
                    </button>
                    <button class="ghost-btn cluster-card__action cluster-card__action--export" data-action="export" data-cluster-id="${item.cluster_id}">
                        <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
                            <path d="M12 3V14"/>
                            <path d="M8 10L12 14L16 10"/>
                            <path d="M5 17H19"/>
                            <path d="M7 21H17"/>
                        </svg>
                        <span>${I18N.t('exportCluster')}</span>
                    </button>
                </div>
            </article>
        `;
    }).join('');

}

function renderClusterMap(results, cluster) {
    const width = Math.max(320, refs.clusterMap.clientWidth - 12);
    const height = 360;
    const points = cluster.embedding;
    const xs = points.map(point => point[0]);
    const ys = points.map(point => point[1]);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const padding = 32;
    const scaleX = value => padding + ((value - minX) / Math.max(maxX - minX, 1e-6)) * (width - padding * 2);
    const scaleY = value => height - padding - ((value - minY) / Math.max(maxY - minY, 1e-6)) * (height - padding * 2);

    const labelMap = buildLabelMap(cluster);
    const resultMap = buildResultMap(results);
    state.clusterLabelMap = labelMap;
    state.clusterResultMap = resultMap;
    refs.clusterMap.innerHTML = `
        <svg class="cluster-svg" viewBox="0 0 ${width} ${height}">
            ${points.map((point, index) => `
                <g class="cluster-point ${getPointStateClass(cluster.image_names[index], labelMap[cluster.image_names[index]])}" data-filename="${cluster.image_names[index]}">
                    <circle cx="${scaleX(point[0])}" cy="${scaleY(point[1])}" r="${getPointRadius(cluster.image_names[index])}" fill="${clusterColor(labelMap[cluster.image_names[index]])}" />
                    <text x="${scaleX(point[0]) + 12}" y="${scaleY(point[1]) + 4}">${cluster.image_names[index]}</text>
                </g>
            `).join('')}
        </svg>
    `;

}

function renderDendrogram(cluster) {
    const data = cluster?.dendrogram;
    if (!data?.nodes?.length) {
        refs.clusterDendrogram.innerHTML = '<p class="cluster-detail__placeholder">暂无树状图数据。</p>';
        return;
    }

    const width = Math.max(320, refs.clusterDendrogram.clientWidth - 12);
    const height = 280;
    const maxX = Math.max(...data.nodes.map(node => node.x), 1);
    const maxY = Math.max(data.max_height || 1, 1);
    const scaleX = value => 24 + (value / maxX) * (width - 48);
    const scaleY = value => height - 30 - (value / maxY) * (height - 58);

    refs.clusterDendrogram.innerHTML = `
        <svg class="cluster-svg cluster-svg--dendrogram" viewBox="0 0 ${width} ${height}">
            ${data.links.map(link => `
                <path
                    class="dendrogram-link ${isDendrogramLinkActive(link, data) ? 'is-active' : ''}"
                    d="M ${scaleX(link.x1)} ${scaleY(link.y1)} V ${scaleY(link.y2)} H ${scaleX(link.x2)}"
                />
            `).join('')}
            ${data.nodes.map(node => `
                <g class="dendrogram-node ${getDendrogramNodeState(node)}" data-node-id="${node.id}">
                    <circle cx="${scaleX(node.x)}" cy="${scaleY(node.y)}" r="${node.height > 0 ? 6 : 4}"></circle>
                </g>
            `).join('')}
            ${data.leaves.map(leaf => `
                <text class="dendrogram-label" x="${scaleX(leaf.x)}" y="${height - 8}" text-anchor="middle">${leaf.name}</text>
            `).join('')}
        </svg>
    `;

    refreshDendrogramHighlight(data);
}

function syncBatchDetail(cluster, results) {
    void cluster;
    void results;
}

function toggleClusterComparison(clusterId) {
    const next = new Set(state.comparisonClusterIds);
    if (next.has(clusterId)) {
        next.delete(clusterId);
    } else {
        next.add(clusterId);
    }
    state.comparisonClusterIds = [...next];
    renderBatch(state.batchResult);
}

function exportCluster(clusterId) {
    const runId = state.batchRunId || state.batchResult?.run_id;
    if (!runId) {
        setStatus(I18N.t('exportFailedNoRunId'), 'error');
        renderStatusCard();
        return;
    }
    window.open(`/api/export-cluster/${runId}/${clusterId}`, '_blank', 'noopener');
}

function renderClusterCompare(cluster) {
    const selected = (cluster?.clusters || []).filter(item => state.comparisonClusterIds.includes(item.cluster_id));
    if (!selected.length) {
        refs.clusterCompare.classList.add('hidden');
        refs.clusterCompareSummary.innerHTML = '';
        refs.clusterCompareChart.innerHTML = '';
        refs.clusterCompareGallery.innerHTML = '';
        return;
    }

    refs.clusterCompare.classList.remove('hidden');
    const selectedCountText = I18N.t('selectedClusters', { count: selected.length });
    const clusterPillLabel = (item) => {
        const clusterLabel = I18N.lang === 'zh' ? `第 ${item.cluster_id + 1} 类` : I18N.t('clusterLabel', { n: item.cluster_id + 1 });
        const sampleText = I18N.lang === 'zh' ? `${item.sample_count} 个样本` : I18N.t('sampleCountText', { count: item.sample_count });
        return `${clusterLabel} · ${sampleText}`;
    };
    refs.clusterCompareSummary.innerHTML = `
        <div class="cluster-compare__summary-head">
            <div class="cluster-compare__summary-title">${selectedCountText}</div>
            <button class="ghost-btn cluster-compare__clear" id="clearCompareBtn">${I18N.t('clearCompare')}</button>
        </div>
        <div class="cluster-compare__summary-pills">
            ${selected.map(item => `
                <button class="cluster-compare__pill" data-cluster-id="${item.cluster_id}">${clusterPillLabel(item)} <span class="cluster-compare__pill-close" aria-hidden="true">×</span></button>
            `).join('')}
        </div>
    `;

    if (selected.length < 2) {
        refs.clusterCompareChart.innerHTML = `<div class="compare-empty">${I18N.t('compareNeedTwo')}</div>`;
        refs.clusterCompareGallery.innerHTML = '';
        return;
    }

    const spikeletMetrics = [
        { key: 'mean_spikelet_length_mm', label: I18N.t('metricSpikeletLength'), unit: 'mm', group: I18N.t('groupSpikelet'), groupClass: 'spikelet' },
        { key: 'mean_spikelet_width_mm', label: I18N.t('metricSpikeletWidth'), unit: 'mm', group: I18N.t('groupSpikelet'), groupClass: 'spikelet' },
        { key: 'mean_aspect_ratio', label: I18N.t('metricSpikeletAspectRatio'), unit: '', group: I18N.t('groupSpikelet'), groupClass: 'spikelet' },
        { key: 'mean_attachment_angle', label: I18N.t('metricAttachmentAngle'), unit: '°', group: I18N.t('groupSpikelet'), groupClass: 'spikelet' },
        { key: 'mean_hue_deg', label: I18N.t('metricMeanHue'), unit: '°', group: I18N.t('groupSpikelet'), groupClass: 'spikelet' },
        { key: 'mean_saturation', label: I18N.t('metricMeanSaturation'), unit: '', group: I18N.t('groupSpikelet'), groupClass: 'spikelet' },
        { key: 'std_hue', label: I18N.t('metricStdHue'), unit: '°', group: I18N.t('groupSpikelet'), groupClass: 'spikelet' },
    ];
    const earMetrics = [
        { key: 'mean_spike_length_cm', label: I18N.t('metricSpikeLength'), unit: 'cm', group: I18N.t('groupEar'), groupClass: 'ear' },
        { key: 'spikelet_count', label: I18N.t('metricSpikeletCount'), unit: '', group: I18N.t('groupEar'), groupClass: 'ear' },
        { key: 'spikelet_density', label: I18N.t('metricSpikeletDensity'), unit: '', group: I18N.t('groupEar'), groupClass: 'ear' },
        { key: 'mean_symmetry_index', label: I18N.t('metricSymmetry'), unit: '', group: I18N.t('groupEar'), groupClass: 'ear' },
        { key: 'mean_centroid_offset', label: I18N.t('metricCentroidOffset'), unit: '', group: I18N.t('groupEar'), groupClass: 'ear' },
    ];
    const metrics = [...spikeletMetrics, ...earMetrics];

    refs.clusterCompareChart.innerHTML = `
        <div class="compare-radar-grid">
            <div class="compare-radar-shell">
                <div class="compare-radar__title">${I18N.t('radarChartSpikelet')}</div>
                ${renderClusterRadar(selected, spikeletMetrics)}
            </div>
            <div class="compare-radar-shell">
                <div class="compare-radar__title">${I18N.t('radarChartEar')}</div>
                ${renderClusterRadar(selected, earMetrics)}
            </div>
        </div>
        <div class="compare-bars-shell">
            <div class="compare-bars-shell__title">${I18N.t('barChart')}</div>
            <div class="compare-bars-grid">
                ${metrics.map(metric => {
                    const values = selected.map(item => Number(item.aggregate_metrics?.[metric.key] ?? 0));
                    const minValue = Math.min(...values, 0);
                    const maxValue = Math.max(...values, 1e-6);
                    const range = Math.max(maxValue - minValue, 1e-6);
                    return `
                        <div class="compare-metric compare-metric--compact">
                            <div class="compare-metric__head">
                                <div class="compare-metric__title">${metric.label}</div>
                                <div class="compare-metric__group compare-metric__group--${metric.groupClass}">${metric.group}</div>
                            </div>
                            ${selected.map(item => {
                                const value = Number(item.aggregate_metrics?.[metric.key] ?? 0);
                                const normalizedWidth = ((value - minValue) / range) * 100;
                                const barClusterLabel = I18N.lang === 'zh' ? `第 ${item.cluster_id + 1} 类` : I18N.t('clusterLabel', { n: item.cluster_id + 1 });
                                return `
                                    <div class="compare-bar">
                                        <div class="compare-bar__label"><i class="compare-bar__dot" style="background:${clusterColor(item.cluster_id)}"></i>${barClusterLabel}</div>
                                        <div class="compare-bar__track">
                                            <span style="width:${normalizedWidth}%"></span>
                                        </div>
                                        <div class="compare-bar__value">${formatClusterMetric(value, metric.unit)}</div>
                                    </div>
                                `;
                            }).join('')}
                        </div>
                    `;
                }).join('')}
            </div>
        </div>
    `;

    refs.clusterCompareGallery.innerHTML = '';
}

function saveBatchSession(extra = {}) {
    if (state.mode !== 'batch' || !state.batchRunId) {
        return;
    }
    const snapshot = {
        mode: 'batch',
        runId: state.batchRunId,
        state: extra.state || 'running',
        progress: state.batchProgress,
        batchSortMetric: state.batchSortMetric,
        batchFilterMetric: state.batchFilterMetric,
        batchFilterValue: state.batchFilterValue,
        batchHideUnmatched: state.batchHideUnmatched,
        savedAt: Date.now(),
    };
    try {
        window.sessionStorage.setItem(BATCH_SESSION_KEY, JSON.stringify(snapshot));
    } catch (error) {
        void error;
    }
}

function clearBatchSession() {
    try {
        window.sessionStorage.removeItem(BATCH_SESSION_KEY);
    } catch (error) {
        void error;
    }
}

function loadBatchSession() {
    try {
        const raw = window.sessionStorage.getItem(BATCH_SESSION_KEY);
        if (!raw) {
            return null;
        }
        return JSON.parse(raw);
    } catch (error) {
        void error;
        return null;
    }
}

async function restoreBatchSession() {
    const snapshot = loadBatchSession();
    if (!snapshot?.runId) {
        return;
    }

    if (snapshot.mode === 'batch' && state.mode !== 'batch') {
        state.mode = 'batch';
        renderMode({ instant: true });
        renderFiles();
    }

    state.batchRunId = snapshot.runId;
    state.batchSortMetric = snapshot.batchSortMetric || state.batchSortMetric;
    state.batchFilterMetric = snapshot.batchFilterMetric || state.batchFilterMetric;
    state.batchFilterValue = snapshot.batchFilterValue || '';
    state.batchHideUnmatched = Boolean(snapshot.batchHideUnmatched);
    refs.clusterFilterValue.value = state.batchFilterValue;
    refs.clusterHideUnmatched.checked = state.batchHideUnmatched;

    setStatus(I18N.t('restoringBatch'), 'running');
    renderStatusCard();

    try {
        const response = await fetch(`/api/batch-status/${snapshot.runId}`);
        const status = await response.json();
        if (!response.ok || status.error) {
            if (status?.code === 'job_expired' || status?.code === 'job_not_found') {
                clearBatchSession();
                state.batchRunId = null;
                state.batchProgress = null;
                setStatus(I18N.t('taskExpired'), 'idle');
                renderStatusCard();
                return;
            }
            throw new Error(status.error || I18N.t('errorTaskUnavailable'));
        }
        updateBatchProgress(status);
        if (status.state === 'completed') {
            await fetchBatchResult(snapshot.runId);
            return;
        }
        if (status.state === 'error') {
            setStatus(I18N.t('lastBatchFailed', { error: status.error || I18N.t('errorLastBatchFailed') }), 'error');
            renderStatusCard();
            saveBatchSession({ state: 'error' });
            return;
        }
        startBatchPolling(snapshot.runId);
    } catch (error) {
        clearBatchSession();
        state.batchRunId = null;
        state.batchProgress = null;
        setStatus(I18N.t('restoreFailed', { error: error.message }), 'error');
        renderStatusCard();
    }
}

function isRealPointerBoundary(target, relatedTarget) {
    return !relatedTarget || (relatedTarget !== target && !target.contains(relatedTarget));
}

function handleClusterCardMouseOver(event) {
    const card = event.target.closest('.cluster-card');
    if (!card || !isRealPointerBoundary(card, event.relatedTarget)) {
        return;
    }
    const clusterId = Number.parseInt(card.dataset.clusterId, 10);
    if (!Number.isInteger(clusterId)) {
        return;
    }
    state.hoveredClusterId = clusterId;
    hideClusterHoverCard();
    refreshDendrogramHighlight(state.batchResult?.cluster?.dendrogram);
}

function handleClusterCardMouseOut(event) {
    const card = event.target.closest('.cluster-card');
    if (!card || !isRealPointerBoundary(card, event.relatedTarget)) {
        return;
    }
    state.hoveredClusterId = null;
    hideClusterHoverCard();
    refreshDendrogramHighlight(state.batchResult?.cluster?.dendrogram);
}

function handleClusterCardClick(event) {
    const actionButton = event.target.closest('.cluster-card__action');
    if (actionButton) {
        event.stopPropagation();
        const clusterId = Number.parseInt(actionButton.dataset.clusterId, 10);
        if (!Number.isInteger(clusterId)) {
            return;
        }
        if (actionButton.dataset.action === 'compare') {
            toggleClusterComparison(clusterId);
            return;
        }
        if (actionButton.dataset.action === 'export') {
            exportCluster(clusterId);
        }
        return;
    }

    const card = event.target.closest('.cluster-card');
    if (!card) {
        return;
    }
    const clusterId = Number.parseInt(card.dataset.clusterId, 10);
    if (!Number.isInteger(clusterId)) {
        return;
    }
    openClusterModal(clusterId);
}

function handleClusterMapMouseOver(event) {
    const point = event.target.closest('.cluster-point');
    if (!point || !isRealPointerBoundary(point, event.relatedTarget)) {
        return;
    }
    const filename = point.dataset.filename;
    state.hoveredSampleName = filename;
    refreshDendrogramHighlight(state.batchResult?.cluster?.dendrogram);
}

function handleClusterMapMouseMove(event) {
    const point = event.target.closest('.cluster-point');
    if (!point) {
        return;
    }
    const filename = point.dataset.filename;
    showClusterHoverCardForSample(event, state.clusterResultMap[filename], state.clusterLabelMap[filename]);
}

function handleClusterMapMouseOut(event) {
    const point = event.target.closest('.cluster-point');
    if (!point || !isRealPointerBoundary(point, event.relatedTarget)) {
        return;
    }
    state.hoveredSampleName = null;
    hideClusterHoverCard();
    refreshDendrogramHighlight(state.batchResult?.cluster?.dendrogram);
}

function handleClusterMapClick(event) {
    const point = event.target.closest('.cluster-point');
    if (!point) {
        return;
    }
    const filename = point.dataset.filename;
    state.hoveredSampleName = null;
    hideClusterHoverCard();
    const nextClusterId = state.clusterLabelMap[filename];
    if (state.selectedSampleName === filename) {
        state.selectedSampleName = null;
        state.selectedClusterId = null;
    } else {
        state.selectedSampleName = filename;
        state.selectedClusterId = nextClusterId;
    }
    renderBatch(state.batchResult);
}

function handleDendrogramMouseOver(event) {
    const node = event.target.closest('.dendrogram-node');
    if (!node || !isRealPointerBoundary(node, event.relatedTarget)) {
        return;
    }
    const nodeId = Number.parseInt(node.dataset.nodeId, 10);
    if (!Number.isInteger(nodeId)) {
        return;
    }
    state.hoveredDendrogramNodeId = nodeId;
    refreshDendrogramHighlight(state.batchResult?.cluster?.dendrogram);
}

function handleDendrogramMouseMove(event) {
    const node = event.target.closest('.dendrogram-node');
    if (!node) {
        return;
    }
    const nodeId = Number.parseInt(node.dataset.nodeId, 10);
    if (!Number.isInteger(nodeId)) {
        return;
    }
    showClusterHoverCardForDendrogramNode(event, state.batchResult?.cluster, nodeId);
}

function handleDendrogramMouseOut(event) {
    const node = event.target.closest('.dendrogram-node');
    if (!node || !isRealPointerBoundary(node, event.relatedTarget)) {
        return;
    }
    state.hoveredDendrogramNodeId = null;
    hideClusterHoverCard();
    refreshDendrogramHighlight(state.batchResult?.cluster?.dendrogram);
}

function handleCompareSummaryClick(event) {
    const clearButton = event.target.closest('#clearCompareBtn');
    if (clearButton) {
        state.comparisonClusterIds = [];
        renderBatch(state.batchResult);
        return;
    }
    const pill = event.target.closest('.cluster-compare__pill');
    if (!pill) {
        return;
    }
    const clusterId = Number.parseInt(pill.dataset.clusterId, 10);
    if (!Number.isInteger(clusterId)) {
        return;
    }
    toggleClusterComparison(clusterId);
}

function renderClusterRadar(selected, metrics) {
    const width = 360;
    const height = 320;
    const cx = width / 2;
    const cy = height / 2;
    const radius = 108;
    const rings = [0.25, 0.5, 0.75, 1];
    const angleStep = (Math.PI * 2) / metrics.length;
    const series = selected.map(item => {
        const points = metrics.map((metric, index) => {
            const values = selected.map(target => Number(target.aggregate_metrics?.[metric.key] ?? 0));
            const minValue = Math.min(...values, 0);
            const maxValue = Math.max(...values, 1e-6);
            const range = Math.max(maxValue - minValue, 1e-6);
            const value = (Number(item.aggregate_metrics?.[metric.key] ?? 0) - minValue) / range;
            const angle = -Math.PI / 2 + angleStep * index;
            const px = cx + Math.cos(angle) * radius * value;
            const py = cy + Math.sin(angle) * radius * value;
            return `${px},${py}`;
        });
        return {
            clusterId: item.cluster_id,
            color: clusterColor(item.cluster_id),
            points: points.join(' '),
        };
    });

    return `
        <svg class="compare-radar" viewBox="0 0 ${width} ${height}">
            ${rings.map(ratio => `
                <polygon
                    class="compare-radar__ring"
                    points="${metrics.map((metric, index) => {
                        const angle = -Math.PI / 2 + angleStep * index;
                        const px = cx + Math.cos(angle) * radius * ratio;
                        const py = cy + Math.sin(angle) * radius * ratio;
                        return `${px},${py}`;
                    }).join(' ')}"
                ></polygon>
            `).join('')}
            ${metrics.map((metric, index) => {
                const angle = -Math.PI / 2 + angleStep * index;
                const px = cx + Math.cos(angle) * radius;
                const py = cy + Math.sin(angle) * radius;
                const lx = cx + Math.cos(angle) * (radius + 26);
                const ly = cy + Math.sin(angle) * (radius + 20);
                return `
                    <line class="compare-radar__axis" x1="${cx}" y1="${cy}" x2="${px}" y2="${py}"></line>
                    <text class="compare-radar__label" x="${lx}" y="${ly}" text-anchor="middle">${metric.label}</text>
                `;
            }).join('')}
            ${series.map(item => `
                <polygon
                    class="compare-radar__shape"
                    points="${item.points}"
                    fill="${item.color}22"
                    stroke="${item.color}"
                ></polygon>
            `).join('')}
        </svg>
        <div class="compare-radar__legend">
            ${selected.map(item => `<span><i style="background:${clusterColor(item.cluster_id)}"></i>${I18N.t('clusterLabel', { n: item.cluster_id + 1 })}</span>`).join('')}
        </div>
    `;
}

function buildLabelMap(cluster) {
    if (!cluster) {
        return {};
    }
    const map = {};
    cluster.image_names.forEach((name, index) => {
        map[name] = cluster.labels[index];
    });
    return map;
}

function clusterColor(label) {
    const palette = ['#3cf2ff', '#78ffb8', '#ffbb54', '#ff7d92', '#a48cff', '#7ce0ff'];
    return palette[(label ?? 0) % palette.length];
}

function buildResultMap(results) {
    return Object.fromEntries(results.map(item => [item.image_name || item.filename, item]));
}

function getClusterMetricOptions(cluster) {
    const preferredMetrics = [
        { key: 'mean_spikelet_length_mm', labelKey: 'metricSpikeletLength', group: 'spikelet', groupLabelKey: 'groupSpikelet' },
        { key: 'mean_spikelet_width_mm', labelKey: 'metricSpikeletWidth', group: 'spikelet', groupLabelKey: 'groupSpikelet' },
        { key: 'mean_aspect_ratio', labelKey: 'metricSpikeletAspectRatio', group: 'spikelet', groupLabelKey: 'groupSpikelet' },
        { key: 'mean_attachment_angle', labelKey: 'metricAttachmentAngle', group: 'spikelet', groupLabelKey: 'groupSpikelet' },
        { key: 'mean_hue_deg', labelKey: 'metricMeanHue', group: 'spikelet', groupLabelKey: 'groupSpikelet' },
        { key: 'mean_saturation', labelKey: 'metricMeanSaturation', group: 'spikelet', groupLabelKey: 'groupSpikelet' },
        { key: 'std_hue', labelKey: 'metricStdHue', group: 'spikelet', groupLabelKey: 'groupSpikelet' },
        { key: 'mean_spike_length_cm', labelKey: 'metricSpikeLength', group: 'ear', groupLabelKey: 'groupEar' },
        { key: 'spikelet_count', labelKey: 'metricSpikeletCount', group: 'ear', groupLabelKey: 'groupEar' },
        { key: 'spikelet_density', labelKey: 'metricSpikeletDensity', group: 'ear', groupLabelKey: 'groupEar' },
        { key: 'mean_symmetry_index', labelKey: 'metricSymmetry', group: 'ear', groupLabelKey: 'groupEar' },
        { key: 'mean_centroid_offset', labelKey: 'metricCentroidOffset', group: 'ear', groupLabelKey: 'groupEar' },
    ].map(m => ({
        ...m,
        label: I18N.t(m.labelKey),
        groupLabel: I18N.t(m.groupLabelKey)
    }));
    const availableMetrics = cluster?.clusters?.[0]?.aggregate_metrics || {};
    return preferredMetrics.filter(metric => Object.prototype.hasOwnProperty.call(availableMetrics, metric.key));
}

function renderMetricCascadeControls() {
    refs.clusterSortMenu.innerHTML = buildMetricCascadeMenuHtml('sort', state.batchSortMetric);
    refs.clusterFilterMenu.innerHTML = buildMetricCascadeMenuHtml('filter', state.batchFilterMetric);
    updateMetricCascadeTriggerLabel('sort');
    updateMetricCascadeTriggerLabel('filter');
}

function buildMetricCascadeMenuHtml(target, selectedMetricKey) {
    const groups = [];
    const seen = new Set();
    state.clusterMetricOptions.forEach(metric => {
        if (!seen.has(metric.group)) {
            seen.add(metric.group);
            groups.push(metric.group);
        }
    });

    return groups.map(group => {
        const groupMetrics = state.clusterMetricOptions.filter(metric => metric.group === group);
        const groupLabel = groupMetrics[0]?.groupLabel || group;
        const hasSelected = groupMetrics.some(metric => metric.key === selectedMetricKey);
        return `
            <div class="cascade-menu__group ${hasSelected ? 'is-current' : ''}">
                <button type="button" class="cascade-menu__group-btn">${groupLabel}</button>
                <div class="cascade-menu__submenu">
                    ${groupMetrics.map(metric => `
                        <button
                            type="button"
                            class="cascade-menu__item ${metric.key === selectedMetricKey ? 'is-selected' : ''}"
                            data-target="${target}"
                            data-metric-key="${metric.key}"
                        >
                            ${metric.label}
                        </button>
                    `).join('')}
                </div>
            </div>
        `;
    }).join('');
}

function handleMetricCascadeSelect(event, target) {
    const item = event.target.closest('.cascade-menu__item');
    if (!item) {
        return false;
    }
    const metricKey = item.dataset.metricKey;
    if (!metricKey) {
        return false;
    }
    if (target === 'sort') {
        state.batchSortMetric = metricKey;
    } else {
        state.batchFilterMetric = metricKey;
    }
    renderMetricCascadeControls();
    closeMetricCascadeMenus();
    return true;
}

function updateMetricCascadeTriggerLabel(target) {
    const metricKey = target === 'sort' ? state.batchSortMetric : state.batchFilterMetric;
    const metric = state.clusterMetricOptions.find(item => item.key === metricKey);
    const trigger = target === 'sort' ? refs.clusterSortTrigger : refs.clusterFilterTrigger;
    const prefix = target === 'sort' ? I18N.t('sortBy') : I18N.t('filterBy');
    const groupLabel = metric?.groupLabel || I18N.t('selectGroup');
    const label = metric?.label || I18N.t('selectMetric');
    trigger.innerHTML = `
        <span class="cascade-select__prefix">${prefix}</span>
        <span class="cascade-select__value">${groupLabel} / ${label}</span>
        <span class="cascade-select__chevron">▾</span>
    `;
}

function toggleMetricCascadeMenu(target) {
    const menu = target === 'sort' ? refs.clusterSortMenu : refs.clusterFilterMenu;
    const otherMenu = target === 'sort' ? refs.clusterFilterMenu : refs.clusterSortMenu;
    const shouldOpen = menu.classList.contains('hidden');
    otherMenu.classList.add('hidden');
    menu.classList.toggle('hidden', !shouldOpen);
}

function closeMetricCascadeMenus() {
    refs.clusterSortMenu.classList.add('hidden');
    refs.clusterFilterMenu.classList.add('hidden');
}

function getMetricLabel(key) {
    const labelKeyMap = {
        spikelet_count: 'metricSpikeletCount',
        mean_spikelet_length: 'metricSpikeletLength',
        mean_spikelet_width: 'metricSpikeletWidth',
        mean_aspect_ratio: 'metricSpikeletAspectRatio',
        mean_attachment_angle: 'metricAttachmentAngle',
        spike_length: 'metricSpikeLength',
        spikelet_density: 'metricSpikeletDensity',
        symmetry_index: 'metricSymmetry',
        centroid_offset: 'metricCentroidOffset',
        spike_length_cm: 'metricSpikeLength',
        mean_spike_length_cm: 'metricSpikeLength',
        mean_spikelet_length_mm: 'metricSpikeletLength',
        mean_spikelet_width_mm: 'metricSpikeletWidth',
        mean_symmetry_index: 'metricSymmetry',
        mean_centroid_offset: 'metricCentroidOffset',
        mean_hue_deg: 'metricMeanHue',
        mean_saturation: 'metricMeanSaturation',
        std_hue: 'metricStdHue',
    };
    const labelKey = labelKeyMap[key];
    return labelKey ? I18N.t(labelKey) : null;
}

function getVisibleClusters(cluster) {
    const clusters = [...(cluster?.clusters || [])];
    const metric = state.batchSortMetric;
    const filterMetric = state.batchFilterMetric;
    const threshold = Number.parseFloat(state.batchFilterValue);
    const hasThreshold = Number.isFinite(threshold);

    const decorated = clusters.map(item => {
        const value = Number(item.aggregate_metrics?.[filterMetric]);
        const matched = !hasThreshold || (Number.isFinite(value) && value >= threshold);
        return { ...item, __matched: matched };
    });

    decorated.sort((left, right) => Number(right.aggregate_metrics?.[metric] ?? -Infinity) - Number(left.aggregate_metrics?.[metric] ?? -Infinity));
    return state.batchHideUnmatched ? decorated.filter(item => item.__matched) : decorated;
}

function formatClusterMetric(value, unit = '') {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
        return 'N/A';
    }
    return `${Number(value).toFixed(2)}${unit ? ` ${unit}` : ''}`;
}

function buildNineSampleMetricRows(ear) {
    const safeEar = ear || {};
    return [
        { group: I18N.t('groupSpikelet'), label: I18N.t('metricSpikeletLength'), value: formatMetric(safeEar.mean_spikelet_length_mm, 'mm', safeEar.mean_spikelet_length ?? 0, 'px') },
        { group: I18N.t('groupSpikelet'), label: I18N.t('metricSpikeletWidth'), value: formatMetric(safeEar.mean_spikelet_width_mm, 'mm', safeEar.mean_spikelet_width ?? 0, 'px') },
        { group: I18N.t('groupSpikelet'), label: I18N.t('metricSpikeletAspectRatio'), value: Number(safeEar.mean_aspect_ratio ?? 0).toFixed(3) },
        { group: I18N.t('groupSpikelet'), label: I18N.t('metricAttachmentAngle'), value: `${Number(safeEar.mean_attachment_angle ?? 0).toFixed(2)} °` },
        { group: I18N.t('groupSpikelet'), label: I18N.t('metricMeanHue'), value: `${Number(safeEar.mean_hue_deg ?? 0).toFixed(2)} °` },
        { group: I18N.t('groupSpikelet'), label: I18N.t('metricMeanSaturation'), value: Number(safeEar.mean_saturation ?? 0).toFixed(2) },
        { group: I18N.t('groupSpikelet'), label: I18N.t('metricStdHue'), value: Number(safeEar.std_hue ?? 0).toFixed(2) },
        { group: I18N.t('groupEar'), label: I18N.t('metricSpikeLength'), value: formatMetric(safeEar.spike_length_cm, 'cm', safeEar.spike_length_px ?? 0, 'px') },
        { group: I18N.t('groupEar'), label: I18N.t('metricSpikeletCount'), value: String(Math.round(Number(safeEar.spikelet_count ?? 0))) },
        { group: I18N.t('groupEar'), label: I18N.t('metricSpikeletDensity'), value: formatMetric(safeEar.spikelet_density_per_cm, '/cm', safeEar.spikelet_density_px ?? 0, '/px') },
        { group: I18N.t('groupEar'), label: I18N.t('metricSymmetry'), value: Number(safeEar.symmetry_index ?? 0).toFixed(4) },
        { group: I18N.t('groupEar'), label: I18N.t('metricCentroidOffset'), value: Number(safeEar.centroid_offset ?? 0).toFixed(4) },
    ];
}

function buildNineClusterMetricRows(clusterItem) {
    const metrics = clusterItem?.aggregate_metrics || {};
    return [
        { group: I18N.t('groupSpikelet'), label: I18N.t('metricSpikeletLength'), value: formatClusterMetric(metrics.mean_spikelet_length_mm, 'mm') },
        { group: I18N.t('groupSpikelet'), label: I18N.t('metricSpikeletWidth'), value: formatClusterMetric(metrics.mean_spikelet_width_mm, 'mm') },
        { group: I18N.t('groupSpikelet'), label: I18N.t('metricSpikeletAspectRatio'), value: formatClusterMetric(metrics.mean_aspect_ratio, '') },
        { group: I18N.t('groupSpikelet'), label: I18N.t('metricAttachmentAngle'), value: formatClusterMetric(metrics.mean_attachment_angle, '°') },
        { group: I18N.t('groupSpikelet'), label: I18N.t('metricMeanHue'), value: formatClusterMetric(metrics.mean_hue_deg, '°') },
        { group: I18N.t('groupSpikelet'), label: I18N.t('metricMeanSaturation'), value: formatClusterMetric(metrics.mean_saturation, '') },
        { group: I18N.t('groupSpikelet'), label: I18N.t('metricStdHue'), value: formatClusterMetric(metrics.std_hue, '°') },
        { group: I18N.t('groupEar'), label: I18N.t('metricSpikeLength'), value: formatClusterMetric(metrics.mean_spike_length_cm, 'cm') },
        { group: I18N.t('groupEar'), label: I18N.t('metricSpikeletCount'), value: formatClusterMetric(metrics.spikelet_count, '') },
        { group: I18N.t('groupEar'), label: I18N.t('metricSpikeletDensity'), value: formatClusterMetric(metrics.spikelet_density, '') },
        { group: I18N.t('groupEar'), label: I18N.t('metricSymmetry'), value: formatClusterMetric(metrics.mean_symmetry_index, '') },
        { group: I18N.t('groupEar'), label: I18N.t('metricCentroidOffset'), value: formatClusterMetric(metrics.mean_centroid_offset, '') },
    ];
}

function renderGroupedMetricCards(rows) {
    const spikeletRows = rows.filter(item => item.group === I18N.t('groupSpikelet'));
    const earRows = rows.filter(item => item.group === I18N.t('groupEar'));
    const renderSection = (title, groupClass, items) => `
        <section class="cluster-modal__section cluster-modal__section--${groupClass}">
            <h4>${title}</h4>
            <div class="cluster-modal__metrics-grid">
                ${items.map(item => `
                    <div class="cluster-modal__metric-card">
                        <div class="cluster-modal__metric-label">${item.label}</div>
                        <div class="cluster-modal__metric-value">${item.value}</div>
                    </div>
                `).join('')}
            </div>
        </section>
    `;

    return `${renderSection(I18N.t('groupSpikelet'), 'spikelet', spikeletRows)}${renderSection(I18N.t('groupEar'), 'ear', earRows)}`;
}

function renderHoverMetricRows(rows) {
    return rows.map(item => `
        <div class="cluster-hover-card__kv">
            <span class="cluster-hover-card__kv-label">${item.label}</span>
            <strong class="cluster-hover-card__kv-value">${item.value}</strong>
        </div>
    `).join('');
}

function showClusterHoverCardForSample(event, result, clusterLabel, options = {}) {
    if (!result) {
        hideClusterHoverCard();
        return;
    }
    const rows = buildNineSampleMetricRows(result.ear_pheno || {});
    const spikeletRows = rows.filter(item => item.group === I18N.t('groupSpikelet'));
    const earRows = rows.filter(item => item.group === I18N.t('groupEar'));
    const showImage = options.showImage !== false;
    const imageUrl = result.images?.original || result.images?.analysis || '';

    const displayClusterLabel = Number.isFinite(Number(clusterLabel)) ? Number(clusterLabel) + 1 : (clusterLabel ?? '-');
    if (showImage && imageUrl) {
        refs.clusterHoverCard.innerHTML = `
            <div class="cluster-hover-card__layout">
                <div class="cluster-hover-card__body cluster-hover-card__body--metrics-only">
                    <h4>${result.image_name || result.filename}</h4>
                    <p>${I18N.t('clusterLabel', { n: displayClusterLabel })}</p>
                    <div class="cluster-hover-card__group">${I18N.t('groupSpikelet')}</div>
                    ${renderHoverMetricRows(spikeletRows)}
                    <div class="cluster-hover-card__group">${I18N.t('groupEar')}</div>
                    ${renderHoverMetricRows(earRows)}
                </div>
                <div class="cluster-hover-card__image-wrap"><img class="cluster-hover-card__image" src="${imageUrl}" alt="${result.image_name || result.filename}"></div>
            </div>
        `;
        refs.clusterHoverCard.classList.add('cluster-hover-card--with-image');
    } else {
        refs.clusterHoverCard.innerHTML = `
            <div class="cluster-hover-card__body cluster-hover-card__body--metrics-only">
                <h4>${result.image_name || result.filename}</h4>
                <p>${I18N.t('clusterLabel', { n: displayClusterLabel })}</p>
                <div class="cluster-hover-card__group">${I18N.t('groupSpikelet')}</div>
                ${renderHoverMetricRows(spikeletRows)}
                <div class="cluster-hover-card__group">${I18N.t('groupEar')}</div>
                ${renderHoverMetricRows(earRows)}
            </div>
        `;
        refs.clusterHoverCard.classList.remove('cluster-hover-card--with-image');
    }
    moveClusterHoverCard(event);
    refs.clusterHoverCard.classList.remove('hidden');
}

function showClusterHoverCardForCluster(event, clusterItem) {
    if (!clusterItem) {
        hideClusterHoverCard();
        return;
    }

    const rows = buildNineClusterMetricRows(clusterItem);
    const spikeletRows = rows.filter(item => item.group === I18N.t('groupSpikelet'));
    const earRows = rows.filter(item => item.group === I18N.t('groupEar'));

    refs.clusterHoverCard.innerHTML = `
        <div class="cluster-hover-card__body cluster-hover-card__body--metrics-only">
            <h4>${I18N.t('clusterLabel', { n: clusterItem.cluster_id + 1 })}</h4>
            <p>${I18N.t('clusterSamplesCount', { count: clusterItem.sample_count })}</p>
            <div class="cluster-hover-card__group">${I18N.t('groupSpikelet')}</div>
            ${renderHoverMetricRows(spikeletRows)}
            <div class="cluster-hover-card__group">${I18N.t('groupEar')}</div>
            ${renderHoverMetricRows(earRows)}
        </div>
    `;
    refs.clusterHoverCard.classList.remove('cluster-hover-card--with-image');
    moveClusterHoverCard(event);
    refs.clusterHoverCard.classList.remove('hidden');
}

function showClusterHoverCardForDendrogramNode(event, cluster, nodeId) {
    const node = cluster?.dendrogram?.nodes?.find(item => item.id === nodeId);
    if (!node || !node.sample_names?.length) {
        hideClusterHoverCard();
        return;
    }
    const resultMap = buildResultMap(state.batchResult?.results || []);
    const sample = resultMap[node.sample_names[0]];
    if (!sample) {
        hideClusterHoverCard();
        return;
    }
    const extra = node.sample_names.length > 1 ? I18N.t('extraSamples', { count: node.sample_names.length - 1 }) : '';
    const rows = buildNineSampleMetricRows(sample.ear_pheno || {});
    const spikeletRows = rows.filter(item => item.group === I18N.t('groupSpikelet'));
    const earRows = rows.filter(item => item.group === I18N.t('groupEar'));
    const imageUrl = sample.images?.original || sample.images?.analysis || '';
    refs.clusterHoverCard.innerHTML = `
        <div class="cluster-hover-card__layout">
            <div class="cluster-hover-card__body cluster-hover-card__body--metrics-only">
                <h4>${sample.image_name || sample.filename}${extra}</h4>
                <p>${I18N.t('treeNodeCoverSamples', { count: node.sample_names.length })}</p>
                <div class="cluster-hover-card__group">${I18N.t('groupSpikelet')}</div>
                ${renderHoverMetricRows(spikeletRows)}
                <div class="cluster-hover-card__group">${I18N.t('groupEar')}</div>
                ${renderHoverMetricRows(earRows)}
            </div>
            ${imageUrl ? `<div class="cluster-hover-card__image-wrap"><img class="cluster-hover-card__image" src="${imageUrl}" alt="${sample.image_name || sample.filename}"></div>` : ''}
        </div>
    `;
    refs.clusterHoverCard.classList.add('cluster-hover-card--with-image');
    moveClusterHoverCard(event);
    refs.clusterHoverCard.classList.remove('hidden');
}

function moveClusterHoverCard(event) {
    const cardWidth = refs.clusterHoverCard.offsetWidth || 300;
    const cardHeight = refs.clusterHoverCard.offsetHeight || 220;
    const x = Math.min(window.innerWidth - cardWidth - 16, event.clientX + 18);
    const y = Math.min(window.innerHeight - cardHeight - 16, event.clientY + 18);
    refs.clusterHoverCard.style.left = `${Math.max(16, x)}px`;
    refs.clusterHoverCard.style.top = `${Math.max(16, y)}px`;
}

function hideClusterHoverCard() {
    refs.clusterHoverCard.classList.remove('cluster-hover-card--with-image');
    refs.clusterHoverCard.classList.add('hidden');
}

function getPointStateClass(filename, clusterId) {
    const activeNames = new Set(getSampleNamesForHoveredDendrogramNode(state.batchResult?.cluster));
    const isActiveSample = filename === state.hoveredSampleName || filename === state.selectedSampleName || activeNames.has(filename);
    const isActiveCluster = clusterId === state.hoveredClusterId || clusterId === state.selectedClusterId;
    if (isActiveSample) {
        return 'is-active';
    }
    if (isActiveCluster) {
        return 'is-cluster-active';
    }
    return '';
}

function getPointRadius(filename) {
    return (filename === state.hoveredSampleName || filename === state.selectedSampleName) ? 12 : 8;
}

function getSampleNamesForHoveredDendrogramNode(cluster) {
    const nodeId = state.hoveredDendrogramNodeId;
    if (nodeId === null || nodeId === undefined) {
        return [];
    }
    const node = cluster?.dendrogram?.nodes?.find(item => item.id === nodeId);
    return node?.sample_names || [];
}

function getDendrogramNodeState(node) {
    const activeNames = new Set(getSampleNamesForHoveredDendrogramNode(state.batchResult?.cluster));
    const clusterNames = new Set((state.batchResult?.cluster?.clusters || [])
        .find(item => item.cluster_id === (state.hoveredClusterId ?? state.selectedClusterId))?.sample_names || []);
    const nodeNames = new Set(node.sample_names || []);
    const isDendrogramActive = node.id === state.hoveredDendrogramNodeId;
    const overlapsHovered = [...nodeNames].some(name => activeNames.has(name));
    const overlapsCluster = [...nodeNames].some(name => clusterNames.has(name));
    if (isDendrogramActive || overlapsHovered) {
        return 'is-active';
    }
    if (overlapsCluster) {
        return 'is-cluster-active';
    }
    return '';
}

function isDendrogramLinkActive(link, data) {
    const activeNames = getActiveDendrogramNames(data);
    if (!activeNames.size) {
        return false;
    }
    const child = data.nodes.find(item => item.id === link.child);
    return (child?.sample_names || []).some(name => activeNames.has(name));
}

function getActiveDendrogramNames(data) {
    const names = new Set();

    if (state.hoveredSampleName) {
        names.add(state.hoveredSampleName);
    }
    if (state.selectedSampleName) {
        names.add(state.selectedSampleName);
    }

    const hoveredNode = data?.nodes?.find(item => item.id === state.hoveredDendrogramNodeId);
    (hoveredNode?.sample_names || []).forEach(name => names.add(name));

    const activeClusterId = state.hoveredClusterId ?? state.selectedClusterId;
    const activeCluster = (state.batchResult?.cluster?.clusters || []).find(item => item.cluster_id === activeClusterId);
    (activeCluster?.sample_names || []).forEach(name => names.add(name));

    return names;
}

function refreshDendrogramHighlight(data) {
    if (!refs.clusterDendrogram || !data?.nodes) {
        return;
    }
    refs.clusterDendrogram.querySelectorAll('.dendrogram-link').forEach((element, index) => {
        const link = data?.links?.[index];
        element.classList.toggle('is-active', Boolean(link) && isDendrogramLinkActive(link, data));
    });
    refs.clusterDendrogram.querySelectorAll('.dendrogram-node').forEach(nodeElement => {
        const nodeId = Number.parseInt(nodeElement.dataset.nodeId, 10);
        const node = data.nodes.find(item => item.id === nodeId);
        const stateClass = node ? getDendrogramNodeState(node) : '';
        nodeElement.classList.remove('is-active', 'is-cluster-active');
        if (stateClass) {
            nodeElement.classList.add(stateClass);
        }
    });
}

function openClusterModal(clusterId) {
    const cluster = state.batchResult?.cluster?.clusters?.find(item => item.cluster_id === clusterId);
    if (!cluster) {
        return;
    }
    state.selectedClusterId = clusterId;
    refs.clusterModalTitle.textContent = I18N.t('clusterDetailTitleWithId', { n: clusterId + 1 });
    refs.clusterModalSummary.innerHTML = renderGroupedMetricCards(buildNineClusterMetricRows(cluster));
    refs.clusterModalGrid.innerHTML = cluster.samples.map(sample => `
        <button class="cluster-thumb" data-sample-name="${sample.image_name || sample.filename}" data-image-url="${sample.images?.original || sample.images?.analysis || ''}">
            <img src="${sample.images?.analysis || sample.images?.original || ''}" alt="${sample.image_name || sample.filename}">
            <span>${sample.image_name || sample.filename}</span>
        </button>
    `).join('');
    refs.clusterModalGrid.querySelectorAll('.cluster-thumb').forEach(button => {
        const sampleName = button.dataset.sampleName;
        button.addEventListener('mouseenter', () => {
            state.hoveredSampleName = sampleName;
            syncBatchDetail(state.batchResult.cluster, state.batchResult.results);
            renderBatch(state.batchResult);
        });
        button.addEventListener('mousemove', (event) => {
            const resultMap = buildResultMap(state.batchResult?.results || []);
            const sample = resultMap[sampleName];
            const labelMap = buildLabelMap(state.batchResult?.cluster);
            showClusterHoverCardForSample(event, sample, labelMap[sampleName], { showImage: false });
        });
        button.addEventListener('mouseleave', () => {
            state.hoveredSampleName = null;
            syncBatchDetail(state.batchResult.cluster, state.batchResult.results);
            hideClusterHoverCard();
            renderBatch(state.batchResult);
        });
        button.addEventListener('click', () => {
            openImagePreview(sampleName, button.dataset.imageUrl);
        });
    });
    refs.clusterModal.classList.remove('hidden');
    document.body.classList.add('modal-open');
    renderBatch(state.batchResult);
}

function closeClusterModal() {
    refs.clusterModal.classList.add('hidden');
    hideClusterHoverCard();
    if (!refs.previewModal || refs.previewModal.classList.contains('hidden')) {
        document.body.classList.remove('modal-open');
    }
}

