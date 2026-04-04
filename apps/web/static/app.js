const state = {
    mode: 'single',
    files: [],
    previewUrls: [],
    singleResult: null,
    batchResult: null,
    batchRunId: null,
    batchStatusPoller: null,
    statusText: '等待上传',
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
        if (!runBtn || runBtn.disabled) {
            return;
        }
        runAnalysis();
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
}

function renderMode(options = {}) {
    const { instant = false } = options;
    refs.modeButtons.forEach(btn => btn.classList.toggle('active', btn.dataset.mode === state.mode));
    refs.dropHint.textContent = state.mode === 'single'
        ? '单张模式下请选择 1 张图片'
        : '批量模式下可一次导入多张图片';
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
                    <button class="file-action-btn" data-action="preview" data-index="${index}">预览</button>
                    <button class="file-action-btn file-action-btn--danger" data-action="remove" data-index="${index}" aria-label="移除文件">×</button>
                </div>
            </div>
        `).join('')
        : '<div class="file-list__empty">暂无文件，拖拽或点击上方按钮选择图片</div>';
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
    const showTime = !(state.statusType === 'idle' && state.statusText === '等待上传');
    const showInlineRunBtn = state.statusType === 'ready' && !state.isAnalyzing && state.files.length > 0;
    const showBatchProgress = state.mode === 'batch' && state.statusType === 'running' && state.batchProgress && state.batchProgress.total > 0;
    let statusTextHtml = escapeHtml(state.statusText);

    if (showInlineRunBtn) {
        const inlineButtonHtml = `
            <button id="runBtn" class="status-run-btn status-run-btn--inline" aria-label="开始分析">
                <span class="status-run-btn__shine" aria-hidden="true"></span>
                <span class="status-run-btn__text">开始分析</span>
            </button>
        `.trim();
        if (statusTextHtml.includes('开始分析')) {
            statusTextHtml = statusTextHtml.replace('开始分析', inlineButtonHtml);
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
                    <div class="status-card__progress-label">${escapeHtml(state.batchProgress.label || '批量分析中')}</div>
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
    setStatus(state.mode === 'single' ? '单张图片分析中...' : '批量分析与聚类中...', 'running');

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
            setStatus('单张分析完成', 'success');
        } else {
            state.batchRunId = payload.run_id;
            updateBatchProgress({ stage: 'queued', current: 0, total: state.files.length, percent: 0 });
            startBatchPolling(payload.run_id);
            setStatus('批量任务已启动，正在分析中...', 'running');
            saveBatchSession({ state: 'queued' });
        }
    } catch (error) {
        setStatus(`分析失败：${error.message}`, 'error');
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
            setStatus(`批量分析失败：${error.message}`, 'error');
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
    renderBatch(payload);
    setStatus('批量分析完成', 'success');
    renderStatusCard();
    saveBatchSession({ state: 'completed' });
}

function updateBatchProgress(status) {
    const stageMap = {
        queued: '等待进入分析队列',
        analyzing: `正在分析 ${status.current_file || '当前样本'}`,
        clustering: '正在执行聚类与树状图计算',
        completed: '批量分析与聚类完成',
        error: '批量分析失败',
    };
    const nextProgress = {
        label: stageMap[status.stage] || '批量任务处理中',
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
        setStatus(`正在调整聚类类别到 ${nextCount} 类...`, 'running');
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
        setStatus(`已更新为 ${nextCount} 类聚类结果`, 'success');
    } catch (error) {
        state.batchProgress = null;
        setStatus(`重聚类失败：${error.message}`, 'error');
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
    refs.previewTitle.textContent = title || '图片预览';
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
        setStatus('等待上传', 'idle');
        return;
    }
    if (state.mode === 'single') {
        setStatus('已选择 1 张图片，点击“开始分析”', 'ready');
        return;
    }
    setStatus(`已选择 ${state.files.length} 张图片，点击“开始分析”`, 'ready');
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

    const baseImageUrl = result.images.original;
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
    const stemGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    stemGroup.setAttribute('class', 'skeleton-group');
    stemGroup.innerHTML = `
        <path class="skeleton-path skeleton-path--stem" data-path-id="stem" d="${stemPath}"></path>
        <path class="skeleton-path skeleton-path--stem skeleton-path--glow" data-path-id="stem" d="${stemPath}"></path>
        <path class="skeleton-hit skeleton-hit--stem" data-path-id="stem" d="${stemPath}"></path>
    `;
    refs.skeletonSvg.appendChild(stemGroup);

    (state.skeletonOverlay.spikelets || []).forEach(spikelet => {
        const branchPath = buildSvgPath([spikelet.highest_point, spikelet.lowest_point]);
        const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        group.setAttribute('class', 'skeleton-group');
        group.dataset.pathId = `spikelet-${spikelet.index}`;
        group.dataset.side = spikelet.side;
        group.innerHTML = `
            <path class="skeleton-path skeleton-path--branch ${spikelet.side}" data-path-id="spikelet-${spikelet.index}" d="${branchPath}"></path>
            <path class="skeleton-path skeleton-path--branch skeleton-path--glow ${spikelet.side}" data-path-id="spikelet-${spikelet.index}" d="${branchPath}"></path>
            <circle class="skeleton-node ${spikelet.side}" data-node-id="spikelet-${spikelet.index}" cx="${spikelet.lowest_point[0]}" cy="${spikelet.lowest_point[1]}" r="2.8"></circle>
            <circle class="skeleton-node ${spikelet.side}" data-node-id="spikelet-${spikelet.index}" cx="${spikelet.highest_point[0]}" cy="${spikelet.highest_point[1]}" r="2.5"></circle>
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
}

function handleSkeletonPathHover(event) {
    if (state.currentView !== 'skeleton_extract' || refs.skeletonLayer.classList.contains('hidden')) {
        return;
    }

    const pathId = event.currentTarget.dataset.pathId;
    if (!pathId) {
        return;
    }

    const targetIds = pathId === 'stem' ? ['stem'] : ['stem', pathId];
    setActiveSkeletonPaths(targetIds);

    const rect = refs.viewer.getBoundingClientRect();
    state.skeletonFxPointer = {
        x: (event.clientX - rect.left - state.translateX) / state.scale,
        y: (event.clientY - rect.top - state.translateY) / state.scale,
    };
    ensureSkeletonFxLoop();
}

function setActiveSkeletonPaths(pathIds) {
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
    refs.tooltip.innerHTML = `
        <h4>小穗 #${record.order}</h4>
        <p>长度: ${record.length.toFixed(2)} px</p>
        <p>宽度: ${record.width.toFixed(2)} px</p>
        <p>长宽比: ${record.aspect_ratio.toFixed(3)}</p>
        <p>着生角度: ${record.attachment_angle.toFixed(2)}°</p>
        <p>侧别: ${record.side === 'left' ? '左侧' : '右侧'}</p>
    `;
    refs.tooltip.classList.remove('hidden');
    moveTooltip(event);
}

function moveTooltip(event) {
    const rect = refs.viewer.getBoundingClientRect();
    refs.tooltip.style.left = `${event.clientX - rect.left + 18}px`;
    refs.tooltip.style.top = `${event.clientY - rect.top + 18}px`;
}

function hideTooltip() {
    refs.tooltip.classList.add('hidden');
}

function renderSingleMetrics(result) {
    const ear = result.ear_pheno;
    const calibration = result.calibration || {};
    const metrics = [
        ['平均长度', formatMetric(ear.mean_spikelet_length_mm, 'mm', ear.mean_spikelet_length, 'px')],
        ['平均宽度', formatMetric(ear.mean_spikelet_width_mm, 'mm', ear.mean_spikelet_width, 'px')],
        ['平均长宽比', ear.mean_aspect_ratio.toFixed(3)],
        ['平均着生角', `${ear.mean_attachment_angle.toFixed(2)}°`],
        ['穗长', formatMetric(ear.spike_length_cm, 'cm', ear.spike_length_px, 'px')],
        ['小穗数', `${ear.spikelet_count}`],
        ['着生密度', formatMetric(ear.spikelet_density_per_cm, '/cm', ear.spikelet_density_px, '/px')],
        ['对称度指数', ear.asymmetry_index.toFixed(4)],
        ['重心偏移度', ear.centroid_offset.toFixed(4)],
    ];

    if (Number.isFinite(Number(ear.mean_color_l))) {
        metrics.push(['平均亮度 L*', Number(ear.mean_color_l).toFixed(2)]);
    }
    if (Number.isFinite(Number(ear.mean_color_a))) {
        metrics.push(['平均色轴 a*', Number(ear.mean_color_a).toFixed(2)]);
    }
    if (Number.isFinite(Number(ear.mean_color_b))) {
        metrics.push(['平均色轴 b*', Number(ear.mean_color_b).toFixed(2)]);
    }
    if (Number.isFinite(Number(ear.color_std_l))) {
        metrics.push(['亮度离散度', Number(ear.color_std_l).toFixed(2)]);
    }
    if (Number.isFinite(Number(ear.left_right_color_delta_e))) {
        metrics.push(['左右色差 ΔE', Number(ear.left_right_color_delta_e).toFixed(2)]);
    }

    metrics.push(['色卡标定', calibration.color_calibration_ok ? '已启用' : '未启用']);
    if (calibration.color_calibration_ok) {
        if (Number.isFinite(Number(calibration.color_delta_e_mean))) {
            metrics.push(['色卡ΔE均值', Number(calibration.color_delta_e_mean).toFixed(2)]);
        }
        if (Number.isFinite(Number(calibration.color_quality_score))) {
            metrics.push(['色卡质量分', Number(calibration.color_quality_score).toFixed(3)]);
        }
        if (Number.isFinite(Number(calibration.color_card_confidence))) {
            metrics.push(['色卡置信度', Number(calibration.color_card_confidence).toFixed(3)]);
        }
    } else if (calibration.color_error) {
        metrics.push(['色彩标定状态', calibration.color_error]);
    }

    refs.singleMetrics.innerHTML = metrics.map(([label, value]) => `
        <div class="metric-card">
            <div class="metric-card__label">${label}</div>
            <div class="metric-card__value">${value}</div>
        </div>
    `).join('');
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
        return `
            <article class="cluster-card ${active ? 'is-active' : ''} ${item.__matched ? '' : 'is-muted'}" data-cluster-id="${item.cluster_id}">
                <div class="cluster-card__eyebrow">Cluster ${item.cluster_id + 1}</div>
                <div class="cluster-card__header">
                    <div>
                        <h3>第 ${item.cluster_id + 1} 类</h3>
                        <p>${item.sample_count} 个样本</p>
                    </div>
                    <div class="cluster-card__badge">${formatClusterMetric(item.aggregate_metrics[state.batchSortMetric])}</div>
                </div>
                <div class="cluster-card__cover">
                    ${item.representative_image ? `<img src="${item.representative_image}" alt="cluster ${item.cluster_id} representative">` : '<div class="cluster-card__cover-empty">No Image</div>'}
                </div>
                <div class="cluster-card__hint">点击查看该簇详情</div>
                <div class="cluster-card__actions">
                    <button class="ghost-btn cluster-card__action cluster-card__action--compare ${inComparison ? 'is-active' : ''}" data-action="compare" data-cluster-id="${item.cluster_id}">
                        ${inComparison ? '取消对比' : '加入对比'}
                    </button>
                    <button class="ghost-btn cluster-card__action cluster-card__action--export" data-action="export" data-cluster-id="${item.cluster_id}">
                        <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
                            <path d="M12 3V14"/>
                            <path d="M8 10L12 14L16 10"/>
                            <path d="M5 17H19"/>
                            <path d="M7 21H17"/>
                        </svg>
                        <span>导出</span>
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
        setStatus('导出失败：缺少批量任务 run_id，请重新执行一次批量分析。', 'error');
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
    refs.clusterCompareSummary.innerHTML = `
        <div class="cluster-compare__summary-head">
            <div class="cluster-compare__summary-title">已选择 ${selected.length} 个类簇</div>
            <button class="ghost-btn cluster-compare__clear" id="clearCompareBtn">清空对比</button>
        </div>
        <div class="cluster-compare__summary-pills">
            ${selected.map(item => `
                <button class="cluster-compare__pill" data-cluster-id="${item.cluster_id}">第 ${item.cluster_id + 1} 类 · ${item.sample_count} 个样本 <span class="cluster-compare__pill-close" aria-hidden="true">×</span></button>
            `).join('')}
        </div>
    `;

    if (selected.length < 2) {
        refs.clusterCompareChart.innerHTML = '<div class="compare-empty">至少选择 2 个类簇后展示多类对比图表。</div>';
        refs.clusterCompareGallery.innerHTML = '';
        return;
    }

    const spikeletMetrics = [
        { key: 'mean_spikelet_length_mm', label: '平均小穗长度', unit: 'mm', group: '小穗级特征', groupClass: 'spikelet' },
        { key: 'mean_spikelet_width_mm', label: '平均小穗宽度', unit: 'mm', group: '小穗级特征', groupClass: 'spikelet' },
        { key: 'mean_aspect_ratio', label: '平均小穗长宽比', unit: '', group: '小穗级特征', groupClass: 'spikelet' },
        { key: 'mean_attachment_angle', label: '平均着生角', unit: '°', group: '小穗级特征', groupClass: 'spikelet' },
    ];
    const earMetrics = [
        { key: 'mean_spike_length_cm', label: '穗长', unit: 'cm', group: '穗级特征', groupClass: 'ear' },
        { key: 'spikelet_count', label: '小穗数', unit: '', group: '穗级特征', groupClass: 'ear' },
        { key: 'spikelet_density', label: '着生密度', unit: '', group: '穗级特征', groupClass: 'ear' },
        { key: 'mean_asymmetry_index', label: '对称度', unit: '', group: '穗级特征', groupClass: 'ear' },
        { key: 'mean_centroid_offset', label: '重心偏移度', unit: '', group: '穗级特征', groupClass: 'ear' },
    ];
    const colorMetrics = [
        { key: 'mean_color_l', label: '平均亮度 L*', unit: '', group: '颜色特征', groupClass: 'ear' },
        { key: 'mean_color_a', label: '平均色轴 a*', unit: '', group: '颜色特征', groupClass: 'ear' },
        { key: 'mean_color_b', label: '平均色轴 b*', unit: '', group: '颜色特征', groupClass: 'ear' },
        { key: 'color_std_l', label: '亮度离散度', unit: '', group: '颜色特征', groupClass: 'ear' },
        { key: 'left_right_color_delta_e', label: '左右色差 ΔE', unit: '', group: '颜色特征', groupClass: 'ear' },
    ];
    const availableColorMetrics = colorMetrics.filter(metric =>
        selected.some(item => Number.isFinite(Number(item.aggregate_metrics?.[metric.key])))
    );
    const metrics = [...spikeletMetrics, ...earMetrics, ...availableColorMetrics];

    refs.clusterCompareChart.innerHTML = `
        <div class="compare-radar-grid">
            <div class="compare-radar-shell">
                <div class="compare-radar__title">雷达图 • 小穗级特征</div>
                ${renderClusterRadar(selected, spikeletMetrics)}
            </div>
            <div class="compare-radar-shell">
                <div class="compare-radar__title">雷达图 • 穗级特征</div>
                ${renderClusterRadar(selected, earMetrics)}
            </div>
            ${availableColorMetrics.length ? `
                <div class="compare-radar-shell">
                    <div class="compare-radar__title">雷达图 • 颜色特征</div>
                    ${renderClusterRadar(selected, availableColorMetrics)}
                </div>
            ` : ''}
        </div>
        <div class="compare-bars-shell">
            <div class="compare-bars-shell__title">柱状图</div>
            <div class="compare-bars-grid">
                ${metrics.map(metric => {
                    const values = selected.map(item => Number(item.aggregate_metrics?.[metric.key] ?? 0));
                    const maxValue = Math.max(...values, 1e-6);
                    return `
                        <div class="compare-metric compare-metric--compact">
                            <div class="compare-metric__head">
                                <div class="compare-metric__title">${metric.label}</div>
                                <div class="compare-metric__group compare-metric__group--${metric.groupClass}">${metric.group}</div>
                            </div>
                            ${selected.map(item => {
                                const value = Number(item.aggregate_metrics?.[metric.key] ?? 0);
                                return `
                                    <div class="compare-bar">
                                        <div class="compare-bar__label"><i class="compare-bar__dot" style="background:${clusterColor(item.cluster_id)}"></i>第 ${item.cluster_id + 1} 类</div>
                                        <div class="compare-bar__track">
                                            <span style="width:${(value / maxValue) * 100}%"></span>
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

    setStatus('已恢复上次批量任务，正在同步状态...', 'running');
    renderStatusCard();

    try {
        const response = await fetch(`/api/batch-status/${snapshot.runId}`);
        const status = await response.json();
        if (!response.ok || status.error) {
            if (status?.code === 'job_expired' || status?.code === 'job_not_found') {
                clearBatchSession();
                state.batchRunId = null;
                state.batchProgress = null;
                setStatus('上次任务已过期或不存在，请重新上传后开始分析。', 'idle');
                renderStatusCard();
                return;
            }
            throw new Error(status.error || '任务状态不可用');
        }
        updateBatchProgress(status);
        if (status.state === 'completed') {
            await fetchBatchResult(snapshot.runId);
            return;
        }
        if (status.state === 'error') {
            setStatus(`上次批量任务失败：${status.error || '请重新上传后重试'}`, 'error');
            renderStatusCard();
            saveBatchSession({ state: 'error' });
            return;
        }
        startBatchPolling(snapshot.runId);
    } catch (error) {
        clearBatchSession();
        state.batchRunId = null;
        state.batchProgress = null;
        setStatus(`任务恢复失败：${error.message}`, 'error');
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
            const maxValue = Math.max(...selected.map(target => Number(target.aggregate_metrics?.[metric.key] ?? 0)), 1e-6);
            const value = Number(item.aggregate_metrics?.[metric.key] ?? 0) / maxValue;
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
            ${selected.map(item => `<span><i style="background:${clusterColor(item.cluster_id)}"></i>第 ${item.cluster_id + 1} 类</span>`).join('')}
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
        { key: 'mean_spikelet_length_mm', label: '平均小穗长度', group: 'spikelet', groupLabel: '小穗级特征' },
        { key: 'mean_spikelet_width_mm', label: '平均小穗宽度', group: 'spikelet', groupLabel: '小穗级特征' },
        { key: 'mean_aspect_ratio', label: '平均小穗长宽比', group: 'spikelet', groupLabel: '小穗级特征' },
        { key: 'mean_attachment_angle', label: '平均着生角', group: 'spikelet', groupLabel: '小穗级特征' },
        { key: 'mean_spike_length_cm', label: '穗长', group: 'ear', groupLabel: '穗级特征' },
        { key: 'spikelet_count', label: '小穗数', group: 'ear', groupLabel: '穗级特征' },
        { key: 'spikelet_density', label: '着生密度', group: 'ear', groupLabel: '穗级特征' },
        { key: 'mean_asymmetry_index', label: '对称度', group: 'ear', groupLabel: '穗级特征' },
        { key: 'mean_centroid_offset', label: '重心偏移度', group: 'ear', groupLabel: '穗级特征' },
        { key: 'mean_color_l', label: '平均亮度 L*', group: 'color', groupLabel: '颜色特征' },
        { key: 'mean_color_a', label: '平均色轴 a*', group: 'color', groupLabel: '颜色特征' },
        { key: 'mean_color_b', label: '平均色轴 b*', group: 'color', groupLabel: '颜色特征' },
        { key: 'color_std_l', label: '亮度离散度', group: 'color', groupLabel: '颜色特征' },
        { key: 'left_right_color_delta_e', label: '左右色差 ΔE', group: 'color', groupLabel: '颜色特征' },
    ];
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
    const prefix = target === 'sort' ? '排序' : '筛选';
    const groupLabel = metric?.groupLabel || '未选择分组';
    const label = metric?.label || '请选择指标';
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
    const labelMap = {
        spikelet_count: '小穗数',
        mean_spikelet_length: '平均小穗长度',
        mean_spikelet_width: '平均小穗宽度',
        mean_aspect_ratio: '平均长宽比',
        mean_attachment_angle: '平均着生角',
        spike_length: '穗长',
        spikelet_density: '小穗密度',
        asymmetry_index: '对称度',
        centroid_offset: '重心偏移度',
        spike_length_cm: '穗长',
        mean_spike_length_cm: '平均穗长',
        mean_spikelet_length_mm: '平均小穗长度',
        mean_spikelet_width_mm: '平均小穗宽度',
        mean_asymmetry_index: '平均对称度',
        mean_centroid_offset: '平均重心偏移度',
        mean_color_l: '平均亮度 L*',
        mean_color_a: '平均色轴 a*',
        mean_color_b: '平均色轴 b*',
        color_std_l: '亮度离散度',
        left_right_color_delta_e: '左右色差 ΔE',
    };
    return labelMap[key] || null;
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
        { group: '小穗级特征', label: '平均小穗长度', value: formatMetric(safeEar.mean_spikelet_length_mm, 'mm', safeEar.mean_spikelet_length ?? 0, 'px') },
        { group: '小穗级特征', label: '平均小穗宽度', value: formatMetric(safeEar.mean_spikelet_width_mm, 'mm', safeEar.mean_spikelet_width ?? 0, 'px') },
        { group: '小穗级特征', label: '平均小穗长宽比', value: Number(safeEar.mean_aspect_ratio ?? 0).toFixed(3) },
        { group: '小穗级特征', label: '平均着生角', value: `${Number(safeEar.mean_attachment_angle ?? 0).toFixed(2)} °` },
        { group: '穗级特征', label: '穗长', value: formatMetric(safeEar.spike_length_cm, 'cm', safeEar.spike_length_px ?? 0, 'px') },
        { group: '穗级特征', label: '小穗数', value: String(Math.round(Number(safeEar.spikelet_count ?? 0))) },
        { group: '穗级特征', label: '着生密度', value: formatMetric(safeEar.spikelet_density_per_cm, '/cm', safeEar.spikelet_density_px ?? 0, '/px') },
        { group: '穗级特征', label: '对称度', value: Number(safeEar.asymmetry_index ?? 0).toFixed(4) },
        { group: '穗级特征', label: '重心偏移度', value: Number(safeEar.centroid_offset ?? 0).toFixed(4) },
    ];
}

function buildNineClusterMetricRows(clusterItem) {
    const metrics = clusterItem?.aggregate_metrics || {};
    return [
        { group: '小穗级特征', label: '平均小穗长度', value: formatClusterMetric(metrics.mean_spikelet_length_mm, 'mm') },
        { group: '小穗级特征', label: '平均小穗宽度', value: formatClusterMetric(metrics.mean_spikelet_width_mm, 'mm') },
        { group: '小穗级特征', label: '平均小穗长宽比', value: formatClusterMetric(metrics.mean_aspect_ratio, '') },
        { group: '小穗级特征', label: '平均着生角', value: formatClusterMetric(metrics.mean_attachment_angle, '°') },
        { group: '穗级特征', label: '穗长', value: formatClusterMetric(metrics.mean_spike_length_cm, 'cm') },
        { group: '穗级特征', label: '小穗数', value: formatClusterMetric(metrics.spikelet_count, '') },
        { group: '穗级特征', label: '着生密度', value: formatClusterMetric(metrics.spikelet_density, '') },
        { group: '穗级特征', label: '对称度', value: formatClusterMetric(metrics.mean_asymmetry_index, '') },
        { group: '穗级特征', label: '重心偏移度', value: formatClusterMetric(metrics.mean_centroid_offset, '') },
    ];
}

function renderGroupedMetricCards(rows) {
    const spikeletRows = rows.filter(item => item.group === '小穗级特征');
    const earRows = rows.filter(item => item.group === '穗级特征');
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

    return `${renderSection('小穗级特征', 'spikelet', spikeletRows)}${renderSection('穗级特征', 'ear', earRows)}`;
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
    const spikeletRows = rows.filter(item => item.group === '小穗级特征');
    const earRows = rows.filter(item => item.group === '穗级特征');
    const showImage = options.showImage !== false;
    const imageUrl = result.images?.original || result.images?.analysis || '';

    const displayClusterLabel = Number.isFinite(Number(clusterLabel)) ? Number(clusterLabel) + 1 : (clusterLabel ?? '-');
    if (showImage && imageUrl) {
        refs.clusterHoverCard.innerHTML = `
            <div class="cluster-hover-card__layout">
                <div class="cluster-hover-card__body cluster-hover-card__body--metrics-only">
                    <h4>${result.image_name || result.filename}</h4>
                    <p>Cluster ${displayClusterLabel}</p>
                    <div class="cluster-hover-card__group">小穗级特征</div>
                    ${renderHoverMetricRows(spikeletRows)}
                    <div class="cluster-hover-card__group">穗级特征</div>
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
                <p>Cluster ${displayClusterLabel}</p>
                <div class="cluster-hover-card__group">小穗级特征</div>
                ${renderHoverMetricRows(spikeletRows)}
                <div class="cluster-hover-card__group">穗级特征</div>
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
    const spikeletRows = rows.filter(item => item.group === '小穗级特征');
    const earRows = rows.filter(item => item.group === '穗级特征');

    refs.clusterHoverCard.innerHTML = `
        <div class="cluster-hover-card__body cluster-hover-card__body--metrics-only">
            <h4>第 ${clusterItem.cluster_id + 1} 类</h4>
            <p>样本数：${clusterItem.sample_count}</p>
            <div class="cluster-hover-card__group">小穗级特征</div>
            ${renderHoverMetricRows(spikeletRows)}
            <div class="cluster-hover-card__group">穗级特征</div>
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
    const extra = node.sample_names.length > 1 ? ` +${node.sample_names.length - 1} 个样本` : '';
    const rows = buildNineSampleMetricRows(sample.ear_pheno || {});
    const spikeletRows = rows.filter(item => item.group === '小穗级特征');
    const earRows = rows.filter(item => item.group === '穗级特征');
    const imageUrl = sample.images?.original || sample.images?.analysis || '';
    refs.clusterHoverCard.innerHTML = `
        <div class="cluster-hover-card__layout">
            <div class="cluster-hover-card__body cluster-hover-card__body--metrics-only">
                <h4>${sample.image_name || sample.filename}${extra}</h4>
                <p>树节点覆盖 ${node.sample_names.length} 个样本</p>
                <div class="cluster-hover-card__group">小穗级特征</div>
                ${renderHoverMetricRows(spikeletRows)}
                <div class="cluster-hover-card__group">穗级特征</div>
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
    refs.clusterModalTitle.textContent = `第 ${clusterId + 1} 类详情`;
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

