const state = {
    mode: 'single',
    files: [],
    previewUrls: [],
    singleResult: null,
    batchResult: null,
    statusText: '等待上传',
    statusType: 'idle',
    isAnalyzing: false,
    currentView: 'spikelet',
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
};

const refs = {};
let panelTransitionToken = 0;

document.addEventListener('DOMContentLoaded', () => {
    bindRefs();
    bindEvents();
    renderMode({ instant: true });
    renderFiles();
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
    refs.clusterDetail = document.getElementById('clusterDetail');
    refs.artifactGrid = document.getElementById('artifactGrid');
    refs.batchCards = document.getElementById('batchCards');
    refs.downloadLinks = document.getElementById('downloadLinks');
    refs.previewModal = document.getElementById('previewModal');
    refs.previewBackdrop = document.getElementById('previewBackdrop');
    refs.previewClose = document.getElementById('previewClose');
    refs.previewImage = document.getElementById('previewImage');
    refs.previewTitle = document.getElementById('previewTitle');
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
    window.addEventListener('keydown', (event) => {
        if (event.key === 'Escape' && !refs.previewModal.classList.contains('hidden')) {
            closePreview();
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
        </div>
    `;
    refs.statusCard.className = `status-card status-card--${state.statusType}`;
}

function resetPanels() {
    refs.singleMetrics.innerHTML = '';
    refs.batchCards.innerHTML = '';
    refs.clusterMap.innerHTML = '';
    refs.clusterDetail.innerHTML = '<p class="cluster-detail__placeholder">完成批量分析后，点击样本点查看详情。</p>';
    refs.artifactGrid.innerHTML = '';
    refs.downloadLinks.innerHTML = '';
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
            state.batchResult = payload;
            renderBatch(payload);
            setStatus('批量分析完成', 'success');
        }
    } catch (error) {
        setStatus(`分析失败：${error.message}`, 'error');
    } finally {
        state.isAnalyzing = false;
        renderStatusCard();
    }
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
    refs.previewTitle.textContent = state.files[index].name;
    refs.previewImage.src = state.previewUrls[index];
    refs.previewModal.classList.remove('hidden');
    document.body.classList.add('modal-open');
}

function closePreview() {
    refs.previewModal.classList.add('hidden');
    refs.previewImage.removeAttribute('src');
    document.body.classList.remove('modal-open');
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
    const cluster = payload.cluster;
    const results = payload.results || [];
    renderBatchCards(results, cluster);
    renderClusterMap(results, cluster);
    renderArtifacts(payload);
}

function renderBatchCards(results, cluster) {
    const labelMap = buildLabelMap(cluster);
    refs.batchCards.innerHTML = results.map(result => {
        const clusterLabel = labelMap[result.image_name || result.filename] ?? '-';
        return `
            <div class="batch-card" data-filename="${result.image_name || result.filename}">
                <div class="batch-card__cluster">Cluster ${clusterLabel}</div>
                <div class="batch-card__name">${result.image_name || result.filename}</div>
                <div class="batch-card__metrics">
                    <div>穗长: ${formatMetric(result.ear_pheno.spike_length_cm, 'cm', result.ear_pheno.spike_length_px, 'px')}</div>
                    <div>平均着生角: ${result.ear_pheno.mean_attachment_angle.toFixed(2)}°</div>
                    <div>对称度: ${result.ear_pheno.asymmetry_index.toFixed(4)}</div>
                </div>
            </div>
        `;
    }).join('');

    refs.batchCards.querySelectorAll('.batch-card').forEach(card => {
        card.addEventListener('click', () => {
            const filename = card.dataset.filename;
            const item = results.find(entry => (entry.image_name || entry.filename) === filename);
            if (item) {
                renderBatchDetail(item, cluster ? buildLabelMap(cluster)[item.image_name || item.filename] : null);
            }
        });
    });
}

function renderClusterMap(results, cluster) {
    if (!cluster || !cluster.embedding || !cluster.embedding.length) {
        refs.clusterMap.innerHTML = '<p class="cluster-detail__placeholder">样本不足，未生成聚类嵌入图。</p>';
        return;
    }

    const width = refs.clusterMap.clientWidth - 10;
    const height = 300;
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
    refs.clusterMap.innerHTML = `
        <div class="metric-card">
            <div class="metric-card__label">层次聚类 + t-SNE</div>
            <div class="metric-card__value">${cluster.silhouette_score ? cluster.silhouette_score.toFixed(3) : 'N/A'}</div>
        </div>
        <svg class="cluster-svg" viewBox="0 0 ${width} ${height}">
            ${points.map((point, index) => `
                <g class="cluster-point" data-filename="${cluster.image_names[index]}">
                    <circle cx="${scaleX(point[0])}" cy="${scaleY(point[1])}" r="9" fill="${clusterColor(labelMap[cluster.image_names[index]])}" />
                    <text x="${scaleX(point[0]) + 12}" y="${scaleY(point[1]) + 4}" fill="#dff7ff" font-size="11">${cluster.image_names[index]}</text>
                </g>
            `).join('')}
        </svg>
    `;

    refs.clusterMap.querySelectorAll('.cluster-point').forEach(point => {
        point.addEventListener('click', () => {
            const filename = point.dataset.filename;
            const result = results.find(entry => (entry.image_name || entry.filename) === filename);
            if (result) {
                renderBatchDetail(result, labelMap[filename]);
            }
        });
    });
}

function renderBatchDetail(result, clusterLabel) {
    refs.clusterDetail.innerHTML = `
        <div class="cluster-detail__sample">
            <h3>${result.image_name || result.filename}</h3>
            <p>Cluster: ${clusterLabel ?? '-'}</p>
            <p>穗长: ${formatMetric(result.ear_pheno.spike_length_cm, 'cm', result.ear_pheno.spike_length_px, 'px')}</p>
            <p>平均小穗长度: ${formatMetric(result.ear_pheno.mean_spikelet_length_mm, 'mm', result.ear_pheno.mean_spikelet_length, 'px')}</p>
            <p>平均小穗宽度: ${formatMetric(result.ear_pheno.mean_spikelet_width_mm, 'mm', result.ear_pheno.mean_spikelet_width, 'px')}</p>
            <p>平均着生角: ${result.ear_pheno.mean_attachment_angle.toFixed(2)}°</p>
            <p>对称度: ${result.ear_pheno.asymmetry_index.toFixed(4)}</p>
            <p>重心偏移度: ${result.ear_pheno.centroid_offset.toFixed(4)}</p>
            <p><a href="${result.images.analysis}" target="_blank">查看综合分析图</a></p>
        </div>
    `;
}

function renderArtifacts(payload) {
    refs.downloadLinks.innerHTML = `
        <a href="${payload.downloads.phenotypes_csv}" target="_blank">表型 CSV</a>
        <a href="${payload.downloads.features_csv}" target="_blank">特征 CSV</a>
    `;

    if (!payload.cluster) {
        refs.artifactGrid.innerHTML = '';
        return;
    }

    refs.artifactGrid.innerHTML = `
        <div class="artifact-card">
            <div>聚类嵌入图</div>
            <a href="${payload.cluster.artifacts.embedding}" target="_blank"><img src="${payload.cluster.artifacts.embedding}" alt="cluster embedding"></a>
        </div>
        <div class="artifact-card">
            <div>样本距离热图</div>
            <a href="${payload.cluster.artifacts.heatmap}" target="_blank"><img src="${payload.cluster.artifacts.heatmap}" alt="sample heatmap"></a>
        </div>
        <div class="artifact-card">
            <div>层次聚类树状图</div>
            <a href="${payload.cluster.artifacts.dendrogram}" target="_blank"><img src="${payload.cluster.artifacts.dendrogram}" alt="cluster dendrogram"></a>
        </div>
        <div class="artifact-card">
            <div><a href="${payload.cluster.artifacts.labels_csv}" target="_blank">下载聚类标签 CSV</a></div>
            <div style="margin-top:10px;"><a href="${payload.cluster.artifacts.centers_csv}" target="_blank">下载聚类中心 CSV</a></div>
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
