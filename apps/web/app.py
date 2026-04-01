from __future__ import annotations

import os
import uuid
from pathlib import Path

import cv2
from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from wheat_analysis.pipeline import BatchImagePipeline, SingleImagePipeline


PROJECT_ROOT = Path(__file__).parent.parent.parent
WEB_ROOT = Path(__file__).parent
UPLOAD_FOLDER = WEB_ROOT / 'uploads'
RESULT_FOLDER = WEB_ROOT / 'results'

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
# Default to 128 MB for batch uploads and allow override via env.
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH_MB', '128')) * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp'}

MODEL_PATH = str(PROJECT_ROOT / 'runs/obb/yolo11_1440_4/weights/best.pt')
single_pipeline = SingleImagePipeline(model_path=MODEL_PATH, imgsz=1440, conf=0.5)
batch_pipeline = BatchImagePipeline(model_path=MODEL_PATH, imgsz=1440, conf=0.5)


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def to_json_safe(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    if hasattr(value, 'item'):
        return value.item()
    if isinstance(value, dict):
        return {key: to_json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [to_json_safe(item) for item in value]
    return value


def result_image_urls(run_id: str, stem_name: str):
    return {
        'original': f'/results/{run_id}/{stem_name}_original.jpg',
        'analysis': f'/results/{run_id}/{stem_name}_analysis.jpg',
        'skeleton': f'/results/{run_id}/{stem_name}_skeleton.jpg',
        'detection': f'/results/{run_id}/{stem_name}_detection.jpg',
    }


def serialize_skeleton_overlay(result: dict):
    skeleton = result.get('skeleton')
    detection = result.get('detection')
    if not skeleton or not detection:
        return None

    centers = to_json_safe(detection.get('centers'))
    highest_points = to_json_safe(skeleton.get('spikelet_highest_points'))
    lowest_points = to_json_safe(skeleton.get('spikelet_lowest_points'))
    tangents = to_json_safe(skeleton.get('spikelet_tangent'))
    sides = to_json_safe(skeleton.get('spikelet_side'))
    orders = to_json_safe(skeleton.get('spikelet_order'))
    positions = to_json_safe(skeleton.get('spikelet_s'))
    stem_points = to_json_safe(skeleton.get('stem_points'))
    stem_fit_points = to_json_safe(skeleton.get('stem_fit_points'))

    if centers is None or highest_points is None or lowest_points is None or stem_points is None:
        return None

    order_rank = {}
    if orders is not None:
        for rank, spikelet_index in enumerate(orders, start=1):
            order_rank[int(spikelet_index)] = rank

    spikelets = []
    for index, center in enumerate(centers):
        spikelets.append({
            'index': index,
            'order': order_rank.get(index, index + 1),
            'side': 'right' if sides is not None and float(sides[index]) >= 0 else 'left',
            'center': center,
            'highest_point': highest_points[index],
            'lowest_point': lowest_points[index],
            'tangent': tangents[index] if tangents is not None else None,
            'stem_position': positions[index] if positions is not None else None,
        })

    return {
        'stem_points': stem_points,
        'stem_fit_points': stem_fit_points,
        'spikelets': spikelets,
    }


def serialize_single_result(result: dict, run_id: str, stem_name: str):
    if result.get('error'):
        return {
            'filename': stem_name,
            'error': result['error'],
        }

    detection = result['detection']
    return {
        'filename': stem_name,
        'image_name': Path(detection['image_path']).name,
        'image_shape': list(detection['image_shape']) if detection.get('image_shape') is not None else None,
        'images': result_image_urls(run_id, stem_name),
        'ear_pheno': to_json_safe(result['ear_pheno']),
        'calibration': to_json_safe(result['calibration']),
        'skeleton_overlay': serialize_skeleton_overlay(result),
        'spikelet_pheno': to_json_safe(result['spikelet_pheno']),
        'spikelet_records': to_json_safe(result['spikelet_records']),
        'feature_names': result['feature_names'],
        'feature_vector': to_json_safe(result['feature_vector']),
    }


def serialize_cluster_result(cluster_result: dict, run_id: str):
    if cluster_result is None:
        return None

    return {
        'method': cluster_result['method'],
        'labels': to_json_safe(cluster_result['labels']),
        'embedding': to_json_safe(cluster_result['embedding']),
        'image_names': cluster_result['image_names'],
        'feature_names': cluster_result['feature_names'],
        'cluster_centers': to_json_safe(cluster_result['cluster_centers']),
        'silhouette_score': cluster_result['silhouette_score'],
        'artifacts': {
            'embedding': f'/results/{run_id}/cluster_embedding.png',
            'heatmap': f'/results/{run_id}/sample_similarity_heatmap.png',
            'dendrogram': f'/results/{run_id}/cluster_dendrogram.png',
            'labels_csv': f'/results/{run_id}/clustering_results.csv',
            'centers_csv': f'/results/{run_id}/cluster_centers.csv',
        },
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.errorhandler(RequestEntityTooLarge)
def handle_request_too_large(exc):
    max_mb = app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)
    return jsonify({'error': f'上传内容过大，当前上限为 {max_mb} MB，请减少图片数量或压缩后重试'}), 413


@app.route('/api/analyze-single', methods=['POST'])
def analyze_single_api():
    if 'file' not in request.files:
        return jsonify({'error': '没有文件'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': '不允许的文件类型'}), 400

    run_id = f"single_{uuid.uuid4().hex[:8]}"
    upload_dir = UPLOAD_FOLDER / run_id
    result_dir = RESULT_FOLDER / run_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    filename = secure_filename(file.filename)
    filepath = upload_dir / filename
    file.save(filepath)

    try:
        result = single_pipeline.analyze(str(filepath), str(result_dir))
        return jsonify(serialize_single_result(result, run_id, filepath.stem))
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(exc)}), 500


@app.route('/api/analyze-batch', methods=['POST'])
def analyze_batch_api():
    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return jsonify({'error': '没有选择文件'}), 400

    run_id = f"batch_{uuid.uuid4().hex[:8]}"
    upload_dir = UPLOAD_FOLDER / run_id
    result_dir = RESULT_FOLDER / run_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    image_paths = []
    for file in files:
        if not allowed_file(file.filename):
            continue
        filename = secure_filename(file.filename)
        filepath = upload_dir / filename
        file.save(filepath)
        image_paths.append(str(filepath))

    if not image_paths:
        return jsonify({'error': '没有有效图片'}), 400

    try:
        analysis = batch_pipeline.analyze_paths(image_paths, str(result_dir))
        serialized_results = []
        for image_path, result in zip(image_paths, analysis['results']):
            serialized_results.append(serialize_single_result(result, run_id, Path(image_path).stem))

        response = {
            'run_id': run_id,
            'results': serialized_results,
            'cluster': serialize_cluster_result(analysis['cluster'], run_id),
            'downloads': {
                'phenotypes_csv': f'/results/{run_id}/phenotype_results.csv',
                'features_csv': f'/results/{run_id}/feature_vectors.csv',
            },
        }
        return jsonify(response)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(exc)}), 500


@app.route('/results/<run_id>/<path:filename>')
def get_result_file(run_id, filename):
    return send_from_directory(RESULT_FOLDER / run_id, filename)


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(RESULT_FOLDER, exist_ok=True)
    app.run(debug=True, port=5000)
