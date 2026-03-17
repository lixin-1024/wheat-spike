from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import cv2
import numpy as np
from pathlib import Path
from werkzeug.utils import secure_filename

from wheat_analysis.pipeline import WheatAnalysisPipeline

# 配置
app = Flask(__name__)
CORS(app)  # 启用CORS支持
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB最大文件大小
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp'}

# 初始化分析管线
pipeline = WheatAnalysisPipeline(
    model_path='runs/obb/yolo11_1440_4/weights/best.pt',
    imgsz=1440,
    conf=0.5
)


RESULT_FOLDER = 'results/web'


def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def serialize_result(result, stem_name):
    """将分析结果转换为 JSON 可序列化的字典，并保存可视化图片"""
    if result.get('error'):
        return {
            'error': result['error'],
            'spikelet_count': result['detection']['count'] if result.get('detection') else 0,
        }

    # 保存可视化图片到结果目录
    out_dir = Path(RESULT_FOLDER)
    out_dir.mkdir(parents=True, exist_ok=True)

    if result.get('vis_image') is not None:
        cv2.imwrite(str(out_dir / f"{stem_name}_analysis.jpg"), result['vis_image'])

    if result.get('skeleton') and result.get('detection'):
        from wheat_analysis.visualizer import Visualizer
        vis = Visualizer()
        image = cv2.imread(result['detection']['image_path'])
        skeleton_vis = vis.draw_skeleton(image, result['detection'], result['skeleton'])
        cv2.imwrite(str(out_dir / f"{stem_name}_skeleton.jpg"), skeleton_vis)
        detect_vis = vis.draw_detection(image, result['detection'], draw_index=False)
        cv2.imwrite(str(out_dir / f"{stem_name}_detection.jpg"), detect_vis)

    # 构造 JSON 安全的响应
    resp = {
        'filename': stem_name,
        'images': {
            'analysis': f'/results_img/{stem_name}_analysis.jpg',
            'skeleton': f'/results_img/{stem_name}_skeleton.jpg',
            'detection': f'/results_img/{stem_name}_detection.jpg',
        },
    }

    if result.get('ear_pheno'):
        resp['ear_pheno'] = {
            k: float(v) if hasattr(v, 'item') else v
            for k, v in result['ear_pheno'].items()
        }

    if result.get('spikelet_pheno'):
        sp = result['spikelet_pheno']
        resp['spikelet_pheno'] = {
            k: v.tolist() if hasattr(v, 'tolist') else v
            for k, v in sp.items()
        }

    if result.get('ear_vector') is not None:
        resp['ear_vector'] = result['ear_vector'].tolist()

    return resp


@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """上传图片并进行分析"""
    if 'file' not in request.files:
        return jsonify({'error': '没有文件'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            stem_name = Path(filename).stem
            result = pipeline.analyze_single(filepath)
            resp = serialize_result(result, stem_name)
            return jsonify(resp)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': '不允许的文件类型'}), 400


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """获取上传的文件"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/batch_analysis', methods=['POST'])
def batch_analysis():
    """批量分析"""
    if 'files' not in request.files:
        return jsonify({'error': '没有文件'}), 400

    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return jsonify({'error': '没有选择文件'}), 400

    results = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                stem_name = Path(filename).stem
                result = pipeline.analyze_single(filepath)
                resp = serialize_result(result, stem_name)
                results.append(resp)
            except Exception as e:
                results.append({
                    'filename': filename,
                    'error': str(e)
                })

    return jsonify({'results': results})


@app.route('/results_img/<filename>')
def get_result_image(filename):
    """获取分析结果图片"""
    return send_from_directory(RESULT_FOLDER, filename)


@app.route('/phenotype/<filename>')
def get_phenotype(filename):
    """获取表型数据CSV"""
    return send_from_directory(RESULT_FOLDER, "phenotype_results.csv")


if __name__ == '__main__':
    # 确保上传目录存在
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(RESULT_FOLDER, exist_ok=True)

    app.run(debug=True, port=5000)