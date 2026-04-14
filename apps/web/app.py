from __future__ import annotations

import csv
import os
import threading
import time
import uuid
from io import StringIO
from pathlib import Path

import cv2
from flask import Flask, Response, jsonify, render_template, request, send_from_directory
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from wheat_analysis.clustering import PhenotypeClusterAnalyzer
from wheat_analysis.pipeline import BatchImagePipeline, SingleImagePipeline


PROJECT_ROOT = Path(__file__).parent.parent.parent
WEB_ROOT = Path(__file__).parent
UPLOAD_FOLDER = WEB_ROOT / "uploads"
RESULT_FOLDER = WEB_ROOT / "results"

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_CONTENT_LENGTH_MB", "128")) * 1024 * 1024
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg", "bmp"}

MODEL_PATH = str(PROJECT_ROOT / "runs/obb/yolo11_1440_4/weights/best.pt")
single_pipeline = SingleImagePipeline(model_path=MODEL_PATH, imgsz=1440, conf=0.5)
batch_pipeline = BatchImagePipeline(model_path=MODEL_PATH, imgsz=1440, conf=0.5)

batch_jobs: dict[str, dict] = {}
batch_jobs_lock = threading.Lock()
JOB_TTL_SECONDS = int(os.getenv("BATCH_JOB_TTL_SECONDS", "7200"))


def now_ts() -> float:
    return time.time()


def is_job_expired(job: dict) -> bool:
    if job.get("state") not in {"completed", "error"}:
        return False
    updated_at = float(job.get("updated_at") or now_ts())
    return (now_ts() - updated_at) > JOB_TTL_SECONDS


def cleanup_expired_jobs():
    with batch_jobs_lock:
        expired = [run_id for run_id, job in batch_jobs.items() if is_job_expired(job)]
        for run_id in expired:
            batch_jobs.pop(run_id, None)


def error_payload(message: str, code: str):
    return {"error": message, "code": code}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]


def to_json_safe(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {key: to_json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [to_json_safe(item) for item in value]
    return value


def result_image_urls(run_id: str, stem_name: str):
    return {
        "original": f"/results/{run_id}/{stem_name}_original.jpg",
        "analysis": f"/results/{run_id}/{stem_name}_analysis.jpg",
        "skeleton": f"/results/{run_id}/{stem_name}_skeleton.jpg",
        "detection": f"/results/{run_id}/{stem_name}_detection.jpg",
    }


def serialize_skeleton_overlay(result: dict):
    skeleton = result.get("skeleton")
    detection = result.get("detection")
    if not skeleton or not detection:
        return None

    centers = to_json_safe(detection.get("centers"))
    highest_points = to_json_safe(skeleton.get("spikelet_highest_points"))
    lowest_points = to_json_safe(skeleton.get("spikelet_lowest_points"))
    tangents = to_json_safe(skeleton.get("spikelet_tangent"))
    sides = to_json_safe(skeleton.get("spikelet_side"))
    orders = to_json_safe(skeleton.get("spikelet_order"))
    positions = to_json_safe(skeleton.get("spikelet_s"))
    stem_points = to_json_safe(skeleton.get("stem_points"))
    stem_fit_points = to_json_safe(skeleton.get("stem_fit_points"))

    if centers is None or highest_points is None or lowest_points is None or stem_points is None:
        return None

    order_rank = {}
    if orders is not None:
        for rank, spikelet_index in enumerate(orders, start=1):
            order_rank[int(spikelet_index)] = rank

    spikelets = []
    for index, center in enumerate(centers):
        spikelets.append(
            {
                "index": index,
                "order": order_rank.get(index, index + 1),
                "side": "right" if sides is not None and float(sides[index]) >= 0 else "left",
                "center": center,
                "highest_point": highest_points[index],
                "lowest_point": lowest_points[index],
                "tangent": tangents[index] if tangents is not None else None,
                "stem_position": positions[index] if positions is not None else None,
            }
        )

    return {
        "stem_points": stem_points,
        "stem_fit_points": stem_fit_points,
        "spikelets": spikelets,
    }


def serialize_single_result(result: dict, run_id: str, stem_name: str):
    if result.get("error"):
        return {
            "filename": stem_name,
            "error": result["error"],
        }

    detection = result["detection"]
    return {
        "filename": stem_name,
        "image_name": Path(detection["image_path"]).name,
        "image_shape": list(detection["image_shape"]) if detection.get("image_shape") is not None else None,
        "images": result_image_urls(run_id, stem_name),
        "ear_pheno": to_json_safe(result["ear_pheno"]),
        "calibration": to_json_safe(result["calibration"]),
        "skeleton_overlay": serialize_skeleton_overlay(result),
        "spikelet_pheno": to_json_safe(result["spikelet_pheno"]),
        "spikelet_records": to_json_safe(result["spikelet_records"]),
        "feature_names": result["feature_names"],
        "feature_vector": to_json_safe(result["feature_vector"]),
    }


def enrich_cluster_result(cluster_result: dict | None, serialized_results: list[dict], run_id: str):
    if cluster_result is None:
        return None

    def normalize_aggregate_metrics(metrics: dict | None) -> dict:
        source = dict(metrics or {})

        def pick(*keys):
            for key in keys:
                value = source.get(key)
                if value is not None:
                    return value
            return None

        source.setdefault("mean_spike_length_cm", pick("mean_spike_length_cm", "spike_length_cm", "spike_length"))
        source.setdefault(
            "mean_spikelet_length_mm",
            pick("mean_spikelet_length_mm", "mean_spikelet_length"),
        )
        source.setdefault(
            "mean_spikelet_width_mm",
            pick("mean_spikelet_width_mm", "mean_spikelet_width"),
        )
        source.setdefault("mean_attachment_angle", pick("mean_attachment_angle"))
        source.setdefault("mean_symmetry_index", pick("mean_symmetry_index", "symmetry_index"))
        source.setdefault("mean_centroid_offset", pick("mean_centroid_offset", "centroid_offset"))
        return source

    result_map = {item.get("image_name") or item.get("filename"): item for item in serialized_results}
    clusters = []
    for cluster in cluster_result.get("clusters", []):
        clusters.append(
            {
                "cluster_id": cluster["cluster_id"],
                "sample_count": cluster["sample_count"],
                "sample_names": cluster["sample_names"],
                "representative_image": cluster.get("representative_image")
                or (result_map.get(cluster.get("representative_name")) or {}).get("images", {}).get("analysis"),
                "representative_name": cluster.get("representative_name"),
                "samples": [result_map[name] for name in cluster["sample_names"] if name in result_map],
                "aggregate_metrics": to_json_safe(normalize_aggregate_metrics(cluster["aggregate_metrics"])),
                "metric_ranges": to_json_safe(cluster["metric_ranges"]),
            }
        )

    return {
        "method": cluster_result["method"],
        "labels": to_json_safe(cluster_result["labels"]),
        "embedding": to_json_safe(cluster_result["embedding"]),
        "image_names": cluster_result["image_names"],
        "feature_names": cluster_result["feature_names"],
        "cluster_centers": to_json_safe(cluster_result["cluster_centers"]),
        "silhouette_score": cluster_result["silhouette_score"],
        "clusters": clusters,
        "dendrogram": to_json_safe(cluster_result["dendrogram"]),
        "cluster_options": to_json_safe(cluster_result["cluster_options"]),
        "artifacts": {
            "labels_csv": f"/results/{run_id}/clustering_results.csv",
            "centers_csv": f"/results/{run_id}/cluster_centers.csv",
            "dendrogram": f"/results/{run_id}/cluster_dendrogram.png",
        },
    }


def build_batch_payload(run_id: str, raw_results: list[dict], cluster_result: dict | None):
    serialized_results = []
    for item in raw_results:
        image_path = item.get("detection", {}).get("image_path")
        stem_name = Path(image_path).stem if image_path else item.get("filename", "sample")
        serialized_results.append(serialize_single_result(item, run_id, stem_name))

    return {
        "run_id": run_id,
        "results": serialized_results,
        "cluster": enrich_cluster_result(cluster_result, serialized_results, run_id),
        "downloads": {
            "phenotypes_csv": f"/results/{run_id}/phenotype_results.csv",
            "features_csv": f"/results/{run_id}/feature_vectors.csv",
        },
    }


def serialize_job_status(job: dict):
    total = max(int(job.get("total", 0)), 1)
    current = min(int(job.get("current", 0)), total)
    return {
        "run_id": job["run_id"],
        "state": job["state"],
        "stage": job.get("stage", "queued"),
        "current": current,
        "total": int(job.get("total", 0)),
        "current_file": job.get("current_file"),
        "error": job.get("error"),
        "percent": 100 if not job.get("total") and job["state"] == "completed" else round(current / total * 100, 1),
        "updated_at": job.get("updated_at"),
    }


def update_job(run_id: str, **updates):
    with batch_jobs_lock:
        if run_id not in batch_jobs:
            return
        updates["updated_at"] = now_ts()
        batch_jobs[run_id].update(updates)


def run_batch_job(run_id: str, image_paths: list[str]):
    result_dir = RESULT_FOLDER / run_id

    def progress_callback(snapshot: dict):
        current = int(snapshot.get("current", 0))
        total = int(snapshot.get("total", len(image_paths)))
        if snapshot.get("stage") == "analyzing":
            current = min(current + 1, total)
        update_job(
            run_id,
            stage=snapshot.get("stage", "analyzing"),
            current=current,
            total=total,
            current_file=snapshot.get("current_file"),
            state="running" if snapshot.get("stage") != "completed" else "completed",
        )

    try:
        analysis = batch_pipeline.analyze_paths(image_paths, str(result_dir), progress_callback=progress_callback)
        payload = build_batch_payload(run_id, analysis["results"], analysis["cluster"])
        update_job(
            run_id,
            payload=payload,
            state="completed",
            stage="completed",
            current=len(image_paths),
            total=len(image_paths),
            current_file=None,
            error=None,
        )
    except Exception as exc:
        import traceback

        traceback.print_exc()
        update_job(
            run_id,
            state="error",
            stage="error",
            current=0,
            error=str(exc),
            current_file=None,
        )


def feature_samples_from_payload(payload: dict) -> list[dict]:
    samples = []
    for result in payload.get("results", []):
        if result.get("error") or not result.get("feature_vector"):
            continue
        samples.append(
            {
                "image": result.get("image_name") or result.get("filename"),
                "feature_names": result.get("feature_names"),
                "features": result.get("feature_vector"),
                "image_url": result.get("images", {}).get("analysis"),
                "ear_pheno": result.get("ear_pheno"),
            }
        )
    return samples


def build_cluster_export_csv(payload: dict, cluster_id: int) -> str | None:
    cluster = payload.get("cluster") or {}
    target = next((item for item in cluster.get("clusters", []) if int(item.get("cluster_id", -1)) == int(cluster_id)), None)
    if target is None:
        return None

    buffer = StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["cluster_id", target["cluster_id"]])
    writer.writerow(["sample_count", target["sample_count"]])
    writer.writerow([])
    writer.writerow(["aggregate_metric", "value"])
    for key, value in (target.get("aggregate_metrics") or {}).items():
        writer.writerow([key, value])
    writer.writerow([])

    feature_keys = [
        "spike_length_cm",
        "mean_spikelet_length_mm",
        "mean_spikelet_width_mm",
        "mean_attachment_angle",
        "symmetry_index",
        "centroid_offset",
    ]
    writer.writerow(["image", *feature_keys])
    for sample in target.get("samples", []):
        ear = sample.get("ear_pheno") or {}
        writer.writerow([sample.get("image_name") or sample.get("filename"), *[ear.get(key) for key in feature_keys]])

    return buffer.getvalue()


@app.route("/")
def index():
    return render_template("index.html")


@app.errorhandler(RequestEntityTooLarge)
def handle_request_too_large(exc):
    max_mb = app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)
    return jsonify({"error": f"上传文件过大，最大支持 {max_mb} MB"}), 413


@app.route("/api/analyze-single", methods=["POST"])
def analyze_single_api():
    if "file" not in request.files:
        return jsonify({"error": "未接收到图片文件"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "请选择图片文件"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "不支持的图片格式"}), 400

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
        return jsonify({"error": str(exc)}), 500


@app.route("/api/analyze-batch", methods=["POST"])
def analyze_batch_api():
    cleanup_expired_jobs()
    files = request.files.getlist("files")
    if not files or files[0].filename == "":
        return jsonify({"error": "请选择图片文件"}), 400

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
        return jsonify({"error": "未检测到可用图片"}), 400

    with batch_jobs_lock:
        now = now_ts()
        batch_jobs[run_id] = {
            "run_id": run_id,
            "state": "queued",
            "stage": "queued",
            "current": 0,
            "total": len(image_paths),
            "current_file": None,
            "payload": None,
            "error": None,
            "created_at": now,
            "updated_at": now,
        }

    worker = threading.Thread(target=run_batch_job, args=(run_id, image_paths), daemon=True)
    worker.start()
    return jsonify({"run_id": run_id, "status_url": f"/api/batch-status/{run_id}", "result_url": f"/api/batch-result/{run_id}"})


@app.route("/api/batch-status/<run_id>", methods=["GET"])
def batch_status_api(run_id: str):
    with batch_jobs_lock:
        job = batch_jobs.get(run_id)
    if not job:
        return jsonify(error_payload("任务不存在，请重新上传并发起分析。", "job_not_found")), 404
    if is_job_expired(job):
        with batch_jobs_lock:
            batch_jobs.pop(run_id, None)
        return jsonify(error_payload("任务结果已过期，请重新上传并发起分析。", "job_expired")), 410
    return jsonify(serialize_job_status(job))


@app.route("/api/batch-result/<run_id>", methods=["GET"])
def batch_result_api(run_id: str):
    with batch_jobs_lock:
        job = batch_jobs.get(run_id)
    if not job:
        return jsonify(error_payload("任务不存在，请重新上传并发起分析。", "job_not_found")), 404
    if is_job_expired(job):
        with batch_jobs_lock:
            batch_jobs.pop(run_id, None)
        return jsonify(error_payload("任务结果已过期，请重新上传并发起分析。", "job_expired")), 410
    if job["state"] == "error":
        return jsonify(error_payload(job.get("error") or "批量分析失败", "job_failed")), 500
    if job["state"] != "completed" or not job.get("payload"):
        return jsonify(error_payload("任务尚未完成", "job_not_completed")), 409
    return jsonify(job["payload"])


@app.route("/api/recluster", methods=["POST"])
def recluster_api():
    payload = request.get_json(silent=True) or {}
    run_id = payload.get("run_id")
    n_clusters = payload.get("n_clusters")

    if not run_id or n_clusters is None:
        return jsonify({"error": "缺少 run_id 或 n_clusters"}), 400

    with batch_jobs_lock:
        job = batch_jobs.get(run_id)
    if not job or not job.get("payload"):
        return jsonify({"error": "未找到可重聚类的批量结果"}), 404

    try:
        analyzer = PhenotypeClusterAnalyzer(n_clusters=int(n_clusters))
        samples = feature_samples_from_payload(job["payload"])
        if len(samples) < 2:
            return jsonify({"error": "有效样本不足，无法重聚类"}), 400

        cluster_result = analyzer.cluster(samples, str(RESULT_FOLDER / run_id))
        updated_payload = dict(job["payload"])
        updated_payload["cluster"] = enrich_cluster_result(cluster_result, updated_payload["results"], run_id)
        update_job(run_id, payload=updated_payload)
        return jsonify({"cluster": updated_payload["cluster"]})
    except Exception as exc:
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.route("/api/export-cluster/<run_id>/<int:cluster_id>", methods=["GET"])
def export_cluster_api(run_id: str, cluster_id: int):
    with batch_jobs_lock:
        job = batch_jobs.get(run_id)
    if not job or not job.get("payload"):
        return jsonify({"error": "未找到可导出的批量结果"}), 404

    csv_content = build_cluster_export_csv(job["payload"], cluster_id)
    if csv_content is None:
        return jsonify({"error": "未找到对应聚类"}), 404

    return Response(
        csv_content,
        mimetype="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="cluster_{cluster_id}_summary.csv"',
        },
    )


@app.route("/results/<run_id>/<path:filename>")
def get_result_file(run_id, filename):
    return send_from_directory(RESULT_FOLDER / run_id, filename)


if __name__ == "__main__":
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(RESULT_FOLDER, exist_ok=True)
    app.run(debug=True, port=5000)
