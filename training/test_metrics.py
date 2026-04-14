import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 允许重复加载libiomp5md.dll

import argparse
import json
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def to_serializable(value):
    """Convert numpy/tensor-like values to JSON serializable primitives."""
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(v) for v in value]
    return value


def run_validation(
    model_path: Path,
    data_path: Path,
    imgsz: int,
    split: str,
    device: str,
    batch: int,
    workers: int,
    conf: float,
    iou: float,
):
    """Run Ultralytics validation and return normalized metrics payload."""
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(data_path),
        imgsz=imgsz,
        split=split,
        device=device,
        batch=batch,
        workers=workers,
        conf=conf,
        iou=iou,
        verbose=False,
    )

    results_dict = {}
    if hasattr(metrics, "results_dict") and isinstance(metrics.results_dict, dict):
        results_dict = {k: to_serializable(v) for k, v in metrics.results_dict.items()}

    speed = {}
    if hasattr(metrics, "speed") and isinstance(metrics.speed, dict):
        speed = {k: to_serializable(v) for k, v in metrics.speed.items()}

    return {
        "results_dict": results_dict,
        "speed": speed,
    }


def print_metrics(metrics_payload: dict, split: str) -> None:
    """Pretty print validation metrics to terminal."""
    print(f"\n=== Validation Metrics ({split}) ===")
    results_dict = metrics_payload.get("results_dict", {})
    if results_dict:
        for key, value in results_dict.items():
            if isinstance(value, float):
                print(f"{key}: {value:.6f}")
            else:
                print(f"{key}: {value}")
    else:
        print("No scalar metrics found in results_dict.")

    speed = metrics_payload.get("speed", {})
    if speed:
        print(f"\n=== Speed ({split}, ms/image) ===")
        for key, value in speed.items():
            if isinstance(value, float):
                print(f"{key}: {value:.3f}")
            else:
                print(f"{key}: {value}")


def build_metric_delta(val_payload: dict, test_payload: dict) -> dict:
    """Build test-val deltas for scalar metrics present in both splits."""
    val_metrics = val_payload.get("results_dict", {})
    test_metrics = test_payload.get("results_dict", {})

    delta = {}
    for key, test_value in test_metrics.items():
        val_value = val_metrics.get(key)
        if isinstance(test_value, (int, float)) and isinstance(val_value, (int, float)):
            delta[key] = float(test_value) - float(val_value)
    return delta


def print_metric_delta(delta: dict) -> None:
    """Pretty print test-val metric differences."""
    print("\n=== Delta (test - val) ===")
    if not delta:
        print("No comparable scalar metrics found between val and test.")
        return

    for key, value in delta.items():
        print(f"{key}: {value:+.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate trained YOLO OBB model metrics")
    parser.add_argument(
        "--data",
        type=str,
        default=str(SCRIPT_DIR / "data.yaml"),
        help="Path to dataset yaml",
    )
    parser.add_argument("--imgsz", type=int, default=1440, help="Validation image size")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--device", type=str, default="0", help="CUDA device id or 'cpu'")
    parser.add_argument("--batch", type=int, default=1, help="Validation batch size (use 1 for low-memory stability)")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers (Windows recommends 0)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.6, help="NMS IoU threshold")
    parser.add_argument(
        "--compare-val-test",
        dest="compare_val_test",
        action="store_true",
        default=True,
        help="Run both val and test splits and output side-by-side comparison (default)",
    )
    parser.add_argument(
        "--single-split",
        dest="compare_val_test",
        action="store_false",
        help="Disable default comparison mode and run only --split",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "results" / "test_model_metrics"),
        help="Directory to save metric JSON",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional suffix for output filename",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_path = Path(args.data).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Data yaml not found: {data_path}")

    # Restrict evaluation to the single weight file beside this script.
    model_path = (SCRIPT_DIR / "best.pt").resolve()

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Place your trained best.pt in the same directory as this script."
        )

    print(f"Model: {model_path}")
    print(f"Data:  {data_path}")
    print(f"Split: {args.split}")
    print(f"Batch: {args.batch}")
    print(f"Workers: {args.workers}")

    compare_mode = args.compare_val_test

    if compare_mode:
        print("Comparison mode: running both val and test")
        val_payload = run_validation(
            model_path=model_path,
            data_path=data_path,
            imgsz=args.imgsz,
            split="val",
            device=args.device,
            batch=args.batch,
            workers=args.workers,
            conf=args.conf,
            iou=args.iou,
        )
        test_payload = run_validation(
            model_path=model_path,
            data_path=data_path,
            imgsz=args.imgsz,
            split="test",
            device=args.device,
            batch=args.batch,
            workers=args.workers,
            conf=args.conf,
            iou=args.iou,
        )
        print_metrics(val_payload, split="val")
        print_metrics(test_payload, split="test")
        delta_payload = build_metric_delta(val_payload, test_payload)
        print_metric_delta(delta_payload)
    else:
        metrics_payload = run_validation(
            model_path=model_path,
            data_path=data_path,
            imgsz=args.imgsz,
            split=args.split,
            device=args.device,
            batch=args.batch,
            workers=args.workers,
            conf=args.conf,
            iou=args.iou,
        )
        print_metrics(metrics_payload, split=args.split)

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    output_file = output_dir / f"metrics_{timestamp}{tag}.json"

    payload = {
        "model": str(model_path),
        "data": str(data_path),
        "split": args.split,
        "imgsz": args.imgsz,
        "device": args.device,
        "batch": args.batch,
        "workers": args.workers,
        "conf": args.conf,
        "iou": args.iou,
        "compare_val_test": compare_mode,
    }

    if compare_mode:
        payload["metrics_by_split"] = {
            "val": val_payload,
            "test": test_payload,
        }
        payload["delta_test_minus_val"] = delta_payload
    else:
        payload["metrics"] = metrics_payload

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\nSaved metrics JSON to: {output_file}")


if __name__ == "__main__":
    main()
