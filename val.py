"""Evaluate a trained checkpoint on a dataset's validation or test split."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0", help="GPU index or cpu")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--project", default="runs/val")
    parser.add_argument("--name", default="ecafa")
    parser.add_argument("--save-json", action="store_true", help="Also export detection predictions as JSON")
    return parser.parse_args()


def main():
    args = parse_args()
    for path in (args.weights, args.data):
        if not path.is_file():
            raise FileNotFoundError(path)
    from ultralytics import YOLO

    model = YOLO(str(args.weights.resolve()), task="detect")
    metrics = model.val(
        data=str(args.data.resolve()), split=args.split, imgsz=args.imgsz,
        batch=args.batch, device=args.device, workers=args.workers,
        conf=args.conf, iou=args.iou, max_det=args.max_det,
        project=args.project, name=args.name, save_json=args.save_json,
    )
    summary = {
        "weights": str(args.weights.resolve()), "data": str(args.data.resolve()),
        "split": args.split, "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr), "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
    }
    output = Path(metrics.save_dir) / "metrics.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Metrics saved to {output}")


if __name__ == "__main__":
    main()
