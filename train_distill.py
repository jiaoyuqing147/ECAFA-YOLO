"""Offline feature distillation for ECAFA-YOLO on TT100K or GTSDB."""

import argparse
import re
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Prepared dataset YAML")
    parser.add_argument("--teacher", type=Path, required=True, help="Trained teacher checkpoint (.pt)")
    parser.add_argument("--student", type=Path, required=True, help="Trained student checkpoint (.pt) or model YAML")
    parser.add_argument("--scale", choices=("n", "s", "m", "l", "x"), help="Student YAML scale; omit for .pt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0", help="One GPU index (e.g. 0 or 1), or cpu")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--optimizer", default="SGD")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--project", default="runs/distill")
    parser.add_argument("--name", default="ecafa_kd")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    for path in (args.data, args.teacher, args.student):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.teacher.suffix.lower() != ".pt":
        raise ValueError("--teacher must be a trained .pt checkpoint")
    student_path = args.student.resolve()
    if student_path.suffix.lower() in {".yaml", ".yml"}:
        scale = args.scale
        if scale is None and not re.match(r"yolo11[nslmx]", student_path.stem):
            scale = "n"
        if scale is not None:
            if not re.match(r"yolo11(?:[nslmx])?(?:-|$)", student_path.stem):
                raise ValueError("--scale expects a yolo11[-...].yaml filename")
            student_path = student_path.with_name(re.sub(r"^yolo11[nslmx]?", "yolo11" + scale, student_path.name))
    elif student_path.suffix.lower() != ".pt" or args.scale is not None:
        raise ValueError("Use a student YAML with --scale, or a .pt checkpoint without --scale")

    from ultralytics import YOLO
    teacher = YOLO(str(args.teacher.resolve()), task="detect")
    teacher.model.model[-1].set_Distillation = True
    student = YOLO(str(student_path), task="detect")
    # Use the same teacher hand-off as train_distillation_GTSDB.py.
    student.train(
        data=str(args.data.resolve()), model_t=teacher.model,
        epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, workers=args.workers, patience=args.patience,
        optimizer=args.optimizer, lr0=args.lr0,
        cache=False, amp=not args.no_amp, project=args.project, name=args.name,
        single_cls=False, close_mosaic=10,
    )


if __name__ == "__main__":
    main()
