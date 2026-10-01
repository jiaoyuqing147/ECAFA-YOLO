"""Convert the full GTSDB gt.txt and prepare YOLO train/val/test folders.

Uses the original full-900-image workflow: annotated images only, 43 original
class IDs, multilabel K-fold partitioning towards 70/15/15, and a search for
validation/test class coverage. Ratios are approximate because whole folds
are selected. Images without annotations are excluded; classes are not remapped.
"""

import argparse
import csv
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "datasets" / "gtsdb"
IMAGE_WIDTH, IMAGE_HEIGHT = 1360, 800  # Original gt-to-YOLO conversion dimensions.
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15
BASE_SEED, MAX_TRIALS = 42, 200

NAMES = [
    "speed limit 20", "speed limit 30", "speed limit 50", "speed limit 60",
    "speed limit 70", "speed limit 80", "restriction ends 80", "speed limit 100",
    "speed limit 120", "no overtaking", "no overtaking (trucks)", "priority at next intersection",
    "priority road", "give way", "stop", "no traffic both ways", "no trucks", "no entry",
    "danger", "bend left", "bend right", "bend", "uneven road", "slippery road", "road narrows",
    "construction", "traffic signal", "pedestrian crossing", "school crossing", "cycles crossing",
    "snow", "animals", "restriction ends", "go right", "go left", "go straight",
    "go right or straight", "go left or straight", "keep right", "keep left", "roundabout",
    "restriction ends (overtaking)", "restriction ends (overtaking trucks)",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="FullIJCNN2013 directory containing images and gt.txt")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="New or empty output directory (default: PROJECT/datasets/gtsdb)")
    parser.add_argument("--annotations", type=Path, help="Defaults to SOURCE/gt.txt")
    return parser.parse_args()


def read_annotations(path):
    records = defaultdict(list)
    for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.strip().split(";")
        if len(parts) != 6:
            raise ValueError(f"Expected six semicolon-separated fields at line {number}")
        filename = parts[0]
        if Path(filename).name != filename or "\\" in filename:
            raise ValueError(f"Expected an image filename at line {number}")
        x1, y1, x2, y2 = map(int, parts[1:5])
        class_id = int(parts[5])
        if not 0 <= class_id < len(NAMES):
            raise ValueError(f"Invalid GTSDB class ID at line {number}: {class_id}")
        records[filename].append((class_id, x1, y1, x2, y2))
    if not records:
        raise ValueError("No annotations found")
    stems = [Path(name).stem for name in records]
    if len(stems) != len(set(stems)):
        raise ValueError("Duplicate image stems in gt.txt")
    return dict(sorted(records.items(), key=lambda item: Path(item[0]).stem + ".txt"))


def build_split(records):
    import numpy as np
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

    # Same order as sorted(glob('labels/*.txt')) in the original splitting script.
    filenames = sorted(records, key=lambda name: Path(name).stem + ".txt")
    labels = np.zeros((len(filenames), len(NAMES)), dtype=int)
    for i, name in enumerate(filenames):
        for class_id, *_ in records[name]:
            labels[i, class_id] = 1
    if len(filenames) < 14:
        raise ValueError("Too few annotated images for the original K-fold split; use the full dataset")

    def select_fold(indices, ratio, state):
        count = len(indices)
        target = max(1, min(int(round(count * ratio)), count - 1))
        splitter = MultilabelStratifiedKFold(n_splits=max(2, int(round(1 / ratio))), shuffle=True, random_state=state)
        best, min_gap = None, float("inf")
        for remain, part in splitter.split(np.zeros((count, 1)), labels[indices]):
            gap = abs(len(part) - target)
            if gap < min_gap:
                best, min_gap = (remain, part), gap
        remain, part = best
        return [indices[i] for i in remain], [indices[i] for i in part]

    best_score, best_result = (-1, -1), None
    # Preserve the original search: test first, then val from the remainder;
    # maximize (val coverage, test coverage) lexicographically, retaining ties.
    for state in range(BASE_SEED, BASE_SEED + MAX_TRIALS):
        indices = list(range(len(filenames)))
        random.Random(state).shuffle(indices)
        remain, test = select_fold(indices, TEST_RATIO, state)
        train, val = select_fold(remain, VAL_RATIO / (1.0 - TEST_RATIO), state + 1)
        score = (int((labels[val].sum(0) > 0).sum()), int((labels[test].sum(0) > 0).sum()))
        if score > best_score:
            best_score, best_result = score, (train, val, test)
            if score == (len(NAMES), len(NAMES)):
                break
    return {split: [filenames[i] for i in indices]
            for split, indices in zip(("train", "val", "test"), best_result)}


def find_source_image(source, filename):
    """Use the original image lookup priority: ppm, jpg, jpeg, then png."""
    for extension in (".ppm", ".jpg", ".jpeg", ".png"):
        candidate = source / (Path(filename).stem + extension)
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No source image found for {filename} in {source}")


def prepare(source, output=DEFAULT_OUTPUT, annotations=None):
    import yaml
    from PIL import Image

    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Output must be a new or empty directory: {output}")
    records = read_annotations(annotations or source / "gt.txt")
    splits = build_split(records)
    source_images = {name: find_source_image(source, name) for name in records}
    output.mkdir(parents=True, exist_ok=True)
    summary = {"dataset": "GTSDB", "classes": len(NAMES), "splits": {}}
    total_counts = Counter(row[0] for rows in records.values() for row in rows)
    with (output / "class_count.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("class_id", "count"))
        writer.writerows(sorted(total_counts.items()))
    split_counts = {}
    for split, filenames in splits.items():
        image_dir, label_dir = output / "images" / split, output / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        files, present, box_count = [], set(), 0
        counts = Counter()
        for filename in filenames:
            width, height = IMAGE_WIDTH, IMAGE_HEIGHT
            rows = []
            for class_id, x1, y1, x2, y2 in records[filename]:
                if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                    raise ValueError(f"Invalid box in {filename}: {(x1, y1, x2, y2)}")
                rows.append(f"{class_id} {(x1+x2)/2/width:.6f} {(y1+y2)/2/height:.6f} {(x2-x1)/width:.6f} {(y2-y1)/height:.6f}")
                present.add(class_id)
                counts[class_id] += 1
            stem = Path(filename).stem
            source_image = source_images[filename]
            destination = image_dir / f"{stem}.jpg"
            # Match original step 4: copy JPEGs, convert PPM/PNG with Pillow.
            if source_image.suffix.lower() in (".jpg", ".jpeg"):
                shutil.copy2(source_image, destination)
            else:
                with Image.open(source_image) as image:
                    image.convert("RGB").save(destination, format="JPEG", quality=95, optimize=True)
            (label_dir / f"{stem}.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
            files.append(f"images/{split}/{stem}.jpg")
            box_count += len(rows)
        (output / f"{split}.txt").write_text("\n".join(files) + "\n", encoding="utf-8")
        summary["splits"][split] = {"images": len(files), "boxes": box_count, "classes_present": len(present),
                                    "missing_class_ids": [i for i in range(len(NAMES)) if i not in present]}
        split_counts[split] = counts
    with (output / "class_distribution.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("class_id", "train_count", "val_count", "test_count"))
        for class_id in range(len(NAMES)):
            writer.writerow((class_id, *(split_counts[split][class_id] for split in ("train", "val", "test"))))
    config = {"path": output.as_posix(), "train": "images/train", "val": "images/val", "test": "images/test",
              "nc": len(NAMES), "names": NAMES}
    (output / "gtsdb.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (output / "classes.txt").write_text("\n".join(NAMES) + "\n", encoding="utf-8")
    (output / "split_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Dataset YAML: {output / 'gtsdb.yaml'}")
    return summary


if __name__ == "__main__":
    args = parse_args()
    prepare(args.source, args.output, args.annotations)
