"""Convert TT100K annotations and prepare the final train/val/test split.

The official training subset is split 80:20 using multilabel stratification.
Classes absent from either partition are removed, IDs are remapped in the
original `types` order, and images with no remaining boxes are omitted.
The SAME class mapping is applied to the official test subset.

The historical val/test exchange is already applied to the output:
  train = 80% partition of official train
  val   = filtered official test
  test  = 20% held-out partition of official train
Do not swap the generated folders or YAML entries again.
"""

import argparse
import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "datasets" / "tt100k"
# New output -> folders in the author's existing prepared dataset.
EXISTING_FOLDER_MAPPING = {"train": "train", "test": "val", "val": "test"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="Extracted TT100K root containing train/ and test/")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="New or empty output directory (default: PROJECT/datasets/tt100k)")
    parser.add_argument("--annotations", type=Path, help="Defaults to SOURCE/annotations_all.json")
    return parser.parse_args()


def build_split(annotation):
    import numpy as np
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

    names = annotation["types"]
    if not isinstance(names, list) or len(names) != len(set(names)):
        raise ValueError("Expected a unique ordered list in annotations['types']")
    class_ids = {name: i for i, name in enumerate(names)}
    subsets = {"train": [], "test": []}
    for item in annotation["imgs"].values():
        relative = Path(item["path"].replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Expected a relative image path: {relative}")
        if relative.parts[0] in subsets:
            subsets[relative.parts[0]].append(item)
    for subset in subsets.values():
        # Match the original sorted label filenames, not numeric image IDs.
        subset.sort(key=lambda item: Path(item["path"]).stem + ".txt")
        stems = [Path(item["path"]).stem for item in subset]
        if len(stems) != len(set(stems)):
            raise ValueError("Duplicate image stems in a source subset")
    records = subsets["train"]
    if len(records) < 2 or not subsets["test"]:
        raise ValueError("Annotations must include both official train and test images")
    labels = np.zeros((len(records), len(names)), dtype=int)
    for i, item in enumerate(records):
        for obj in item["objects"]:
            if obj["category"] in class_ids:
                labels[i, class_ids[obj["category"]]] = 1
    splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_indices, holdout_indices = next(splitter.split(np.arange(len(records)), labels))
    kept = np.flatnonzero((labels[train_indices].sum(0) > 0) & (labels[holdout_indices].sum(0) > 0))
    if len(kept) == 0:
        raise ValueError("No classes are shared by the training partitions")
    selected_names = [names[int(i)] for i in kept]
    remap = {name: i for i, name in enumerate(selected_names)}
    final = {
        "train": [records[i] for i in train_indices],
        "val": subsets["test"],  # Existing test folder -> final val folder.
        "test": [records[i] for i in holdout_indices],  # Existing val folder -> final test folder.
    }
    return selected_names, remap, final


def prepare(source, output=DEFAULT_OUTPUT, annotations=None):
    import yaml
    from PIL import Image

    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Output must be a new or empty directory: {output}")
    annotations = Path(annotations) if annotations else source / "annotations_all.json"
    annotation = json.loads(annotations.read_text(encoding="utf-8"))
    names, remap, splits = build_split(annotation)
    # Check all required image paths before starting the copy.
    for items in splits.values():
        for item in items:
            if any(obj["category"] in remap for obj in item["objects"]):
                image_path = source / item["path"].replace("\\", "/")
                if not image_path.is_file():
                    raise FileNotFoundError(image_path)
    output.mkdir(parents=True, exist_ok=True)
    summary = {"dataset": "TT100K", "classes": len(names), "val_test_exchange_applied": True,
               "output_to_existing_folder": EXISTING_FOLDER_MAPPING, "splits": {}}
    for split, items in splits.items():
        image_dir, label_dir = output / "images" / split, output / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        files, present, box_count = [], set(), 0
        for item in items:
            objects = [obj for obj in item["objects"] if obj["category"] in remap]
            if not objects:
                continue
            source_image = source / item["path"].replace("\\", "/")
            with Image.open(source_image) as image:
                width, height = image.size
            rows = []
            for obj in objects:
                box = obj["bbox"]
                x1, y1, x2, y2 = (float(box[k]) for k in ("xmin", "ymin", "xmax", "ymax"))
                if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                    raise ValueError(f"Invalid box in {source_image}: {box}")
                class_id = remap[obj["category"]]
                rows.append(f"{class_id} {(x1+x2)/2/width:.6f} {(y1+y2)/2/height:.6f} {(x2-x1)/width:.6f} {(y2-y1)/height:.6f}")
                present.add(class_id)
            shutil.copy2(source_image, image_dir / source_image.name)
            (label_dir / (source_image.stem + ".txt")).write_text("\n".join(rows) + "\n", encoding="utf-8")
            files.append(f"images/{split}/{source_image.name}")
            box_count += len(rows)
        (output / f"{split}.txt").write_text("\n".join(files) + "\n", encoding="utf-8")
        summary["splits"][split] = {"images": len(files), "boxes": box_count, "classes_present": len(present),
                                    "missing_classes": [names[i] for i in range(len(names)) if i not in present]}
    config = {"path": output.as_posix(), "train": "images/train", "val": "images/val", "test": "images/test",
              "nc": len(names), "names": names}
    (output / "tt100k.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    (output / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    (output / "class_mapping.json").write_text(json.dumps(remap, indent=2) + "\n", encoding="utf-8")
    (output / "split_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Dataset YAML: {output / 'tt100k.yaml'}")
    return summary


if __name__ == "__main__":
    args = parse_args()
    prepare(args.source, args.output, args.annotations)
