# ECAFA-YOLO

Code for **ECAFA-YOLO: A YOLO-based framework with cross alternating feature aggregation and knowledge distillation for traffic sign detection**.

This repository includes the custom CAFA module, the E-ASFF detection head, and their combined ECAFA-YOLO architecture. The public entry points below cover dataset preparation, supervised training, offline feature distillation, and validation/test evaluation.

## Installation

Use Python 3.9 or later and a PyTorch environment appropriate for your hardware. Run the following from the repository root:

```bash
python -m pip install -r requirements.txt
```

This installs the local modified Ultralytics package in editable mode, plus `iterative-stratification` for dataset preparation. The custom models require this repository's implementation. GPU training requires a CUDA-enabled PyTorch installation; `--device cpu` is available for CPU execution.

## Files

```text
ECAFA-YOLO/
├── train.py                         # Supervised training
├── train_distill.py                 # Offline teacher-student feature distillation
├── val.py                           # Validation/test evaluation
├── requirements.txt
├── scripts/
│   ├── prepare_tt100k.py             # TT100K conversion, filtering, and final split
│   └── prepare_gtsdb.py              # Full GTSDB conversion and three-way split
└── ultralytics/
    ├── nn/ECAFA_modules/             # CAFA and E-ASFF implementation
    └── cfg/models/11/
        ├── yolo11.yaml              # Baseline
        ├── yolo11-CAFA.yaml
        ├── yolo11-EASFFHead.yaml
        └── yolo11-CAFA-EASFFHead.yaml
```

## Supplementary materials and pretrained models

Download [ECAFA-YOLO Supplementary Materials from Google Drive](https://drive.google.com/file/d/1IUi6OOLZxmTx5NPoAt1LkdY374ns2U0K/view?usp=drive_link). The package contains experimental records and pretrained checkpoints for **TT100K** and **GTSDB**:

- `Experimental_Results/`: training configurations (`args.yaml`) and epoch-wise training/validation results (`results.csv`).
- `Models/`: best checkpoints for YOLOv11, YOLOv11 + CAFA, YOLOv11 + E-ASFF, ECAFA-YOLO, ECAFA-YOLO + KD, and the YOLO11x-based distillation teacher.
- `README.md`: the correspondence between the folders and the paper's models, software versions, and evaluation examples.

After downloading and extracting the package, place `ECAFA-YOLO-Supplementary-Materials/` in the project root. Readers can use the checkpoints with this repository and the prepared datasets to run inference and evaluate the proposed method. For example, evaluate the TT100K model trained with knowledge distillation:

```bash
python val.py --weights ECAFA-YOLO-Supplementary-Materials/Models/TT100K/ECAFA_YOLO_KD_best.pt --data ultralytics/cfg/datasets/tt100k.yaml --split test --device 0
```

## Data availability and downloads

The data supporting the findings of this study are publicly available from these official dataset pages:

- **TT100K:** [Download and dataset information](https://cg.cs.tsinghua.edu.cn/traffic-sign).
- **GTSDB:** [Download and dataset information](https://benchmark.ini.rub.de/gtsdb_dataset.html). Use the full 900-image release.

Download and extract the original datasets yourself. All local paths below are relative to the project root; run the commands from that directory. Place the extracted raw data in `datasets/raw/tt100k/` and `datasets/raw/FullIJCNN2013/`. The preparation scripts read local files; they do not download datasets. Dataset images and annotations are not distributed with the code.

### TT100K

Use the TT100K release containing `annotations_all.json` with the original category list, together with the official `train/` and `test/` images:

```text
datasets/raw/tt100k/
├── annotations_all.json
├── train/
└── test/
```

```bash
python scripts/prepare_tt100k.py --source datasets/raw/tt100k
```

The script converts the annotations to YOLO format, filters categories, and generates the training, validation, and test sets under `datasets/tt100k/`:

```text
datasets/tt100k/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── tt100k.yaml
```

The prepared dataset contains 130 categories, with 4,838 training images, 3,014 validation images, and 1,191 test images. Image and label folders are created automatically. The script also saves the actual split counts in `split_summary.json`.

### GTSDB

Download the **full 900-image GTSDB release**, including its complete `gt.txt`. Point `--source` to the extracted directory that directly contains the images and annotation file:

```text
datasets/raw/FullIJCNN2013/
├── gt.txt
├── 00000.ppm
├── 00001.ppm
└── ...
```

```bash
python scripts/prepare_gtsdb.py --source datasets/raw/FullIJCNN2013
```

The single script follows the original five-step full-900-image preparation workflow:

1. Convert `gt.txt` to YOLO labels using the original fixed image dimensions of **1360 × 800**, preserving class IDs 0–42 and six decimal places.
2. Count the annotation instances for each class and write `class_count.csv`.
3. Sort by label filename, exclude empty/unannotated images, and build a multi-label class-presence vector for each image. Use `MultilabelStratifiedKFold` to first select the test partition, then select validation from the remainder, targeting 70%/15%/15%. Select the fold closest to the target size. Search up to 200 candidates starting at 42, compare validation coverage first and test coverage second, and stop when both cover all 43 classes. Whole-fold selection makes the final ratios approximate.
4. Find each selected image in `.ppm`, `.jpg`, `.jpeg`, `.png` order. Copy existing JPEGs directly. Convert PPM/PNG to RGB JPEG with Pillow using `quality=95, optimize=True`, matching the original image-conversion script.
5. Count classes and boxes in each output split and write `class_distribution.csv` and `split_summary.json`.

Use the full release rather than the smaller training-only archive. The local prepared dataset checked against this workflow contains 524 training, 108 validation, and 109 test images. The resulting split sizes and missing classes, if any, are recorded in `split_summary.json`.

### Preparation outputs

By default, both scripts write **inside this project**, regardless of the terminal's current directory:

| Dataset | Default output directory | Generated YAML |
| --- | --- | --- |
| TT100K | `datasets/tt100k/` | `datasets/tt100k/tt100k.yaml` |
| GTSDB | `datasets/gtsdb/` | `datasets/gtsdb/gtsdb.yaml` |

Use `--output datasets/custom_output` to override the default with a new or empty output directory.

The output directory must be new or empty. The scripts leave the original downloaded data intact; they do not rename or overwrite your existing prepared folders. Each writes:

- `images/{train,val,test}/` and matching `labels/{train,val,test}/`;
- `tt100k.yaml` or `gtsdb.yaml`, ready for the training and evaluation commands;
- `classes.txt`, three relative-path split manifests, and `split_summary.json`;
- TT100K additionally writes `class_mapping.json` with category-name-to-new-ID mappings;
- GTSDB additionally writes `class_count.csv` and `class_distribution.csv`.

Use `--annotations datasets/raw/tt100k/annotations_all.json` or `--annotations datasets/raw/FullIJCNN2013/gt.txt` if the annotation file is stored separately. Update the generated YAML's `path` field if you move the prepared data. Quote paths containing spaces on Windows or Linux. The split manifests are records of the prepared images; the YAML uses the image directories directly.

### Dataset configurations used by the commands

The training, distillation, and evaluation commands below use:

- `ultralytics/cfg/datasets/GTSDB.yaml`
- `ultralytics/cfg/datasets/tt100k.yaml`

After preparing both datasets, copy the generated YAML files to the configuration paths used by the commands below. Run this from the project root:

```bash
python -c "from shutil import copyfile; copyfile('datasets/gtsdb/gtsdb.yaml', 'ultralytics/cfg/datasets/GTSDB.yaml'); copyfile('datasets/tt100k/tt100k.yaml', 'ultralytics/cfg/datasets/tt100k.yaml')"
```

These configurations contain the dataset root, class names, and paths to `images/train`, `images/val`, and `images/test`.

## Training

Run the commands from the repository root after synchronizing the dataset configurations as described above. Each dataset has four model configurations below. Experiment names are distinct so their checkpoints can be compared separately.

### TT100K

```bash
# YOLO11 baseline
python train.py --data ultralytics/cfg/datasets/tt100k.yaml --model ultralytics/cfg/models/11/yolo11.yaml --scale n --epochs 200 --batch 64 --device 0 --name tt100k_yolo11

# YOLO11 + CAFA
python train.py --data ultralytics/cfg/datasets/tt100k.yaml --model ultralytics/cfg/models/11/yolo11-CAFA.yaml --scale n --epochs 200 --batch 64 --device 0 --name tt100k_cafa

# YOLO11 + E-ASFF
python train.py --data ultralytics/cfg/datasets/tt100k.yaml --model ultralytics/cfg/models/11/yolo11-EASFFHead.yaml --scale n --epochs 200 --batch 64 --device 0 --name tt100k_easff

# ECAFA-YOLO: CAFA + E-ASFF (final model)
python train.py --data ultralytics/cfg/datasets/tt100k.yaml --model ultralytics/cfg/models/11/yolo11-CAFA-EASFFHead.yaml --scale n --epochs 200 --batch 64 --device 0 --name tt100k_ecafa
```

### GTSDB

```bash
# YOLO11 baseline
python train.py --data ultralytics/cfg/datasets/GTSDB.yaml --model ultralytics/cfg/models/11/yolo11.yaml --scale n --epochs 200 --batch 64 --device 0 --name gtsdb_yolo11

# YOLO11 + CAFA
python train.py --data ultralytics/cfg/datasets/GTSDB.yaml --model ultralytics/cfg/models/11/yolo11-CAFA.yaml --scale n --epochs 200 --batch 64 --device 0 --name gtsdb_cafa

# YOLO11 + E-ASFF
python train.py --data ultralytics/cfg/datasets/GTSDB.yaml --model ultralytics/cfg/models/11/yolo11-EASFFHead.yaml --scale n --epochs 200 --batch 64 --device 0 --name gtsdb_easff

# ECAFA-YOLO: CAFA + E-ASFF (final model)
python train.py --data ultralytics/cfg/datasets/GTSDB.yaml --model ultralytics/cfg/models/11/yolo11-CAFA-EASFFHead.yaml --scale n --epochs 200 --batch 64 --device 0 --name gtsdb_ecafa
```

### Model selection and defaults

Select the architecture with `--model`:

| Configuration | `--model` |
| --- | --- |
| YOLO11 baseline | `ultralytics/cfg/models/11/yolo11.yaml` |
| YOLO11 + CAFA | `ultralytics/cfg/models/11/yolo11-CAFA.yaml` |
| YOLO11 + E-ASFF | `ultralytics/cfg/models/11/yolo11-EASFFHead.yaml` |
| ECAFA-YOLO | `ultralytics/cfg/models/11/yolo11-CAFA-EASFFHead.yaml` |

If `--model` is omitted, `train.py` uses `yolo11-CAFA.yaml`. To train the final combined model, explicitly pass `--model ultralytics/cfg/models/11/yolo11-CAFA-EASFFHead.yaml` as shown above. `--scale` accepts `n`, `s`, `m`, `l`, or `x`; unscaled YAML filenames default to `n`. For a larger supervised model, use `--scale x` and reduce `--batch` as needed. All model configurations are already in the repository.

The active training defaults match the original GTSDB and TT100K training scripts; dataset and output paths remain configurable:

| Setting | Default |
| --- | --- |
| Task | Detection (`detect`) |
| Epochs / image size / batch size | 200 / 640 / 64 |
| Device | `cuda` (the commands above explicitly select GPU `0`) |
| Data-loader workers / CPU threads | 16 / 8 |
| Dataset cache | `ram` |
| Optimizer / initial learning rate | SGD / 0.01 |
| Learning-rate schedule | Linear decay, `lrf=0.01`, `cos_lr=False` |
| Mosaic / MixUp / Copy-Paste | 0.0 / 0.0 / 0.0 |
| `close_mosaic` | 9999 |
| Random seed / deterministic training | 42 / enabled |
| Resume / single-class training | Disabled / disabled |
| Automatic mixed precision (AMP) | Enabled |
| Output project / default experiment name | `runs/train` / `exp` |

Model structure and GFLOPs are printed before training. Adjust `--batch`, `--workers`, `--cpu-threads`, and `--device` for your machine. Cache can be changed with `--cache ram`, `--cache disk`, or `--cache false`. Use `--no-amp` to disable mixed precision; `--seed` and `--no-deterministic` override the initialization settings when needed.

A YAML initializes a model from scratch. Passing a local `.pt` file to `--model` starts a new fine-tuning run from that checkpoint; omit `--scale` in that case. Add `--resume` with a resumable checkpoint such as `last.pt` to continue an interrupted run. `--single-cls` enables single-class training.

Results are written under `runs/train/<name>/`, including `weights/best.pt`, `weights/last.pt`, `args.yaml`, and `results.csv`. Existing experiment names may receive a numeric suffix; use the output directory printed by training.

No pretrained ECAFA-YOLO weights are automatically downloaded by these scripts.

## Knowledge distillation

Use `train_distill.py` on either dataset. This is a command-line version of the existing `train_distillation_GTSDB.py` workflow: it loads the teacher, sets `set_Distillation = True` on its detection head, loads the student, and passes `model_t=teacher.model` to `student.train()`. It uses the repository's existing distillation implementation.

First train a teacher and a student on the same prepared dataset and class mapping. The intended pair is ECAFA-YOLO-x as teacher and ECAFA-YOLO-n as student.

For example, train the GTSDB teacher (`--scale x`) and use the supervised student from the previous section:

```bash
python train.py --data ultralytics/cfg/datasets/GTSDB.yaml --model ultralytics/cfg/models/11/yolo11-CAFA-EASFFHead.yaml --scale x --epochs 200 --batch 16 --device 0 --name gtsdb_teacher
python train_distill.py --data ultralytics/cfg/datasets/GTSDB.yaml --teacher runs/train/gtsdb_teacher/weights/best.pt --student runs/train/gtsdb_ecafa/weights/best.pt --epochs 200 --batch 16 --device 0 --name gtsdb_ecafa_kd
```

For TT100K, prepare a teacher in the same way and pass the corresponding checkpoints:

```bash
python train.py --data ultralytics/cfg/datasets/tt100k.yaml --model ultralytics/cfg/models/11/yolo11-CAFA-EASFFHead.yaml --scale x --epochs 200 --batch 16 --device 0 --name tt100k_teacher
python train_distill.py --data ultralytics/cfg/datasets/tt100k.yaml --teacher runs/train/tt100k_teacher/weights/best.pt --student runs/train/tt100k_ecafa/weights/best.pt --epochs 200 --batch 16 --device 0 --name tt100k_ecafa_kd
```

The student can also start from a YAML by replacing `--student` with `ultralytics/cfg/models/11/yolo11-CAFA-EASFFHead.yaml --scale n`. A `.pt` student initializes from its saved weights; omit `--scale` in that case. Checkpoints must use the same dataset class order. Use `--device 0` to select the first GPU.

Defaults follow the original GTSDB distillation script: 200 epochs, batch size 16, input size 640, SGD, early-stopping patience 50, and `close_mosaic=10`. Cache is disabled and AMP is enabled; use `--no-amp` to disable it. Other KD settings are controlled by the existing trainer implementation.

KD results are saved under `runs/distill/<name>/`. Evaluate the trained student using `weights/best.pt` with the same `val.py`:

```bash
python val.py --weights runs/distill/gtsdb_ecafa_kd/weights/best.pt --data ultralytics/cfg/datasets/GTSDB.yaml --split test --device 0 --name gtsdb_ecafa_kd_test
```

## Validation and test evaluation

Evaluate the final TT100K test partition:

```bash
python val.py --weights runs/train/tt100k_ecafa/weights/best.pt --data ultralytics/cfg/datasets/tt100k.yaml --split test --device 0 --name tt100k_ecafa_test
```

Evaluate GTSDB:

```bash
python val.py --weights runs/train/gtsdb_ecafa/weights/best.pt --data ultralytics/cfg/datasets/GTSDB.yaml --split test --device 0 --name gtsdb_ecafa_test
```

Use `--split val` to evaluate the validation partition. The default evaluation settings are image size 640, confidence threshold 0.001, NMS IoU threshold 0.7, and at most 300 detections per image. `val.py` reports Precision, Recall, mAP50, and mAP50-95, and saves them to `runs/val/<name>/metrics.json` alongside the framework's evaluation outputs. Add `--save-json` to export detection predictions.

```bash
python train.py --help
python train_distill.py --help
python val.py --help
python scripts/prepare_tt100k.py --help
python scripts/prepare_gtsdb.py --help
```

## TT100K degraded test images

Download `test_images_degraded.rar` from [Google Drive — TT100K degraded test dataset](https://drive.google.com/file/d/1nulD0Iq9i3qmf-qPGlDW6ne3lA02ZXkJ/view?usp=drive_link). It contains seven conditions, each with 1,191 images: `test_fog`, `test_jpeg`, `test_lowlight`, `test_motionblur`, `test_occlusion`, `test_rain`, and `test_scale`. These images are for robustness testing with an already trained model.

Extract the archive and place the seven condition folders directly under `datasets/tt100k/images/` (remove the extra `test_images_degraded/` wrapper level if present). The image names correspond to the clean test images. Reuse the clean-test bounding-box labels from `datasets/tt100k/labels/test/` and keep the same 130-class mapping.

The degraded-image folders sit alongside the clean `test/` folder inside the project:

```text
datasets/tt100k/images/
├── train/
├── val/
├── test/
├── test_fog/
├── test_jpeg/
├── test_lowlight/
├── test_motionblur/
├── test_occlusion/
├── test_rain/
└── test_scale/
```

For example, after placing fog images in `datasets/tt100k/images/test_fog/`, run this Python snippet from the project root to copy the matching clean-test labels and create a separate evaluation YAML:

```python
from pathlib import Path
import shutil
import yaml

root = Path("datasets/tt100k").resolve()
condition = "test_fog"
shutil.copytree(root / "labels/test", root / "labels" / condition, dirs_exist_ok=True)
data = yaml.safe_load(Path("ultralytics/cfg/datasets/tt100k.yaml").read_text(encoding="utf-8"))
data.update(path=root.as_posix(), train="images/train", val="images/val", test=f"images/{condition}")
(root / f"{condition}.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
```

Evaluate the trained student on that condition:

```bash
python val.py --weights runs/distill/tt100k_ecafa_kd/weights/best.pt --data datasets/tt100k/test_fog.yaml --split test --device 0 --name tt100k_fog
```

Change `condition` and the evaluation YAML/output name for the other six conditions. Use the same checkpoint and evaluation settings for clean and degraded tests. The labels must retain the same file stems as the degraded images; keep separate `labels/<condition>/` directories so Ultralytics can locate them from `images/<condition>/`.

## Acknowledgements and license

This project builds on [Ultralytics YOLO](https://github.com/ultralytics/ultralytics). We thank the TT100K and GTSDB authors for making their datasets available. See [LICENSE](LICENSE) for the repository license.
