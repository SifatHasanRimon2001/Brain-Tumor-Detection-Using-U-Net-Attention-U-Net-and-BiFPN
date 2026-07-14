# Brain Tumor Detection Using U-Net, Attention U-Net & BiFPN

[![GitHub contributors](https://img.shields.io/github/contributors/SifatHasanRimon2001/brain-tumor-detection-using-unet-attunet-bifpn)](https://github.com/SifatHasanRimon2001/brain-tumor-detection-using-unet-attunet-bifpn/graphs/contributors)

Multi-task deep learning project for brain tumor analysis using MRI scans from the **BRISC2025** dataset. Implements three architectures across three tasks with segmentation, classification, and joint segmentation + classification.

| Task      | Description                                                                    |
| --------- | ------------------------------------------------------------------------------ |
| `seg`   | Binary segmentation — predict tumor mask                                      |
| `cls`   | 4-class classification — tumor type (glioma, meningioma, no_tumor, pituitary) |
| `joint` | Segmentation + classification in a single model                                |

| Model                     | Key Feature                                     | Paper                                               |
| ------------------------- | ----------------------------------------------- | --------------------------------------------------- |
| **U-Net**           | Baseline encoder–decoder with skip connections | [arXiv:1505.04597](https://arxiv.org/abs/1505.04597) |
| **Attention U-Net** | Additive attention gates on skip connections    | [arXiv:1804.03999](https://arxiv.org/abs/1804.03999) |
| **BiFPN U-Net**     | Weighted bi-directional feature pyramid fusion  | [arXiv:1911.09070](https://arxiv.org/abs/1911.09070) |

---

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Results](#results)
- [Recommended Training Config](#recommended-training-config-rtx-4070-laptop--8-gb-vram)
- [Joint Model Checkpoint Issue](#joint-model-checkpoint-issue--root-cause)
- [Issues Fixed](#issues-fixed)
- [Structural Improvements](#structural-improvements)
- [Performance Optimizations](#performance-optimizations)
- [Remaining Limitations](#remaining-limitations)
- [Future Improvements](#future-improvements)
- [Contributors & Credits](#contributors--credits)
- [References](#references)

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/SifatHasanRimon2001/brain-tumor-detection-using-unet-attunet-bifpn.git
cd brain-tumor-detection-using-unet-attunet-bifpn
```

### 2. Create virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux / macOS
```

### 3. Install PyTorch (CUDA)

Choose the command that matches your CUDA version:

```bash
# CUDA 12.8 (latest) — recommended for RTX 40-series
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 5. Verify installation

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0))"
```

Expected output (example):

```
2.11.0+cu128
True
1
NVIDIA GeForce RTX 4070 Laptop GPU
```

---

## Dataset

We use the **BRISC 2025 MRI dataset**:

- [Download from Kaggle](https://www.kaggle.com/datasets/briscdataset/brisc2025?resource=download)

Place the dataset in a folder called `./brisc2025` at the project root.

The dataset contains T1-weighted contrast-enhanced MRI scans organized into four classes:
`glioma`, `meningioma`, `no_tumor`, and `pituitary`. Each sample has both an MRI image and
a corresponding tumor segmentation mask.

---

## Project Structure

```
├── train.py                          # Training entry point
├── infer.py                          # Inference / evaluation entry point
├── visualize.py                      # Model comparison visualization
├── pyproject.toml                    # Project metadata & dependencies (PEP 621)
├── requirements.txt                  # Python dependencies
├── .gitignore
├── .editorconfig                     # Editor formatting rules
├── .pre-commit-config.yaml           # Pre-commit hooks (ruff linting)
├── README.md
│
├── docs/                             # Reference papers & guidelines
│   ├── U-Net.pdf
│   ├── Attention U-Net.pdf
│   ├── EfficientDet.pdf
│   └── Project Guidelines in Summary.pdf
│
├── scripts/                          # Automation scripts
│   ├── run_all.ps1                   # Full batch pipeline (train + infer × 9 configs)
│   └── check_torch_version.py        # CUDA / PyTorch compatibility check
│
├── tests/                            # pytest test suite
│   ├── __init__.py
│   ├── conftest.py                   # Shared fixtures (models, dummy data)
│   ├── test_models.py                # Forward pass, shapes, gradient flow
│   ├── test_losses.py                # DiceLoss, CombinedJointLoss
│   ├── test_metrics.py               # Dice, IoU, classification metrics
│   └── test_config.py                # Config dataclass tests
│
├── src/                              # Source package
│   ├── __init__.py                   # Package root (v1.0.0)
│   ├── py.typed                      # PEP 561 type-checking marker
│   ├── config.py                     # Central dataclass-based configuration
│   ├── data/
│   │   ├── __init__.py               # Exports BriscDataset
│   │   └── dataset.py                # BRISC Dataset with transforms
│   ├── models/
│   │   ├── __init__.py               # Model registry & create_model() factory
│   │   ├── components.py             # Shared blocks: DoubleConv, AttentionBlock
│   │   ├── unet.py                   # Standard U-Net
│   │   ├── attention_unet.py         # Attention U-Net
│   │   ├── bifpn_unet.py             # BiFPN-enhanced U-Net
│   │   ├── losses.py                 # DiceLoss, CombinedJointLoss
│   │   └── metrics.py                # Dice, IoU, classification metrics
│   ├── training/
│   │   ├── __init__.py               # Exports Trainer
│   │   └── trainer.py                # Training loop with AMP, compile, WandB
│   ├── inference/
│   │   ├── __init__.py               # Exports Predictor
│   │   └── predictor.py              # Inference pipeline
│   └── utils/
│       ├── __init__.py               # Exports common utilities
│       ├── transforms.py             # Albumentations pipelines
│       ├── checkpoint.py             # Save / load checkpoints
│       ├── logging.py                # Logger setup
│       ├── reporting.py              # CSV logging & confusion matrices
│       ├── plotting.py               # All plot generation (loss, metrics, ROC, etc.)
│       ├── visualization.py          # Model comparison figures
│       ├── export.py                 # ONNX export for deployment
│       └── misc.py                   # Seed setting
│
├── brisc2025/                        # Dataset (gitignored)
├── runs/                             # Saved checkpoints (auto-created, gitignored)
├── reports/                          # Experiment CSV logs & confusion matrices (gitignored)
├── outputs/                          # Prediction masks from inference (gitignored)
├── exports/                          # ONNX models (auto-created, gitignored)
└── results/                          # Visualization figures (auto-created, gitignored)
```

---

## Usage

### Arguments reference

| Argument            | Default        | Description                                      |
| ------------------- | -------------- | ------------------------------------------------ |
| `--task`          | required       | `seg`, `cls`, or `joint`                   |
| `--model`         | required       | `unet`, `attunet`, or `bifpn`              |
| `--data_root`     | required       | Path to BRISC2025 dataset                        |
| `--img_size`      | `256`        | Image size                                       |
| `--batch`         | `8`          | Batch size                                       |
| `--epochs`        | `50`         | Number of epochs                                 |
| `--lr`            | `3e-4`       | Learning rate                                    |
| `--weight_decay`  | `1e-4`       | Weight decay                                     |
| `--num_workers`   | `4`          | DataLoader workers                               |
| `--no_amp`        | off            | Disable mixed precision                          |
| `--compile`       | off            | Enable `torch.compile()` for 10-20% speedup |
| `--n_folds`       | `0`          | K-fold CV folds (0=disabled, 2-10=enabled)  |
| `--wandb`         | off            | Enable Weights & Biases experiment tracking      |
| `--wandb_project` | `brisc-tumor` | WandB project name                               |
| `--wandb_run_name`| (auto)        | WandB run name (defaults to `{task}_{model}`)   |
| `--seed`          | `42`         | Random seed                                      |

---

### Segmentation

#### U-Net

```powershell
python train.py --task seg --model unet --data_root ./brisc2025 --img_size 256 --batch 8 --epochs 40
python infer.py --task seg --model unet --data_root ./brisc2025 --size 256 --ckpt runs/seg_unet/best.ckpt
```

#### Attention U-Net

```powershell
python train.py --task seg --model attunet --data_root ./brisc2025 --img_size 256 --batch 8 --epochs 40
python infer.py --task seg --model attunet --data_root ./brisc2025 --size 256 --ckpt runs/seg_attunet/best.ckpt
```

#### BiFPN

```powershell
python train.py --task seg --model bifpn --data_root ./brisc2025 --img_size 256 --batch 8 --epochs 40
python infer.py --task seg --model bifpn --data_root ./brisc2025 --size 256 --ckpt runs/seg_bifpn/best.ckpt
```

---

### Classification

#### U-Net encoder

```powershell
python train.py --task cls --model unet --data_root ./brisc2025 --img_size 256 --batch 8 --epochs 25
python infer.py --task cls --model unet --data_root ./brisc2025 --size 256 --ckpt runs/cls_unet/best.ckpt
```

#### Attention U-Net encoder

```powershell
python train.py --task cls --model attunet --data_root ./brisc2025 --img_size 256 --batch 8 --epochs 25
python infer.py --task cls --model attunet --data_root ./brisc2025 --size 256 --ckpt runs/cls_attunet/best.ckpt
```

#### BiFPN encoder

```powershell
python train.py --task cls --model bifpn --data_root ./brisc2025 --img_size 256 --batch 8 --epochs 25
python infer.py --task cls --model bifpn --data_root ./brisc2025 --size 256 --ckpt runs/cls_bifpn/best.ckpt
```

---

### Joint (Segmentation + Classification)

#### U-Net

```powershell
python train.py --task joint --model unet --data_root ./brisc2025 --img_size 256 --batch 8 --epochs 50
python infer.py --task joint --model unet --data_root ./brisc2025 --size 256 --ckpt runs/joint_unet/best.ckpt
```

#### Attention U-Net

```powershell
python train.py --task joint --model attunet --data_root ./brisc2025 --img_size 256 --batch 8 --epochs 50
python infer.py --task joint --model attunet --data_root ./brisc2025 --size 256 --ckpt runs/joint_attunet/best.ckpt
```

#### BiFPN

```powershell
python train.py --task joint --model bifpn --data_root ./brisc2025 --img_size 256 --batch 8 --epochs 50
python infer.py --task joint --model bifpn --data_root ./brisc2025 --size 256 --ckpt runs/joint_bifpn/best.ckpt
```

---

### Additional commands

**Visualize model comparison:**

```bash
python visualize.py --data_root ./brisc2025 --index 42
```

**Run full pipeline (all 9 configurations):**

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_all.ps1
```

**Check CUDA / PyTorch compatibility:**

```bash
python scripts/check_torch_version.py
```

---

## Advanced Features

### K-Fold Cross-Validation

Train with stratified K-fold cross-validation for more robust evaluation:

```powershell
python train.py --task seg --model unet --data_root ./brisc2025 --n_folds 5 --epochs 40
```

Each fold creates its own run directory (e.g., `runs/seg_unet_fold0/`, `runs/seg_unet_fold1/`, ...). Labels are stratified to maintain class balance across folds.

### torch.compile() Acceleration

Enable `torch.compile()` for 10-20% training speedup on RTX 40-series GPUs:

```powershell
python train.py --task seg --model unet --data_root ./brisc2025 --compile
```

### Weights & Biases Experiment Tracking

Track experiments, compare runs, and perform hyperparameter sweeps:

```powershell
# Install wandb first
pip install wandb

# Train with WandB tracking
python train.py --task seg --model unet --data_root ./brisc2025 --wandb --wandb_project my-project

# Custom run name
python train.py --task seg --model bifpn --data_root ./brisc2025 --wandb --wandb_run_name bifpn-seg-lr3e4
```

### ONNX Export

Export trained models to ONNX format for deployment:

```bash
python -m src.utils.export --model unet --task seg --ckpt runs/seg_unet/best.ckpt --output_dir exports
```

### Running Tests

The project includes a comprehensive pytest test suite:

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_models.py -v
```

Tests cover:
- **Model forward passes** — output shapes, dtype, gradient flow, deterministic eval
- **Loss functions** — DiceLoss, CombinedJointLoss, differentiability
- **Metrics** — Dice, IoU, classification metrics, per-class breakdowns
- **Configuration** — dataclass defaults, path resolution, K-fold settings

### Pre-commit Hooks

Set up automated linting and formatting before each commit:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks (runs automatically on `git commit`)
pre-commit install

# Run manually on all files
pre-commit run --all-files
```

Hooks include:
- **ruff** — linting and auto-fix
- **ruff-format** — code formatting
- **trailing-whitespace** / **end-of-file-fixer** — whitespace cleanup
- **check-added-large-files** — prevent committing large files (>500KB)

---

## Results

Evaluation metrics computed on the held-out test set. Segmentation models report **Dice** and **IoU**; classification models report **Accuracy**, **Precision**, **Recall**, and **F1**; joint models report both segmentation and classification metrics.

### Segmentation

| Model   | Dice ↑          | IoU ↑           |
| ------- | ---------------- | ---------------- |
| UNet    | 0.8442           | 0.7697           |
| AttUNet | **0.8491** | **0.7743** |
| BiFPN   | 0.8379           | 0.7614           |

### Classification

| Model   | Accuracy ↑     | Precision        | Recall           | F1               |
| ------- | --------------- | ---------------- | ---------------- | ---------------- |
| UNet    | **0.9480** | **0.9436** | **0.9528** | **0.9476** |
| AttUNet | 0.9400           | 0.9356           | 0.9466           | 0.9394           |
| BiFPN   | 0.9270           | 0.9204           | 0.9364           | 0.9269           |

### Joint (Segmentation + Classification)

| Model   | Accuracy ↑      | Dice ↑          | IoU ↑           |
| ------- | ---------------- | ---------------- | ---------------- |
| UNet    | **0.9884** | **0.8408** | **0.7636** |
| AttUNet | 0.9849           | 0.8311           | 0.7514           |
| BiFPN   | 0.9849           | 0.8367           | 0.7572           |

> **Note:** Results are from the best checkpoint (lowest validation loss) across all training epochs. Metrics computed using `reports/summary.csv`.

---

## Recommended Training Config (RTX 4070 Laptop — 8 GB VRAM)

Based on **Dell Alienware M16 R2** (Intel Core Ultra 7 155H, RTX 4070 8 GB, 32 GB RAM):

| Setting                         | Recommended Value                          | Rationale                                        |
| ------------------------------- | ------------------------------------------ | ------------------------------------------------ |
| **Batch size**            | `8`                                      | Fits 8 GB VRAM with 256x256 images              |
| **Epochs**                | `40` (seg), `25` (cls), `50` (joint) | Sufficient for convergence                       |
| **num_workers**           | `4`                                      | Matches P-core count on Ultra 7 155H (6P+8E+2LP) |
| **pin_memory**            | `True`                                   | Faster CPU->GPU transfer                         |
| **persistent_workers**    | `True`                                   | Avoids worker spawn overhead every epoch         |
| **prefetch_factor**       | `2`                                      | Double-buffers data loading                      |
| **Mixed precision (AMP)** | `ON` (~40% memory reduction)             | Uses RTX 4070 Tensor Cores                       |
| **Gradient accumulation** | `1` (not needed at batch 8)              | Increase to 2 if using batch 4                   |
| **Learning rate**         | `3e-4` (AdamW)                           | Standard for AdamW, well-tested                  |
| **Weight decay**          | `1e-4`                                   | Light regularization                             |
| **Scheduler**             | CosineAnnealingLR                          | Smooth LR decay without plateau detection        |
| **Gradient clipping**     | `5.0`                                    | Prevents gradient explosion                      |
| **Checkpoint**            | `best.ckpt` (task+model combo score)     | Saves whenever validation improves               |
| **cuDNN benchmark**       | `True` (set via seed)                    | Auto-tunes conv algorithms for fixed input size  |
| **torch.compile**         | Optional                                   | Can provide 10-20% speedup on Ada Lovelace       |

---

## Joint Model Checkpoint Issue — Root Cause

**Symptom**: The joint U-Net could not properly load its 20-epoch checkpoint, while seg and cls models worked fine.

**Root cause**: The joint model training **only logged 19 epochs** (1–19) in `report/unet/joint/log.csv`, while seg logged 20 epochs. The 20th epoch likely **crashed during validation** due to **out-of-memory (OOM)** — the original `validate()` function concatenated *all* validation outputs (`torch.cat(seg_outputs)`) into a single giant tensor, which exceeded 8 GB VRAM for the joint task because both seg outputs and cls outputs were being stored and concatenated simultaneously.

**Fix applied**:

1. Validation now computes metrics **per-batch and accumulates** (no more `torch.cat` of all outputs)
2. Dice is now computed during inference for joint tasks (was missing — only IoU was computed)
3. Metrics return Python `float` instead of `torch.Tensor` (was causing `"tensor(0.08, device='cuda:0')"` in CSVs)
4. Test transforms no longer include augmentations (HorizontalFlip, etc. were being applied at test time)

---

## Issues Fixed

| #  | Issue                                           | Severity    | Fix                                           |
| -- | ----------------------------------------------- | ----------- | --------------------------------------------- |
| 1  | Validation OOM — concatenated all outputs      | Critical | Per-batch metric accumulation                 |
| 2  | Dice missing in joint inference                 | Critical | Added dice computation for joint in predictor |
| 3  | Metrics returning tensors instead of floats     | High     | Added `.item()` / `float()` conversion     |
| 4  | Test transforms with augmentations              | High     | Separate val transforms (no augmentations)    |
| 5  | BiFPN dynamic Conv2d in forward pass            | High     | Proper projection layers in `__init__`       |
| 6  | Duplicated `get_model()` in train & infer      | Medium   | Single `create_model()` in model registry    |
| 7  | Dead code (`augment.py`, `save_checkpoint`) | Low      | Removed unused functions                      |
| 8  | Inconsistent normalization                      | Low      | Unified to mean=0.5, std=0.5 in config        |
| 9  | No type hints                                   | Low      | Full type hints on all functions              |
| 10 | Flat/unnested project structure                 | Low      | Professional package structure                |
| 11 | Deprecated `torch.cuda.amp` imports             | High     | Migrated to `torch.amp` (PyTorch 2.4+)       |
| 12 | `Config.device` lazy import on every access     | Medium   | Resolved once at `__post_init__`             |
| 13 | `set_seed()` disabled cuDNN benchmark           | Medium   | Enabled benchmark for performance (matches docs) |
| 14 | Predictor ignores `batch_size` config           | Medium   | Uses `cfg.data.batch_size`                    |
| 15 | `print_metrics()` used `print()` not logger     | Low      | Replaced with structured logger calls         |
| 16 | Hardcoded magic number in `visualize.py`        | Low      | Derived from actual file count                |
| 17 | Missing `cv2.imread` None checks in dataset     | Medium   | Added `FileNotFoundError` on read failure     |
| 18 | `setup.py` outdated (use `pyproject.toml`)      | High     | Migrated to PEP 621 `pyproject.toml`         |
| 19 | Reference PDFs in project root                   | Low      | Moved to `docs/` directory                    |
| 20 | Empty `__init__.py` files                        | Low      | Added docstrings and public exports           |
| 21 | Missing `.editorconfig`                          | Low      | Added for consistent formatting               |
| 22 | Missing `py.typed` marker                        | Low      | Added PEP 561 compliance marker               |
| 23 | Narrow exception handling in predictor           | Medium   | Broadened to `(PermissionError, OSError)`     |
| 24 | Optimizer used raw model after `torch.compile`  | Critical | Uses `self.model.parameters()` post-compile  |
| 25 | `autocast`/`GradScaler` hardcoded to `"cuda"`  | High     | Device-aware via `self.device.type`           |
| 26 | Predictor `.item()` crashes when batch > 1     | High     | Uses `.cpu().tolist()` with batch indexing    |
| 27 | `final_results.csv` header appended every epoch | High     | Overwrites with `mode="w"` instead of append |
| 28 | Missing `import argparse` in `train.py`         | Critical | Added `import argparse`                       |
| 29 | Import placed after logger in `metrics.py`      | Medium   | Moved to top-level imports                    |
| 30 | `Optional` without type parameter in checkpoint | Low     | Added `torch.optim.lr_scheduler.LRScheduler` |
| 31 | Unused imports in test files                     | Low      | Cleaned up all F401 warnings                  |
| 32 | Missing trailing newlines in source files        | Low      | Added via ruff auto-fix                       |
| 33 | Broken `[project.scripts]` in `pyproject.toml`  | Medium  | Removed — scripts run directly, not installed |
| 34 | `run_all.ps1` unquoted paths + no visualization | Medium | Quoted paths, added visualize.py call         |
| 35 | `check_torch_version.py` crashes without CUDA   | Medium | Added device check, multi-GPU info            |

---

## Structural Improvements

- **New package structure**: `src/` with subpackages for `data/`, `models/`, `training/`, `inference/`, `utils/`
- **Central config**: All hyperparameters in `src/config.py` dataclasses
- **Model registry**: Single `create_model()` factory function
- **Entry-point scripts**: `train.py`, `infer.py`, `visualize.py` at root
- **Separated concerns**: Losses, metrics, transforms in dedicated modules
- **Batch script**: `scripts/run_all.ps1` for full pipeline automation
- **Modern packaging**: `pyproject.toml` (PEP 621) replaces legacy `setup.py`
- **Type annotations**: PEP 561 `py.typed` marker + comprehensive type hints
- **Editor config**: `.editorconfig` for cross-editor formatting consistency
- **Reference docs**: PDFs moved to `docs/` to keep project root clean
- **Test suite**: `tests/` with pytest — models, losses, metrics, config
- **ONNX export**: `src/utils/export.py` for model deployment
- **Pre-commit hooks**: `.pre-commit-config.yaml` with ruff linting
- **K-fold CV**: Stratified cross-validation via `--n_folds` flag
- **Experiment tracking**: Optional WandB integration via `--wandb` flag
- **torch.compile**: Optional compilation via `--compile` flag

---

## Performance Optimizations

- **Mixed precision (AMP)** with `GradScaler` — reduces VRAM usage by ~40%
- **Gradient accumulation** support for larger effective batch sizes
- **`pin_memory=True`** + **`persistent_workers=True`** + **`prefetch_factor=2`** — faster data loading
- **`optimizer.zero_grad(set_to_none=True)`** — reduces memory traffic
- **Per-batch validation** — avoids OOM from concatenating all outputs
- **cuDNN auto-tuner** enabled — selects fastest convolution algorithms
- **Gradient clipping** — prevents training instability
- **Modern `torch.amp` API** — forward-compatible with PyTorch 2.4+

---

## Remaining Limitations

1. **Single-GPU only** — no DDP support for multi-GPU training
2. **No model checkpoint snapshots** — only best checkpoint is saved, not periodic snapshots
3. **No test-time augmentation** (TTA) — could improve inference scores

---

## Future Improvements

- Convert to PyTorch Lightning for cleaner multi-GPU support
- Add TensorBoard logging alongside CSV
- Implement test-time augmentation (TTA) for better inference accuracy
- Add periodic checkpoint snapshots (every N epochs) for mid-training recovery
- Add data validation and versioning pipeline
- Add model profiling and FLOP counting utilities

---

## Contributors & Credits

- **[Sifat Hasan Rimon](https://github.com/SifatHasanRimon2001)** — Project Lead, UNet & AttUNet implementation, pipeline design
- **[Emdadul Hq Iram](https://github.com/EmdadulHqIram013)** — BiFPN implementation, model optimization, code contributions

---

## References

- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) — Ronneberger et al., 2015
- [Attention U-Net: Learning Where to Look for the Pancreas](https://arxiv.org/abs/1804.03999) — Oktay et al., 2018
- [EfficientDet: Scalable and Efficient Object Detection (BiFPN)](https://arxiv.org/abs/1911.09070) — Tan et al., 2019
