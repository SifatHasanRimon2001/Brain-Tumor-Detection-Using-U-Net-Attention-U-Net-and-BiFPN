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

## 📖 Table of Contents

- [Installation](#-installation)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Usage](#-usage)
- [Results](#-results)
- [Recommended Training Config](#%EF%B8%8F-recommended-training-config-rtx-4070-laptop--8-gb-vram)
- [Joint Model Checkpoint Issue](#-joint-model-checkpoint-issue--root-cause)
- [Issues Fixed](#-issues-fixed)
- [Structural Improvements](#-structural-improvements)
- [Performance Optimizations](#-performance-optimizations)
- [Remaining Limitations](#-remaining-limitations)
- [Future Improvements](#-future-improvements)
- [Contributors &amp; Credits](#contributors--credits)
- [References](#references)

---

## 📦 Installation

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

## 📁 Dataset

We use the **BRISC 2025 MRI dataset**:

- [Download from Kaggle](https://www.kaggle.com/datasets/briscdataset/brisc2025?resource=download)

Place the dataset in a folder called `./brisc2025` at the project root.

The dataset contains T1-weighted contrast-enhanced MRI scans organized into four classes:
`glioma`, `meningioma`, `no_tumor`, and `pituitary`. Each sample has both an MRI image and
a corresponding tumor segmentation mask.

---

## 📁 Project Structure

```
├── train.py                          # Training entry point
├── infer.py                          # Inference / evaluation entry point
├── visualize.py                      # Model comparison visualization
├── setup.py                          # Package metadata
├── requirements.txt                  # Python dependencies
├── check_torch_version.py            # CUDA / PyTorch compatibility check
├── .gitignore
├── README.md
│
├── src/                              # Source package
│   ├── config.py                     # Central dataclass-based configuration
│   ├── data/
│   │   └── dataset.py                # BRISC Dataset with transforms
│   ├── models/
│   │   ├── components.py             # Shared blocks: DoubleConv, AttentionBlock
│   │   ├── unet.py                   # Standard U-Net
│   │   ├── attention_unet.py         # Attention U-Net
│   │   ├── bifpn_unet.py             # BiFPN-enhanced U-Net
│   │   ├── losses.py                 # DiceLoss, CombinedJointLoss
│   │   └── metrics.py                # Dice, IoU, classification metrics
│   ├── training/
│   │   └── trainer.py                # Training loop with AMP & gradient accumulation
│   ├── inference/
│   │   └── predictor.py              # Inference pipeline
│   └── utils/
│       ├── transforms.py             # Albumentations pipelines
│       ├── checkpoint.py             # Save / load checkpoints
│       ├── logging.py                # Logger setup
│       ├── reporting.py              # CSV logging & confusion matrices
│       ├── visualization.py          # Model comparison figures
│       └── misc.py                   # Seed setting
│
├── scripts/
│   └── run_all.ps1                   # Full batch pipeline (train + infer × 9 configs)
│
├── runs/                             # Saved checkpoints (auto-created)
│   ├── seg_unet/best.ckpt
│   ├── seg_attunet/best.ckpt
│   ├── seg_bifpn/best.ckpt
│   ├── cls_unet/best.ckpt
│   ├── cls_attunet/best.ckpt
│   ├── cls_bifpn/best.ckpt
│   ├── joint_unet/best.ckpt
│   ├── joint_attunet/best.ckpt
│   └── joint_bifpn/best.ckpt
│
├── reports/                          # Experiment CSV logs & confusion matrices
│   ├── summary.csv                   # Aggregated results across all models
│   ├── unet/{seg,cls,joint}/         # Per-model CSV logs & confusion matrices
│   ├── attunet/{seg,cls,joint}/
│   └── bifpn/{seg,cls,joint}/
│
├── outputs/                          # Prediction masks from inference
│   ├── unet/{seg,cls,joint}/
│   ├── attunet/{seg,cls,joint}/
│   └── bifpn/{seg,cls,joint}/
│
├── results/                          # Visualization figures (auto-created)
├── .venv/                            # Virtual environment
│
├── U-Net.pdf                         # U-Net reference paper
├── Attention U-Net.pdf               # Attention U-Net reference paper
├── EfficientDet.pdf                  # BiFPN / EfficientDet reference paper
└── Project Guidelines in Summary.pdf # Project guidelines reference
```

---

## 🚀 Usage

### Arguments reference

| Argument          | Default  | Description                         |
| ----------------- | -------- | ----------------------------------- |
| `--task`        | required | `seg`, `cls`, or `joint`      |
| `--model`       | required | `unet`, `attunet`, or `bifpn` |
| `--data_root`   | required | Path to BRISC2025 dataset           |
| `--img_size`    | `256`  | Image size                          |
| `--batch`       | `8`    | Batch size                          |
| `--epochs`      | `50`   | Number of epochs                    |
| `--lr`          | `3e-4` | Learning rate                       |
| `--num_workers` | `4`    | DataLoader workers                  |
| `--no_amp`      | off      | Disable mixed precision             |
| `--seed`        | `42`   | Random seed                         |

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
powershell -File scripts/run_all.ps1
```

---

## 📊 Results

Evaluation metrics computed on the held-out test set. Segmentation models report **Dice** and **IoU**; classification models report **Accuracy**, **Precision**, **Recall**, and **F1**; joint models report both segmentation and classification metrics.

### Segmentation

| Model   | Dice ↑          | IoU ↑           |
| ------- | ---------------- | ---------------- |
| UNet    | 0.8453           | 0.7699           |
| AttUNet | 0.8443           | 0.7692           |
| BiFPN   | **0.8470** | **0.7722** |

### Classification

| Model   | Accuracy ↑     | Precision        | Recall           | F1               |
| ------- | --------------- | ---------------- | ---------------- | ---------------- |
| UNet    | 0.949           | 0.9453           | 0.9536           | 0.9488           |
| AttUNet | **0.952** | **0.9530** | 0.9569           | **0.9543** |
| BiFPN   | 0.951           | 0.9420           | **0.9584** | 0.9486           |

### Joint (Segmentation + Classification)

| Model   | Accuracy ↑      | Dice ↑          | IoU ↑           |
| ------- | ---------------- | ---------------- | ---------------- |
| UNet    | 0.9884           | 0.8348           | 0.7554           |
| AttUNet | 0.9872           | 0.8218           | 0.7417           |
| BiFPN   | **0.9895** | **0.8362** | **0.7583** |

> **Note:** Results are from the best checkpoint (lowest validation loss) across all training epochs. Metrics computed using `reports/summary.csv`.

---

## ⚙️ Recommended Training Config (RTX 4070 Laptop — 8 GB VRAM)

Based on **Dell Alienware M16 R2** (Intel Core Ultra 7 155H, RTX 4070 8 GB, 32 GB RAM):

| Setting                         | Recommended Value                          | Rationale                                        |
| ------------------------------- | ------------------------------------------ | ------------------------------------------------ |
| **Batch size**            | `8`                                      | Fits 8 GB VRAM with 256×256 images              |
| **Epochs**                | `40` (seg), `25` (cls), `50` (joint) | Sufficient for convergence                       |
| **num_workers**           | `4`                                      | Matches P-core count on Ultra 7 155H (6P+8E+2LP) |
| **pin_memory**            | `True`                                   | Faster CPU→GPU transfer                         |
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

## 🐛 Joint Model Checkpoint Issue — Root Cause

**Symptom**: The joint U-Net could not properly load its 20-epoch checkpoint, while seg and cls models worked fine.

**Root cause**: The joint model training **only logged 19 epochs** (1–19) in `report/unet/joint/log.csv`, while seg logged 20 epochs. The 20th epoch likely **crashed during validation** due to **out-of-memory (OOM)** — the original `validate()` function concatenated *all* validation outputs (`torch.cat(seg_outputs)`) into a single giant tensor, which exceeded 8 GB VRAM for the joint task because both seg outputs and cls outputs were being stored and concatenated simultaneously.

**Fix applied**:

1. Validation now computes metrics **per-batch and accumulates** (no more `torch.cat` of all outputs)
2. Dice is now computed during inference for joint tasks (was missing — only IoU was computed)
3. Metrics return Python `float` instead of `torch.Tensor` (was causing `"tensor(0.08, device='cuda:0')"` in CSVs)
4. Test transforms no longer include augmentations (HorizontalFlip, etc. were being applied at test time)

---

## 🧹 Issues Fixed

| #  | Issue                                           | Severity    | Fix                                           |
| -- | ----------------------------------------------- | ----------- | --------------------------------------------- |
| 1  | Validation OOM — concatenated all outputs      | 🔴 Critical | Per-batch metric accumulation                 |
| 2  | Dice missing in joint inference                 | 🔴 Critical | Added dice computation for joint in predictor |
| 3  | Metrics returning tensors instead of floats     | 🟡 High     | Added`.item()` / `float()` conversion     |
| 4  | Test transforms with augmentations              | 🟡 High     | Separate val transforms (no augmentations)    |
| 5  | BiFPN dynamic Conv2d in forward pass            | 🟡 High     | Proper projection layers in`__init__`       |
| 6  | Duplicated`get_model()` in train & infer      | 🟡 Medium   | Single`create_model()` in model registry    |
| 7  | Dead code (`augment.py`, `save_checkpoint`) | 🟢 Low      | Removed unused functions                      |
| 8  | Inconsistent normalization                      | 🟢 Low      | Unified to mean=0.5, std=0.5 in config        |
| 9  | No type hints                                   | 🟢 Low      | Full type hints on all functions              |
| 10 | Flat/unnested project structure                 | 🟢 Low      | Professional package structure                |

---

## 🏗️ Structural Improvements

- **New package structure**: `src/` with subpackages for `data/`, `models/`, `training/`, `inference/`, `utils/`
- **Central config**: All hyperparameters in `src/config.py` dataclasses
- **Model registry**: Single `create_model()` factory function
- **Entry-point scripts**: `train.py`, `infer.py`, `visualize.py` at root
- **Separated concerns**: Losses, metrics, transforms in dedicated modules
- **Batch script**: `scripts/run_all.ps1` for full pipeline automation

---

## 📊 Performance Optimizations

- **Mixed precision (AMP)** with `GradScaler` — reduces VRAM usage by ~40%
- **Gradient accumulation** support for larger effective batch sizes
- **`pin_memory=True`** + **`persistent_workers=True`** + **`prefetch_factor=2`** — faster data loading
- **`optimizer.zero_grad(set_to_none=True)`** — reduces memory traffic
- **Per-batch validation** — avoids OOM from concatenating all outputs
- **cuDNN auto-tuner** enabled — selects fastest convolution algorithms
- **Gradient clipping** — prevents training instability

---

## 📋 Remaining Limitations

1. **Single-GPU only** — no DDP support for multi-GPU training
2. **No experiment tracking** — consider integrating MLflow or WandB
3. **No model checkpoint snapshots** — only best checkpoint is saved, not periodic snapshots
4. **No test-time augmentation** (TTA) — could improve inference scores
5. **No cross-validation** — train/test split is fixed

---

## 🔮 Future Improvements

- Add `torch.compile()` for 10-20% speedup on RTX 40-series
- Integrate Weights & Biases for experiment tracking & hyperparameter sweeps
- Add periodic checkpoint snapshots (every N epochs) for mid-training recovery
- Implement test-time augmentation (TTA) for better inference accuracy
- Add K-fold cross-validation
- Convert to PyTorch Lightning for cleaner multi-GPU support
- Add ONNX export for deployment
- Add TensorBoard logging alongside CSV

---

## Contributors & Credits

- **[Sifat Hasan Rimon](https://github.com/SifatHasanRimon2001)** — Project Lead, UNet & AttUNet implementation, pipeline design
- **[Emdadul Hq Iram](https://github.com/EmdadulHqIram013)** — BiFPN implementation, model optimization, code contributions

---

## References

- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) — Ronneberger et al., 2015
- [Attention U-Net: Learning Where to Look for the Pancreas](https://arxiv.org/abs/1804.03999) — Oktay et al., 2018
- [EfficientDet: Scalable and Efficient Object Detection (BiFPN)](https://arxiv.org/abs/1911.09070) — Tan et al., 2019
