# ────────────────────────────────────────────────────────────────────────────
# BRISC2025 — Full Training & Evaluation Pipeline
# ────────────────────────────────────────────────────────────────────────────
# Run from the project root:
#   powershell -File scripts/run_all.ps1
# ────────────────────────────────────────────────────────────────────────────

$ErrorActionPreference = "Continue"
$Py = "python"

# ── 1. SEGMENTATION ───────────────────────────────────────────────────────
Write-Output "=" * 60
Write-Output ">>> Starting SEGMENTATION training"
Write-Output "=" * 60

$models = @("unet", "attunet", "bifpn")

# ── 1. SEGMENTATION ───────────────────────────────────────────────────────
foreach ($model in $models) {
    Write-Output "[SEG] Training $model..."
    & $Py train.py --task seg --model $model `
        --data_root ./brisc2025 --img_size 256 --batch 8 --epochs 40

    Write-Output "[SEG] Evaluating $model..."
    & $Py infer.py --task seg --model $model `
        --data_root ./brisc2025 --size 256 `
        --ckpt runs/seg_${model}/best.ckpt
}

# ── 2. CLASSIFICATION ──────────────────────────────────────────────────────
foreach ($model in $models) {
    Write-Output "[CLS] Training $model..."
    & $Py train.py --task cls --model $model `
        --data_root ./brisc2025 --img_size 256 --batch 8 --epochs 25

    Write-Output "[CLS] Evaluating $model..."
    & $Py infer.py --task cls --model $model `
        --data_root ./brisc2025 --size 256 `
        --ckpt runs/cls_${model}/best.ckpt
}

# ── 3. JOINT (SEG + CLS) ──────────────────────────────────────────────────
foreach ($model in $models) {
    Write-Output "[JOINT] Training $model..."
    & $Py train.py --task joint --model $model `
        --data_root ./brisc2025 --img_size 256 --batch 8 --epochs 50

    Write-Output "[JOINT] Evaluating $model..."
    & $Py infer.py --task joint --model $model `
        --data_root ./brisc2025 --size 256 `
        --ckpt runs/joint_${model}/best.ckpt
}

Write-Output "=" * 60
Write-Output ">>> All trainings and evaluations completed!"
Write-Output "=" * 60
