# ────────────────────────────────────────────────────────────────────────────
# BRISC2025 — Full Training & Evaluation Pipeline
# Run from project root:
#   powershell -ExecutionPolicy Bypass -File scripts/run_all.ps1
# ────────────────────────────────────────────────────────────────────────────

$ErrorActionPreference = "Stop"
$Py = "python"

$Line = "=" * 60

Write-Host $Line
Write-Host ">>> Starting BRISC2025 Training Pipeline"
Write-Host $Line

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

$SegModels   = @("unet", "attunet", "bifpn")
$ClsModels   = @("unet", "attunet", "bifpn")   # Change if using classification backbones
$JointModels = @("unet", "attunet", "bifpn")

# ---------------------------------------------------------------------------
# 1. Segmentation
# ---------------------------------------------------------------------------

Write-Host "`n=== SEGMENTATION ==="

foreach ($model in $SegModels) {

    Write-Host "[SEG] Training $model"

    & $Py train.py `
        --task seg `
        --model $model `
        --data_root "./brisc2025" `
        --img_size 256 `
        --batch 8 `
        --epochs 40

    if ($LASTEXITCODE -ne 0) {
        Write-Error "Training failed for $model"
        continue
    }

    $ckpt = "runs/seg_$($model)/best.ckpt"

    Write-Host "[SEG] Evaluating $model"

    & $Py infer.py `
        --task seg `
        --model $model `
        --data_root "./brisc2025" `
        --size 256 `
        --ckpt $ckpt
}

# ---------------------------------------------------------------------------
# 2. Classification
# ---------------------------------------------------------------------------

Write-Host "`n=== CLASSIFICATION ==="

foreach ($model in $ClsModels) {

    Write-Host "[CLS] Training $model"

    & $Py train.py `
        --task cls `
        --model $model `
        --data_root "./brisc2025" `
        --img_size 256 `
        --batch 8 `
        --epochs 25

    if ($LASTEXITCODE -ne 0) {
        Write-Error "Training failed for $model"
        continue
    }

    $ckpt = "runs/cls_$($model)/best.ckpt"

    Write-Host "[CLS] Evaluating $model"

    & $Py infer.py `
        --task cls `
        --model $model `
        --data_root "./brisc2025" `
        --size 256 `
        --ckpt $ckpt
}

# ---------------------------------------------------------------------------
# 3. Joint Training
# ---------------------------------------------------------------------------

Write-Host "`n=== JOINT TRAINING ==="

foreach ($model in $JointModels) {

    Write-Host "[JOINT] Training $model"

    & $Py train.py `
        --task joint `
        --model $model `
        --data_root "./brisc2025" `
        --img_size 256 `
        --batch 8 `
        --epochs 50

    if ($LASTEXITCODE -ne 0) {
        Write-Error "Training failed for $model"
        continue
    }

    $ckpt = "runs/joint_$($model)/best.ckpt"

    Write-Host "[JOINT] Evaluating $model"

    & $Py infer.py `
        --task joint `
        --model $model `
        --data_root "./brisc2025" `
        --size 256 `
        --ckpt $ckpt
}

Write-Host ""
Write-Host $Line
Write-Host ">>> All trainings and evaluations completed!"
Write-Host $Line