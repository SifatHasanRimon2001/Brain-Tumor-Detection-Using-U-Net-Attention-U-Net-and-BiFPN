# 1. SEGMENTATION TRAINING
# =========================
Write-Output ">>> Starting SEGMENTATION training"
# --- U-Net ---
Write-Output "[SEG] Training U-Net..."
python run_train.py --task seg --model unet `
  --data_root ./brisc2025 `
  --img_size 256 `
  --batch 8 `
  --epochs 40
Write-Output "[SEG] Evaluating U-Net..."
python run_infer.py --task seg --model unet `
  --data_root ./brisc2025 `
  --size 256 `
  --ckpt runs/seg_unet/best.ckpt
# --- Attention U-Net ---
Write-Output "[SEG] Training Attention U-Net..."
python run_train.py --task seg --model attunet `
  --data_root ./brisc2025 `
  --img_size 256 `
  --batch 8 `
  --epochs 40
Write-Output "[SEG] Evaluating Attention U-Net..."
python run_infer.py --task seg --model attunet `
  --data_root ./brisc2025 `
  --size 256 `
  --ckpt runs/seg_attunet/best.ckpt
# --- BiFPN ---
Write-Output "[SEG] Training BiFPN..."
python run_train.py --task seg --model bifpn `
  --data_root ./brisc2025 `
  --img_size 256 `
  --batch 8 `
  --epochs 40
Write-Output "[SEG] Evaluating BiFPN..."
python run_infer.py --task seg --model bifpn `
  --data_root ./brisc2025 `
  --size 256 `
  --ckpt runs/seg_bifpn/best.ckpt
# =========================
# 2. CLASSIFICATION TRAINING
# =========================
Write-Output ">>> Starting CLASSIFICATION training"
# --- U-Net encoder head ---
Write-Output "[CLS] Training U-Net..."
python run_train.py --task cls --model unet `
  --data_root ./brisc2025 `
  --img_size 256 `
  --batch 8 `
  --epochs 25
Write-Output "[CLS] Evaluating U-Net..."
python run_infer.py --task cls --model unet `
  --data_root ./brisc2025 `
  --size 256 `
  --ckpt runs/cls_unet/best.ckpt
# --- Attention U-Net encoder ---
Write-Output "[CLS] Training Attention U-Net..."
python run_train.py --task cls --model attunet `
  --data_root ./brisc2025 `
  --img_size 256 `
  --batch 8 `
  --epochs 25
Write-Output "[CLS] Evaluating Attention U-Net..."
python run_infer.py --task cls --model attunet `
  --data_root ./brisc2025 `
  --size 256 `
  --ckpt runs/cls_attunet/best.ckpt
# --- BiFPN encoder ---
Write-Output "[CLS] Training BiFPN..."
python run_train.py --task cls --model bifpn `
  --data_root ./brisc2025 `
  --img_size 256 `
  --batch 8 `
  --epochs 25
Write-Output "[CLS] Evaluating BiFPN..."
python run_infer.py --task cls --model bifpn `
  --data_root ./brisc2025 `
  --size 256 `
  --ckpt runs/cls_bifpn/best.ckpt

# =========================
# 3. JOINT TRAINING (Seg + Cls)
# =========================
Write-Output ">>> Starting JOINT (Seg+Cls) training"
# --- U-Net ---
Write-Output "[JOINT] Training U-Net..."
python run_train.py --task joint --model unet `
  --data_root ./brisc2025 `
  --img_size 256 `
  --batch 8 `
  --epochs 50
Write-Output "[JOINT] Evaluating U-Net..."
python run_infer.py --task joint --model unet `
  --data_root ./brisc2025 `
  --size 256 `
  --ckpt runs/joint_unet/best.ckpt
# --- Attention U-Net ---
Write-Output "[JOINT] Training Attention U-Net..."
python run_train.py --task joint --model attunet `
  --data_root ./brisc2025 `
  --img_size 256 `
  --batch 8 `
  --epochs 50
Write-Output "[JOINT] Evaluating Attention U-Net..."
python run_infer.py --task joint --model attunet `
  --data_root ./brisc2025 `
  --size 256 `
  --ckpt runs/joint_attunet/best.ckpt
# --- BiFPN ---
Write-Output "[JOINT] Training BiFPN..."
python run_train.py --task joint --model bifpn `
  --data_root ./brisc2025 `
  --img_size 256 `
  --batch 8 `
  --epochs 50
Write-Output "[JOINT] Evaluating BiFPN..."
python run_infer.py --task joint --model bifpn `
  --data_root ./brisc2025 `
  --size 256 `
  --ckpt runs/joint_bifpn/best.ckpt
Write-Output ">>> All trainings and evaluations completed!"