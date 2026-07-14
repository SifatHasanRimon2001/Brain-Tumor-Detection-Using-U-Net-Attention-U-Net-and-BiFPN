"""Inference-time visualisation comparing separate vs. joint models.
Given a single image, loads all six models (seg + cls + joint × 3 architectures)
and produces a comprehensive side-by-side comparison figure.
"""
from __future__ import annotations
import os
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from src.config import IMAGE_MEAN, IMAGE_STD, NUM_CLASSES_CLS, NUM_CLASSES_SEG
from src.models import create_model
from src.models.metrics import dice_coefficient, iou_score
from src.utils.checkpoint import load_checkpoint
from src.utils.logging import get_logger
matplotlib.use("Agg")
logger = get_logger(__name__)
ARCHITECTURES = ["unet", "attunet", "bifpn"]
def _overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.4,
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    overlay = image.copy()
    mask_rgb = np.zeros_like(image)
    mask_rgb[mask > 0] = color
    cv2.addWeighted(mask_rgb, alpha, overlay, 1 - alpha, 0, overlay)
    return overlay
def _load_model(
    name: str,
    ckpt_path: str,
    device: str,
) -> torch.nn.Module:
    model = create_model(name, num_classes=NUM_CLASSES_SEG, num_cls_labels=NUM_CLASSES_CLS)
    load_checkpoint(model, ckpt_path, device=device)
    return model.to(device).eval()
def _prepare_tensor(image_path: str, img_size: int, device: str) -> torch.Tensor:
    """Load, resize, normalise, and batch an image."""
    raw = cv2.imread(image_path)
    if raw is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    orig = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(orig, (img_size, img_size))
    tensor = (
        torch.tensor(resized / 255.0, dtype=torch.float32)
        .permute(2, 0, 1)
        .unsqueeze(0)
    )
    # Normalise: (x - mean) / std
    mean = torch.tensor(IMAGE_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGE_STD).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.to(device)
def visualize_models(
    image_path: str,
    mask_path: str | None,
    models_root: str = "runs",
    device: str = "cuda",
    img_size: int = 256,
    save_path: str | None = None,
    dpi: int = 150,
) -> None:
    """Produce a large comparison figure and save to *save_path*."""
    sns.set_style("whitegrid")
    sns.set_context("talk")
    orig = cv2.imread(image_path)
    if orig is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(orig, (img_size, img_size))
    tensor = _prepare_tensor(image_path, img_size, device)
    # Ground-truth mask
    gt_mask: np.ndarray | None = None
    overlay_gt: np.ndarray | None = None
    if mask_path is not None:
        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if gt_mask is not None:
            gt_mask = cv2.resize(gt_mask, (img_size, img_size))
            gt_mask = (gt_mask > 127).astype(np.uint8)
            overlay_gt = _overlay_mask(image_resized, gt_mask, color=(255, 0, 0))
    results: dict[str, dict[str, object]] = {"separate": {}, "joint": {}}
    for arch in ARCHITECTURES:
        # Separate models
        seg_ckpt = os.path.join(models_root, f"seg_{arch}", "best.ckpt")
        cls_ckpt = os.path.join(models_root, f"cls_{arch}", "best.ckpt")
        seg_model = _load_model(arch, seg_ckpt, device)
        cls_model = _load_model(arch, cls_ckpt, device)
        with torch.no_grad():
            seg_out, _ = seg_model(tensor)
            prob_mask = torch.sigmoid(seg_out)[0, 0].cpu().numpy()
            pred_mask_sep = (prob_mask > 0.5).astype(np.uint8)
            _, cls_out = cls_model(tensor)
            cls_pred = torch.argmax(cls_out, dim=1).item()
            cls_probs = torch.softmax(cls_out, dim=1)[0].cpu().numpy()
        results["separate"][arch] = (
            pred_mask_sep, cls_pred, cls_probs, image_resized, gt_mask, overlay_gt
        )
        # Joint model
        joint_ckpt = os.path.join(models_root, f"joint_{arch}", "best.ckpt")
        joint_model = _load_model(arch, joint_ckpt, device)
        with torch.no_grad():
            seg_out, cls_out = joint_model(tensor)
            prob_mask = torch.sigmoid(seg_out)[0, 0].cpu().numpy()
            pred_mask_joint = (prob_mask > 0.5).astype(np.uint8)
            cls_pred_joint = torch.argmax(cls_out, dim=1).item()
            cls_probs_joint = torch.softmax(cls_out, dim=1)[0].cpu().numpy()
        results["joint"][arch] = (
            pred_mask_joint, cls_pred_joint, cls_probs_joint,
            image_resized, gt_mask, overlay_gt,
        )
    # ── Build the figure ────────────────────────────────────────────
    total_sections = len(["separate", "joint"]) * len(ARCHITECTURES)
    rows_needed = 1 + total_sections * 3
    height_ratios = [1]
    for _ in range(total_sections):
        height_ratios += [1, 3, 3]
    fig, axs = plt.subplots(
        rows_needed, 3,
        figsize=(26, sum(height_ratios)),
        gridspec_kw={"height_ratios": height_ratios, "wspace": 0.02},
    )
    fig.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.01, hspace=0.35)
    for ax in axs[0]:
        ax.axis("off")
    axs[0, 1].text(
        0.5, 0.5,
        "Task: Classification with Segmentation (Separate Models) vs Joint Models\n"
        "Accuracy & IoU for UNet, AttUNet, and BiFPN",
        ha="center", va="center", fontsize=22, weight="bold",
    )
    row = 1
    for mode in ("separate", "joint"):
        for arch in ARCHITECTURES:
            data = results[mode][arch]
            pred_mask, cls_pred, cls_probs, img_resized, gt_mask_i, overlay_gt_i = data
            overlay_pred = _overlay_mask(img_resized, pred_mask)
            mode_label = (
                "classification with segmentation (separate models)"
                if mode == "separate"
                else "joint (a single joint model)"
            )
            # Compute metrics
            dice_val = iou_val = None
            if gt_mask_i is not None:
                pred_t = torch.tensor(pred_mask).unsqueeze(0).unsqueeze(0).float()
                gt_t = torch.tensor(gt_mask_i).unsqueeze(0).unsqueeze(0).float()
                dice_val = dice_coefficient(pred_t, gt_t, num_classes=1)
                iou_val = iou_score(pred_t, gt_t, num_classes=1)
            iou_str = f"{iou_val:.3f}" if iou_val is not None else "N/A"
            dice_str = f"{dice_val:.3f}" if dice_val is not None else "N/A"
            # Section header
            for ax in axs[row]:
                ax.axis("off")
            axs[row, 1].text(
                0.5, 0.6,
                f"{arch.upper()}: {mode_label}\n"
                f"Dice={dice_str}  |  IoU={iou_str}",
                ha="center", va="center", fontsize=16, weight="bold", color="darkblue",
            )
            row += 1
            # Row: ground truth
            axs[row, 0].imshow(img_resized)
            axs[row, 0].set_title("Original", fontsize=12)
            axs[row, 0].axis("off")
            axs[row, 1].imshow(gt_mask_i if gt_mask_i is not None else np.zeros_like(pred_mask), cmap="gray")
            axs[row, 1].set_title("Ground Truth Mask", fontsize=12)
            axs[row, 1].axis("off")
            axs[row, 2].imshow(overlay_gt_i if overlay_gt_i is not None else img_resized)
            axs[row, 2].set_title("Ground Truth Overlay", fontsize=12)
            axs[row, 2].axis("off")
            row += 1
            # Row: prediction
            axs[row, 0].imshow(img_resized)
            axs[row, 0].set_title("Image", fontsize=12)
            axs[row, 0].axis("off")
            axs[row, 1].imshow(pred_mask, cmap="gray")
            axs[row, 1].set_title("Predicted Mask", fontsize=12)
            axs[row, 1].axis("off")
            axs[row, 2].imshow(overlay_pred)
            axs[row, 2].set_title("Predicted Overlay", fontsize=12, color="darkred")
            axs[row, 2].axis("off")
            row += 1
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info("Visualization saved to %s", save_path)
    plt.close(fig)
