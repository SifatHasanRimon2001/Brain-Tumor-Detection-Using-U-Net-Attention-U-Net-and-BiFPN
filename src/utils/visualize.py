import os
import cv2
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from src.models.metrics import dice_coefficient, iou_score
from src.data.datasets import CLASS_MAP
from src.models.unet import UNet
from src.models.att_unet import AttUNet
from src.models.bifpn import BiFPNUNet
def overlay_mask_on_image(image, mask, alpha=0.4, color=(0, 255, 0)):
    overlay = image.copy()
    mask_rgb = np.zeros_like(image)
    mask_rgb[mask > 0] = color
    cv2.addWeighted(mask_rgb, alpha, overlay, 1 - alpha, 0, overlay)
    return overlay
def load_model(model_type, ckpt_path, device, num_classes=1, num_cls_labels=4):
    if model_type == "unet":
        model = UNet(in_ch=3, num_classes=num_classes, num_cls_labels=num_cls_labels)
    elif model_type == "attunet":
        model = AttUNet(in_ch=3, num_classes=num_classes, num_cls_labels=num_cls_labels)
    elif model_type == "bifpn":
        model = BiFPNUNet(in_ch=3, num_classes=num_classes, num_cls_labels=num_cls_labels)
    else:
        raise ValueError(f"Unknown model: {model_type}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model.to(device).eval()
def visualize_models(image_path, mask_path, models_root="runs",
                     device="cuda", img_size=256, save_path=None, dpi=150):
    sns.set_style("whitegrid")
    sns.set_context("talk")
    orig = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(orig, (img_size, img_size))
    tensor = torch.tensor(image_resized / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    tensor = (tensor - 0.5) / 0.5
    tensor = tensor.to(device)
    gt_mask, overlay_gt = None, None
    if mask_path is not None:
        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.resize(gt_mask, (img_size, img_size))
        gt_mask = (gt_mask > 127).astype(np.uint8)
        overlay_gt = overlay_mask_on_image(image_resized, gt_mask, alpha=0.4, color=(255, 0, 0))
    archs = ["unet", "attunet", "bifpn"]
    results = {"separate": {}, "joint": {}}
    for arch in archs:
        seg_model = load_model(arch, os.path.join(models_root, f"seg_{arch}", "best.ckpt"), device)
        cls_model = load_model(arch, os.path.join(models_root, f"cls_{arch}", "best.ckpt"), device)
        with torch.no_grad():
            seg_out, _ = seg_model(tensor)
            prob_mask = torch.sigmoid(seg_out)[0, 0].cpu().numpy()
            pred_mask_sep = (prob_mask > 0.5).astype(np.uint8)
            _, cls_out = cls_model(tensor)
            cls_pred = torch.argmax(cls_out, dim=1).item()
            cls_probs = torch.softmax(cls_out, dim=1)[0].cpu().numpy()
        results["separate"][arch] = (pred_mask_sep, cls_pred, cls_probs, image_resized, gt_mask, overlay_gt)
        joint_model = load_model(arch, os.path.join(models_root, f"joint_{arch}", "best.ckpt"), device)
        with torch.no_grad():
            seg_out, cls_out = joint_model(tensor)
            prob_mask = torch.sigmoid(seg_out)[0, 0].cpu().numpy()
            pred_mask_joint = (prob_mask > 0.5).astype(np.uint8)
            cls_pred_joint = torch.argmax(cls_out, dim=1).item()
            cls_probs_joint = torch.softmax(cls_out, dim=1)[0].cpu().numpy()
        results["joint"][arch] = (pred_mask_joint, cls_pred_joint, cls_probs_joint, image_resized, gt_mask, overlay_gt)
    total_sections = len(["separate", "joint"]) * len(archs)
    rows_needed = 1 + total_sections * 3
    height_ratios = [1] + [item for _ in range(total_sections) for item in [1, 3, 3]]
    fig, axs = plt.subplots(
        rows_needed, 3,
        figsize=(26, sum(height_ratios)),
        gridspec_kw={"height_ratios": height_ratios, "wspace": 0.02},
        constrained_layout=False
    )
    fig.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.01, hspace=0.35)
    for ax in axs[0]:
        ax.axis("off")
    axs[0, 1].text(
        0.5, 0.5,
        "Task: Classification with Segmentation (Separate Models) vs Joint Models\n"
        "Accuracy & IoU for UNet, AttUNet, and BiFPN",
        ha="center", va="center", fontsize=22, weight="bold"
    )
    row = 1
    for mode in ["separate", "joint"]:
        for arch in archs:
            pred_mask, cls_pred, cls_probs, image_resized, gt_mask, overlay_gt = results[mode][arch]
            overlay_pred = overlay_mask_on_image(image_resized, pred_mask)
            dice = iou = acc = None
            if gt_mask is not None:
                pred_t = torch.tensor(pred_mask).unsqueeze(0).unsqueeze(0).float()
                gt_t = torch.tensor(gt_mask).unsqueeze(0).unsqueeze(0).float()
                dice = dice_coefficient(pred_t, gt_t, num_classes=1)
                iou = iou_score(pred_t, gt_t, num_classes=1)
                acc = (pred_mask == gt_mask).sum() / gt_mask.size
            iou_val = float(iou.item()) if iou is not None else None
            cls_acc_val = float(cls_probs[cls_pred]) if cls_probs is not None else None
            iou_str = f"{iou_val:.3f}" if iou_val is not None else "N/A"
            cls_acc_str = f"{cls_acc_val:.3f}" if cls_acc_val is not None else "N/A"
            for ax in axs[row]:
                ax.axis("off")
            axs[row, 1].text(
                0.5, 0.6,
                f"{arch.upper()}: {'classification with segmentation (separate models)' if mode=='separate' else 'joint (a single joint model)'}\n"
                f"Classification Accuracy={cls_acc_str} | IoU={iou_str}",
                ha="center", va="center", fontsize=16, weight="bold", color="darkblue"
            )
            row += 1
            axs[row, 0].imshow(image_resized)
            axs[row, 0].text(0.5, -0.05, "Ground Truth - Original", ha="center", va="top", fontsize=12, transform=axs[row, 0].transAxes)
            axs[row, 0].axis("off")
            axs[row, 1].imshow(gt_mask if gt_mask is not None else np.zeros_like(pred_mask), cmap="gray")
            axs[row, 1].text(0.5, -0.05, "Ground Truth - Mask", ha="center", va="top", fontsize=12, transform=axs[row, 1].transAxes)
            axs[row, 1].axis("off")
            axs[row, 2].imshow(overlay_gt if overlay_gt is not None else image_resized)
            axs[row, 2].text(0.5, -0.05, "Ground Truth - Overlay", ha="center", va="top", fontsize=12, transform=axs[row, 2].transAxes)
            axs[row, 2].axis("off")
            row += 1
            axs[row, 0].imshow(image_resized)
            axs[row, 0].text(0.5, -0.05, "Predicted - Image", ha="center", va="top", fontsize=12, transform=axs[row, 0].transAxes)
            axs[row, 0].axis("off")
            axs[row, 1].imshow(pred_mask, cmap="gray")
            axs[row, 1].text(0.5, -0.05, "Predicted - Mask", ha="center", va="top", fontsize=12, transform=axs[row, 1].transAxes)
            axs[row, 1].axis("off")
            axs[row, 2].imshow(overlay_pred)
            axs[row, 2].text(0.5, -0.05, "Predicted - Overlay", ha="center", va="top", fontsize=12, color="darkred", transform=axs[row, 2].transAxes)
            axs[row, 2].axis("off")
            row += 1
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"[INFO] Visualization saved to {save_path}")
    plt.close()