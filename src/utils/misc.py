import os
import random
import numpy as np
import torch
def set_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def save_checkpoint(state, save_dir, filename="best.ckpt"):
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, filename)
    torch.save(state, ckpt_path)
    print(f"[INFO] Saved checkpoint to {ckpt_path}")
def load_checkpoint(model, ckpt_path, optimizer=None, device="cpu"):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("[INFO] Loaded optimizer state from checkpoint.")
    else:
        model.load_state_dict(checkpoint)
    print(f"[INFO] Loaded checkpoint from {ckpt_path}")
    return model, checkpoint
def print_metrics(task, metrics: dict):
    if task == "seg":
        print(f" Dice: {metrics.get('dice', 0):.4f} | IoU: {metrics.get('iou', 0):.4f}")
    elif task == "cls":
        print(f" Acc: {metrics.get('acc', 0):.4f} | "
              f"Prec: {metrics.get('precision', 0):.4f} | "
              f"Recall: {metrics.get('recall', 0):.4f} | "
              f"F1: {metrics.get('f1', 0):.4f}")
    elif task == "joint":
        print("[JOINT] Metrics:")
        seg_metrics = {k: v for k, v in metrics.items() if k in ["dice", "iou"]}
        cls_metrics = {k: v for k, v in metrics.items() if k in ["acc", "precision", "recall", "f1"]}
        print("--- Segmentation ---")
        print_metrics("seg", seg_metrics)
        print("--- Classification ---")
        print_metrics("cls", cls_metrics)