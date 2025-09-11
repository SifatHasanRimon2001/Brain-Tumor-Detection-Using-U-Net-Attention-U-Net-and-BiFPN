import os
import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.data.datasets import BriscDataset, CLASS_MAP
from src.models.unet import UNet
from src.models.att_unet import AttUNet
from src.models.bifpn import BiFPNUNet
from src.models.metrics import iou_score, classification_metrics
from src.utils.misc import load_checkpoint
from src.utils.report_utils import ReportManager
def get_model(name, num_classes=1, num_cls_labels=4):
    if name == "unet":
        return UNet(in_ch=3, num_classes=num_classes, num_cls_labels=num_cls_labels)
    elif name == "attunet":
        return AttUNet(in_ch=3, num_classes=num_classes, num_cls_labels=num_cls_labels)
    elif name == "bifpn":
        return BiFPNUNet(in_ch=3, num_classes=num_classes, num_cls_labels=num_cls_labels)
    else:
        raise ValueError(f"Unknown model: {name}")
def infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = BriscDataset(args.data_root, split="test", task=args.task, img_size=args.size)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    num_classes_seg = 1
    num_classes_cls = len(CLASS_MAP)
    model = get_model(args.model, num_cls_labels=num_classes_cls, num_classes=num_classes_seg).to(device)
    model, _ = load_checkpoint(model, args.ckpt, device=device)
    model.eval()
    print(f"[INFO] Loaded checkpoint from {args.ckpt}")
    report = ReportManager(root_dir="report", model_name=args.model, task=args.task)
    base_out_dir = os.path.join("outputs", args.model, args.task)
    os.makedirs(base_out_dir, exist_ok=True)
    total_iou, cls_preds, cls_labels = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Infer")):
            if args.task == "seg":
                imgs, masks = batch
                imgs, masks = imgs.to(device), masks.to(device)
                seg_out, _ = model(imgs)
                iou = iou_score(seg_out, masks, num_classes=1)
                total_iou.append(iou.item())
                pred_mask = (torch.sigmoid(seg_out) > 0.5).cpu().numpy()[0, 0]
                cv2.imwrite(
                    os.path.join(base_out_dir, f"pred_mask_{i}.png"),
                    (pred_mask * 255).astype(np.uint8)
                )
            elif args.task == "cls":
                imgs, labels = batch
                imgs, labels = imgs.to(device), labels.to(device)
                _, cls_out = model(imgs)
                pred = torch.argmax(cls_out, dim=1).item()
                gt = labels.item()
                cls_preds.append(pred)
                cls_labels.append(gt)
            elif args.task == "joint":
                imgs, masks, labels = batch
                imgs, masks, labels = imgs.to(device), masks.to(device), labels.to(device)
                seg_out, cls_out = model(imgs)
                iou = iou_score(seg_out, masks, num_classes=1)
                total_iou.append(iou.item())
                cls_pred = torch.argmax(cls_out, dim=1).item()
                gt_cls = labels.item()
                cls_preds.append(cls_pred)
                cls_labels.append(gt_cls)
                pred_mask = (torch.sigmoid(seg_out) > 0.5).cpu().numpy()[0, 0]
                cv2.imwrite(
                    os.path.join(base_out_dir, f"pred_mask_{i}.png"),
                    (pred_mask * 255).astype(np.uint8)
                )
    test_metrics = {
        "acc": 0, "precision": 0, "recall": 0, "f1": 0,
        "dice": 0, "iou": 0
    }
    if total_iou:
        test_metrics["iou"] = float(np.mean(total_iou))
        print(f"[RESULT] Mean IoU = {test_metrics['iou']:.4f}")
    if cls_preds:
        cls_preds_tensor = torch.tensor(cls_preds)
        cls_labels_tensor = torch.tensor(cls_labels)
        cls_metrics = classification_metrics(cls_preds_tensor, cls_labels_tensor)
        test_metrics.update(cls_metrics)
        print(f"[RESULT] Classification metrics: {cls_metrics}")
    try:
        report.log_epoch(
            epoch=0,
            train_loss=0,
            test_loss=0,
            train_metrics={},
            test_metrics=test_metrics
        )
    except PermissionError as e:
        print(f"[WARN] Could not write report (file locked): {e}")
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=["seg", "cls", "joint"], required=True)
    parser.add_argument("--model", type=str, choices=["unet", "attunet", "bifpn"], required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--ckpt", type=str, required=True)
    args = parser.parse_args()
    infer(args)