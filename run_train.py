import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from src.data.datasets import BriscDataset, CLASS_MAP
from src.models.unet import UNet
from src.models.att_unet import AttUNet
from src.models.bifpn import BiFPNUNet
from src.models.metrics import dice_coefficient, iou_score, classification_metrics
from src.utils.report_utils import ReportManager
from torch.cuda.amp import autocast, GradScaler
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, pred, target):
        if pred.shape[1] > 1:
            pred = F.softmax(pred, dim=1)
            target_one_hot = F.one_hot(
                target.long(), num_classes=pred.shape[1]
            ).permute(0, 3, 1, 2).float()
            target = target_one_hot
        else:
            pred = torch.sigmoid(pred)
        inter = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * inter + self.eps) / (union + self.eps)
        return (1 - dice).mean()
def get_model(name, task, num_classes=1, num_cls_labels=4):
    if name == "unet":
        return UNet(in_ch=3, num_classes=num_classes, num_cls_labels=num_cls_labels)
    elif name == "attunet":
        return AttUNet(in_ch=3, num_classes=num_classes, num_cls_labels=num_cls_labels)
    elif name == "bifpn":
        return BiFPNUNet(in_ch=3, num_classes=num_classes, num_cls_labels=num_cls_labels)
    else:
        raise ValueError(f"Unknown model: {name}")
def train_one_epoch(model, loader, optimizer, criterion_seg, criterion_cls, task, device, scaler):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training", leave=False):
        optimizer.zero_grad()
        with autocast():
            if task == "seg":
                imgs, masks = batch
                imgs, masks = imgs.to(device), masks.to(device)
                seg_out, _ = model(imgs)
                loss = criterion_seg(seg_out, masks)
            elif task == "cls":
                imgs, labels = batch
                imgs, labels = imgs.to(device), labels.to(device)
                _, cls_out = model(imgs)
                loss = criterion_cls(cls_out, labels)
            elif task == "joint":
                imgs, masks, labels = batch
                imgs, masks, labels = imgs.to(device), masks.to(device), labels.to(device)
                seg_out, cls_out = model(imgs)
                loss_seg = criterion_seg(seg_out, masks)
                loss_cls = criterion_cls(cls_out, labels)
                loss = loss_seg + loss_cls
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)
def validate(model, loader, criterion_seg, criterion_cls, task, device, num_classes):
    model.eval()
    total_loss, n = 0, 0
    seg_outputs, seg_targets = [], []
    cls_preds, cls_targets = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating", leave=False):
            with autocast():
                if task == "seg":
                    imgs, masks = batch
                    imgs, masks = imgs.to(device), masks.to(device)
                    seg_out, _ = model(imgs)
                    loss = criterion_seg(seg_out, masks)
                    seg_outputs.append(seg_out)
                    seg_targets.append(masks)
                elif task == "cls":
                    imgs, labels = batch
                    imgs, labels = imgs.to(device), labels.to(device)
                    _, cls_out = model(imgs)
                    loss = criterion_cls(cls_out, labels)
                    cls_preds.append(cls_out)
                    cls_targets.append(labels)
                elif task == "joint":
                    imgs, masks, labels = batch
                    imgs, masks, labels = imgs.to(device), masks.to(device), labels.to(device)
                    seg_out, cls_out = model(imgs)
                    loss_seg = criterion_seg(seg_out, masks)
                    loss_cls = criterion_cls(cls_out, labels)
                    loss = loss_seg + loss_cls
                    seg_outputs.append(seg_out)
                    seg_targets.append(masks)
                    cls_preds.append(cls_out)
                    cls_targets.append(labels)
            total_loss += loss.item()
            n += 1
    results = {"loss": total_loss / max(n, 1)}
    if task in ["seg", "joint"]:
        seg_preds = torch.cat(seg_outputs)
        seg_labels = torch.cat(seg_targets)
        results["dice"] = dice_coefficient(seg_preds, seg_labels, num_classes)
        results["iou"] = iou_score(seg_preds, seg_labels, num_classes)
    if task in ["cls", "joint"]:
        cls_preds_tensor = torch.cat(cls_preds)
        cls_labels_tensor = torch.cat(cls_targets)
        results.update(classification_metrics(cls_preds_tensor, cls_labels_tensor))
    return results, cls_preds, cls_targets
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train_ds = BriscDataset(args.data_root, split="train", task=args.task, img_size=args.img_size)
    val_ds = BriscDataset(args.data_root, split="test", task=args.task, img_size=args.img_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)
    num_classes_seg = 1
    num_classes_cls = len(CLASS_MAP)
    model = get_model(args.model, args.task, num_classes=num_classes_seg, num_cls_labels=num_classes_cls).to(device)
    criterion_seg = DiceLoss()
    criterion_cls = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()
    reporter = ReportManager(root_dir="report", model_name=args.model, task=args.task)
    best_score, best_path = -1, f"runs/{args.task}_{args.model}"
    os.makedirs(best_path, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs} | LR={scheduler.get_last_lr()[0]:.6f}")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion_seg, criterion_cls, args.task, device, scaler)
        results, cls_preds, cls_targets = validate(model, val_loader, criterion_seg, criterion_cls, args.task, device, num_classes_seg)
        print(f"Train loss: {train_loss:.4f} | Val loss: {results['loss']:.4f}")
        cm = None
        if args.task in ["cls", "joint"] and cls_preds:
            preds = torch.argmax(torch.cat(cls_preds), dim=1).cpu().numpy()
            gts = torch.cat(cls_targets).cpu().numpy()
            cm = confusion_matrix(gts, preds)
        reporter.log_epoch(epoch, train_loss, results["loss"], {}, results, confusion_matrix=cm)
        if args.task == "seg":
            score = results.get("dice", 0)
        elif args.task == "cls":
            score = results.get("f1", 0)
        elif args.task == "joint":
            score = results.get("dice", 0) + results.get("f1", 0)
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), os.path.join(best_path, "best.ckpt"))
            print(f"Saved best model (score={best_score:.4f})")
        scheduler.step()
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=["seg", "cls", "joint"], required=True)
    parser.add_argument("--model", type=str, choices=["unet", "attunet", "bifpn"], required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    main(args)