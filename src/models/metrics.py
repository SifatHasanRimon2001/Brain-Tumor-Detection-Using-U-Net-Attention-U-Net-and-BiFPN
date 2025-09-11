import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
def dice_coefficient(pred, target, num_classes, eps=1e-6):
    dice_scores = []
    if num_classes > 1:
        pred_softmax = F.softmax(pred, dim=1)
        pred_labels = torch.argmax(pred_softmax, dim=1)
        for class_idx in range(1, num_classes):
            pred_mask = (pred_labels == class_idx).float()
            target_mask = (target == class_idx).float()
            inter = torch.sum(pred_mask * target_mask)
            union = torch.sum(pred_mask) + torch.sum(target_mask)
            dice = (2.0 * inter + eps) / (union + eps)
            dice_scores.append(dice.item())
        return sum(dice_scores) / len(dice_scores) if dice_scores else 0.0
    else:
        pred_binary = (torch.sigmoid(pred) > 0.5).float()
        target_binary = target.float()
        inter = torch.sum(pred_binary * target_binary)
        union = torch.sum(pred_binary) + torch.sum(target_binary)
        return (2.0 * inter + eps) / (union + eps)
def iou_score(pred, target, num_classes, eps=1e-6):
    iou_scores = []
    if num_classes > 1:
        pred_softmax = F.softmax(pred, dim=1)
        pred_labels = torch.argmax(pred_softmax, dim=1)
        for class_idx in range(1, num_classes):
            pred_mask = (pred_labels == class_idx).float()
            target_mask = (target == class_idx).float()
            inter = torch.sum(pred_mask * target_mask)
            union = torch.sum(pred_mask) + torch.sum(target_mask) - inter
            iou_scores.append(((inter + eps) / (union + eps)).item())
        return sum(iou_scores) / len(iou_scores) if iou_scores else 0.0
    else:
        pred_binary = (torch.sigmoid(pred) > 0.5).float()
        target_binary = target.float()
        inter = torch.sum(pred_binary * target_binary)
        union = torch.sum(pred_binary) + torch.sum(target_binary) - inter
        return (inter + eps) / (union + eps)
def classification_metrics(pred_logits_or_labels, target, average="macro"):
    if pred_logits_or_labels.ndim > 1:
        preds = torch.argmax(pred_logits_or_labels, dim=1).cpu().numpy()
    else:
        preds = pred_logits_or_labels.cpu().numpy()
    target_np = target.cpu().numpy()
    acc = accuracy_score(target_np, preds)
    prec = precision_score(target_np, preds, average=average, zero_division=0)
    rec = recall_score(target_np, preds, average=average, zero_division=0)
    f1 = f1_score(target_np, preds, average=average, zero_division=0)
    return {"acc": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1)}