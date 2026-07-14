"""BRISC2025 brain tumour dataset.
Supports three tasks:
- ``seg``    — binary segmentation (image → mask)
- ``cls``    — 4-class classification (image → tumour type)
- ``joint``  — both segmentation and classification
"""
from __future__ import annotations
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from src.config import CLASS_MAP, FILENAME_CLASS_MAP
from src.utils.logging import get_logger
from src.utils.transforms import (
    train_image_mask_transform,
    train_image_transform,
    train_normalize_transform,
    val_image_mask_transform,
    val_image_transform,
    val_normalize_transform,
)
logger = get_logger(__name__)
class BriscDataset(Dataset):
    """BRISC2025 brain tumour MRI dataset.
    Parameters
    ----------
    data_root : str
        Path to the dataset root directory.
    split : str
        ``"train"`` or ``"test"``.
    task : str
        ``"seg"``, ``"cls"``, or ``"joint"``.
    img_size : int
        Spatial size to resize images / masks to.
    """
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        task: str = "seg",
        img_size: int = 256,
    ) -> None:
        super().__init__()
        self.task = task
        self.img_size = img_size
        self.split = split
        self._setup_paths(data_root)
        self._setup_transforms()
    # ── Path resolution ────────────────────────────────────────────────
    def _setup_paths(self, data_root: str) -> None:
        if self.task in ("seg", "joint"):
            self.img_dir = os.path.join(data_root, "segmentation_task", self.split, "images")
            self.mask_dir = os.path.join(data_root, "segmentation_task", self.split, "masks")
            self.img_files: list[str] = sorted(os.listdir(self.img_dir))
        else:  # cls
            self.img_dir = os.path.join(data_root, "classification_task", self.split)
            self.img_files = []
            for cls_name, cls_idx in CLASS_MAP.items():
                cls_dir = os.path.join(self.img_dir, cls_name)
                if not os.path.isdir(cls_dir):
                    logger.warning("Classification sub-directory not found: %s", cls_dir)
                    continue
                for fname in os.listdir(cls_dir):
                    self.img_files.append((os.path.join(cls_dir, fname), cls_idx))
    # ── Transform setup ────────────────────────────────────────────────
    def _setup_transforms(self) -> None:
        is_train = self.split == "train"
        if self.task in ("seg", "joint"):
            self.image_mask_transform = (
                train_image_mask_transform(self.img_size)
                if is_train
                else val_image_mask_transform(self.img_size)
            )
            self.normalize_transform = (
                train_normalize_transform() if is_train
                else val_normalize_transform()
            )
        else:
            self.img_transform = (
                train_image_transform(self.img_size)
                if is_train
                else val_image_transform(self.img_size)
            )
    # ── Dataset interface ──────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.img_files)
    def __getitem__(self, idx: int):
        if self.task in ("seg", "joint"):
            return self._get_seg_or_joint(idx)
        else:
            return self._get_cls(idx)
    # ── Internal helpers ───────────────────────────────────────────────
    def _get_seg_or_joint(self, idx: int):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(
            self.mask_dir,
            img_name.replace(".jpg", ".png"),
        )
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Failed to read mask: {mask_path}")
        mask = (mask > 127).astype(np.float32)
        # Apply geometric augmentations jointly
        augmented = self.image_mask_transform(image=image, mask=mask)
        image_aug = augmented["image"]  # uint8/np array, not normalised yet
        mask_aug = augmented["mask"]    # same spatial shape as image_aug
        # Normalise image
        image_norm = self.normalize_transform(image=image_aug)["image"]
        # Convert mask to tensor
        mask_tensor = torch.from_numpy(mask_aug).unsqueeze(0).float()
        if self.task == "seg":
            return image_norm, mask_tensor
        # ── joint: also extract class label from filename ──
        label: int | None = None
        for key, cls_idx in FILENAME_CLASS_MAP.items():
            if f"_{key}_" in img_name.lower():
                label = cls_idx
                break
        if label is None:
            raise ValueError(
                f"Could not determine class label from filename: {img_name}. "
                f"Expected one of: {list(FILENAME_CLASS_MAP.keys())}"
            )
        return image_norm, mask_tensor, torch.tensor(label, dtype=torch.long)
    def _get_cls(self, idx: int):
        img_path, label = self.img_files[idx]
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        aug = self.img_transform(image=image)
        return aug["image"], torch.tensor(label, dtype=torch.long)
