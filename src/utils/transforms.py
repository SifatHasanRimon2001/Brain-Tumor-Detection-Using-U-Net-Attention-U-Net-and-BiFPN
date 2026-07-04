"""Image augmentation & normalisation pipelines.

All transforms use the same normalisation constants defined in
``src.config`` so that train / val / inference are consistent.
"""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.config import IMAGE_MEAN, IMAGE_STD


def _normalize() -> A.Compose:
    """Return normalisation + ToTensorV2."""
    return A.Compose([
        A.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
        ToTensorV2(),
    ])


# ── Training transforms ───────────────────────────────────────────────────

def train_image_transform(img_size: int = 256) -> A.Compose:
    """Transform for classification (image only)."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1, p=0.3),
        A.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
        ToTensorV2(),
    ])


def train_image_mask_transform(img_size: int = 256) -> A.Compose:
    """Transform for segmentation / joint (image + mask, no normalisation)."""
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ],
        additional_targets={"mask": "mask"},
    )


def train_normalize_transform() -> A.Compose:
    """Normalisation step applied *after* the geometric augmentations."""
    return _normalize()


# ── Validation / test transforms ──────────────────────────────────────────

def val_image_transform(img_size: int = 256) -> A.Compose:
    """Transform for classification validation (no augmentation)."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
        ToTensorV2(),
    ])


def val_image_mask_transform(img_size: int = 256) -> A.Compose:
    """Transform for segmentation / joint validation (no augmentation)."""
    return A.Compose(
        [A.Resize(img_size, img_size)],
        additional_targets={"mask": "mask"},
    )


def val_normalize_transform() -> A.Compose:
    return _normalize()
