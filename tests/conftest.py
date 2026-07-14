"""Shared pytest fixtures for model creation and dummy data."""
from __future__ import annotations
import pytest
import torch
from src.models import create_model
@pytest.fixture(params=["unet", "attunet", "bifpn"])
def model(request: pytest.FixtureRequest) -> torch.nn.Module:
    """Create a model instance for each architecture."""
    return create_model(request.param, num_classes=1, num_cls_labels=4)
@pytest.fixture
def dummy_seg_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Random segmentation batch: (images, masks)."""
    imgs = torch.randn(2, 3, 256, 256)
    masks = torch.zeros(2, 1, 256, 256)
    masks[:, :, 64:192, 64:192] = 1.0  # synthetic tumour square
    return imgs, masks
@pytest.fixture
def dummy_cls_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Random classification batch: (images, labels)."""
    imgs = torch.randn(2, 3, 256, 256)
    labels = torch.tensor([0, 2])
    return imgs, labels
@pytest.fixture
def dummy_joint_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random joint batch: (images, masks, labels)."""
    imgs = torch.randn(2, 3, 256, 256)
    masks = torch.zeros(2, 1, 256, 256)
    masks[:, :, 64:192, 64:192] = 1.0
    labels = torch.tensor([0, 3])
    return imgs, masks, labels
@pytest.fixture
def device() -> torch.device:
    """Return CPU device for tests (deterministic, no GPU required)."""
    return torch.device("cpu")
