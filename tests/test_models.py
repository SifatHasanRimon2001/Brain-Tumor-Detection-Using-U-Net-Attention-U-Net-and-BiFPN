"""Tests for model forward passes and output shapes."""
from __future__ import annotations
import pytest
import torch
from src.models import MODEL_REGISTRY, create_model
from src.models.attention_unet import AttentionUNet
from src.models.bifpn_unet import BiFPNUNet
from src.models.unet import UNet
class TestModelRegistry:
    """Tests for the model registry and factory function."""
    def test_all_architectures_registered(self) -> None:
        assert set(MODEL_REGISTRY.keys()) == {"unet", "attunet", "bifpn"}
    def test_create_model_returns_correct_type(self) -> None:
        assert isinstance(create_model("unet"), UNet)
        assert isinstance(create_model("attunet"), AttentionUNet)
        assert isinstance(create_model("bifpn"), BiFPNUNet)
    def test_create_model_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown model"):
            create_model("nonexistent")
class TestUNetForward:
    """Tests for standard U-Net forward pass."""
    def test_output_shapes(self, model: torch.nn.Module, dummy_seg_batch: tuple) -> None:
        imgs, masks = dummy_seg_batch
        seg_out, cls_out = model(imgs)
        assert seg_out.shape == (2, 1, 256, 256), f"Seg shape: {seg_out.shape}"
        assert cls_out.shape == (2, 4), f"Cls shape: {cls_out.shape}"
    def test_output_dtype(self, model: torch.nn.Module, dummy_seg_batch: tuple) -> None:
        imgs, _ = dummy_seg_batch
        seg_out, cls_out = model(imgs)
        assert seg_out.dtype == torch.float32
        assert cls_out.dtype == torch.float32
    def test_grad_flow(self, model: torch.nn.Module, dummy_seg_batch: tuple) -> None:
        imgs, masks = dummy_seg_batch
        seg_out, cls_out = model(imgs)
        loss = seg_out.mean() + cls_out.mean()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
    def test_different_input_sizes(self, model: torch.nn.Module) -> None:
        for size in [128, 256]:
            imgs = torch.randn(1, 3, size, size)
            seg_out, cls_out = model(imgs)
            assert seg_out.shape == (1, 1, size, size)
            assert cls_out.shape == (1, 4)
    def test_deterministic_eval(self, model: torch.nn.Module, dummy_seg_batch: tuple) -> None:
        model.eval()
        imgs, _ = dummy_seg_batch
        with torch.no_grad():
            out1_seg, out1_cls = model(imgs)
            out2_seg, out2_cls = model(imgs)
        assert torch.allclose(out1_seg, out2_seg)
        assert torch.allclose(out1_cls, out2_cls)
class TestModelParameterCount:
    """Sanity check that models have reasonable parameter counts."""
    @pytest.mark.parametrize("name,expected_min_millions", [
        ("unet", 30),
        ("attunet", 34),
        ("bifpn", 32),
    ])
    def test_parameter_count(self, name: str, expected_min_millions: int) -> None:
        model = create_model(name)
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > expected_min_millions * 1e6, (
            f"{name} has {n_params / 1e6:.1f}M params, expected > {expected_min_millions}M"
        )
