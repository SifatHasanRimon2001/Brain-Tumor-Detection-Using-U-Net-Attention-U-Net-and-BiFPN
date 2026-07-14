"""Model registry — maps model names to constructors."""
from __future__ import annotations
from src.models.attention_unet import AttentionUNet
from src.models.bifpn_unet import BiFPNUNet
from src.models.unet import UNet
MODEL_REGISTRY: dict[str, type] = {
    "unet": UNet,
    "attunet": AttentionUNet,
    "bifpn": BiFPNUNet,
}
def create_model(
    name: str,
    in_channels: int = 3,
    num_classes: int = 1,
    num_cls_labels: int = 4,
) -> UNet | AttentionUNet | BiFPNUNet:
    """Create a model instance by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name](
        in_ch=in_channels,
        num_classes=num_classes,
        num_cls_labels=num_cls_labels,
    )
