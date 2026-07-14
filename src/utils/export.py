"""ONNX export utility for model deployment.
Exports a trained model to ONNX format for inference optimization
and deployment on edge devices, TensorRT, ONNX Runtime, etc.
Usage
-----
    python -m src.utils.export --model unet --ckpt runs/seg_unet/best.ckpt \\
        --task seg --output_dir exports
"""
from __future__ import annotations
import argparse
import os
import torch
from src.config import NUM_CLASSES_CLS, NUM_CLASSES_SEG
from src.models import create_model
from src.utils.checkpoint import load_checkpoint
from src.utils.logging import get_logger
logger = get_logger(__name__)
def export_onnx(
    model: torch.nn.Module,
    save_path: str,
    img_size: int = 256,
    opset: int = 17,
    dynamic_batch: bool = True,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
) -> str:
    """Export a model to ONNX format.
    Parameters
    ----------
    model : nn.Module
        The model to export (should already be on CPU and in eval mode).
    save_path : str
        Path to save the .onnx file.
    img_size : int
        Spatial size of input images.
    opset : int
        ONNX operator set version.
    dynamic_batch : bool
        If True, batch dimension is dynamic.
    input_names : list[str] | None
        Names for input tensors.
    output_names : list[str] | None
        Names for output tensors.
    Returns
    -------
    str
        Path to the saved ONNX file.
    """
    model.eval()
    model.cpu()
    dummy_input = torch.randn(1, 3, img_size, img_size)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "seg_output": {0: "batch_size"},
            "cls_output": {0: "batch_size"},
        }
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        opset_version=opset,
        input_names=input_names or ["input"],
        output_names=output_names or ["seg_output", "cls_output"],
        dynamic_axes=dynamic_axes,
    )
    logger.info("ONNX model exported to %s", save_path)
    return save_path
def main() -> None:
    parser = argparse.ArgumentParser(description="Export a trained model to ONNX")
    parser.add_argument("--model", type=str, choices=["unet", "attunet", "bifpn"], required=True)
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--task", type=str, choices=["seg", "cls", "joint"], default="seg")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--output_dir", type=str, default="exports")
    args = parser.parse_args()
    model = create_model(args.model, num_classes=NUM_CLASSES_SEG, num_cls_labels=NUM_CLASSES_CLS)
    load_checkpoint(model, args.ckpt, device="cpu")
    save_path = os.path.join(args.output_dir, f"{args.task}_{args.model}.onnx")
    export_onnx(model, save_path, img_size=args.img_size, opset=args.opset)
if __name__ == "__main__":
    main()
