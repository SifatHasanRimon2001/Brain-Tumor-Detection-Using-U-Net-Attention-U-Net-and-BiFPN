#!/usr/bin/env python3
"""Quick CUDA / PyTorch compatibility check.
Usage:
    python scripts/check_torch_version.py
"""
from __future__ import annotations
def main() -> None:
    import torch
    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("No CUDA device found — training will run on CPU.")
        return
    print(f"CUDA devices    : {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  Device {i}      : {torch.cuda.get_device_name(i)}")
    # Quick benchmark: matrix multiply on GPU
    try:
        a = torch.randn(1000, 1000, device="cuda")
        b = torch.randn(1000, 1000, device="cuda")
        _ = torch.mm(a, b)
        torch.cuda.synchronize()
        print("GPU compute    : OK (matrix multiply succeeded)")
    except Exception as e:
        print(f"GPU compute    : FAILED ({e})")
if __name__ == "__main__":
    main()
