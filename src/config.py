"""Central configuration using Python dataclasses.
All hyperparameters, paths, and settings are defined here
to avoid magic numbers scattered across the codebase.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
# ── Dataset constants ──────────────────────────────────────────────────────
CLASS_MAP: dict[str, int] = {
    "glioma": 0,
    "meningioma": 1,
    "no_tumor": 2,
    "pituitary": 3,
}
# Mapping from filename substrings to class indices (must match CLASS_MAP order)
FILENAME_CLASS_MAP: dict[str, int] = {
    "gl": 0,  # glioma
    "me": 1,  # meningioma
    "nt": 2,  # no_tumor
    "pi": 3,  # pituitary
}
NUM_CLASSES_SEG: int = 1     # binary segmentation
NUM_CLASSES_CLS: int = len(CLASS_MAP)
# ── Normalisation constants ────────────────────────────────────────────────
IMAGE_MEAN: tuple[float, float, float] = (0.5, 0.5, 0.5)
IMAGE_STD: tuple[float, float, float] = (0.5, 0.5, 0.5)
# ── Dataclass configs ──────────────────────────────────────────────────────
@dataclass
class DataConfig:
    """Paths and data-loading settings."""
    data_root: str = "./brisc2025"
    img_size: int = 256
    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
@dataclass
class OptimizerConfig:
    """Optimiser & scheduler settings."""
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    scheduler_t_max: int | None = None  # defaults to epochs
@dataclass
class TrainingConfig:
    """Training hyper-parameters."""
    task: str = "seg"          # "seg" | "cls" | "joint"
    model: str = "unet"        # "unet" | "attunet" | "bifpn"
    epochs: int = 50
    use_amp: bool = True
    use_compile: bool = False  # torch.compile() for 10-20% speedup on RTX 40+
    gradient_accumulation_steps: int = 1
    grad_clip_max_norm: float = 5.0
    seed: int = 42
    log_interval: int = 1      # log every N epochs
    # ── K-fold cross-validation ───────────────────────────────────────
    n_folds: int = 0           # 0 = disabled; 2-10 = K-fold CV
    fold: int = 0              # current fold index (set automatically)
    # ── Experiment tracking (optional) ────────────────────────────────
    use_wandb: bool = False
    wandb_project: str = "brisc-tumor"
    wandb_run_name: str = ""
@dataclass
class PathConfig:
    """All output / cache directories."""
    checkpoint_dir: str = "runs"
    report_dir: str = "reports"
    output_dir: str = "outputs"
    result_dir: str = "results"
    def ensure_dirs(self) -> None:
        for d in [self.checkpoint_dir, self.report_dir,
                  self.output_dir, self.result_dir]:
            os.makedirs(d, exist_ok=True)
@dataclass
class Config:
    """Top-level config aggregating all sub-configs."""
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    device: str = ""
    def __post_init__(self) -> None:
        if self.optim.scheduler_t_max is None:
            self.optim.scheduler_t_max = self.training.epochs
        if not self.device:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
    # ── Shortcut helpers ────────────────────────────────────────────────
    @property
    def checkpoint_path(self) -> str:
        return os.path.join(
            self.paths.checkpoint_dir,
            f"{self.training.task}_{self.training.model}",
        )
    @property
    def best_ckpt_path(self) -> str:
        return os.path.join(self.checkpoint_path, "best.ckpt")
    def task_in(self, *tasks: str) -> bool:
        return self.training.task in tasks
