"""Tests for configuration dataclasses."""
from __future__ import annotations
from src.config import Config, OptimizerConfig, PathConfig, TrainingConfig
class TestConfig:
    """Tests for the top-level Config dataclass."""
    def test_default_config(self) -> None:
        cfg = Config()
        assert cfg.training.task == "seg"
        assert cfg.training.model == "unet"
        assert cfg.data.img_size == 256
        assert cfg.optim.learning_rate == 3e-4
    def test_scheduler_t_max_defaults_to_epochs(self) -> None:
        cfg = Config(training=TrainingConfig(epochs=30))
        assert cfg.optim.scheduler_t_max == 30
    def test_scheduler_t_max_respects_explicit_value(self) -> None:
        cfg = Config(
            training=TrainingConfig(epochs=50),
            optim=OptimizerConfig(scheduler_t_max=20),
        )
        assert cfg.optim.scheduler_t_max == 20
    def test_device_resolves(self) -> None:
        cfg = Config()
        assert cfg.device in ("cuda", "cpu")
    def test_checkpoint_path(self) -> None:
        cfg = Config(
            training=TrainingConfig(task="seg", model="unet"),
            paths=PathConfig(checkpoint_dir="runs"),
        )
        assert cfg.checkpoint_path == "runs/seg_unet"
    def test_best_ckpt_path(self) -> None:
        cfg = Config(
            training=TrainingConfig(task="cls", model="attunet"),
        )
        assert cfg.best_ckpt_path.endswith("best.ckpt")
    def test_task_in(self) -> None:
        cfg = Config(training=TrainingConfig(task="seg"))
        assert cfg.task_in("seg", "joint")
        assert not cfg.task_in("cls")
    def test_ensure_dirs(self, tmp_path: object) -> None:
        import os
        d = str(tmp_path)
        cfg = Config(paths=PathConfig(
            checkpoint_dir=os.path.join(d, "runs"),
            report_dir=os.path.join(d, "reports"),
            output_dir=os.path.join(d, "outputs"),
            result_dir=os.path.join(d, "results"),
        ))
        cfg.paths.ensure_dirs()
        for subdir in ["runs", "reports", "outputs", "results"]:
            assert os.path.isdir(os.path.join(d, subdir))
