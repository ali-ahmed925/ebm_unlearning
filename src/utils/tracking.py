from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


class Tracker:
    def log_scalar(self, name: str, value: float, step: int) -> None:  # pragma: no cover
        raise NotImplementedError

    def log_scalars(self, prefix: str, values: Dict[str, float], step: int) -> None:
        for k, v in values.items():
            self.log_scalar(f"{prefix}/{k}", float(v), step)

    def close(self) -> None:  # pragma: no cover
        pass


class NullTracker(Tracker):
    def log_scalar(self, name: str, value: float, step: int) -> None:
        return


@dataclass(frozen=True)
class TensorBoardConfig:
    log_dir: str


class TensorBoardTracker(Tracker):
    def __init__(self, cfg: TensorBoardConfig):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "TensorBoard logging requires `tensorboard`. Install it with: pip install tensorboard"
            ) from e
        self._writer = SummaryWriter(log_dir=cfg.log_dir)

    def log_scalar(self, name: str, value: float, step: int) -> None:
        self._writer.add_scalar(name, float(value), int(step))

    def close(self) -> None:
        self._writer.flush()
        self._writer.close()


@dataclass(frozen=True)
class WandbConfig:
    project: str
    name: Optional[str] = None
    dir: Optional[str] = None
    config: Optional[Dict[str, Any]] = None


class WandbTracker(Tracker):
    def __init__(self, cfg: WandbConfig):
        try:
            import wandb
        except Exception as e:  # pragma: no cover
            raise ImportError("wandb logging requires `wandb`. Install it with: pip install wandb") from e
        self._wandb = wandb
        self._run = wandb.init(project=cfg.project, name=cfg.name, dir=cfg.dir, config=cfg.config)

    def log_scalar(self, name: str, value: float, step: int) -> None:
        self._wandb.log({name: float(value)}, step=int(step))

    def close(self) -> None:
        self._run.finish()


def make_tracker(backend: str, **kwargs: Any) -> Tracker:
    b = (backend or "none").lower()
    if b in ("none", "null", "off", "disable", "disabled"):
        return NullTracker()
    if b in ("tb", "tensorboard"):
        return TensorBoardTracker(TensorBoardConfig(**kwargs))
    if b in ("wandb",):
        return WandbTracker(WandbConfig(**kwargs))
    raise ValueError(f"Unsupported tracking backend: {backend}")


