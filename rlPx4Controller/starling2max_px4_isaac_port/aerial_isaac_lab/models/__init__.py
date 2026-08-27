"""Reusable drone-model configuration data."""

from .model_cfg import MultirotorModelCfg
from .starling2max import STARLING2MAX_MODEL, STARLING2MAX_URDF_PATH

__all__ = [
    "MultirotorModelCfg",
    "STARLING2MAX_MODEL",
    "STARLING2MAX_URDF_PATH",
]
