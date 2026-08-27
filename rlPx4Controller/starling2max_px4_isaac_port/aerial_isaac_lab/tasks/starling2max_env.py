"""Compatibility imports for Starling 2 Max task classes."""

from .starling2max_attitude_delta_env import (
    Starling2MaxAttitudeDeltaEnv,
    Starling2MaxAttitudeDeltaEnvCfg,
)
from .starling2max_depth_attitude_delta_env import (
    Starling2MaxDepthAttitudeDeltaEnv,
    Starling2MaxDepthAttitudeDeltaEnvCfg,
)
from .starling2max_navigation_env import (
    Starling2MaxNavigationEnv,
    Starling2MaxNavigationEnvCfg,
)
from .starling2max_position_env import Starling2MaxPositionEnv, Starling2MaxPositionEnvCfg
from .starling2max_rates_env import Starling2MaxRatesEnv, Starling2MaxRatesEnvCfg
from .starling2max_velocity_env import Starling2MaxVelocityEnv, Starling2MaxVelocityEnvCfg

__all__ = [
    "Starling2MaxAttitudeDeltaEnv",
    "Starling2MaxAttitudeDeltaEnvCfg",
    "Starling2MaxDepthAttitudeDeltaEnv",
    "Starling2MaxDepthAttitudeDeltaEnvCfg",
    "Starling2MaxNavigationEnv",
    "Starling2MaxNavigationEnvCfg",
    "Starling2MaxPositionEnv",
    "Starling2MaxPositionEnvCfg",
    "Starling2MaxRatesEnv",
    "Starling2MaxRatesEnvCfg",
    "Starling2MaxVelocityEnv",
    "Starling2MaxVelocityEnvCfg",
]
