"""Isaac Lab task registrations.

Import this module only after :class:`isaaclab.app.AppLauncher` has started the
simulation application.
"""

import gymnasium as gym

from .starling2max_env import (
    Starling2MaxAttitudeDeltaEnv,
    Starling2MaxAttitudeDeltaEnvCfg,
    Starling2MaxDepthAttitudeDeltaEnv,
    Starling2MaxDepthAttitudeDeltaEnvCfg,
    Starling2MaxNavigationEnv,
    Starling2MaxNavigationEnvCfg,
    Starling2MaxPositionEnv,
    Starling2MaxPositionEnvCfg,
    Starling2MaxRatesEnv,
    Starling2MaxRatesEnvCfg,
    Starling2MaxVelocityEnv,
    Starling2MaxVelocityEnvCfg,
)

STARLING2MAX_VELOCITY_TASK_ID = "AerialIsaac-Starling2Max-Velocity-v0"
STARLING2MAX_POSITION_TASK_ID = "AerialIsaac-Starling2Max-Position-v0"
STARLING2MAX_ATTITUDE_DELTA_TASK_ID = "AerialIsaac-Starling2Max-AttitudeDelta-v0"
STARLING2MAX_RATES_TASK_ID = "AerialIsaac-Starling2Max-Rates-v0"
STARLING2MAX_NAVIGATION_TASK_ID = "AerialIsaac-Starling2Max-Navigation-v0"
STARLING2MAX_DEPTH_ATTITUDE_DELTA_TASK_ID = "AerialIsaac-Starling2Max-DepthAttitudeDelta-v0"

if STARLING2MAX_VELOCITY_TASK_ID not in gym.registry:
    gym.register(
        id=STARLING2MAX_VELOCITY_TASK_ID,
        entry_point="aerial_isaac_lab.tasks:Starling2MaxVelocityEnv",
    )

if STARLING2MAX_POSITION_TASK_ID not in gym.registry:
    gym.register(
        id=STARLING2MAX_POSITION_TASK_ID,
        entry_point="aerial_isaac_lab.tasks:Starling2MaxPositionEnv",
    )

if STARLING2MAX_ATTITUDE_DELTA_TASK_ID not in gym.registry:
    gym.register(
        id=STARLING2MAX_ATTITUDE_DELTA_TASK_ID,
        entry_point="aerial_isaac_lab.tasks:Starling2MaxAttitudeDeltaEnv",
    )

if STARLING2MAX_RATES_TASK_ID not in gym.registry:
    gym.register(
        id=STARLING2MAX_RATES_TASK_ID,
        entry_point="aerial_isaac_lab.tasks:Starling2MaxRatesEnv",
    )

if STARLING2MAX_NAVIGATION_TASK_ID not in gym.registry:
    gym.register(
        id=STARLING2MAX_NAVIGATION_TASK_ID,
        entry_point="aerial_isaac_lab.tasks:Starling2MaxNavigationEnv",
    )

if STARLING2MAX_DEPTH_ATTITUDE_DELTA_TASK_ID not in gym.registry:
    gym.register(
        id=STARLING2MAX_DEPTH_ATTITUDE_DELTA_TASK_ID,
        entry_point="aerial_isaac_lab.tasks:Starling2MaxDepthAttitudeDeltaEnv",
    )

__all__ = [
    "STARLING2MAX_ATTITUDE_DELTA_TASK_ID",
    "STARLING2MAX_DEPTH_ATTITUDE_DELTA_TASK_ID",
    "STARLING2MAX_NAVIGATION_TASK_ID",
    "STARLING2MAX_POSITION_TASK_ID",
    "STARLING2MAX_RATES_TASK_ID",
    "STARLING2MAX_VELOCITY_TASK_ID",
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
