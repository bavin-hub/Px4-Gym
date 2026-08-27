"""Shared obstacle layout and reset sampling for depth-navigation tasks."""

from __future__ import annotations

import math

import torch

from isaaclab.envs import ViewerCfg
from isaaclab.utils import configclass

from aerial_isaac_lab.core.math import euler_xyz_from_quat, wrap_to_pi
from aerial_isaac_lab.scene import ObstacleFieldCfg, sample_perimeter_spawn_positions


OBSTACLE_CORE_RADIUS = 3.5
SPAWN_RING_MIN_RADIUS = 4.2
SPAWN_RING_MAX_RADIUS = 4.8
SPAWN_Z_HALF_RANGE = 2.0
SPAWN_YAW_HALF_RANGE = math.pi / 6.0
TARGET_RATIO = (0.5, 0.5, 0.5)
RESET_LINEAR_VELOCITY = 0.2
RESET_ANGULAR_VELOCITY = 0.2


def make_depth_navigation_viewer_cfg() -> ViewerCfg:
    """Build the shared yaw-relative chase-camera configuration."""

    return ViewerCfg(
        eye=(-3.0, 0.0, 1.2),
        lookat=(1.0, 0.0, 0.15),
        origin_type="world",
        asset_name="robot",
        env_index=0,
    )


def update_depth_chase_view(sim, robot, viewer: ViewerCfg, enabled: bool) -> None:
    """Place the GUI camera behind the selected drone in its yaw frame."""

    if not enabled or not sim.has_gui():
        return

    position = robot.data.root_pos_w[viewer.env_index]
    _, _, yaw = euler_xyz_from_quat(
        robot.data.root_quat_w[viewer.env_index : viewer.env_index + 1]
    )
    cos_yaw = torch.cos(yaw[0])
    sin_yaw = torch.sin(yaw[0])

    def rotate_yaw(offset: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            (
                cos_yaw * offset[0] - sin_yaw * offset[1],
                sin_yaw * offset[0] + cos_yaw * offset[1],
                offset[2],
            )
        )

    eye = position + rotate_yaw(position.new_tensor(viewer.eye))
    target = position + rotate_yaw(position.new_tensor(viewer.lookat))
    sim.set_camera_view(eye=eye.tolist(), target=target.tolist())


@configclass
class DepthNavigationCurriculumCfg:
    """Common obstacle-count curriculum for depth navigation."""

    min_level: int = 45
    max_level: int = 140
    check_after_log_instances: int = 2048
    increase_step: int = 2
    decrease_step: int = 1
    success_rate_for_increase: float = 0.7
    success_rate_for_decrease: float = 0.6


def make_depth_navigation_obstacle_cfg() -> ObstacleFieldCfg:
    """Build the dense central obstacle field and empty perimeter shell."""

    return ObstacleFieldCfg(
        obstacle_core_radius=OBSTACLE_CORE_RADIUS,
        spawn_forward_only=False,
    )


def sample_depth_navigation_spawn(
    obstacle_cfg: ObstacleFieldCfg,
    count: int,
    target_local: torch.Tensor,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return perimeter-shell positions and goal-facing wrapped yaws."""

    pos_local = sample_perimeter_spawn_positions(
        obstacle_cfg,
        count,
        SPAWN_RING_MIN_RADIUS,
        SPAWN_RING_MAX_RADIUS,
        SPAWN_Z_HALF_RANGE,
        device,
    )
    goal_bearing = torch.atan2(
        target_local[:, 1] - pos_local[:, 1],
        target_local[:, 0] - pos_local[:, 0],
    )
    yaw_offset = torch.empty(count, device=device).uniform_(
        -SPAWN_YAW_HALF_RANGE, SPAWN_YAW_HALF_RANGE
    )
    return pos_local, wrap_to_pi(goal_bearing + yaw_offset)
