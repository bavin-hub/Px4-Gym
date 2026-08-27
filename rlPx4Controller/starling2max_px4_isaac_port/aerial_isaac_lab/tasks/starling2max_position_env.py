"""Starling 2 Max position-setpoint task using Aerial Gym dynamics on Isaac Lab physics.

This is a separate task from ``starling2max_velocity_env``.  It reuses the same Starling 2 Max
mass/inertia, motor allocation, target sampling, observations, and sim2real
reward, but swaps the control contract to a Lee position controller:

policy action -> drone-relative ``[x, y, z]`` displacement + absolute ``yaw``.
"""

from __future__ import annotations

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils import configclass

from aerial_isaac_lab.core import LeePositionController
from aerial_isaac_lab.core.math import quat_apply_inverse

from .starling2max_velocity_env import Starling2MaxVelocityEnv, Starling2MaxVelocityEnvCfg


@configclass
class Starling2MaxPositionEnvCfg(Starling2MaxVelocityEnvCfg):
    """Configuration for Starling 2 Max policy outputs interpreted as position setpoints."""

    # Same Aerial Gym Starling 2 Max sim2real target/reward/observation settings as the
    # velocity task, but with a position-controller action contract.
    #
    # ``self._actions`` stores the clipped [-1, 1] policy output (so the
    # sim2real action penalties act on the native range).  In _apply_action the
    # first three channels scale to a metric displacement applied RELATIVE to
    # the current position, and the yaw channel scales to an absolute world-yaw
    # setpoint.
    action_position_scale = 5.0 / math.sqrt(3.0)
    action_yaw_scale = math.pi


class Starling2MaxPositionEnv(Starling2MaxVelocityEnv):
    """Direct-RL port whose policy emits a drone-relative ``[x, y, z]`` displacement and absolute ``yaw``."""

    cfg: Starling2MaxPositionEnvCfg

    def __init__(self, cfg: Starling2MaxPositionEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._controller = LeePositionController(self.num_envs, self.device)

        # Viewport-only marker for the current commanded position setpoint.
        setpoint_marker_cfg = VisualizationMarkersCfg(
            prim_path="/World/Visuals/Starling2MaxPositionSetpoints",
            markers={
                "setpoint": sim_utils.SphereCfg(
                    radius=0.08,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.75, 1.0)),
                )
            },
        )
        self._setpoint_gizmo = VisualizationMarkers(setpoint_marker_cfg)
        self._setpoint_gizmo_indices = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Store the clipped [-1, 1] policy output un-scaled.  The inherited
        # sim2real reward penalizes action magnitude/rate on this native range,
        # so action_penalty/action_difference no longer saturate the way they
        # would on metre/radian setpoints.  Scaling into a metric displacement
        # and an absolute yaw is deferred to _apply_action.
        self._previous_actions[:] = self._actions
        self._actions[:] = actions.clamp(-1.0, 1.0)
        self._previous_distance[:] = self._position_error_w().norm(dim=-1)

    def _apply_action(self) -> None:
        root_quat_wxyz = self._robot.data.root_quat_w
        root_lin_vel_w = self._robot.data.root_lin_vel_w
        root_ang_vel_b = quat_apply_inverse(root_quat_wxyz, self._robot.data.root_ang_vel_w)
        # Scale the normalized action into a metric command. The position
        # channels are applied as a displacement RELATIVE to the current
        # position: the observation is purely relative (target error), so a
        # relative setpoint is directly learnable and reachable at any range,
        # rather than an absolute env-local point bounded to the spawn box.
        # The yaw channel is an absolute world-yaw setpoint.
        command_w = torch.empty_like(self._actions)
        command_w[:, 0:3] = (
            self._robot.data.root_pos_w + self._actions[:, 0:3] * self.cfg.action_position_scale
        )
        command_w[:, 3] = self._actions[:, 3] * self.cfg.action_yaw_scale
        wrench_b = self._controller.compute(
            command_w,
            self._robot.data.root_pos_w,
            root_quat_wxyz,
            root_lin_vel_w,
            root_ang_vel_b,
            self._mass,
            self._inertia_b,
        )
        body_lin_vel = quat_apply_inverse(root_quat_wxyz, root_lin_vel_w)
        motor_forces_b, motor_torques_b = self._allocator.allocate_wrench(
            wrench_b, body_velocity_b=body_lin_vel
        )
        self._motor_forces_b.zero_()
        self._motor_torques_b.zero_()
        self._motor_forces_b[:, self._motor_body_ids] = motor_forces_b
        self._motor_torques_b[:, self._motor_body_ids] = motor_torques_b

        # Preserve the same Aerial Gym aerodynamic-drag hook as the velocity
        # task. Starling 2 Max's coefficients are currently zero, so this is exact today.
        drag_force = -self._linear_drag * body_lin_vel - self._quadratic_drag * body_lin_vel.norm(
            dim=-1, keepdim=True
        ) * body_lin_vel
        drag_torque = -self._angular_drag * root_ang_vel_b - self._angular_quadratic_drag * root_ang_vel_b.abs() * root_ang_vel_b
        self._motor_forces_b[:, self._root_body_id] += drag_force
        self._motor_torques_b[:, self._root_body_id] += drag_torque
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=torch.arange(self._robot.num_bodies, device=self.device),
            forces=self._motor_forces_b,
            torques=self._motor_torques_b,
        )

    def _get_observations(self) -> dict[str, torch.Tensor]:
        observations = super()._get_observations()
        self._setpoint_gizmo.visualize(
            translations=self._robot.data.root_pos_w
            + self._actions[:, 0:3] * self.cfg.action_position_scale,
            marker_indices=self._setpoint_gizmo_indices,
        )
        return observations
