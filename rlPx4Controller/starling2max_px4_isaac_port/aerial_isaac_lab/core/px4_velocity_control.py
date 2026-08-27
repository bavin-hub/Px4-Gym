"""Torch port of this repository's PX4-like velocity outer loop.

The equations mirror ``include/SimplePositionController.hpp`` in
``CTRL_VEL_ONLY`` mode, with the Starling 2 Max hover throttle supplied by the
caller instead of the C++ controller's vehicle-specific hard-coded value.
"""

from __future__ import annotations

import torch

from .math import euler_xyz_from_quat, quat_apply, quat_from_euler_xyz


class PX4VelocityController:
    """Batched body-FLU velocity -> collective-thrust/attitude controller.

    The navigation policy command is ``[vx, vy, vz, raw_delta_yaw]`` in the
    vehicle's yaw-only FLU frame.  The linear command is rotated into the world
    frame before applying the same velocity-P, acceleration clamp, collective
    thrust, roll, and pitch equations as the C++ ``SimplePositionController``.

    The fourth channel is intentionally not consumed here. The environment
    converts it to an absolute measured-yaw-relative setpoint and applies that
    after this outer loop returns.
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        hover_thrust: float = 0.13,
        velocity_gain: tuple[float, float, float] = (1.5, 1.5, 1.5),
        max_acceleration: float = 4.0,
        gravity: float = 9.80665,
    ):
        self.num_envs = int(num_envs)
        self.device = device
        self.hover_thrust = float(hover_thrust)
        self.max_acceleration = float(max_acceleration)
        self.gravity = float(gravity)
        self.velocity_gain = torch.tensor(
            velocity_gain, dtype=torch.float32, device=device
        ).expand(self.num_envs, -1)

    def compute(
        self,
        command_b: torch.Tensor,
        root_quat_wxyz: torch.Tensor,
        root_lin_vel_w: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``[collective_thrust, roll_sp, pitch_sp, yaw_sp]``.

        Args:
            command_b: Body-FLU ``[vx, vy, vz, raw_delta_yaw]`` command. The
                delta-yaw channel is preserved for the caller but not used here.
            root_quat_wxyz: Body attitude in the world frame.
            root_lin_vel_w: Measured world-frame linear velocity.
        """
        if command_b.shape != (self.num_envs, 4):
            raise ValueError(
                f"Expected command shape {(self.num_envs, 4)}, got {tuple(command_b.shape)}"
            )

        _, _, yaw = euler_xyz_from_quat(root_quat_wxyz)
        zeros = torch.zeros_like(yaw)
        vehicle_quat = quat_from_euler_xyz(zeros, zeros, yaw)
        velocity_sp_w = quat_apply(vehicle_quat, command_b[:, :3])

        desired_acceleration_w = self.velocity_gain * (velocity_sp_w - root_lin_vel_w)
        desired_acceleration_w = desired_acceleration_w.clamp(
            -self.max_acceleration, self.max_acceleration
        )

        collective_thrust = self.hover_thrust * (
            1.0 + desired_acceleration_w[:, 2] / self.gravity
        )
        sin_yaw = torch.sin(yaw)
        cos_yaw = torch.cos(yaw)
        roll_sp = (
            desired_acceleration_w[:, 0] * sin_yaw
            - desired_acceleration_w[:, 1] * cos_yaw
        ) / self.gravity
        pitch_sp = (
            desired_acceleration_w[:, 0] * cos_yaw
            + desired_acceleration_w[:, 1] * sin_yaw
        ) / self.gravity

        return torch.stack((collective_thrust, roll_sp, pitch_sp, yaw), dim=-1)
