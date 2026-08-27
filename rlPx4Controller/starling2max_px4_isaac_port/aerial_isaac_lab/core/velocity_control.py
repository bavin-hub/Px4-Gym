"""Aerial Gym Lee velocity controller rewritten for Isaac Lab ``wxyz`` states."""

from __future__ import annotations

import math

import torch

from .math import (
    euler_xyz_from_quat,
    matrix_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_from_euler_xyz,
    quat_from_matrix,
    quat_inv,
    quat_mul,
)


class LeeVelocityController:
    """Velocity-to-wrench controller preserving the legacy Lee-control constants."""

    def __init__(self, num_envs: int, device: torch.device):
        self.num_envs = num_envs
        self.device = device
        # Means of Aerial Gym's non-randomized gain ranges.
        self.k_vel = torch.tensor((2.5, 2.5, 2.5), device=device).expand(num_envs, -1)
        self.k_rot = torch.tensor((1.0, 1.0, 0.5), device=device).expand(num_envs, -1)
        self.k_angvel = torch.tensor((0.15, 0.15, 0.15), device=device).expand(num_envs, -1)
        self.max_yaw_rate = math.pi / 3.0
        self.gravity_w = torch.tensor((0.0, 0.0, -9.81), device=device)

    def compute(
        self,
        command_b: torch.Tensor,
        root_pos_w: torch.Tensor,
        root_quat_wxyz: torch.Tensor,
        root_lin_vel_w: torch.Tensor,
        root_ang_vel_b: torch.Tensor,
        mass: torch.Tensor,
        inertia_b: torch.Tensor,
    ) -> torch.Tensor:
        """Compute local-root wrench for body-FLU ``[vx, vy, vz, yaw_rate]`` commands."""
        roll, pitch, yaw = euler_xyz_from_quat(root_quat_wxyz)
        vehicle_quat = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
        desired_velocity_w = quat_apply(vehicle_quat, command_b[:, :3])
        acceleration_w = self.k_vel * (desired_velocity_w - root_lin_vel_w)
        force_w = (acceleration_w - self.gravity_w) * mass.unsqueeze(-1)

        wrench_b = torch.zeros((self.num_envs, 6), device=self.device)
        body_z_w = matrix_from_quat(root_quat_wxyz)[..., :, 2]
        wrench_b[:, 2] = (force_w * body_z_w).sum(dim=-1)

        desired_quat = self._desired_orientation(force_w, yaw)
        desired_angvel_b = self._yaw_rate_to_body_rates(command_b[:, 3], roll, pitch)
        relative_quat = quat_mul(quat_inv(root_quat_wxyz), desired_quat)
        relative_rotation = matrix_from_quat(relative_quat)
        skew = relative_rotation.transpose(-2, -1) - relative_rotation
        rotation_error = 0.5 * torch.stack(
            (-skew[:, 1, 2], skew[:, 0, 2], -skew[:, 0, 1]), dim=-1
        )
        angular_velocity_error = root_ang_vel_b - quat_apply(relative_quat, desired_angvel_b)
        gyroscopic_term = torch.cross(
            root_ang_vel_b, torch.bmm(inertia_b, root_ang_vel_b.unsqueeze(-1)).squeeze(-1), dim=-1
        )
        wrench_b[:, 3:] = (
            -self.k_rot * rotation_error - self.k_angvel * angular_velocity_error + gyroscopic_term
        )
        return wrench_b

    def _desired_orientation(self, force_w: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        b3 = force_w / force_w.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
        heading = torch.stack((torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)), dim=-1)
        b2 = torch.cross(b3, heading, dim=-1)
        b2 = b2 / b2.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
        b1 = torch.cross(b2, b3, dim=-1)
        return quat_from_matrix(torch.stack((b1, b2, b3), dim=-1))

    def _yaw_rate_to_body_rates(self, yaw_rate: torch.Tensor, roll: torch.Tensor, pitch: torch.Tensor) -> torch.Tensor:
        yaw_rate = yaw_rate.clamp(-self.max_yaw_rate, self.max_yaw_rate)
        result = torch.zeros((self.num_envs, 3), device=self.device)
        result[:, 0] = -torch.sin(pitch) * yaw_rate
        result[:, 1] = torch.sin(roll) * torch.cos(pitch) * yaw_rate
        result[:, 2] = torch.cos(roll) * torch.cos(pitch) * yaw_rate
        return result
