"""Aerial Gym Lee attitude-yaw controller rewritten for Isaac Lab ``wxyz`` states."""

from __future__ import annotations

import torch

from .math import matrix_from_quat, quat_apply, quat_from_euler_xyz, quat_inv, quat_mul


class LeeAttitudeYawController:
    """Absolute roll/pitch/yaw + normalized-thrust controller from Aerial Gym."""

    def __init__(self, num_envs: int, device: torch.device):
        self.num_envs = num_envs
        self.device = device
        # Means of Aerial Gym's non-randomized Lee-controller gain ranges. The nominal
        # values are kept so a DomainRandomizer can rescale from them each reset
        # instead of compounding scales onto already-scaled gains.
        self.nominal_k_rot = torch.tensor((1.0, 1.0, 0.5), device=device).expand(num_envs, -1)
        self.nominal_k_angvel = torch.tensor((0.15, 0.15, 0.15), device=device).expand(num_envs, -1)
        self.k_rot = self.nominal_k_rot
        self.k_angvel = self.nominal_k_angvel
        self.gravity_norm = 9.81

    def compute(
        self,
        command: torch.Tensor,
        root_quat_wxyz: torch.Tensor,
        root_ang_vel_b: torch.Tensor,
        mass: torch.Tensor,
        inertia_b: torch.Tensor,
    ) -> torch.Tensor:
        """Compute local-root wrench for ``[thrust, roll, pitch, yaw]`` commands.

        ``command[:, 0]`` is Aerial Gym's normalized thrust.  Hover is zero,
        because the controller applies ``(thrust + 1) * mass * g``.
        """
        wrench_b = torch.zeros((self.num_envs, 6), device=self.device)
        wrench_b[:, 2] = (command[:, 0] + 1.0) * mass * self.gravity_norm

        desired_quat = quat_from_euler_xyz(command[:, 1], command[:, 2], command[:, 3])
        desired_angvel_b = torch.zeros((self.num_envs, 3), device=self.device)
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
