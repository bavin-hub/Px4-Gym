"""Torch-native PX4-style attitude and rate control for Isaac Lab training.

This mirrors the quaternion attitude controller and rate PID structure in this
repository's C++ ``Px4AttitudeController`` and ``Px4RateController`` while
staying on batched Torch tensors for large Isaac Lab rollouts.
"""

from __future__ import annotations

import math

import torch

from .math import matrix_from_quat, quat_from_euler_xyz, quat_inv, quat_mul


def _canonical_quat(quat: torch.Tensor) -> torch.Tensor:
    sign = torch.where(quat[..., :1] < 0.0, -1.0, 1.0)
    return quat * sign


def _normalize_quat(quat: torch.Tensor) -> torch.Tensor:
    return quat / quat.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)


def _quat_from_two_vectors(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Shortest rotation from ``src`` to ``dst`` using ``wxyz`` quaternions."""
    cross = torch.cross(src, dst, dim=-1)
    dot = (src * dst).sum(dim=-1)
    norm_product = (src.square().sum(dim=-1) * dst.square().sum(dim=-1)).sqrt()
    quat = torch.cat(((dot + norm_product).unsqueeze(-1), cross), dim=-1)

    opposite = torch.logical_and(cross.norm(dim=-1) < 1.0e-5, dot < 0.0)
    abs_src = src.abs()
    use_x = torch.logical_and(abs_src[:, 0] < abs_src[:, 1], abs_src[:, 0] < abs_src[:, 2])
    use_y = torch.logical_and(~use_x, abs_src[:, 1] < abs_src[:, 2])
    basis = torch.zeros_like(src)
    basis[:, 0] = use_x.to(src.dtype)
    basis[:, 1] = use_y.to(src.dtype)
    basis[:, 2] = torch.logical_not(torch.logical_or(use_x, use_y)).to(src.dtype)
    opposite_cross = torch.cross(src, basis, dim=-1)
    opposite_quat = torch.cat((torch.zeros_like(dot).unsqueeze(-1), opposite_cross), dim=-1)
    quat = torch.where(opposite.unsqueeze(-1), opposite_quat, quat)
    return _normalize_quat(quat)


class PX4AttitudeRateController:
    """Batched PX4-like attitude -> body-rate -> normalized control controller."""

    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        hover_thrust: float = 0.13,
        min_thrust: float = 0.04,
        max_thrust: float = 0.60,
        attitude_interval: int = 1,
    ):
        self.num_envs = num_envs
        self.device = device
        self.hover_thrust = float(hover_thrust)
        self.min_thrust = float(min_thrust)
        self.max_thrust = float(max_thrust)
        self.yaw_weight = 2.8 / 8.0
        self.yawspeed_setpoint = torch.zeros(num_envs, device=device)

        attitude_p = torch.tensor((8.0, 8.0, 8.0), device=device)
        rate_p = torch.tensor((0.064, 0.064, 0.070), device=device)
        rate_i = torch.tensor((0.100, 0.096, 0.500), device=device)
        rate_d = torch.tensor((0.0013, 0.0011, 0.0), device=device)
        self.nominal_attitude_p = attitude_p.expand(num_envs, -1).clone()
        self.nominal_rate_p = rate_p.expand(num_envs, -1).clone()
        self.nominal_rate_i = rate_i.expand(num_envs, -1).clone()
        self.nominal_rate_d = rate_d.expand(num_envs, -1).clone()
        self.attitude_p = self.nominal_attitude_p.clone()
        self.rate_p = self.nominal_rate_p.clone()
        self.rate_i = self.nominal_rate_i.clone()
        self.rate_d = self.nominal_rate_d.clone()
        self.rate_integral = torch.zeros((num_envs, 3), device=device)
        self.previous_ang_vel = torch.zeros((num_envs, 3), device=device)
        # Multi-rate loop: compute() is called once per physics tick (the rate
        # loop). The attitude loop refreshes rate_sp only every attitude_interval
        # ticks and holds it in between, so it runs slower than the rate loop.
        self.attitude_interval = max(int(attitude_interval), 1)
        self._tick = 0
        self.rate_sp = torch.zeros((num_envs, 3), device=device)
        self.integral_limit = torch.tensor((0.3, 0.3, 0.3), device=device)
        # PX4 MC_ROLLRATE_MAX / MC_PITCHRATE_MAX / MC_YAWRATE_MAX (deg/s) from the
        # 4007_gz_starling2max airframe: 130 / 130 / 150.
        self.rate_limit = torch.tensor(
            (math.radians(130.0), math.radians(130.0), math.radians(150.0)), device=device
        )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.rate_integral.zero_()
            self.previous_ang_vel.zero_()
            self.rate_sp.zero_()
            self._tick = 0
            return
        self.rate_integral[env_ids] = 0.0
        self.previous_ang_vel[env_ids] = 0.0
        self.rate_sp[env_ids] = 0.0

    def compute(
        self,
        command: torch.Tensor,
        root_quat_wxyz: torch.Tensor,
        root_ang_vel_b: torch.Tensor,
        dt: float,
        angular_accel_b: torch.Tensor | None = None,
        landed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return normalized ``[roll, pitch, yaw, thrust]`` control setpoints.

        ``command`` preserves the attitude-delta task contract after scaling:
        ``[px4_normalized_collective_thrust, roll_sp, pitch_sp, yaw_sp]``.
        """
        if self._tick % self.attitude_interval == 0:
            desired_quat = quat_from_euler_xyz(command[:, 1], command[:, 2], command[:, 3])
            self.rate_sp = self._attitude_update(root_quat_wxyz, desired_quat)
        self._tick += 1
        torque_control = self._rate_update(
            self.rate_sp,
            root_ang_vel_b,
            dt,
            angular_accel_b=angular_accel_b,
            landed=landed,
        )
        collective = command[:, 0].clamp(self.min_thrust, self.max_thrust)
        return torch.cat((torque_control.clamp(-1.0, 1.0), collective.unsqueeze(-1)), dim=-1)

    def compute_rate(
        self,
        command: torch.Tensor,
        root_ang_vel_b: torch.Tensor,
        dt: float,
        angular_accel_b: torch.Tensor | None = None,
        landed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run only the PX4 rate loop for a direct CTBR command.

        ``command`` is ``[roll_rate_sp, pitch_rate_sp, yaw_rate_sp, collective]``
        with body-rate setpoints in rad/s and normalized collective thrust.

        The rate setpoint is clamped to the airframe's MC_*RATE_MAX, exactly as
        ``_attitude_update`` already does for the cascaded path.  Without this a
        CTBR policy scaled to +/-180 deg/s can command rates PX4 will silently
        clip on the vehicle (130/130/150), so the region it explores in sim
        would not exist in flight.
        """
        rate_sp = command[:, :3].clamp(-self.rate_limit, self.rate_limit)
        torque_control = self._rate_update(
            rate_sp,
            root_ang_vel_b,
            dt,
            angular_accel_b=angular_accel_b,
            landed=landed,
        )
        collective = command[:, 3].clamp(self.min_thrust, self.max_thrust)
        return torch.cat((torque_control.clamp(-1.0, 1.0), collective.unsqueeze(-1)), dim=-1)

    def _attitude_update(self, q: torch.Tensor, qd: torch.Tensor) -> torch.Tensor:
        rotation = matrix_from_quat(q)
        desired_rotation = matrix_from_quat(qd)
        e_z = rotation[:, :, 2]
        e_z_d = desired_rotation[:, :, 2]

        qd_red = _quat_from_two_vectors(e_z, e_z_d)
        corner_case = torch.logical_or(
            qd_red[:, 1].abs() > 1.0 - 1.0e-5,
            qd_red[:, 2].abs() > 1.0 - 1.0e-5,
        )
        qd_red = quat_mul(qd_red, q)
        qd_red = torch.where(corner_case.unsqueeze(-1), qd, qd_red)

        q_mix = quat_mul(quat_inv(qd_red), qd)
        q_mix = _canonical_quat(q_mix)
        mix_w = q_mix[:, 0].clamp(-1.0, 1.0)
        mix_z = q_mix[:, 3].clamp(-1.0, 1.0)
        yaw_mix = torch.stack(
            (
                torch.cos(self.yaw_weight * torch.acos(mix_w)),
                torch.zeros_like(mix_w),
                torch.zeros_like(mix_w),
                torch.sin(self.yaw_weight * torch.asin(mix_z)),
            ),
            dim=-1,
        )
        qd_weighted = quat_mul(qd_red, yaw_mix)
        qe = _canonical_quat(quat_mul(quat_inv(q), qd_weighted))
        attitude_error = 2.0 * qe[:, 1:4]
        rate_sp = attitude_error * self.attitude_p
        world_z_in_body = matrix_from_quat(quat_inv(q))[:, :, 2]
        rate_sp = rate_sp + world_z_in_body * self.yawspeed_setpoint.unsqueeze(-1)
        return rate_sp.clamp(-self.rate_limit, self.rate_limit)

    def _rate_update(
        self,
        rate_sp: torch.Tensor,
        root_ang_vel_b: torch.Tensor,
        dt: float,
        angular_accel_b: torch.Tensor | None = None,
        landed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if angular_accel_b is None:
            # PX4 damps the angular acceleration (derivative on the measured
            # body rate, not the error) so the rate loop runs as a true PID.
            angular_accel_b = (root_ang_vel_b - self.previous_ang_vel) / max(dt, 1.0e-8)
        self.previous_ang_vel[:] = root_ang_vel_b
        rate_error = rate_sp - root_ang_vel_b
        torque_control = self.rate_p * rate_error + self.rate_integral - self.rate_d * angular_accel_b

        i_factor = rate_error / math.radians(400.0)
        i_factor = (1.0 - i_factor.square()).clamp_min(0.0)
        dt_tensor = torch.as_tensor(dt, dtype=rate_error.dtype, device=self.device)
        next_integral = self.rate_integral + i_factor * self.rate_i * rate_error * dt_tensor
        next_integral = torch.max(
            torch.min(next_integral, self.integral_limit), -self.integral_limit
        )
        if landed is not None:
            next_integral = torch.where(landed.unsqueeze(-1), self.rate_integral, next_integral)
        self.rate_integral[:] = next_integral
        return torque_control
