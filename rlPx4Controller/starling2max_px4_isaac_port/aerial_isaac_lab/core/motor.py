"""Aerial Gym motor dynamics and allocation, independent of Isaac Lab.

The implementation intentionally retains the force/RPM first-order model and
RK4 integration used by ``aerial_gym.control.motor_model``.
"""

from __future__ import annotations

import torch

from aerial_isaac_lab.models.model_cfg import MultirotorModelCfg


class MotorDynamics:
    """Per-motor thrust state with Aerial Gym's RPM time-constant model."""

    def __init__(self, model: MultirotorModelCfg, num_envs: int, dt: float, device: torch.device):
        self.model = model
        self.num_envs = num_envs
        self.dt = dt
        self.device = device
        self.num_motors = len(model.motor_body_names)
        self.min_thrust = torch.full((num_envs, self.num_motors), model.min_thrust, device=device)
        self.max_thrust = torch.full((num_envs, self.num_motors), model.max_thrust, device=device)
        self.max_rate = torch.full((num_envs, self.num_motors), model.max_thrust_rate, device=device)
        self.thrust_constant = torch.full((num_envs, self.num_motors), model.thrust_constant, device=device)
        self.tau_inc = torch.full((num_envs, self.num_motors), model.motor_tau_increasing, device=device)
        self.tau_dec = torch.full((num_envs, self.num_motors), model.motor_tau_decreasing, device=device)
        self.current_thrust = torch.zeros((num_envs, self.num_motors), device=device)
        self.reset()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        # Aerial Gym randomizes the motor's latent initial state, rather than
        # silently starting every motor at hover thrust.
        self.current_thrust[env_ids] = torch.rand(
            (len(env_ids), self.num_motors), device=self.device
        ) * (self.max_thrust[env_ids] - self.min_thrust[env_ids]) + self.min_thrust[env_ids]

    def update(self, reference_thrust: torch.Tensor) -> torch.Tensor:
        reference_thrust = torch.clamp(reference_thrust, self.min_thrust, self.max_thrust)
        error = reference_thrust - self.current_thrust
        tau = torch.where(
            torch.sign(self.current_thrust) * torch.sign(error) < 0.0, self.tau_dec, self.tau_inc
        )
        mixing_factor = 1.0 / tau
        current_rpm = torch.sqrt((self.current_thrust / self.thrust_constant).clamp_min(0.0))
        desired_rpm = torch.sqrt((reference_thrust / self.thrust_constant).clamp_min(0.0))

        def rate(rpm: torch.Tensor) -> torch.Tensor:
            return torch.clamp(mixing_factor * (desired_rpm - rpm), -self.max_rate, self.max_rate)

        # Same RK4 form as the legacy Aerial Gym model.
        k1 = rate(current_rpm)
        k2 = rate(current_rpm + 0.5 * self.dt * k1)
        k3 = rate(current_rpm + 0.5 * self.dt * k2)
        k4 = rate(current_rpm + self.dt * k3)
        current_rpm = current_rpm + self.dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        self.current_thrust[:] = self.thrust_constant * current_rpm.square()
        return self.current_thrust


class MultirotorAllocator:
    """Map body wrench commands to motor-link local force and torque tensors."""

    def __init__(self, model: MultirotorModelCfg, num_envs: int, dt: float, device: torch.device):
        self.model = model
        self.num_envs = num_envs
        self.device = device
        self.motor_directions = torch.tensor(model.motor_directions, device=device)
        allocation = torch.tensor(model.allocation_matrix, dtype=torch.float32, device=device)
        self.inverse_allocation = torch.linalg.pinv(allocation).expand(num_envs, -1, -1)
        self.motor_dynamics = MotorDynamics(model, num_envs, dt, device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        self.motor_dynamics.reset(env_ids)

    def allocate_wrench(self, wrench_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(force_b, torque_b)`` for motors in ``model.motor_body_names`` order."""
        requested_thrust = torch.bmm(self.inverse_allocation, wrench_b.unsqueeze(-1)).squeeze(-1)
        thrust = self.motor_dynamics.update(requested_thrust)
        force_b = torch.zeros((*thrust.shape, 3), device=self.device)
        force_b[..., 2] = thrust
        torque_b = torch.zeros_like(force_b)
        torque_b[..., 2] = -self.model.thrust_to_torque_ratio * thrust * self.motor_directions
        return force_b, torque_b
