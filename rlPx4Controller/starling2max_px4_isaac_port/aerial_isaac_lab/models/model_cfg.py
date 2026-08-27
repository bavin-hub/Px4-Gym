"""Simulator-independent configuration shared by all multirotor models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MultirotorModelCfg:
    """Aerial Gym parameters needed by the generic motor/allocation backend."""

    urdf_path: str
    root_body_name: str
    motor_body_names: tuple[str, ...]
    motor_directions: tuple[float, ...]
    allocation_matrix: tuple[tuple[float, ...], ...]
    thrust_constant: float
    motor_tau_increasing: float
    motor_tau_decreasing: float
    min_thrust: float
    max_thrust: float
    max_thrust_rate: float
    thrust_to_torque_ratio: float
    rigid_linear_damping: float
    rigid_angular_damping: float
    aerodynamic_linear_damping: tuple[float, float, float]
    aerodynamic_quadratic_damping: tuple[float, float, float]
    aerodynamic_angular_linear_damping: tuple[float, float, float]
    aerodynamic_angular_quadratic_damping: tuple[float, float, float]
