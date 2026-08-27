"""PX4/Gazebo-style motor allocation and motor physics for Isaac Lab.

This module keeps the training controllers independent from PX4 internals:
controllers may continue to output an SI body wrench, while this backend
normalizes that wrench into a PX4-like control setpoint, applies PX4's motor
output transforms, and evaluates the Gazebo-classic multicopter motor model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ALLOCATION_PATH = _PROJECT_ROOT / "config" / "allocation_starling2max.yaml"


def _tuple_float(values: list[Any] | tuple[Any, ...]) -> tuple[float, ...]:
    return tuple(float(v) for v in values)


def _matrix_float(values: list[list[Any]]) -> tuple[tuple[float, ...], ...]:
    return tuple(_tuple_float(row) for row in values)


@dataclass(frozen=True)
class PX4AllocationConfig:
    """Static parameters required by the PX4/Gazebo-style backend."""

    model_name: str
    root_body_name: str
    motor_body_names: tuple[str, ...]
    rotor_axis: tuple[float, float, float]
    max_thrust: float
    max_roll_moment: float
    max_pitch_moment: float
    max_yaw_moment: float
    clip_normalized_control: bool
    actuator_min: float
    actuator_max: float
    slew_rate: float
    motor_positions: tuple[tuple[float, float, float], ...]
    direction_signs: tuple[float, ...]
    thrust_model_factor: float
    reversible_flags: int
    output_min: float
    output_max: float
    output_disarmed: float
    output_failsafe: float
    reverse_mask: int
    ramp_up: bool
    ramp_time: float
    max_rot_velocity: float
    motor_constant: float
    moment_constant: float
    time_constant_up: float
    time_constant_down: float
    rotor_drag_coefficient: float
    rolling_moment_coefficient: float
    rotor_velocity_slowdown_sim: float
    max_relative_airspeed: float
    reversible: bool
    enable_air_drag: bool
    enable_rolling_moment: bool

    @property
    def num_motors(self) -> int:
        return len(self.motor_body_names)


def load_px4_allocation_config(path: str | Path = DEFAULT_ALLOCATION_PATH) -> PX4AllocationConfig:
    """Load the YAML-backed PX4/Gazebo motor-allocation config."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)

    model = data["model"]
    frames = data["frames"]
    normalization = data["normalization"]
    allocation = data["allocation"]
    function_motors = data["function_motors"]
    mixing_output = data["mixing_output"]
    motor_model = data["motor_model"]

    motor_body_names = tuple(str(name) for name in model["motor_body_names"])
    motor_positions = _matrix_float(allocation["motor_positions"])
    direction_signs = _tuple_float(allocation["direction_signs"])

    if len(motor_positions) != len(motor_body_names):
        raise ValueError("motor_positions must have one entry per motor_body_name")
    if len(direction_signs) != len(motor_body_names):
        raise ValueError("direction_signs must have one entry per motor_body_name")

    return PX4AllocationConfig(
        model_name=str(model["name"]),
        root_body_name=str(model["root_body_name"]),
        motor_body_names=motor_body_names,
        rotor_axis=_tuple_float(frames["rotor_axis"]),  # type: ignore[arg-type]
        max_thrust=float(normalization["max_thrust"]),
        max_roll_moment=float(normalization["max_roll_moment"]),
        max_pitch_moment=float(normalization["max_pitch_moment"]),
        max_yaw_moment=float(normalization["max_yaw_moment"]),
        clip_normalized_control=bool(normalization.get("clip", True)),
        actuator_min=float(allocation["actuator_min"]),
        actuator_max=float(allocation["actuator_max"]),
        slew_rate=float(allocation.get("slew_rate", 0.0)),
        motor_positions=motor_positions,  # type: ignore[arg-type]
        direction_signs=direction_signs,
        thrust_model_factor=float(function_motors["thrust_model_factor"]),
        reversible_flags=int(function_motors["reversible_flags"]),
        output_min=float(mixing_output["output_min"]),
        output_max=float(mixing_output["output_max"]),
        output_disarmed=float(mixing_output["output_disarmed"]),
        output_failsafe=float(mixing_output["output_failsafe"]),
        reverse_mask=int(mixing_output["reverse_mask"]),
        ramp_up=bool(mixing_output["ramp_up"]),
        ramp_time=float(mixing_output["ramp_time"]),
        max_rot_velocity=float(motor_model["max_rot_velocity"]),
        motor_constant=float(motor_model["motor_constant"]),
        moment_constant=float(motor_model["moment_constant"]),
        time_constant_up=float(motor_model["time_constant_up"]),
        time_constant_down=float(motor_model["time_constant_down"]),
        rotor_drag_coefficient=float(motor_model["rotor_drag_coefficient"]),
        rolling_moment_coefficient=float(motor_model["rolling_moment_coefficient"]),
        rotor_velocity_slowdown_sim=float(motor_model["rotor_velocity_slowdown_sim"]),
        max_relative_airspeed=float(motor_model["max_relative_airspeed"]),
        reversible=bool(motor_model["reversible"]),
        enable_air_drag=bool(motor_model["enable_air_drag"]),
        enable_rolling_moment=bool(motor_model["enable_rolling_moment"]),
    )


class PX4GazeboMotorBackend:
    """PX4-like allocator plus Gazebo-classic motor model.

    Public methods mirror the existing allocator where possible. The main entry
    point is ``allocate_wrench(wrench_b)``, which accepts the current controller
    output in SI units and returns per-motor force/torque tensors in body frame.
    """

    def __init__(
        self,
        num_envs: int,
        dt: float,
        device: torch.device,
        config: PX4AllocationConfig | None = None,
        config_path: str | Path = DEFAULT_ALLOCATION_PATH,
    ):
        self.cfg = config if config is not None else load_px4_allocation_config(config_path)
        self.num_envs = num_envs
        self.dt = float(dt)
        self.device = device
        self.num_motors = self.cfg.num_motors

        self.motor_positions = torch.tensor(
            self.cfg.motor_positions, dtype=torch.float32, device=device
        )
        self.direction_signs = torch.tensor(
            self.cfg.direction_signs, dtype=torch.float32, device=device
        )
        self.rotor_axis = torch.tensor(self.cfg.rotor_axis, dtype=torch.float32, device=device)

        self.max_motor_thrust = self.cfg.motor_constant * self.cfg.max_rot_velocity**2
        self.effectiveness = self._build_effectiveness_matrix()
        self.inverse_effectiveness = torch.linalg.pinv(self.effectiveness).expand(num_envs, -1, -1)

        self.actuator_min = torch.full(
            (num_envs, self.num_motors), self.cfg.actuator_min, device=device
        )
        self.actuator_max = torch.full(
            (num_envs, self.num_motors), self.cfg.actuator_max, device=device
        )
        self.rotor_velocity = torch.zeros((num_envs, self.num_motors), device=device)
        self.joint_velocity = torch.zeros_like(self.rotor_velocity)
        self.previous_actuator_sp = torch.zeros_like(self.rotor_velocity)
        self.armed_time = torch.zeros(num_envs, device=device)
        self.armed = True
        # Optional per-environment multipliers set by a DomainRandomizer.  None
        # means "use the config scalar", so the nominal path is untouched.
        self.motor_constant_scale: torch.Tensor | None = None
        self.time_constant_up_scale: torch.Tensor | None = None
        self.time_constant_down_scale: torch.Tensor | None = None

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.rotor_velocity[env_ids] = 0.0
        self.joint_velocity[env_ids] = 0.0
        self.previous_actuator_sp[env_ids] = 0.0
        self.armed_time[env_ids] = 0.0

    def _build_effectiveness_matrix(self) -> torch.Tensor:
        """Return PX4-like normalized effectiveness, rows [roll, pitch, yaw, thrust]."""

        positions = self.motor_positions
        full_force = torch.full((self.num_motors,), self.max_motor_thrust, device=self.device)
        roll = positions[:, 1] * full_force / self.cfg.max_roll_moment
        pitch = -positions[:, 0] * full_force / self.cfg.max_pitch_moment
        yaw = (
            -self.direction_signs
            * full_force
            * self.cfg.moment_constant
            / self.cfg.max_yaw_moment
        )
        thrust = full_force / self.cfg.max_thrust
        return torch.stack((roll, pitch, yaw, thrust), dim=0)

    def normalize_wrench(self, wrench_b: torch.Tensor) -> torch.Tensor:
        """Convert SI body wrench [Fx, Fy, Fz, Mx, My, Mz] to PX4-like controls."""

        control = torch.stack(
            (
                wrench_b[:, 3] / self.cfg.max_roll_moment,
                wrench_b[:, 4] / self.cfg.max_pitch_moment,
                wrench_b[:, 5] / self.cfg.max_yaw_moment,
                wrench_b[:, 2] / self.cfg.max_thrust,
            ),
            dim=-1,
        )
        if self.cfg.clip_normalized_control:
            control[:, :3] = control[:, :3].clamp(-1.0, 1.0)
            control[:, 3] = control[:, 3].clamp(0.0, 1.0)
        return control

    def allocate_normalized_control(self, control: torch.Tensor) -> torch.Tensor:
        """PX4 ControlAllocator-style normalized control -> motor setpoints."""

        actuator_sp = torch.bmm(self.inverse_effectiveness, control.unsqueeze(-1)).squeeze(-1)
        actuator_sp = actuator_sp.clamp(self.actuator_min, self.actuator_max)

        if self.cfg.slew_rate > 0.0:
            max_delta = self.cfg.slew_rate * self.dt
            actuator_sp = torch.max(
                torch.min(actuator_sp, self.previous_actuator_sp + max_delta),
                self.previous_actuator_sp - max_delta,
            )

        self.previous_actuator_sp[:] = actuator_sp
        return actuator_sp

    def function_motors(self, actuator_sp: torch.Tensor) -> torch.Tensor:
        """Apply PX4 FunctionMotors transforms to actuator_motors.control."""

        values = actuator_sp.clone()
        factor = self.cfg.thrust_model_factor

        if factor > torch.finfo(values.dtype).eps and factor <= 1.0:
            a = factor
            b = 1.0 - factor
            tmp1 = b / (2.0 * a)
            tmp2 = b * b / (4.0 * a * a)
            positive = values > torch.finfo(values.dtype).eps
            negative = values < -torch.finfo(values.dtype).eps
            values = torch.where(
                positive,
                -tmp1 + torch.sqrt(tmp2 + values.clamp_min(0.0) / a),
                values,
            )
            values = torch.where(
                negative,
                tmp1 - torch.sqrt(tmp2 - values.clamp_max(0.0) / a),
                values,
            )

        for motor_idx in range(self.num_motors):
            reversible = bool(self.cfg.reversible_flags & (1 << motor_idx))
            if not reversible:
                invalid = values[:, motor_idx] < -torch.finfo(values.dtype).eps
                values[:, motor_idx] = torch.where(
                    invalid,
                    torch.full_like(values[:, motor_idx], torch.nan),
                    values[:, motor_idx] * 2.0 - 1.0,
                )

        return values

    def mixing_output(self, motor_values: torch.Tensor, armed: bool | None = None) -> torch.Tensor:
        """Apply the motor portion of PX4 MixingOutput scaling."""

        if armed is None:
            armed = self.armed
        command = torch.empty_like(motor_values)

        for motor_idx in range(self.num_motors):
            value = motor_values[:, motor_idx]
            value = torch.where(torch.isfinite(value), value, torch.full_like(value, -1.0))
            if self.cfg.reverse_mask & (1 << motor_idx):
                value = -value
            command[:, motor_idx] = self.cfg.output_min + (value + 1.0) * 0.5 * (
                self.cfg.output_max - self.cfg.output_min
            )

        if not armed:
            return torch.full_like(command, self.cfg.output_disarmed)

        if self.cfg.ramp_up:
            self.armed_time += self.dt
            progress = (self.armed_time / self.cfg.ramp_time).clamp(0.0, 1.0).unsqueeze(-1)
            command = self.cfg.output_disarmed + progress * (command - self.cfg.output_disarmed)

        return command

    def gazebo_motor_model(
        self,
        motor_speed_command: torch.Tensor,
        body_velocity_b: torch.Tensor | None = None,
        wind_velocity_b: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate Gazebo-classic motor force/torque equations in body frame."""

        command = motor_speed_command.clamp(max=self.cfg.max_rot_velocity)
        if not self.cfg.reversible:
            command = command.clamp_min(0.0)
        else:
            command = command.clamp(min=-self.cfg.max_rot_velocity)

        tau_up = torch.full_like(command, self.cfg.time_constant_up)
        tau_down = torch.full_like(command, self.cfg.time_constant_down)
        if self.time_constant_up_scale is not None:
            tau_up = tau_up * self.time_constant_up_scale
        if self.time_constant_down_scale is not None:
            tau_down = tau_down * self.time_constant_down_scale
        tau = torch.where(command.abs() > self.rotor_velocity.abs(), tau_up, tau_down)
        alpha = torch.exp(torch.full_like(command, -self.dt) / tau.clamp_min(1.0e-8))
        self.rotor_velocity[:] = alpha * self.rotor_velocity + (1.0 - alpha) * command

        self.joint_velocity[:] = (
            self.direction_signs.unsqueeze(0)
            * self.rotor_velocity
            / self.cfg.rotor_velocity_slowdown_sim
        )
        real_motor_velocity = self.joint_velocity * self.cfg.rotor_velocity_slowdown_sim
        motor_constant = self.cfg.motor_constant
        if self.motor_constant_scale is not None:
            motor_constant = motor_constant * self.motor_constant_scale
        force = real_motor_velocity * real_motor_velocity.abs() * motor_constant
        if not self.cfg.reversible:
            force = force.abs()

        force_b = torch.zeros((self.num_envs, self.num_motors, 3), device=self.device)
        torque_b = torch.zeros_like(force_b)

        scalar = torch.ones_like(force)
        velocity_perp = None
        if body_velocity_b is not None:
            wind = torch.zeros_like(body_velocity_b) if wind_velocity_b is None else wind_velocity_b
            relative_wind_velocity = body_velocity_b - wind
            parallel = (
                (relative_wind_velocity * self.rotor_axis).sum(dim=-1, keepdim=True)
                * self.rotor_axis
            )
            velocity_perp = relative_wind_velocity - parallel
            scalar = (
                1.0 - parallel.norm(dim=-1, keepdim=True) / self.cfg.max_relative_airspeed
            ).clamp(0.0, 1.0)

        force_b[..., 2] = force * scalar
        torque_b[..., 2] = -self.direction_signs.unsqueeze(0) * force * self.cfg.moment_constant

        if velocity_perp is not None and self.cfg.enable_air_drag:
            air_drag = (
                -self.rotor_velocity.abs().unsqueeze(-1)
                * self.cfg.rotor_drag_coefficient
                * velocity_perp.unsqueeze(1)
            )
            force_b += air_drag

        if velocity_perp is not None and self.cfg.enable_rolling_moment:
            rolling_moment = (
                -self.rotor_velocity.abs().unsqueeze(-1)
                * self.direction_signs.view(1, -1, 1)
                * self.cfg.rolling_moment_coefficient
                * velocity_perp.unsqueeze(1)
            )
            torque_b += rolling_moment

        return force_b, torque_b

    def allocate_wrench(
        self,
        wrench_b: torch.Tensor,
        body_velocity_b: torch.Tensor | None = None,
        wind_velocity_b: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-motor body-frame force and torque tensors for an SI wrench."""

        control = self.normalize_wrench(wrench_b)
        actuator_sp = self.allocate_normalized_control(control)
        motor_values = self.function_motors(actuator_sp)
        motor_speed_command = self.mixing_output(motor_values)
        return self.gazebo_motor_model(motor_speed_command, body_velocity_b, wind_velocity_b)


# Backwards-friendly aliases for callers that want a motor.py-like name.
PX4MultirotorAllocator = PX4GazeboMotorBackend
MultirotorPX4Allocator = PX4GazeboMotorBackend
