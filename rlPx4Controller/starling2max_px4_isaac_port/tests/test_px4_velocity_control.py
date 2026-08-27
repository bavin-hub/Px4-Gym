"""Pure-Torch checks for the PX4 velocity outer-loop port."""

import math

import torch

from aerial_isaac_lab.core.math import quat_from_euler_xyz
from aerial_isaac_lab.core.px4_velocity_control import PX4VelocityController


def _quat_from_yaw(yaw: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros_like(yaw)
    return quat_from_euler_xyz(zeros, zeros, yaw)


def test_hover_command_returns_starling_hover_throttle() -> None:
    controller = PX4VelocityController(2, torch.device("cpu"), hover_thrust=0.13)
    command = torch.zeros((2, 4))
    yaw = torch.tensor((0.0, math.pi / 2.0))

    output = controller.compute(command, _quat_from_yaw(yaw), torch.zeros((2, 3)))

    assert torch.allclose(output[:, 0], torch.full((2,), 0.13))
    assert torch.allclose(output[:, 1:3], torch.zeros((2, 2)), atol=1.0e-7)
    assert torch.allclose(output[:, 3], yaw, atol=1.0e-7)


def test_body_forward_command_is_yaw_rotated_before_velocity_feedback() -> None:
    controller = PX4VelocityController(2, torch.device("cpu"), hover_thrust=0.13)
    command = torch.tensor(((1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)))
    yaw = torch.tensor((0.0, math.pi / 2.0))

    output = controller.compute(command, _quat_from_yaw(yaw), torch.zeros((2, 3)))

    expected_pitch = torch.full((2,), 1.5 / 9.80665)
    assert torch.allclose(output[:, 1], torch.zeros(2), atol=1.0e-6)
    assert torch.allclose(output[:, 2], expected_pitch, atol=1.0e-6)


def test_vertical_feedback_and_acceleration_clamp_match_cpp_equations() -> None:
    controller = PX4VelocityController(1, torch.device("cpu"), hover_thrust=0.13)
    command = torch.tensor(((100.0, 0.0, 100.0, 0.0),))
    quat = _quat_from_yaw(torch.zeros(1))

    output = controller.compute(command, quat, torch.zeros((1, 3)))

    assert torch.allclose(output[:, 0], torch.tensor((0.13 * (1.0 + 4.0 / 9.80665),)))
    assert torch.allclose(output[:, 2], torch.tensor((4.0 / 9.80665,)))


def test_delta_yaw_channel_does_not_change_outer_loop_attitude_or_thrust() -> None:
    controller = PX4VelocityController(2, torch.device("cpu"), hover_thrust=0.13)
    command = torch.tensor(((0.5, 0.0, 0.0, -1.0), (0.5, 0.0, 0.0, 1.0)))
    quat = _quat_from_yaw(torch.zeros(2))

    output = controller.compute(command, quat, torch.zeros((2, 3)))

    assert torch.allclose(output[0], output[1])
