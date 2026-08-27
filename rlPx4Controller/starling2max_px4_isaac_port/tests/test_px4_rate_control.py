"""Pure-Torch checks for the direct PX4 body-rate entry point."""

import torch

from aerial_isaac_lab.core.px4_attitude_rate_control import PX4AttitudeRateController


def test_direct_rate_command_uses_ctbr_order() -> None:
    controller = PX4AttitudeRateController(
        1,
        torch.device("cpu"),
        min_thrust=0.0,
        max_thrust=0.9,
    )
    command = torch.tensor(((0.5, -0.25, 0.1, 0.13),))

    output = controller.compute_rate(command, torch.zeros((1, 3)), dt=0.001)

    expected_torque = command[:, :3] * controller.rate_p
    assert torch.allclose(output[:, :3], expected_torque)
    assert torch.allclose(output[:, 3], torch.tensor((0.13,)))


def test_direct_rate_collective_respects_controller_limits() -> None:
    controller = PX4AttitudeRateController(
        2,
        torch.device("cpu"),
        min_thrust=0.0,
        max_thrust=0.9,
    )
    command = torch.tensor(((0.0, 0.0, 0.0, -0.2), (0.0, 0.0, 0.0, 1.2)))

    output = controller.compute_rate(command, torch.zeros((2, 3)), dt=0.001)

    assert torch.allclose(output[:, 3], torch.tensor((0.0, 0.9)))
