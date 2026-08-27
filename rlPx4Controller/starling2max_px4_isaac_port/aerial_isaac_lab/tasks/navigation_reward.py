"""TorchScript reward for the depth navigation task.

The base reward is ported from Aerial Gym's navigation task and augmented with
the goal-directed velocity-tracking term used by depth attitude-delta.

This lives in its own module deliberately: ``from __future__ import annotations``
turns annotations into strings, which ``torch.jit.script`` cannot always resolve.
Keeping these functions out of the task module lets the task use postponed
annotations while these stay scriptable.
"""

from typing import Dict, Tuple

import torch


@torch.jit.script
def exponential_reward_function(
    magnitude: float, exponent: float, value: torch.Tensor
) -> torch.Tensor:
    """Exponential reward function."""
    return magnitude * torch.exp(-(value * value) * exponent)


@torch.jit.script
def exponential_penalty_function(
    magnitude: float, exponent: float, value: torch.Tensor
) -> torch.Tensor:
    """Exponential penalty function: zero at ``value == 0``, negative elsewhere."""
    return magnitude * (torch.exp(-(value * value) * exponent) - 1.0)


@torch.jit.script
def compute_navigation_reward(
    pos_error: torch.Tensor,
    prev_pos_error: torch.Tensor,
    crashes: torch.Tensor,
    action: torch.Tensor,
    prev_action: torch.Tensor,
    velocity_vehicle: torch.Tensor,
    curriculum_progress_fraction: float,
    parameter_dict: Dict[str, float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Aerial Gym navigation reward plus goal-directed velocity tracking.

    Args:
        pos_error: Goal error in the vehicle (yaw-only) frame, ``(N, 3)``.
        prev_pos_error: The same quantity one policy step earlier.
        crashes: Per-environment collision flag as floats, ``(N,)``.
        action: Command ``[vx, 0, vz, raw_delta_yaw]``.
        prev_action: The command from the previous policy step.
        velocity_vehicle: Measured linear velocity in the vehicle yaw frame.
        curriculum_progress_fraction: ``(level - min_level) / (max_level - min_level)``.
            Scales the positive terms up and gates the absolute-action penalties.
        parameter_dict: ``reward_parameters`` from the task config.

    Returns:
        ``(reward, crashes)``. Crashed environments receive ``collision_penalty``
        in place of the shaped reward.
    """
    MULTIPLICATION_FACTOR_REWARD = 1.0 + (2.0) * curriculum_progress_fraction
    dist = torch.norm(pos_error, dim=1)
    prev_dist_to_goal = torch.norm(prev_pos_error, dim=1)
    pos_reward = exponential_reward_function(
        parameter_dict["pos_reward_magnitude"],
        parameter_dict["pos_reward_exponent"],
        dist,
    )
    very_close_to_goal_reward = exponential_reward_function(
        parameter_dict["very_close_to_goal_reward_magnitude"],
        parameter_dict["very_close_to_goal_reward_exponent"],
        dist,
    )

    getting_closer = prev_dist_to_goal - dist
    getting_closer_reward = torch.where(
        getting_closer > 0,
        parameter_dict["getting_closer_reward_multiplier"] * getting_closer,
        2.0 * parameter_dict["getting_closer_reward_multiplier"] * getting_closer,
    )

    distance_from_goal_reward = (20.0 - dist) / 20.0

    goal_direction = pos_error / dist.clamp_min(1.0e-6).unsqueeze(1)
    desired_speed = parameter_dict["cruise_speed"] * torch.clamp(
        dist / parameter_dict["brake_radius"], 0.0, 1.0
    )
    desired_velocity = desired_speed.unsqueeze(1) * goal_direction
    velocity_error = velocity_vehicle - desired_velocity
    velocity_tracking_reward = parameter_dict["velocity_tracking_magnitude"] * torch.exp(
        -parameter_dict["velocity_tracking_exponent"]
        * (velocity_error * velocity_error).sum(dim=1)
    )

    action_diff = action - prev_action
    x_diff_penalty = exponential_penalty_function(
        parameter_dict["x_action_diff_penalty_magnitude"],
        parameter_dict["x_action_diff_penalty_exponent"],
        action_diff[:, 0],
    )
    z_diff_penalty = exponential_penalty_function(
        parameter_dict["z_action_diff_penalty_magnitude"],
        parameter_dict["z_action_diff_penalty_exponent"],
        action_diff[:, 2],
    )
    delta_yaw_diff_penalty = exponential_penalty_function(
        parameter_dict["delta_yaw_action_diff_penalty_magnitude"],
        parameter_dict["delta_yaw_action_diff_penalty_exponent"],
        action_diff[:, 3],
    )
    action_diff_penalty = x_diff_penalty + z_diff_penalty + delta_yaw_diff_penalty

    # Absolute action penalties ramp in with the curriculum, so early levels do
    # not discourage moving at all.
    x_absolute_penalty = curriculum_progress_fraction * exponential_penalty_function(
        parameter_dict["x_absolute_action_penalty_magnitude"],
        parameter_dict["x_absolute_action_penalty_exponent"],
        action[:, 0],
    )
    z_absolute_penalty = curriculum_progress_fraction * exponential_penalty_function(
        parameter_dict["z_absolute_action_penalty_magnitude"],
        parameter_dict["z_absolute_action_penalty_exponent"],
        action[:, 2],
    )
    delta_yaw_absolute_penalty = curriculum_progress_fraction * exponential_penalty_function(
        parameter_dict["delta_yaw_absolute_action_penalty_magnitude"],
        parameter_dict["delta_yaw_absolute_action_penalty_exponent"],
        action[:, 3],
    )
    absolute_action_penalty = x_absolute_penalty + z_absolute_penalty + delta_yaw_absolute_penalty
    total_action_penalty = action_diff_penalty + absolute_action_penalty

    reward = (
        MULTIPLICATION_FACTOR_REWARD
        * (
            pos_reward
            + very_close_to_goal_reward
            + getting_closer_reward
            + distance_from_goal_reward
        )
        + velocity_tracking_reward
        + total_action_penalty
    )

    reward = torch.where(
        crashes > 0,
        parameter_dict["collision_penalty"] * torch.ones_like(reward),
        reward,
    )
    return reward, crashes
