"""CRL-inspired reward for the depth attitude-delta navigation task.

The active reward combines CRL-style progress/control shaping with the current
repository's CTBR holding terms: a velocity target that tapers to zero near the
goal, a near-goal stopping reward, and a speed-gated recurring dwell bonus.

The previous Aerial-Gym-style dense reward is preserved as comments below the
active implementation so it can be restored without searching git history.
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
def abs_exponential_reward_function(
    magnitude: float, exponent: float, value: torch.Tensor
) -> torch.Tensor:
    """Exponential reward on ``|value|`` rather than ``value**2``.

    Matches the ``_abs_exp`` helper in the velocity task: a sharper peak at zero
    than the squared form, which suits an angular error.
    """
    return magnitude * torch.exp(-exponent * value.abs())


@torch.jit.script
def exponential_penalty_function(
    magnitude: float, exponent: float, value: torch.Tensor
) -> torch.Tensor:
    """Exponential penalty function: zero at ``value == 0``, negative elsewhere."""
    return magnitude * (torch.exp(-(value * value) * exponent) - 1.0)


@torch.jit.script
def compute_depth_attitude_delta_reward(
    pos_error: torch.Tensor,
    prev_pos_error: torch.Tensor,
    crashes: torch.Tensor,
    raw_action: torch.Tensor,
    prev_raw_action: torch.Tensor,
    yaw_error: torch.Tensor,
    speed: torch.Tensor,
    velocity_vehicle: torch.Tensor,
    min_obstacle_dist: torch.Tensor,
    reached_goal: torch.Tensor,
    parameter_dict: Dict[str, float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """CRL-style single-goal reward with attitude-delta action penalties.

    Args:
        pos_error: Goal error in the vehicle (yaw-only) frame, ``(N, 3)``.
        prev_pos_error: The same quantity one policy step earlier.
        crashes: Per-environment collision flag as floats, ``(N,)``.
        raw_action: Clipped policy output ``[delta_thrust, delta_pitch, delta_yaw]``
            in ``[-1, 1]``, shape ``(N, 3)``.
        prev_raw_action: The same from the previous policy step.
        yaw_error: Wrapped heading error to the goal bearing, ``(N,)``, already
            zeroed inside the arrival deadzone by the task.
        speed: World-frame speed magnitude in m/s, ``(N,)``. Retained in the
            interface and used by the stop/dwell terms.
        velocity_vehicle: Linear velocity in the same yaw-only vehicle frame as
            ``pos_error``, shape ``(N, 3)``.
        min_obstacle_dist: Nearest surface in the depth image, metres, ``(N,)``.
            Forward-FOV only, saturating at the sensor far plane.
        reached_goal: One only on the first step that enters the goal radius.
        parameter_dict: ``reward_parameters`` from the task config.

    Returns:
        ``(reward, crashes)``.
    """
    dist = torch.norm(pos_error, dim=1)
    prev_dist_to_goal = torch.norm(prev_pos_error, dim=1)

    # CRL's primary dense signal: signed progress. Moving away loses exactly what
    # the same displacement toward the goal earns, which keeps the potential
    # simple and prevents unrelated positive baselines from rewarding loitering.
    progress_reward = parameter_dict["progress_reward_multiplier"] * (
        prev_dist_to_goal - dist
    )

    yaw_alignment_reward = abs_exponential_reward_function(
        parameter_dict["yaw_alignment_reward_magnitude"],
        parameter_dict["yaw_alignment_reward_exponent"],
        yaw_error,
    )

    # CTBR-style braking: track cruise velocity while far away, then taper the
    # desired velocity continuously to zero over the final braking radius.
    goal_direction = pos_error / (dist.unsqueeze(-1) + 1.0e-6)
    desired_speed = parameter_dict["cruise_speed"] * torch.clamp(
        dist / parameter_dict["brake_radius"], 0.0, 1.0
    )
    desired_velocity = desired_speed.unsqueeze(-1) * goal_direction
    velocity_error = velocity_vehicle - desired_velocity
    velocity_tracking_reward = parameter_dict["velocity_tracking_magnitude"] * torch.exp(
        -parameter_dict["velocity_tracking_exponent"]
        * (velocity_error * velocity_error).sum(dim=1)
    )

    # Reward low total speed increasingly strongly through the braking zone.
    brake_gate = torch.clamp(
        1.0 - dist / parameter_dict["brake_radius"], 0.0, 1.0
    )
    stop_reward = (
        parameter_dict["stop_reward_magnitude"]
        * brake_gate
        * torch.exp(-parameter_dict["stop_reward_exponent"] * speed * speed)
    )

    # Unlike the one-time arrival bonus, this is paid every step only while the
    # vehicle is both tightly positioned and nearly stationary.
    dwelling = torch.logical_and(
        dist < parameter_dict["dwell_distance"],
        speed < parameter_dict["dwell_speed"],
    )
    dwell_reward = parameter_dict["dwell_reward"] * dwelling.to(dist.dtype)

    # CRL uses small L2 costs over the complete raw action rather than large
    # channel-wise exponentials. The raw deltas are the policy's control effort;
    # absolute yaw setpoint must not be penalised because that biases world yaw 0.
    action_penalty = -parameter_dict["action_penalty_multiplier"] * torch.norm(
        raw_action, dim=1
    )
    action_diff_penalty = -parameter_dict["action_diff_penalty_multiplier"] * torch.norm(
        raw_action - prev_raw_action, dim=1
    )

    # Unlike a fixed safe-height floor, the task has a 3-D goal. Penalising the
    # vertical goal error adapts the CRL height term to goals at any altitude.
    vertical_error_penalty = -parameter_dict["vertical_error_penalty_multiplier"] * (
        pos_error[:, 2].abs()
    )

    # CRL-style inverse-distance avoidance, using the nearest forward depth ray.
    # It stays deliberately small; collision is the dominant obstacle event.
    proximity_penalty = -parameter_dict["proximity_penalty_multiplier"] / (
        min_obstacle_dist.clamp_min(0.0) + parameter_dict["proximity_softener"]
    )

    goal_reward = parameter_dict["goal_reached_reward"] * reached_goal
    collision_penalty = parameter_dict["collision_penalty"] * crashes

    reward = (
        progress_reward
        + yaw_alignment_reward
        + velocity_tracking_reward
        + stop_reward
        + dwell_reward
        + goal_reward
        + action_penalty
        + action_diff_penalty
        + vertical_error_penalty
        + proximity_penalty
        + collision_penalty
    )
    return reward, crashes


# --- PREVIOUS AERIAL-GYM-STYLE REWARD (disabled; retained for reversion) -----
# c = curriculum_progress_fraction
# reward = (1 + 2*c) * (
#     5*exp(-dist**2/3.5) + 5*exp(-2*dist**2)
#     + where(delta_dist > 0, 10*delta_dist, 20*delta_dist)
#     + (20-dist)/20 + 2*exp(-3*abs(yaw_error))
# )
# reward += per-channel exponential raw-action-difference penalties
# reward += c * per-channel exponential absolute-action penalties
# reward += 4*(exp(-2*clamp(speed-1, min=0)**2)-1)
# reward += -4*exp(-min_obstacle_dist**2)
# reward = where(crashes, -20, reward)
#
# Previous parameter dictionary:
# "pos_reward_magnitude": 5.0
# "pos_reward_exponent": 1.0 / 3.5
# "very_close_to_goal_reward_magnitude": 5.0
# "very_close_to_goal_reward_exponent": 2.0
# "getting_closer_reward_multiplier": 10.0
# "thrust_action_diff_penalty_magnitude": 0.8
# "thrust_action_diff_penalty_exponent": 5.0
# "pitch_action_diff_penalty_magnitude": 0.8
# "pitch_action_diff_penalty_exponent": 3.333
# "yaw_action_diff_penalty_magnitude": 0.8
# "yaw_action_diff_penalty_exponent": 3.33
# "thrust_absolute_action_penalty_magnitude": 1.5
# "thrust_absolute_action_penalty_exponent": 1.0
# "pitch_absolute_action_penalty_magnitude": 0.1
# "pitch_absolute_action_penalty_exponent": 0.3
# "yaw_absolute_action_penalty_magnitude": 1.5
# "yaw_absolute_action_penalty_exponent": 2.0
# "yaw_alignment_reward_magnitude": 2.0
# "yaw_alignment_reward_exponent": 3.0
# "speed_limit": 1.0
# "speed_penalty_magnitude": 4.0
# "speed_penalty_exponent": 2.0
# "proximity_penalty_magnitude": 4.0
# "proximity_penalty_exponent": 1.0
# "collision_penalty": -20.0
