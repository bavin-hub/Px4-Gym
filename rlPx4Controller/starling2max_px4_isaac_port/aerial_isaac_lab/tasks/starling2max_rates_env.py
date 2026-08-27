"""Starling 2 Max fixed-goal CTBR task using the PX4 body-rate loop."""

from __future__ import annotations

import math

import torch

from isaaclab.utils import configclass

from aerial_isaac_lab.core.math import matrix_from_quat, quat_apply_inverse, quat_from_euler_xyz

from .starling2max_attitude_delta_env import (
    Starling2MaxAttitudeDeltaEnv,
    Starling2MaxAttitudeDeltaEnvCfg,
)


@configclass
class Starling2MaxRatesEnvCfg(Starling2MaxAttitudeDeltaEnvCfg):
    """SimpleFlight-Hover-style CTBR task on the Starling 2 Max plant."""

    # SimpleFlight Hover runs for 500 policy steps at 100 Hz.
    episode_length_s = 5.0

    # Match SimpleFlight's policy transform: each body-rate channel spans
    # +/-180 deg/s.  target_clip = 1.0 in crazyflie.yaml, so its scaling of
    # `target_rate * 180.0 * target_clip` deg/s is exactly +/-pi rad/s.
    body_rate_max = (math.pi, math.pi, math.pi)

    # --- Collective mapping ---------------------------------------------------
    # SimpleFlight maps its thrust channel as clamp((a + 1) / 2, 0.0, 0.9),
    # where the clamped quantity is the fraction of MAXIMUM TOTAL THRUST.  That
    # literal formula is only well-centred because the Crazyflie hovers at
    # 0.6328 of its maximum (T/W = 1.6): hover lands at a = +0.2656 (the value
    # SimpleFlight hardcodes as its hover command) and the reachable band is
    # [0, 1.42] x hover, saturating for a > +0.8.
    #
    # The Starling 2 Max hovers at 0.13 (T/W = 7.7), so copying the [0.0, 0.9]
    # numbers across would put hover at a = -0.74 and make centre stick a 2.8 g
    # climb -- with sigma = 1 at init, essentially the whole action distribution
    # commands a hard ascent.  Clamping tighter does NOT fix this: a clamp
    # saturates, it cannot recentre a mapping.  The band is therefore RESCALED
    # to preserve SimpleFlight's *relative* authority instead of its absolute
    # numbers: the (a + 1) / 2 SLOPE is scaled by the ratio of the two hover
    # points and the ceiling is scaled with it, so the whole curve is the
    # Crazyflie's expressed in units of this airframe's hover.  Hover returns to
    # a = +0.2656, the +/-42% collective margin is restored, and a > +0.8 still
    # saturates exactly as it does upstream.
    #
    # Resolved in __post_init__; the values below are those resolved defaults.
    simpleflight_hover_fraction = 0.6328
    simpleflight_thrust_ratio_min = 0.0
    simpleflight_thrust_ratio_max = 0.9
    px4_hover_thrust = -0.13
    # Multiplies (a + 1) / 2.  1.0 would be SimpleFlight's literal formula.
    collective_thrust_scale = 0.2054361567635904
    collective_thrust_min = 0.0
    collective_thrust_max = 0.18489254108723135
    px4_min_thrust = -0.18489254108723135
    px4_max_thrust = -0.0

    # Port the Hover reset distribution relative to its target [0, 0, 1].
    # Translating by the target puts the fixed target at this box's centre.
    reset_position_offset_min = (-1.0, -1.0, -0.95)
    reset_position_offset_max = (1.0, 1.0, 1.0)
    reset_rpy_min = (-0.2 * math.pi, -0.2 * math.pi, 0.0)
    reset_rpy_max = (0.2 * math.pi, 0.2 * math.pi, 0.5 * math.pi)
    reset_linear_velocity = 0.0
    reset_angular_velocity = 0.0

    # SimpleFlight Hover reward constants. Its checked-in configuration disables
    # action smoothness while retaining the term in the reward definition.
    reward_distance_scale = 10.0
    reward_action_smoothness_weight = 0.0
    position_bonus_radius = 0.02
    heading_bonus_error = 0.02
    position_bonus = 10.0
    heading_bonus = 10.0

    # --- Rate penalties (NOT from SimpleFlight Hover) -------------------------
    # SimpleFlight Hover has no rate term at all, so nothing there costs anything
    # for spinning.  On a CTBR policy that commands the rate loop directly, that
    # leaves the single largest sim2real gap: bang-bang rate commands are free in
    # sim and untrackable on a real airframe.
    #
    # Both are deliberately SMALL relative to the -10 * pos_error position term.
    # In transit the position gradient dominates, so a correction that closes
    # 0.1 m (+1.0) easily outbids the penalty and the policy is not discouraged
    # from manoeuvring.  At the goal pos_error ~ 0 and the position signal goes
    # quiet, so these become the dominant remaining term and damp residual spin
    # -- which is exactly where damping is wanted and nowhere else.
    #
    # body_rate_weight matches the ang_rate_weight = 0.05 this repo already uses
    # in the attitude-delta task.  rate_command_weight is lower because the
    # commanded and measured rates are strongly correlated once the 1 kHz loop
    # tracks, so charging both at full weight would double-bill the same
    # behaviour; it exists to price the COMMAND specifically, which is what PX4
    # has to track and clip on the vehicle.
    body_rate_weight = 0.05
    rate_command_weight = 0.02

    def __post_init__(self) -> None:
        super().__post_init__()
        # Scale SimpleFlight's collective band by the ratio of the two airframes'
        # hover points, so this task keeps its authority in units of hover.
        scale = abs(self.px4_hover_thrust) / self.simpleflight_hover_fraction
        self.collective_thrust_scale = scale
        self.collective_thrust_min = self.simpleflight_thrust_ratio_min * scale
        self.collective_thrust_max = self.simpleflight_thrust_ratio_max * scale
        # The controller clamps against these; keep them the same band.  Note
        # the deliberate sign/order swap the parent __init__ expects:
        # min_thrust=abs(px4_max_thrust), max_thrust=abs(px4_min_thrust).
        self.px4_min_thrust = -self.collective_thrust_max
        self.px4_max_thrust = -self.collective_thrust_min


class Starling2MaxRatesEnv(Starling2MaxAttitudeDeltaEnv):
    """Policy emits ``[roll_rate, pitch_rate, yaw_rate, collective_thrust]``.

    The 98-D policy observation is inherited unchanged in shape from the
    attitude-delta task. Observation channels 13:17, and the corresponding
    history channels, now contain the applied CTBR setpoint in physical units.
    """

    cfg: Starling2MaxRatesEnvCfg

    def __init__(self, cfg: Starling2MaxRatesEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._previous_raw_actions = torch.zeros_like(self._raw_actions)
        self._body_rate_scale = torch.tensor(self.cfg.body_rate_max, device=self.device)
        self._collective_scale = float(self.cfg.collective_thrust_scale)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._previous_actions[:] = self._actions
        self._previous_raw_actions[:] = self._raw_actions
        self._previous_distance[:] = self._position_error_w().norm(dim=-1)

        # SimpleFlight's PIDRateController transform squashes the raw policy
        # output with tanh; it does not clip. The distinction matters: tanh
        # compresses the interior and never saturates, so the Gaussian mean is
        # free to sit outside [-1, 1] while the applied CTBR stays graded,
        # whereas a clamp puts probability atoms on the limits and makes the
        # channels bang-bang. RL-Games' two clipping stages are disabled for
        # this task (config.clip_actions: false and wrapper clip_actions: inf)
        # so the unsquashed sample reaches this line, as it does in SimpleFlight.
        squashed_actions = torch.tanh(actions)
        squashed_actions = self._randomizer.delay_actions(squashed_actions)
        self._raw_actions[:] = squashed_actions

        # Clamp to the airframe's MC_*RATE_MAX (130/130/150 deg/s) rather than
        # letting the +/-180 deg/s scaling stand: PX4 clips there on the vehicle,
        # so anything above it is a region the policy could learn to use in sim
        # and never reach in flight.  Clamped HERE, not only inside the
        # controller, so _actions -- and therefore observation channels 13:17 --
        # report the setpoint that was actually applied rather than the request.
        # NOTE: this leaves a dead zone, |a| > ~0.72 all mapping to the limit.
        # Rescaling _body_rate_scale to the limits would avoid that, but would
        # change SimpleFlight's action scale and would no longer match what PX4
        # physically does to an out-of-range command.
        self._actions[:, :3] = torch.clamp(
            squashed_actions[:, :3] * self._body_rate_scale,
            -self._controller.rate_limit,
            self._controller.rate_limit,
        )
        # SimpleFlight's clamp((a + 1) / 2, min, max) with the slope scaled to
        # this airframe's hover point (see the cfg note).  The clamp is load
        # bearing, not a guard: as upstream, it is what saturates the top of the
        # stick above a = +0.8.
        self._actions[:, 3] = torch.clamp(
            self._collective_scale * (squashed_actions[:, 3] + 1.0) * 0.5,
            self.cfg.collective_thrust_min,
            self.cfg.collective_thrust_max,
        )
        self._apply_random_pushes()

    def _apply_action(self) -> None:
        root_quat_wxyz = self._robot.data.root_quat_w
        root_lin_vel_w = self._robot.data.root_lin_vel_w
        root_ang_vel_b = quat_apply_inverse(root_quat_wxyz, self._robot.data.root_ang_vel_w)
        control = self._controller.compute_rate(
            self._actions,
            root_ang_vel_b,
            self.physics_dt,
        )

        body_lin_vel = quat_apply_inverse(root_quat_wxyz, root_lin_vel_w)
        self._wind_velocity_w = self._randomizer.step_wind()
        wind_velocity_b = quat_apply_inverse(root_quat_wxyz, self._wind_velocity_w)
        relative_air_velocity_b = body_lin_vel - wind_velocity_b

        actuator_sp = self._allocator.allocate_normalized_control(control)
        motor_values = self._allocator.function_motors(actuator_sp)
        motor_speed_command = self._allocator.mixing_output(motor_values)
        motor_forces_b, motor_torques_b = self._allocator.gazebo_motor_model(
            motor_speed_command,
            body_velocity_b=body_lin_vel,
            wind_velocity_b=wind_velocity_b,
        )
        self._motor_forces_b.zero_()
        self._motor_torques_b.zero_()
        self._motor_forces_b[:, self._motor_body_ids] = motor_forces_b
        self._motor_torques_b[:, self._motor_body_ids] = motor_torques_b

        linear_drag = self._linear_drag
        quadratic_drag = self._quadratic_drag
        angular_drag = self._angular_drag
        angular_quadratic_drag = self._angular_quadratic_drag
        if self._randomizer.enabled and self.cfg.domain_randomization.randomize_drag:
            linear_drag = linear_drag + self._randomizer.linear_drag
            quadratic_drag = quadratic_drag + self._randomizer.quadratic_drag
            angular_drag = angular_drag + self._randomizer.angular_drag
            angular_quadratic_drag = (
                angular_quadratic_drag + self._randomizer.angular_quadratic_drag
            )

        drag_force = (
            -linear_drag * relative_air_velocity_b
            - quadratic_drag
            * relative_air_velocity_b.norm(dim=-1, keepdim=True)
            * relative_air_velocity_b
        )
        drag_force = drag_force + self._randomizer.wind_force_b(relative_air_velocity_b)
        drag_torque = (
            -angular_drag * root_ang_vel_b
            - angular_quadratic_drag * root_ang_vel_b.abs() * root_ang_vel_b
        )
        self._motor_forces_b[:, self._root_body_id] += drag_force
        self._motor_torques_b[:, self._root_body_id] += drag_torque
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=torch.arange(self._robot.num_bodies, device=self.device),
            forces=self._motor_forces_b,
            torques=self._motor_torques_b,
        )

    def _get_rewards(self) -> torch.Tensor:
        """Port of SimpleFlight Hover's active reward formula."""
        pos_error = self._position_error_w().norm(dim=-1)
        rotation = matrix_from_quat(self._robot.data.root_quat_w)
        heading = rotation[:, :, 0]
        target_heading = torch.zeros_like(heading)
        target_heading[:, 0] = 1.0
        heading_error = (target_heading - heading).norm(dim=-1)

        reward_pos = -pos_error * self.cfg.reward_distance_scale
        at_position = pos_error <= self.cfg.position_bonus_radius
        reward_pos_bonus = at_position.to(pos_error.dtype) * self.cfg.position_bonus
        reward_heading = -heading_error * at_position.to(pos_error.dtype)
        reward_heading_bonus = (
            torch.logical_and(at_position, heading_error <= self.cfg.heading_bonus_error).to(
                pos_error.dtype
            )
            * self.cfg.heading_bonus
        )
        reward_up = ((rotation[:, 2, 2] + 1.0) * 0.5).square()
        # (1) measured spin and (2) commanded spin.  _actions[:, :3] is the
        # post-clamp setpoint, so this prices what PX4 is actually asked to fly.
        body_ang_vel = quat_apply_inverse(
            self._robot.data.root_quat_w, self._robot.data.root_ang_vel_w
        )
        rate_penalty = self.cfg.body_rate_weight * body_ang_vel.square().sum(dim=-1)
        rate_command_penalty = (
            self.cfg.rate_command_weight * self._actions[:, :3].square().sum(dim=-1)
        )
        action_error = (self._raw_actions - self._previous_raw_actions).norm(dim=-1)
        reward_action_smoothness = (
            self.cfg.reward_action_smoothness_weight * torch.exp(-action_error)
        )
        reward = (
            reward_pos
            + reward_pos_bonus
            + reward_heading
            + reward_heading_bonus
            + reward_up
            + reward_action_smoothness
            - rate_penalty
            - rate_command_penalty
        )
        return torch.where(
            pos_error > self.cfg.crash_distance,
            torch.full_like(reward, -100.0),
            reward,
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Match the delta-attitude task's out-of-range reset while retaining the
        # SimpleFlight Hover timeout.
        timed_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = self._position_error_w().norm(dim=-1) > self.cfg.crash_distance
        return terminated, timed_out

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        super()._reset_idx(env_ids)
        # DirectRLEnv may reset while the parent constructor is still running.
        if not hasattr(self, "_previous_raw_actions"):
            return
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        count = len(env_ids)
        position_min = torch.tensor(self.cfg.reset_position_offset_min, device=self.device)
        position_max = torch.tensor(self.cfg.reset_position_offset_max, device=self.device)
        position_offset = position_min + torch.rand((count, 3), device=self.device) * (
            position_max - position_min
        )

        rpy_min = torch.tensor(self.cfg.reset_rpy_min, device=self.device)
        rpy_max = torch.tensor(self.cfg.reset_rpy_max, device=self.device)
        rpy = rpy_min + torch.rand((count, 3), device=self.device) * (rpy_max - rpy_min)
        pose_w = torch.cat(
            (
                self.scene.env_origins[env_ids] + position_offset,
                quat_from_euler_xyz(rpy[:, 0], rpy[:, 1], rpy[:, 2]),
            ),
            dim=-1,
        )
        velocity_w = torch.zeros((count, 6), device=self.device)
        self._robot.write_root_pose_to_sim(pose_w, env_ids)
        self._robot.write_root_velocity_to_sim(velocity_w, env_ids)

        # Inverse of the collective map in _pre_physics_step, so the episode
        # starts from the action that actually holds hover.  Resolves to +0.2656
        # -- the same constant SimpleFlight hardcodes for its Crazyflie.
        hover_collective = abs(self.cfg.px4_hover_thrust)
        raw_hover = 2.0 * hover_collective / self._collective_scale - 1.0
        self._actions[env_ids, :3] = 0.0
        self._actions[env_ids, 3] = hover_collective
        self._previous_actions[env_ids] = self._actions[env_ids]
        self._raw_actions[env_ids, :3] = 0.0
        self._raw_actions[env_ids, 3] = raw_hover
        self._previous_raw_actions[env_ids] = self._raw_actions[env_ids]
        self._previous_distance[env_ids] = position_offset.norm(dim=-1)
