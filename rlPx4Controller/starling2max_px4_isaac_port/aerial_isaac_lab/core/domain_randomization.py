"""Opt-in sim2real domain randomization, kept separate from the task definitions.

Everything here is inert unless ``DomainRandomizationCfg.enabled`` is True, so a
task that embeds a randomizer still reproduces its nominal dynamics bit-for-bit
when the flag is off.  Each group additionally has its own switch, letting a
sweep isolate which mismatch actually explains a deployment gap.

Randomized quantities fall into three classes, which matter for how they are
applied:

* **Plant** (mass, CoM, inertia, motor constants, drag, wind, pushes) -- these
  change the physics.  The controller keeps its nominal belief, so the mismatch
  between what the controller assumes and what the vehicle does is exactly the
  error the policy must learn to reject.
* **Controller** (gains, hover throttle) -- these change the inner loop the
  policy commands through.
* **Signal path** (sensor noise/bias, estimation delay, action latency) -- these
  degrade what the policy sees and when its command lands.
"""

from __future__ import annotations

import torch

from isaaclab.utils import configclass


@configclass
class DomainRandomizationCfg:
    """Ranges for every randomized quantity.  Master switch is ``enabled``."""

    enabled = True

    # --- Inertial properties -------------------------------------------------
    randomize_mass = True
    # Multiplicative on the URDF mass of every body.
    # mass_scale_range = (0.85, 1.15)
    mass_scale_range = (0.90, 1.10)
    randomize_com = False
    # Additive offset (m) on the root body CoM, per axis.
    com_offset_range = (-0.01, 0.01)
    randomize_inertia = True
    # Multiplicative on the diagonal inertia, applied on top of the mass ratio.
    # inertia_scale_range = (0.85, 1.15)
    inertia_scale_range = (0.90, 1.10)

    # --- Propulsion ----------------------------------------------------------
    randomize_thrust_constant = False
    # Multiplicative on motor_constant, i.e. thrust per rotor-speed squared.
    thrust_constant_scale_range = (0.90, 1.10)
    randomize_hover_throttle = False
    # Multiplicative on |px4_hover_thrust| as the *controller* believes it.  The
    # single most likely cause of a constant altitude offset on deployment.
    # hover_throttle_scale_range = (0.923, 1.077)
    hover_throttle_scale_range = (0.962, 1.038)
    # hover_throttle_scale_range = (0.90, 1.10)
    randomize_motor_lag = True
    # Multiplicative on the rotor time constants (up and down, independently).
    motor_tau_scale_range = (0.70, 1.40)

    # --- Aerodynamics --------------------------------------------------------
    randomize_drag = False
    # Absolute coefficients; the nominal model ships zeros, so any nonzero draw
    # is already a mismatch the policy has to absorb.
    linear_drag_range = (0.0, 0.15)
    quadratic_drag_range = (0.0, 0.10)
    angular_drag_range = (0.0, 0.02)
    angular_quadratic_drag_range = (0.0, 0.02)

    # --- Disturbances --------------------------------------------------------
    # Off: unobservable and, at up to 3 m/s against a 1 m/s cruise, makes some
    # episodes unwinnable.  Re-enable with a narrowed wind_speed_range.
    randomize_wind = False
    # Steady wind speed (m/s), direction uniform over the sphere.
    wind_speed_range = (0.0, 3.0)
    # Ornstein-Uhlenbeck gust on top of the steady component.
    gust_std = 1.0
    gust_correlation_time = 1.0
    # Quadratic wind-force coefficient (N per (m/s)^2) per body axis.
    wind_force_coeff_range = (0.02, 0.10)

    # Off: an impulse is unattributable to any action, so it is pure return
    # variance.  2-3 hits land in an 8 s episode at push_interval_s = 3.
    randomize_pushes = False
    # Expected seconds between impulses; each push is a velocity step.
    push_interval_s = 3.0
    push_linear_velocity_range = (0.0, 1.0)
    push_angular_velocity_range = (0.0, 1.0)

    # --- Controller ----------------------------------------------------------
    randomize_controller_gains = False
    # Multiplicative on the Lee attitude gains, sampled per axis.
    k_rot_scale_range = (0.75, 1.30)
    k_angvel_scale_range = (0.75, 1.30)

    # --- Signal path ---------------------------------------------------------
    # All three delays off: combined they reached 100 ms, and the per-episode
    # draw is absent from the observation, so the policy can only learn the mean
    # of a distribution it cannot identify.  Re-introduce one at a time, and
    # narrow the ranges below to (0, 1) before doing so.
    randomize_action_latency = False
    # Steps of delay between the policy emitting an action and it reaching the
    # controller.  dt = 0.01 s here, so 4 steps is 40 ms.
    action_latency_steps_range = (0, 4)
    randomize_controller_delay = False
    # Extra steps applied on top of action latency, modelling the inner loop.
    controller_delay_steps_range = (0, 2)
    randomize_estimation_delay = False
    # Steps of delay on the observation, modelling EKF/telemetry lag.
    estimation_delay_steps_range = (0, 4)

    randomize_sensor_bias = False
    # Per-episode constant offsets held for the whole episode.  A position bias
    # is indistinguishable from the goal having moved, so no amount of history
    # recovers it -- it sets a hard floor on achievable waypoint accuracy and is
    # kept tight for that reason.
    position_bias_range = (-0.02, 0.02)
    linear_velocity_bias_range = (-0.05, 0.05)
    angular_velocity_bias_range = (-0.02, 0.02)

    # --- Observation slices --------------------------------------------------
    # Index ranges into the 18-dim observation that sensor bias applies to.
    position_obs_slice = (0, 3)
    linear_velocity_obs_slice = (7, 10)
    angular_velocity_obs_slice = (10, 13)


def _uniform(shape, low: float, high: float, device) -> torch.Tensor:
    return torch.empty(shape, device=device).uniform_(low, high)


class DomainRandomizer:
    """Per-environment randomized parameters, resampled on episode reset."""

    def __init__(self, cfg: DomainRandomizationCfg, num_envs: int, dt: float, device):
        self.cfg = cfg
        self.num_envs = num_envs
        self.dt = float(dt)
        self.device = device
        self.enabled = bool(cfg.enabled)
        if not self.enabled:
            return

        # Controller-facing scales; 1.0 means "as nominal".
        self.hover_throttle_scale = torch.ones(num_envs, device=device)
        self.k_rot_scale = torch.ones((num_envs, 3), device=device)
        self.k_angvel_scale = torch.ones((num_envs, 3), device=device)

        # Plant parameters.
        self.linear_drag = torch.zeros((num_envs, 3), device=device)
        self.quadratic_drag = torch.zeros((num_envs, 3), device=device)
        self.angular_drag = torch.zeros((num_envs, 3), device=device)
        self.angular_quadratic_drag = torch.zeros((num_envs, 3), device=device)
        self.thrust_constant_scale = torch.ones((num_envs, 1), device=device)
        self.tau_up_scale = torch.ones((num_envs, 1), device=device)
        self.tau_down_scale = torch.ones((num_envs, 1), device=device)

        # Wind: steady component plus an OU gust, both in world frame.
        self.wind_mean_w = torch.zeros((num_envs, 3), device=device)
        self.wind_gust_w = torch.zeros((num_envs, 3), device=device)
        self.wind_force_coeff = torch.zeros((num_envs, 3), device=device)

        # Pushes are Bernoulli per step with mean interval push_interval_s.
        self.push_probability = self.dt / max(cfg.push_interval_s, 1.0e-6)

        # Signal path.
        self.position_bias = torch.zeros((num_envs, 3), device=device)
        self.linear_velocity_bias = torch.zeros((num_envs, 3), device=device)
        self.angular_velocity_bias = torch.zeros((num_envs, 3), device=device)

        self._max_action_delay = int(
            cfg.action_latency_steps_range[1] + cfg.controller_delay_steps_range[1]
        )
        self._max_obs_delay = int(cfg.estimation_delay_steps_range[1])
        self.action_delay_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.observation_delay_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._action_history: torch.Tensor | None = None
        self._observation_history: torch.Tensor | None = None

        self._env_indices = torch.arange(num_envs, device=device)

    # -- sampling -------------------------------------------------------------

    def reset(self, env_ids: torch.Tensor) -> None:
        """Resample every per-episode parameter for ``env_ids``."""
        if not self.enabled:
            return
        cfg = self.cfg
        count = len(env_ids)
        if count == 0:
            return

        if cfg.randomize_hover_throttle:
            self.hover_throttle_scale[env_ids] = _uniform(
                count, *cfg.hover_throttle_scale_range, device=self.device
            )
        if cfg.randomize_controller_gains:
            self.k_rot_scale[env_ids] = _uniform(
                (count, 3), *cfg.k_rot_scale_range, device=self.device
            )
            self.k_angvel_scale[env_ids] = _uniform(
                (count, 3), *cfg.k_angvel_scale_range, device=self.device
            )
        if cfg.randomize_drag:
            self.linear_drag[env_ids] = _uniform(
                (count, 3), *cfg.linear_drag_range, device=self.device
            )
            self.quadratic_drag[env_ids] = _uniform(
                (count, 3), *cfg.quadratic_drag_range, device=self.device
            )
            self.angular_drag[env_ids] = _uniform(
                (count, 3), *cfg.angular_drag_range, device=self.device
            )
            self.angular_quadratic_drag[env_ids] = _uniform(
                (count, 3), *cfg.angular_quadratic_drag_range, device=self.device
            )
        if cfg.randomize_thrust_constant:
            self.thrust_constant_scale[env_ids] = _uniform(
                (count, 1), *cfg.thrust_constant_scale_range, device=self.device
            )
        if cfg.randomize_motor_lag:
            self.tau_up_scale[env_ids] = _uniform(
                (count, 1), *cfg.motor_tau_scale_range, device=self.device
            )
            self.tau_down_scale[env_ids] = _uniform(
                (count, 1), *cfg.motor_tau_scale_range, device=self.device
            )
        if cfg.randomize_wind:
            direction = torch.randn((count, 3), device=self.device)
            direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
            speed = _uniform(count, *cfg.wind_speed_range, device=self.device)
            self.wind_mean_w[env_ids] = direction * speed.unsqueeze(-1)
            self.wind_gust_w[env_ids] = 0.0
            self.wind_force_coeff[env_ids] = _uniform(
                (count, 3), *cfg.wind_force_coeff_range, device=self.device
            )
        if cfg.randomize_sensor_bias:
            self.position_bias[env_ids] = _uniform(
                (count, 3), *cfg.position_bias_range, device=self.device
            )
            self.linear_velocity_bias[env_ids] = _uniform(
                (count, 3), *cfg.linear_velocity_bias_range, device=self.device
            )
            self.angular_velocity_bias[env_ids] = _uniform(
                (count, 3), *cfg.angular_velocity_bias_range, device=self.device
            )

        delay = torch.zeros(count, dtype=torch.long, device=self.device)
        if cfg.randomize_action_latency:
            low, high = cfg.action_latency_steps_range
            delay += torch.randint(int(low), int(high) + 1, (count,), device=self.device)
        if cfg.randomize_controller_delay:
            low, high = cfg.controller_delay_steps_range
            delay += torch.randint(int(low), int(high) + 1, (count,), device=self.device)
        self.action_delay_steps[env_ids] = delay.clamp(0, max(self._max_action_delay, 0))

        if cfg.randomize_estimation_delay:
            low, high = cfg.estimation_delay_steps_range
            self.observation_delay_steps[env_ids] = torch.randint(
                int(low), int(high) + 1, (count,), device=self.device
            ).clamp(0, max(self._max_obs_delay, 0))

        # Clear history so a fresh episode never reads the previous one's tail.
        if self._action_history is not None:
            self._action_history[env_ids] = 0.0
        if self._observation_history is not None:
            self._observation_history[env_ids] = 0.0

    # -- inertial properties (applied through PhysX, CPU-side) ----------------

    def randomize_body_properties(self, robot, env_ids: torch.Tensor, root_body_id: int) -> None:
        """Scale PhysX mass/inertia/CoM.  The controller keeps nominal values."""
        if not self.enabled:
            return
        cfg = self.cfg
        if not (cfg.randomize_mass or cfg.randomize_inertia or cfg.randomize_com):
            return

        cpu_ids = env_ids.detach().cpu()
        count = len(cpu_ids)
        if count == 0:
            return

        mass_ratio = torch.ones((count, 1), device="cpu")
        if cfg.randomize_mass:
            mass_ratio = _uniform((count, 1), *cfg.mass_scale_range, device="cpu")
            masses = robot.root_physx_view.get_masses()
            masses[cpu_ids] = robot.data.default_mass[cpu_ids].cpu() * mass_ratio
            robot.root_physx_view.set_masses(masses, cpu_ids)

        if cfg.randomize_mass or cfg.randomize_inertia:
            # Inertia scales with mass; the extra factor models a mass
            # distribution that differs from the CAD model, not just its total.
            inertia_scale = mass_ratio
            if cfg.randomize_inertia:
                inertia_scale = inertia_scale * _uniform(
                    (count, 1), *cfg.inertia_scale_range, device="cpu"
                )
            inertias = robot.root_physx_view.get_inertias()
            default_inertia = robot.data.default_inertia[cpu_ids].cpu()
            inertias[cpu_ids] = default_inertia * inertia_scale.unsqueeze(-1)
            robot.root_physx_view.set_inertias(inertias, cpu_ids)

        if cfg.randomize_com:
            coms = robot.root_physx_view.get_coms().clone()
            offset = _uniform((count, 3), *cfg.com_offset_range, device="cpu")
            coms[cpu_ids, root_body_id, :3] += offset
            robot.root_physx_view.set_coms(coms, cpu_ids)

    # -- per-step hooks -------------------------------------------------------

    def delay_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Return the action that actually reaches the controller this step."""
        if not self.enabled or self._max_action_delay <= 0:
            return actions
        if self._action_history is None:
            self._action_history = torch.zeros(
                (self.num_envs, self._max_action_delay + 1, actions.shape[-1]),
                device=self.device,
            )
        self._action_history = torch.roll(self._action_history, shifts=1, dims=1)
        self._action_history[:, 0] = actions
        return self._action_history[self._env_indices, self.action_delay_steps]

    def delay_observations(self, obs: torch.Tensor) -> torch.Tensor:
        """Return a stale observation, modelling state-estimation lag."""
        if not self.enabled or self._max_obs_delay <= 0:
            return obs
        if self._observation_history is None:
            self._observation_history = torch.zeros(
                (self.num_envs, self._max_obs_delay + 1, obs.shape[-1]), device=self.device
            )
        self._observation_history = torch.roll(self._observation_history, shifts=1, dims=1)
        self._observation_history[:, 0] = obs
        return self._observation_history[self._env_indices, self.observation_delay_steps]

    def apply_sensor_bias(self, obs: torch.Tensor) -> torch.Tensor:
        """Add per-episode constant offsets to the biased observation slices."""
        if not self.enabled or not self.cfg.randomize_sensor_bias:
            return obs
        obs = obs.clone()
        for bias, (start, stop) in (
            (self.position_bias, self.cfg.position_obs_slice),
            (self.linear_velocity_bias, self.cfg.linear_velocity_obs_slice),
            (self.angular_velocity_bias, self.cfg.angular_velocity_obs_slice),
        ):
            obs[:, start:stop] += bias
        return obs

    def step_wind(self) -> torch.Tensor:
        """Advance the OU gust and return the world-frame wind velocity."""
        if not self.enabled or not self.cfg.randomize_wind:
            return torch.zeros((self.num_envs, 3), device=self.device)
        theta = self.dt / max(self.cfg.gust_correlation_time, 1.0e-6)
        noise = torch.randn_like(self.wind_gust_w) * self.cfg.gust_std * (2.0 * theta) ** 0.5
        self.wind_gust_w += -theta * self.wind_gust_w + noise
        return self.wind_mean_w + self.wind_gust_w

    def wind_force_b(self, relative_air_velocity_b: torch.Tensor) -> torch.Tensor:
        """Quadratic force from the air moving past the airframe, in body frame."""
        if not self.enabled or not self.cfg.randomize_wind:
            return torch.zeros_like(relative_air_velocity_b)
        return (
            -self.wind_force_coeff
            * relative_air_velocity_b
            * relative_air_velocity_b.abs()
        )

    def sample_pushes(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Return ``(env_ids, d_linear_velocity, d_angular_velocity)`` or None."""
        if not self.enabled or not self.cfg.randomize_pushes:
            return None
        hit = torch.rand(self.num_envs, device=self.device) < self.push_probability
        env_ids = hit.nonzero(as_tuple=False).squeeze(-1)
        if env_ids.numel() == 0:
            return None
        count = len(env_ids)
        d_lin = _uniform((count, 3), *self.cfg.push_linear_velocity_range, device=self.device)
        d_ang = _uniform((count, 3), *self.cfg.push_angular_velocity_range, device=self.device)
        sign_lin = torch.sign(torch.randn_like(d_lin))
        sign_ang = torch.sign(torch.randn_like(d_ang))
        return env_ids, d_lin * sign_lin, d_ang * sign_ang
