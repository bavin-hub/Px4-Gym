"""Starling 2 Max attitude-delta task ported from the Aerial Gym attitude repo."""

from __future__ import annotations

import math

import torch

from isaaclab.utils import configclass

from aerial_isaac_lab.core import PX4AttitudeRateController
from aerial_isaac_lab.core.domain_randomization import (
    DomainRandomizationCfg,
    DomainRandomizer,
)
from aerial_isaac_lab.core.math import (
    euler_xyz_from_quat,
    quat_apply_inverse,
    quat_from_euler_xyz,
    wrap_to_pi,
)

from .starling2max_velocity_env import Starling2MaxVelocityEnv, Starling2MaxVelocityEnvCfg


@configclass
class Starling2MaxAttitudeDeltaEnvCfg(Starling2MaxVelocityEnvCfg):
    """Aerial Gym ``position_setpoint_task_attitude_delta_starling2max`` constants."""

    # Waypoint mode.  False pins the goal to the box centre (the fixed target
    # this task was tuned on); True spreads it across the box and draws a new one
    # on each arrival.  __post_init__ resolves the target_* fields below from it.
    random_waypoints = False

    target_sampling_attempts = 6
    target_reached_distance = 0.20

    # Set in __post_init__ from random_waypoints; the values here are the fixed
    # (centre) defaults and are overwritten when random_waypoints is True.
    target_min_ratio = (0.5, 0.5, 0.5)
    target_max_ratio = (0.5, 0.5, 0.5)
    minimum_target_distance = 0.0
    resample_target_on_reach = False

    px4_hover_thrust = -0.13
    px4_min_thrust = -0.16
    px4_max_thrust = -0.10

    # Multi-rate inner loop mirroring PX4's separate task rates: physics + rate
    # loop at 1 kHz (sim.dt = 0.001, set in __post_init__), attitude loop at
    # 250 Hz, policy at 100 Hz (decimation = 10, so step_dt = 0.001 * 10 = 0.01).
    decimation = 10
    attitude_control_hz = 250.0
    # Holds the PX4 thrust command within [-0.16, -0.10] about hover.
    delta_thrust_scale = 0.03
    # Matched to the collective authority above.  At +/-0.01 about a 0.13 hover
    # the vehicle has +/-7.7% thrust margin, while tilting by theta costs
    # 1 - cos(theta) of vertical thrust: 7.7% is used up by 22.6 degrees.  The
    # inherited 30 degree spawn is therefore unrecoverable by construction, so
    # the attitude-delta task spawns closer to level.
    reset_roll_pitch = math.radians(2.0)
    # Per-step slew of the roll/pitch setpoint (50 deg/s at 100 Hz).
    max_delta_roll_pitch = math.radians(0.5)
    # Absolute cap on the commanded roll/pitch setpoints.  The deltas accumulate
    # onto the measured attitude, so without these bounds a setpoint could walk
    # arbitrarily far.  Split per axis: pitch is the propulsive axis and carries
    # the cruise speed, so it gets the wider envelope, while roll only trims
    # lateral drift and stays gentle.
    #   +/-2 deg -> ~0.34 m/s^2, +/-4 deg -> ~0.69 m/s^2 of horizontal authority.
    # Both back at 2 deg; the split is kept so pitch can be widened on its own
    # when the cruise speed is raised again.
    # PREVIOUS: max_pitch = math.radians(4.0), paired with cruise_speed = 2.0.
    max_roll = math.radians(2.0)
    max_pitch = math.radians(2.0)
    # Yaw-rate AUTHORITY, not an accumulator: the setpoint is rebuilt from the
    # MEASURED yaw every step, so this is a standing lead that the attitude loop
    # turns into a rate command of roughly attitude_p * yaw_weight * lead
    # (8.0 * 0.35 * lead).  At 5 degrees that is a sustainable 14 deg/s, i.e.
    # 12.9 s for a 180 degree correction against an 8.01 s episode -- and spawn
    # yaw is uniform on +/-pi, so half of all episodes could not physically turn
    # to face the goal before timeout.  30 degrees gives ~84 deg/s, so the worst
    # case turns in ~2.2 s and still has ~6 s left to fly.  45 deg -> ~125 deg/s
    # if more margin is wanted; beyond ~60 deg the 150 deg/s rate limit binds.
    # Measured sustained yaw rates: 5 deg -> 14 deg/s (12.9 s per 180 deg turn,
    # unusable), 10 deg -> 28 deg/s (6.43 s), 20 deg -> 56 deg/s (3.22 s),
    # 15 deg -> 42 deg/s (4.29 s), 30 deg -> 84 deg/s (2.15 s).  The episode is
    # 8.01 s, so 30 deg leaves ~5.9 s to fly after a worst-case 180 turn.
    # max_delta_yaw = math.radians(5.0)   # original (revert to this)
    # max_delta_yaw = math.radians(8.0)   # previous
    # max_delta_yaw = math.radians(12.0)  # previous
    # max_delta_yaw = math.radians(15.0)  # previous
    # max_delta_yaw = math.radians(20.0)  # previous
    # max_delta_yaw = math.radians(25.0)  # previous
    # max_delta_yaw = math.radians(30.0)  # previous
    max_delta_yaw = math.radians(10.0)

    # Policy predicts 4 actions [d_thrust, d_roll, d_pitch, d_yaw]; roll is a
    # commanded channel again, so the drone can strafe laterally instead of
    # having to yaw onto the bearing before it can translate.
    # TO REVERT to the roll-locked variant: set action_space = 3, pin
    # _actions[:, 1] = 0.0 and shift the clipped_actions indices in
    # _pre_physics_step down by one (pitch 2 -> 1, yaw 3 -> 2).
    # action_space = 3  # roll-locked variant
    action_space = 4

    # Velocity setpoint: hold cruise_speed until brake_radius, then taper linearly.
    # brake_radius must exceed the stopping distance the pitch envelope can
    # actually produce: at 0.7 m/s against g*tan(2 deg) = 0.34 m/s^2 that is
    # 0.72 m, so the previous 0.4 asked for a stop the airframe cannot fly.
    # cruise_speed = 1.0 ; brake_radius = 0.4  # original (revert to this)
    cruise_speed = 0.7
    brake_radius = 1.5
    # Along-track (speed) and cross-track (sway) velocity-tracking sharpness.
    speed_tracking_sharpness = 5.0
    cross_track_vel_sharpness = 5.0
    # Sharpness of the full-3-vector velocity tracking used inside brake_radius.
    # 2.0 is the commented pursuit reward's value; it is deliberately broader
    # than speed_tracking_sharpness because it now has to hold three components
    # to zero rather than one.
    full_vector_sharpness = 2.0
    # Straight-line corridor half-width (m) about the spawn->goal segment.
    # path_corridor = 0.5  # original (revert to this)
    path_corridor = 0.3
    # Bounded straight-line corridor penalty weight, used in _get_rewards as
    # -w * (1 - exp(-(cross_track / path_corridor)^2)). It SATURATES at -w, so
    # unlike an unbounded quadratic it can never swamp the reward and collapse
    # training. Raise for a straighter path; 0.0 disables (pure #1 revert).
    # Lowered to 1.0 for the turn-and-go reward: a vehicle that is rewarded for
    # pointing at the goal already flies straight, and a heavy corridor term
    # fights the deliberate arc the heading/pitch channels produce.
    # Raised back to 2.0 now that the corridor is ALIGNMENT-gated rather than
    # range-gated: it no longer fights the turn (gate ~0 while turning), so it
    # can afford to do real work on the run-in, which is where the straight-line
    # path is actually wanted.
    # straight_line_weight = 4.0  # previous, paired with cruise_speed = 2.0
    # straight_line_weight = 1.0  # previous (revert to this)
    straight_line_weight = 2.0
    # UNUSED.  Quadratic pull back onto the spawn->goal line: unlike the bounded
    # corridor term, a plain -w * cross_track^2 keeps a restoring gradient at
    # every offset.  Never wired into _get_rewards (it was already dead before
    # the turn-and-go reward), and the arc that reward deliberately flies is not
    # something an unbounded quadratic should be fighting.  Left declared so the
    # option is discoverable.
    cross_track_pos_weight = 4.0
    # Width of the fine terminal position funnel (m).
    fine_goal_sigma = 0.15
    # Per-step distance-closed shaping; 1 m/s at dt=0.01 yields progress_weight * 0.01.
    progress_weight = 100.0
    # Steps held inside target_reached_distance before the dwell bonus saturates.
    dwell_steps = 50
    # Collective-channel smoothness weight. Kept low because the collective
    # barely needs damping at this authority.
    smoothness_thrust_weight = 0.6
    # Body-rate damping, split by axis.  Applied to all three axes at 0.05 it
    # cancelled the turn reward outright: yawing at omega rad/s earns
    # yaw_progress_weight * omega * dt = 0.06 * omega per step, while the penalty
    # charges 0.05 * omega^2 for the same rate.  Those cross at 1.2 rad/s
    # (69 deg/s) and the net optimum sits at 0.6 rad/s (34 deg/s), so the reward
    # had an internal speed limit on heading correction and went NEGATIVE above
    # it.  Roll/pitch rates are what actually need damping for the +/-2 degree
    # tilt envelope; yaw rate is the task, so it gets a token weight only.
    # At 0.005 the crossover moves to 12 rad/s, far above the 2.6 rad/s
    # controller rate limit, so it never binds during a turn but still damps
    # residual spin once the heading is settled.
    # PREVIOUS (revert to this): ang_rate_weight = 0.05 applied to all three
    # axes, i.e. restore `cfg.ang_rate_weight * (body_ang_vel * body_ang_vel)
    # .sum(dim=-1)` in _get_rewards and delete yaw_rate_weight.
    ang_rate_weight = 0.05
    # 0.005 put the reward-optimal yaw rate at 344 deg/s, far above anything the
    # plant delivers, so there was effectively no damping anywhere in the
    # reachable range and the heading oscillated.  0.02 puts the optimum at
    # 86 deg/s -- essentially exactly the 84 deg/s that max_delta_yaw = 30 deg
    # sustains -- so the fastest available turn is also the preferred one, with
    # quadratic damping on anything past it.  This costs nothing on a net turn:
    # yaw_progress telescopes (its episode total is |psi_0| - |psi_final|
    # whatever path is taken), while this penalty is path-dependent, so the pair
    # charges for WASTED yaw motion specifically and not for the turn itself.
    # yaw_rate_weight = 0.005  # previous (revert to this)
    # yaw_rate_weight = 0.02   # previous
    # yaw_rate_weight = 0.03   # previous
    # yaw_rate_weight = 0.05   # previous
    yaw_rate_weight = 0.035
    # Magnitude penalty on the collective channel only.  _raw_actions[:, 1:] are
    # DELTAS, so zero means "hold the current attitude"; penalising their
    # magnitude penalises turning and pitching, i.e. the task itself.  Only
    # channel 0 is absolute (0 = hover), so only it deserves an effort term.
    effort_weight = 0.05

    # --- Coordinated turn-and-go reward ---------------------------------------
    # Heading and pitch are rewarded on independent channels, and NEITHER term
    # carries a velocity or alignment factor that can zero it out.  Yawing pays
    # while the vehicle is still at rest, and pitching forward pays while the
    # heading is still wrong, so neither channel can be maximised by completing
    # the other one first: the optimum is a coordinated arc, not a pivot then a
    # dash.  See the commented pursuit reward in _get_rewards to revert.
    #
    # Turn-rate reward.  Linear in |yaw_error|, so unlike every exp()/cos()
    # heading term it still has a gradient at 180 degrees -- which is where half
    # of all episodes start, since spawn yaw is uniform on +/-pi.
    yaw_progress_weight = 6.0
    # Turn-DIRECTION reward.  yaw_progress above cannot express a preference
    # between turning left and right: it telescopes, so its episode total is
    # |psi_0| - |psi_final| whatever path is taken, and going the long way round
    # merely raises |psi| to pi first and then lowers it by correspondingly more.
    # Measured totals are identical to three decimals (1.745 vs 1.745 from 100
    # degrees, 3.101 vs 3.104 from 178).  At large |psi| the gate-based heading
    # terms are flat -- cos has zero slope at pi -- so yaw_progress is the only
    # live yaw signal exactly where the choice is made, and it has no opinion;
    # the vehicle then commits to the long way and never corrects, because past
    # 180 degrees the long way genuinely IS the short way.
    #
    # sin(psi) * omega_z is positive only when yawing the shortest way: psi > 0
    # means the goal is to the left, which is closed by INCREASING yaw, i.e.
    # omega_z > 0.  It is zero at psi = pi, which is correct -- there the two
    # directions really are equivalent.  Unlike yaw_progress this is a rate
    # reward, not a telescoping one, so it does create a directional preference.
    yaw_direction_weight = 2.0
    # Normaliser for the yaw rate in that term.  max_delta_yaw = 15 deg sustains
    # ~42 deg/s, so this saturates at roughly the achievable turn rate.
    yaw_direction_rate = math.radians(45.0)
    # Heading-error band over which the direction term fades in: off below
    # gate_min, full above gate_max.  Being a rate reward it is maximised by
    # maximum yaw rate at ANY non-zero error, and ungated it beat the yaw rate
    # damping down to 0.44 degrees -- bang-bang yaw, which oscillates once the
    # vehicle is on the bearing.  It only needs to exist where the left/right
    # choice is still open.
    yaw_direction_gate_min = math.radians(20.0)
    yaw_direction_gate_max = math.radians(40.0)
    # Coarse heading hold, 0.5*(1+cos): bounded, smooth, and non-vanishing at
    # every angle.  The fine term locks in the last ~20 degrees.
    heading_weight = 2.0
    # ABS exponential, exp(-k|psi|), NOT exp(-k psi^2).  A Gaussian is flat at
    # its peak, so a squared form has exactly ZERO restoring gradient at zero
    # heading error -- and gate = 0.5*(1+cos psi) is flat there too, so nothing
    # held the heading at alignment.  The plant provides no restoring force
    # either: the yaw setpoint is rebuilt as measured_yaw + delta each step, so
    # commanding zero simply accepts whatever the heading has drifted to.  The
    # nose therefore wandered on the run-in, and with roll pinned level the
    # thrust-tilt direction wanders with it, bowing the approach.
    # exp(-3|psi|) has slope -3 as psi -> 0, which is what the commented pursuit
    # reward's 2.0 * exp(-3|yaw_error|) had and this reward had lost.
    # Sharpness drops 8.0 -> 3.0 because the abs form is far narrower than the
    # squared one at equal k.
    # PREVIOUS (revert to this): heading_fine_sharpness = 8.0 and
    # .square() instead of .abs() in heading_reward.
    heading_fine_weight = 1.5
    heading_fine_sharpness = 3.0
    # Forward-tilt reward, sin(pitch_sp)/sin(max_pitch) * gate.  This is the
    # thrust-tilt direction dotted with the bearing: maximal only when pitched
    # forward AND yawed onto the goal.  Keep it well below speed_tracking_weight
    # or the standing incentive for maximum pitch overrides speed regulation.
    # At 1.0 this paid only ~0.15/step at a realistic cruise trim of 0.3 degrees,
    # against ~7/step available for simply hovering on the bearing -- there was
    # effectively no signal to pitch at all.  3.0 makes the initial acceleration
    # (pitch held at the 2 degree limit) worth 3.0/step.  Kept below
    # speed_tracking_weight so speed regulation still wins at steady state,
    # otherwise this term just demands maximum pitch forever.
    # tilt_weight = 1.0  # previous (revert to this)
    tilt_weight = 3.0
    # Reverse tilt is the pitch-back exploit: with roll pinned level the vehicle
    # can close on the goal tail-first and never correct heading at all.
    # Penalised outside brake_radius; left free inside it because reverse tilt is
    # the only braking authority the +/-2 degree envelope has.
    back_pitch_weight = 2.0
    # Bearing error is faded to zero between these radii rather than switched off
    # at target_reached_distance (what the inherited _goal_yaw_error does).
    # Fading the ERROR instead of the reward keeps full heading credit at the
    # goal, so arriving never costs reward -- a faded-out positive term would be
    # a cliff that discourages the policy from finishing.
    # The bearing itself rotates as the vehicle translates, and it accelerates
    # hard at close range: at 0.5 m/s of lateral drift it swings 9.5 deg/s at
    # 3 m but 47.7 deg/s at 0.6 m.  Fading out only below 0.5 m left the policy
    # chasing a bearing moving nearly as fast as it could yaw, which is a
    # guaranteed limit cycle.  Starting the fade at 1.0 m retires the heading
    # objective before the bearing becomes unchaseable.
    # Narrowed back to 0.30.  At 1.00 the reported heading error was already
    # halved by 0.6 m and near zero by 0.4 m, so heading_fine read ~1.5 (its
    # maximum) against a true value of 0.03 for a 40 degree error -- the reward
    # asserted a perfect heading no matter where the nose pointed, leaving NO
    # heading-holding force on the entire run-in.  With roll pinned level the
    # thrust-tilt direction IS the heading, so a drifting nose rotates the
    # acceleration vector and bows the final approach into a curve.
    # 1.00 was chosen to damp yaw oscillation; direction_gate now does that job,
    # so the fade no longer has to and can go back to being narrow.
    # yaw_fade_max = 0.50  # original
    # yaw_fade_max = 1.00  # previous (revert to this)
    yaw_fade_min = 0.20
    yaw_fade_max = 0.30
    # Scalar closing-speed tracking.  A velocity VECTOR setpoint asks for lateral
    # motion the airframe cannot produce (roll is pinned to 0), so its tracking
    # exp() saturates flat exactly while the heading is wrong.
    speed_tracking_weight = 5.0
    # Sideslip is pure loss on a roll-locked airframe.
    cross_track_vel_weight = 1.0
    # Vertical channel, kept separate from the horizontal pursuit: collective
    # gives ~2.3 m/s^2 against the pitch envelope's 0.34 m/s^2, so folding z into
    # one exponential lets the unachievable horizontal error mask the achievable
    # vertical one.
    # vertical_weight = 2.0  # previous (revert to this)
    vertical_weight = 4.0
    vertical_sharpness = 5.0
    vertical_gain = 1.0
    max_climb_rate = 0.7
    # Direct altitude-POSITION penalty on the vertical goal error, penalty form
    # (same as sway_penalty/vertical_penalty): 0 on altitude, saturating to
    # -altitude_weight as the offset grows.  The vertical_penalty above tracks
    # only climb RATE, so it has no term that charges for merely BEING off
    # altitude -- during the turn the vehicle levels out, the collective (trimmed
    # for the pitched cruise) over-lifts, and the climb-rate term cannot see the
    # resulting steady offset.  Sharpness sets WHERE the penalty bites: at 1.0 it
    # is ~63% of full at 1 m, so a small residual offset costs almost nothing
    # (0.3 m -> 0.26/step) and the policy happily settles there.  Raised to 3.0,
    # which reaches ~63% at ~0.58 m and charges ~0.71/step at 0.3 m.  Push to
    # ~8.0 to squeeze the last few centimetres; beyond ~10 the term saturates
    # before 0.3 m and the fine gradient flattens out again.
    # PREVIOUS (revert to this): altitude_sharpness = 1.0
    altitude_weight = 3.0
    altitude_sharpness = 3.0
    # Terminal funnel: coarse exp(-d) plus the fine_goal_sigma well.
    goal_weight = 2.0
    fine_goal_weight = 3.0
    dwell_weight = 5.0
    # Braking.  The commented pursuit reward stopped at the goal because its
    # near_goal bonus required speed < 0.10 AND its velocity_tracking used a full
    # 3-vector error, so any residual speed was penalised.  The scalar
    # closing-speed term here does neither: overflying the goal sideways leaves
    # closing_speed ~= 0, which scores as perfect tracking.  These restore the
    # stop condition -- dwell now needs the speed gate, and stop_weight rewards
    # actually being stationary, ramping in as the goal is approached.
    goal_speed_threshold = 0.10
    # Matched to the 5.0 that the commented pursuit reward's velocity_tracking
    # carried.  That term drove the whole velocity VECTOR to zero at the goal,
    # which is why it held position well; closing-speed tracking replaced it and
    # is blind to lateral drift (sliding sideways across the goal leaves
    # v_close ~= 0, scoring full marks).  stop_reward is the only term here that
    # uses total speed, so it has to carry that weight on its own.
    # stop_weight = 3.0  # previous (revert to this)
    stop_weight = 5.0
    stop_sharpness = 5.0
    # Per-step clamp on the progress term; a reset or a random push can otherwise
    # spike it by the full displacement in a single step.
    max_progress_per_step = 0.05

    # Observation history.  A single frame cannot distinguish a heavy airframe
    # from a light one, so under domain randomization the policy can only learn
    # the mean of the distribution.  Stacking the recent command/response pair
    # lets it identify the dynamics online.
    #
    # Only the 10 dims that carry cause and effect are stacked: body linear
    # velocity, body angular velocity, and the actions (observation indices
    # 7:17).  Goal error and yaw error are task state, not dynamics, and the
    # attitude quaternion is largely recoverable from the angular-velocity
    # history, so none of them are worth the width.
    history_length = 8
    # Sampled every history_stride steps.  Consecutive frames at dt = 0.01 s are
    # nearly identical; a stride of 5 spans 0.4 s, long enough for acceleration
    # differences between airframes to show up.
    history_stride = 5
    history_slice = (7, 17)
    # 18 current + history_length * 10 stacked.
    observation_space = 98

    # Off by default: with enabled = False the task reproduces its nominal
    # dynamics exactly.  Flip domain_randomization.enabled to turn the whole
    # thing on, or a single group's flag to isolate one mismatch.
    domain_randomization: DomainRandomizationCfg = DomainRandomizationCfg()

    def __post_init__(self) -> None:
        # 1 kHz physics so the rate loop runs every tick. decimation = 10 keeps
        # the policy step at 100 Hz (step_dt = sim.dt * decimation = 0.01), so
        # episode length in policy steps is unchanged.
        self.sim.dt = 0.001

        # Resolve the target sampling from the waypoint mode flag.
        if self.random_waypoints:
            self.target_min_ratio = (0.05, 0.05, 0.10)
            self.target_max_ratio = (0.95, 0.95, 0.95)
            self.minimum_target_distance = 0.75
            self.resample_target_on_reach = True
        else:
            self.target_min_ratio = (0.5, 0.5, 0.5)
            self.target_max_ratio = (0.5, 0.5, 0.5)
            self.minimum_target_distance = 0.0
            self.resample_target_on_reach = False


class Starling2MaxAttitudeDeltaEnv(Starling2MaxVelocityEnv):
    """Direct-RL port whose policy emits ``[d_thrust, d_roll, d_pitch, d_yaw]``."""

    cfg: Starling2MaxAttitudeDeltaEnvCfg

    def __init__(self, cfg: Starling2MaxAttitudeDeltaEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # Rate loop runs every physics tick (1/physics_dt Hz); the attitude loop
        # runs every attitude_interval-th tick to approximate attitude_control_hz.
        attitude_interval = max(
            1, round((1.0 / self.physics_dt) / self.cfg.attitude_control_hz)
        )
        self._controller = PX4AttitudeRateController(
            self.num_envs,
            self.device,
            hover_thrust=abs(self.cfg.px4_hover_thrust),
            min_thrust=abs(self.cfg.px4_max_thrust),
            max_thrust=abs(self.cfg.px4_min_thrust),
            attitude_interval=attitude_interval,
        )
        # self._raw_actions = torch.zeros_like(self._actions)  # original 4-wide (revert)
        self._raw_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        # Spawn point of the current episode; anchors the straight-line corridor.
        self._path_origin_w = self.scene.env_origins.clone()
        # Previous-step bearing error, for the turn-rate reward.  Captured in
        # _pre_physics_step alongside _previous_distance so both differences are
        # taken across the same interval.
        self._previous_yaw_error = torch.zeros(self.num_envs, device=self.device)
        # Consecutive steps held inside target_reached_distance.
        self._dwell_counter = torch.zeros(self.num_envs, device=self.device)
        self._randomizer = DomainRandomizer(
            self.cfg.domain_randomization, self.num_envs, self.physics_dt, self.device
        )
        self._wind_velocity_w = torch.zeros((self.num_envs, 3), device=self.device)

        # Ring buffer of past dynamics frames; index j holds the frame from j
        # steps ago, so the strided taps are a plain gather.
        start, stop = self.cfg.history_slice
        self._history_width = stop - start
        self._history_taps = self.cfg.history_stride * torch.arange(
            1, self.cfg.history_length + 1, device=self.device
        )
        self._obs_history = torch.zeros(
            (self.num_envs, int(self._history_taps[-1]) + 1, self._history_width),
            device=self.device,
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._previous_actions[:] = self._actions
        pos_error_w = self._position_error_w()
        self._previous_distance[:] = pos_error_w.norm(dim=-1)

        clipped_actions = actions.clamp(-1.0, 1.0)
        # Action latency and inner-loop delay: the controller acts on a stale
        # command.  _raw_actions tracks the delayed command, so the effort and
        # smoothness penalties score what the vehicle actually flew.
        clipped_actions = self._randomizer.delay_actions(clipped_actions)
        self._raw_actions[:] = clipped_actions

        roll, pitch, yaw = euler_xyz_from_quat(self._robot.data.root_quat_w)
        self._previous_yaw_error[:] = self._raw_goal_yaw_error(pos_error_w, yaw)
        delta_thrust = clipped_actions[:, 0] * self.cfg.delta_thrust_scale
        # The hover throttle the controller assumes; if it disagrees with what
        # the airframe actually needs, the policy sees a steady thrust bias.
        hover_thrust = self.cfg.px4_hover_thrust
        if self._randomizer.enabled and self.cfg.domain_randomization.randomize_hover_throttle:
            hover_thrust = hover_thrust * self._randomizer.hover_throttle_scale
        px4_thrust = torch.clamp(
            hover_thrust + delta_thrust,
            self.cfg.px4_min_thrust,
            self.cfg.px4_max_thrust,
        )
        self._actions[:, 0] = -px4_thrust
        # clipped_actions index 1 = d_roll.  Same accumulate-and-clamp form as
        # pitch: the delta rides on the MEASURED roll and the setpoint is bounded
        # by max_roll (tighter than the pitch bound -- see the cfg comment).
        # ROLL-LOCKED variant (revert to this): self._actions[:, 1] = 0.0
        self._actions[:, 1] = torch.clamp(
            roll + clipped_actions[:, 1] * self.cfg.max_delta_roll_pitch,
            -self.cfg.max_roll,
            self.cfg.max_roll,
        )
        # clipped_actions index 2 = d_pitch (1 in the roll-locked variant).
        # Two-sided pitch clamp [-max, +max], as it was previously.
        # Forward-only variant (revert here to re-enable): clamp to [0, +max].
        # self._actions[:, 2] = torch.clamp(
        #     pitch + clipped_actions[:, 2] * self.cfg.max_delta_roll_pitch,
        #     0.0,
        #     self.cfg.max_pitch,
        # )
        self._actions[:, 2] = torch.clamp(
            pitch + clipped_actions[:, 2] * self.cfg.max_delta_roll_pitch,
            -self.cfg.max_pitch,
            self.cfg.max_pitch,
        )
        # clipped_actions index 3 = d_yaw (2 in the roll-locked variant).
        self._actions[:, 3] = wrap_to_pi(yaw + clipped_actions[:, 3] * self.cfg.max_delta_yaw)
        self._apply_random_pushes()

    def _apply_random_pushes(self) -> None:
        """Kick a random subset of vehicles with an impulse, as a gust or bump would."""
        push = self._randomizer.sample_pushes()
        if push is None:
            return
        env_ids, delta_linear_w, delta_angular_w = push
        velocity_w = self._robot.data.root_vel_w[env_ids].clone()
        velocity_w[:, :3] += delta_linear_w
        velocity_w[:, 3:] += delta_angular_w
        self._robot.write_root_velocity_to_sim(velocity_w, env_ids)

    def _apply_action(self) -> None:
        root_quat_wxyz = self._robot.data.root_quat_w
        root_lin_vel_w = self._robot.data.root_lin_vel_w
        root_ang_vel_b = quat_apply_inverse(root_quat_wxyz, self._robot.data.root_ang_vel_w)
        control = self._controller.compute(
            self._actions,
            root_quat_wxyz,
            root_ang_vel_b,
            self.physics_dt,
        )
        body_lin_vel = quat_apply_inverse(root_quat_wxyz, root_lin_vel_w)
        # Wind enters twice: through the rotor inflow model and as a body force.
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

        # Drag acts on the airspeed, not the ground speed, so wind biases it.
        linear_drag = self._linear_drag
        quadratic_drag = self._quadratic_drag
        angular_drag = self._angular_drag
        angular_quadratic_drag = self._angular_quadratic_drag
        if self._randomizer.enabled and self.cfg.domain_randomization.randomize_drag:
            linear_drag = linear_drag + self._randomizer.linear_drag
            quadratic_drag = quadratic_drag + self._randomizer.quadratic_drag
            angular_drag = angular_drag + self._randomizer.angular_drag
            angular_quadratic_drag = angular_quadratic_drag + self._randomizer.angular_quadratic_drag

        drag_force = -linear_drag * relative_air_velocity_b - quadratic_drag * relative_air_velocity_b.norm(
            dim=-1, keepdim=True
        ) * relative_air_velocity_b
        drag_force = drag_force + self._randomizer.wind_force_b(relative_air_velocity_b)
        drag_torque = -angular_drag * root_ang_vel_b - angular_quadratic_drag * root_ang_vel_b.abs() * root_ang_vel_b
        self._motor_forces_b[:, self._root_body_id] += drag_force
        self._motor_torques_b[:, self._root_body_id] += drag_torque
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=torch.arange(self._robot.num_bodies, device=self.device),
            forces=self._motor_forces_b,
            torques=self._motor_torques_b,
        )

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        super()._reset_idx(env_ids)
        if not hasattr(self, "_raw_actions"):
            return
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        self._raw_actions[env_ids] = 0.0
        self._controller.reset(env_ids)
        self._path_origin_w[env_ids] = self._robot.data.root_pos_w[env_ids]
        self._dwell_counter[env_ids] = 0.0
        # Zero is safe here only because _pre_physics_step rewrites this every
        # step before _get_rewards reads it; it is never differenced against a
        # stale value from the previous episode.
        self._previous_yaw_error[env_ids] = 0.0
        # Never let a new episode read the previous one's history.
        self._obs_history[env_ids] = 0.0
        self._randomize(env_ids)

    def _randomize(self, env_ids: torch.Tensor) -> None:
        """Resample this episode's plant/controller/signal-path parameters."""
        if not self._randomizer.enabled:
            return
        self._randomizer.reset(env_ids)
        self._randomizer.randomize_body_properties(self._robot, env_ids, self._root_body_id)

        dr_cfg = self.cfg.domain_randomization
        if dr_cfg.randomize_controller_gains:
            self._controller.attitude_p = (
                self._controller.nominal_attitude_p * self._randomizer.k_rot_scale
            )
            self._controller.rate_p = (
                self._controller.nominal_rate_p * self._randomizer.k_angvel_scale
            )
        if dr_cfg.randomize_thrust_constant:
            self._allocator.motor_constant_scale = self._randomizer.thrust_constant_scale
        if dr_cfg.randomize_motor_lag:
            self._allocator.time_constant_up_scale = self._randomizer.tau_up_scale
            self._allocator.time_constant_down_scale = self._randomizer.tau_down_scale

    def _get_observations(self) -> dict[str, torch.Tensor]:
        obs_dict = super()._get_observations()
        obs = obs_dict["policy"]
        # May be called by the base env before this subclass finishes __init__.
        randomizer = getattr(self, "_randomizer", None)
        if randomizer is not None and randomizer.enabled:
            # Bias first, then delay: a biased estimate is what gets buffered.
            obs = randomizer.apply_sensor_bias(obs)
            obs = randomizer.delay_observations(obs)
        # --- goal reparameterization -----------------------------------------
        # Replace the base world-frame pos_error [0:3] with the UNIT direction to
        # the goal in the vehicle (yaw-only) frame, and yaw_error [17] with the
        # RAW distance. Distance is left raw because normalize_input=true whitens
        # the obs anyway, and any scaling+clamp would only discard range info
        # (e.g. distances > max reach saturating). Applied after bias/delay so the
        # corrupted position estimate flows into both. History slice (7,17) is
        # untouched, so observation_space stays 98.
        # TO REVERT: delete this whole block (restores world pos_error + yaw_error).
        _, _, _yaw = euler_xyz_from_quat(self._robot.data.root_quat_w)
        _veh_quat = quat_from_euler_xyz(torch.zeros_like(_yaw), torch.zeros_like(_yaw), _yaw)
        _err_vehicle = quat_apply_inverse(_veh_quat, obs[:, 0:3])
        _dist = _err_vehicle.norm(dim=-1, keepdim=True)
        obs = obs.clone()
        obs[:, 0:3] = _err_vehicle / (_dist + 1.0e-6)
        obs[:, 17] = _dist.squeeze(-1)
        # ---------------------------------------------------------------------
        obs_dict["policy"] = self._append_history(obs)
        return obs_dict

    def _append_history(self, obs: torch.Tensor) -> torch.Tensor:
        """Concatenate strided past dynamics frames onto the current observation.

        History is built from the observation the policy actually receives, so
        it carries the same bias and delay the live signal does.
        """
        if getattr(self, "_obs_history", None) is None:
            return obs
        start, stop = self.cfg.history_slice
        self._obs_history = torch.roll(self._obs_history, shifts=1, dims=1)
        self._obs_history[:, 0] = obs[:, start:stop]
        history = self._obs_history[:, self._history_taps].flatten(1)
        return torch.cat((obs, history), dim=-1)

    # --- LINEAR-PENALTY REWARD, kept for reverting -------------------------
    # Note before reusing: r_alive = 0.5 only offsets -1.0 * pos_error out to
    # 0.5 m, so beyond that every step is negative while crossing crash_distance
    # terminates for a one-off -10.  Ending the episode early therefore beats
    # flying (-10 versus roughly -2000 for holding station at 3 m), and the
    # policy will learn to fly out of bounds.  Raise the crash penalty to the
    # order of -2000, or make the per-step reward non-negative, before using it.
    #
    # def _get_rewards(self) -> torch.Tensor:
    #     pos_error = self._position_error_w().norm(dim=-1)     # ||p* - p||
    #     lin_vel = self._robot.data.root_lin_vel_w.norm(dim=-1)
    #     ang_vel = self._robot.data.root_ang_vel_w.norm(dim=-1)
    #     action_diff = self._actions - self._previous_actions
    #
    #     r_pos = -1.0 * pos_error                              # position tracking
    #     r_vel = -0.05 * lin_vel                               # damp linear velocity
    #     r_angvel = -0.05 * ang_vel                            # damp body rates
    #     r_effort = -0.02 * (self._raw_actions ** 2).sum(-1)   # action magnitude
    #     r_smooth = -0.10 * (action_diff ** 2).sum(-1)         # action smoothness
    #     r_alive = 0.5                                         # per-step survival
    #
    #     reward = r_pos + r_vel + r_angvel + r_effort + r_smooth + r_alive
    #     return torch.where(
    #         pos_error > self.cfg.crash_distance,
    #         torch.full_like(reward, -10.0),
    #         reward,
    #     )
    # -----------------------------------------------------------------------

    # --- PURSUIT REWARD, ACTIVE ---------------------------------------------
    # Restored in place of the turn-and-go reward below.  Known weaknesses,
    # which are why it was superseded once before:
    #   1. velocity_tracking used a velocity VECTOR setpoint whose vehicle-frame
    #      goal_dir carries a lateral component.  Roll is pinned to 0, so that
    #      strafe is unflyable and exp(-2*||vel_error||^2) sits flat in its tail
    #      exactly while the heading is wrong.  It also folded the cheap vertical
    #      channel into the same exponential as the expensive horizontal one.
    #   2. yaw_reward = exp(-2*psi^2) is 0.02 at 90 degrees and ~1e-8 at 180.
    #      Spawn yaw is uniform on +/-pi, so half of all episodes started with no
    #      usable heading gradient at all.
    #   3. Closing speed is sign-agnostic about which body axis produced it, so
    #      pitching BACKWARD closed distance just as well as pitching forward and
    #      heading correction was optional.
    # Two outright bugs also lived here: the effort penalty summed all channels
    # including the two delta channels (penalising turning and pitching), and the
    # smoothness penalty differenced the wrapped yaw setpoint without wrapping,
    # so a heading crossing +/-pi cost ~0.5 * (2*pi)^2 ~= 20 out of nothing.
    #
    def _get_rewards(self) -> torch.Tensor:
        pos_error_w = self._position_error_w()
        distance = pos_error_w.norm(dim=-1)
        root_quat_wxyz = self._robot.data.root_quat_w
        # ADDED for vertical_penalty below; the original pursuit reward only
        # needed the body-frame velocity.  (revert: inline it back into
        # body_lin_vel and delete this line)
        root_lin_vel_w = self._robot.data.root_lin_vel_w
        body_lin_vel = quat_apply_inverse(root_quat_wxyz, root_lin_vel_w)
        body_ang_vel = quat_apply_inverse(root_quat_wxyz, self._robot.data.root_ang_vel_w)
        _, _, yaw = euler_xyz_from_quat(root_quat_wxyz)
        vehicle_quat_wxyz = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
        pos_error_vehicle_frame = quat_apply_inverse(vehicle_quat_wxyz, pos_error_w)

        yaw_error = self._goal_yaw_error(pos_error_w, yaw)
        eps = 1.0e-6
        speed = body_lin_vel.norm(dim=-1)
        # PREVIOUS: 2.0, paired with max_pitch = 4 deg
        cruise_speed = 0.7
        brake_radius = 1.5
        goal_dir = pos_error_vehicle_frame / (distance.unsqueeze(-1) + eps)
        desired_speed = cruise_speed * torch.clamp(distance / brake_radius, 0.0, 1.0)
        desired_vel = desired_speed.unsqueeze(-1) * goal_dir
        vel_error = body_lin_vel - desired_vel
        velocity_tracking = torch.exp(-2.0 * (vel_error * vel_error).sum(dim=-1))
        goal_reward = torch.exp(-distance)
        yaw_reward = torch.exp(-2.0 * yaw_error * yaw_error)
        effort_penalty = (self._raw_actions * self._raw_actions).sum(dim=-1)
        action_diff = self._actions - self._previous_actions
        smoothness_penalty = 2.0 * action_diff[:, 0].square() + 0.5 * action_diff[:, 1:].square().sum(dim=-1)
        ang_rate_penalty = (body_ang_vel * body_ang_vel).sum(dim=-1)
        progress = self._previous_distance - distance

        op_w = self._robot.data.root_pos_w - self._path_origin_w
        line_vec_w = pos_error_w + op_w
        line_dir_w = line_vec_w / (line_vec_w.norm(dim=-1, keepdim=True) + eps)
        along_w = (op_w * line_dir_w).sum(dim=-1, keepdim=True)
        cross_track = (op_w - along_w * line_dir_w).norm(dim=-1)
        corridor_penalty = 1.0 - torch.exp(-((cross_track / self.cfg.path_corridor) ** 2))

        # --- ADDED: direct altitude-POSITION penalty ---------------------------
        # Ported from the commented turn-and-go reward.  Without it this reward
        # has NO term that can see a steady altitude offset: velocity_tracking is
        # a rate objective and is fully satisfied once v ~= v_desired whatever the
        # height, and goal_reward = exp(-distance) gives up very little for a 1 m
        # vertical error.  Climbing costs effort/smoothness/ang_rate, so the
        # converged optimum is to sit high and stop paying to correct -- which is
        # why the offset appears only after long training, not early on.
        # Penalty form (0 on altitude, saturating to -altitude_weight) so it never
        # pays a constant for merely holding height.
        # TO REVERT: delete this term and drop altitude_penalty from the sum.
        altitude_penalty = self.cfg.altitude_weight * (
            1.0 - torch.exp(-self.cfg.altitude_sharpness * pos_error_w[:, 2].square())
        )
        # -----------------------------------------------------------------------

        # --- ADDED: climb-RATE penalty ----------------------------------------
        # Also ported from the commented turn-and-go reward, and the companion to
        # altitude_penalty above.  The position penalty can only charge for an
        # offset that ALREADY exists, so on its own it lets the vehicle climb
        # during a turn and only bills afterwards.  This term tracks the climb
        # rate directly, so leaving the reference altitude costs from the first
        # step -- it is what actually stops the climb rather than punishing it.
        # desired_climb is a P controller on the vertical goal error, saturated at
        # max_climb_rate, so a level goal asks for vz = 0.
        # Penalty form (0 on reference, saturating to -vertical_weight), same
        # reasoning as altitude_penalty: the reward form would pay a constant for
        # simply holding altitude and doing nothing else.
        # TO REVERT: delete this term, drop vertical_penalty from the sum, and
        # (if altitude_penalty is also removed) drop the root_lin_vel_w local.
        desired_climb = torch.clamp(
            self.cfg.vertical_gain * pos_error_w[:, 2],
            -self.cfg.max_climb_rate,
            self.cfg.max_climb_rate,
        )
        vertical_penalty = self.cfg.vertical_weight * (
            1.0
            - torch.exp(
                -self.cfg.vertical_sharpness * (root_lin_vel_w[:, 2] - desired_climb).square()
            )
        )
        # -----------------------------------------------------------------------

        reward = (
            5.0 * velocity_tracking
            + 3.0 * goal_reward
            + 6.0 * yaw_reward
            + self.cfg.progress_weight * progress
            - self.cfg.straight_line_weight * corridor_penalty
            - altitude_penalty            # ADDED (revert: delete this line)
            - vertical_penalty            # ADDED (revert: delete this line)
            - 0.05 * effort_penalty
            - 0.50 * smoothness_penalty
            - 0.30 * ang_rate_penalty
        )
        near_goal = torch.logical_and(distance < 0.20, speed < 0.10)
        reward = torch.where(near_goal, reward + 5.0, reward)
        return torch.where(distance > self.cfg.crash_distance, torch.full_like(reward, -100.0), reward)

    # --- COORDINATED TURN-AND-GO REWARD, kept for reverting -----------------
    # Yaw fast and pitch forward simultaneously: heading and pitch are rewarded
    # on independent channels, so neither term carries a factor the other can
    # drive to zero.  Yawing pays at a standstill and pitching forward pays
    # while the heading is still wrong, which makes the joint optimum an arc
    # into the goal rather than a pivot followed by a dash.  Replaced by the
    # pursuit reward above; restore by swapping the two blocks back.
    #
    # def _get_rewards(self) -> torch.Tensor:
    #     """Coordinated turn-and-go: yaw fast and pitch forward simultaneously.
    #
    #     Heading and pitch are rewarded on independent channels.  Neither term
    #     carries a factor that the other channel can drive to zero, so yawing pays
    #     while the vehicle is at rest and pitching forward pays while the heading
    #     is still wrong.  Neither can be maximised by finishing the other first,
    #     which makes the joint optimum an arc into the goal rather than a pivot
    #     followed by a dash.
    #     """
    #     cfg = self.cfg
    #     eps = 1.0e-6
    #
    #     pos_error_w = self._position_error_w()
    #     distance = pos_error_w.norm(dim=-1)
    #     horizontal_distance = pos_error_w[:, :2].norm(dim=-1)
    #     root_quat_wxyz = self._robot.data.root_quat_w
    #     root_lin_vel_w = self._robot.data.root_lin_vel_w
    #     body_lin_vel = quat_apply_inverse(root_quat_wxyz, root_lin_vel_w)
    #     body_ang_vel = quat_apply_inverse(root_quat_wxyz, self._robot.data.root_ang_vel_w)
    #     _, _, yaw = euler_xyz_from_quat(root_quat_wxyz)
    #
    #     # --- heading channel --------------------------------------------------
    #     # Raw bearing error: the inherited _goal_yaw_error switches to zero inside
    #     # target_reached_distance, and this task fades it instead.
    #     yaw_error = self._raw_goal_yaw_error(pos_error_w, yaw)
    #     fade = torch.clamp(
    #         (horizontal_distance - cfg.yaw_fade_min) / (cfg.yaw_fade_max - cfg.yaw_fade_min),
    #         0.0,
    #         1.0,
    #     )
    #     faded_yaw_error = yaw_error * fade
    #     # Soft alignment gate: 1 aligned, 0.5 at 90 degrees, 0 only at exactly
    #     # 180.  It DISCOUNTS off-bearing translation without ever forbidding it,
    #     # which is what keeps the manoeuvre simultaneous.  A hard clamp(cos, 0, 1)
    #     # would zero the translation reward past 90 degrees and force the policy
    #     # to align first.
    #     gate = 0.5 * (1.0 + torch.cos(faded_yaw_error))
    #
    #     # Turn-rate reward: linear in |yaw_error|, so it still has a gradient at
    #     # 180 degrees where every exp()/cos() heading term is flat.  No velocity
    #     # and no gate factor, so yawing pays even from a standstill facing away.
    #     # Differences the RAW error and scales by fade afterwards -- differencing
    #     # the faded error would manufacture progress as the goal is approached.
    #     yaw_progress = fade * (self._previous_yaw_error.abs() - yaw_error.abs())
    #     # Turn-DIRECTION reward.  yaw_progress is blind to which way the vehicle
    #     # turns (it telescopes, so both directions total the same); this is the
    #     # only term that prefers the shortest arc.  sin(yaw_error) * omega_z is
    #     # positive exactly when yawing the short way, and zero at 180 degrees
    #     # where the two directions are genuinely equivalent.  Uses the RAW error
    #     # for the same reason yaw_progress does, and shares its fade.
    #     #
    #     # Gated OFF near alignment.  This is a RATE reward, so its instantaneous
    #     # optimum is always maximum yaw rate; ungated it out-earned the yaw rate
    #     # damping down to a 0.44 degree heading error, which is bang-bang yaw and
    #     # oscillates once the vehicle has turned onto the bearing.  Its only job
    #     # is breaking the left/right tie at LARGE error, so switching it off
    #     # inside yaw_direction_gate_min costs nothing -- heading_fine handles the
    #     # fine alignment.
    #     # PREVIOUS (revert to this): drop direction_gate from the product below.
    #     direction_gate = torch.clamp(
    #         (yaw_error.abs() - cfg.yaw_direction_gate_min)
    #         / (cfg.yaw_direction_gate_max - cfg.yaw_direction_gate_min),
    #         0.0,
    #         1.0,
    #     )
    #     yaw_direction = fade * direction_gate * torch.clamp(
    #         torch.sin(yaw_error) * body_ang_vel[:, 2] / cfg.yaw_direction_rate,
    #         -1.0,
    #         1.0,
    #     )
    #     # .abs(), not .square(): a Gaussian is flat at its peak and so supplies no
    #     # restoring gradient at zero heading error (see heading_fine_sharpness).
    #     # PREVIOUS (revert to this): faded_yaw_error.square()
    #     heading_reward = cfg.heading_weight * gate + cfg.heading_fine_weight * torch.exp(
    #         -cfg.heading_fine_sharpness * faded_yaw_error.abs()
    #     )
    #
    #     # --- along-track speed reference --------------------------------------
    #     # SCALAR closing speed along the bearing, not a velocity vector: with roll
    #     # pinned level there is no lateral acceleration axis, so a vector setpoint
    #     # asks for a strafe the vehicle can never fly.  Closing speed is always
    #     # achievable, and the only way to raise it is to yaw onto the bearing and
    #     # pitch forward -- the coupling comes out of the geometry.
    #     # Computed ahead of the pitch channel because the tilt reward gates on the
    #     # speed deficit below.
    #     range_taper = torch.clamp(horizontal_distance / cfg.brake_radius, 0.0, 1.0)
    #     goal_dir_xy = pos_error_w[:, :2] / (horizontal_distance.unsqueeze(-1) + eps)
    #     closing_speed = (root_lin_vel_w[:, :2] * goal_dir_xy).sum(dim=-1)
    #     desired_speed = cfg.cruise_speed * range_taper
    #     # Fraction of the reference speed still to be made up: 1 at rest or when
    #     # falling behind, 0 once at or above the reference.
    #     speed_deficit = torch.clamp(
    #         (desired_speed - closing_speed) / cfg.cruise_speed, 0.0, 1.0
    #     )
    #
    #     # --- pitch channel ----------------------------------------------------
    #     # Commanded pitch setpoint; positive tilts the thrust vector forward in
    #     # this FLU body frame.  sin(pitch)/sin(max) * gate is the horizontal
    #     # thrust-tilt direction dotted with the bearing, so it is maximal only
    #     # when pitched forward AND pointed at the goal -- but it carries no
    #     # velocity factor, so it pays from the first step at any heading.
    #     pitch_setpoint = self._actions[:, 2]
    #     tilt_fraction = torch.sin(pitch_setpoint) / math.sin(cfg.max_pitch)
    #     # One-sided: reverse tilt earns ZERO here, never a negative.  Braking
    #     # requires reverse tilt, and inside brake_radius back_pitch_penalty is
    #     # deliberately switched off to allow it -- but a signed tilt_reward then
    #     # charged up to -tilt_weight * range_taper per step for the very
    #     # manoeuvre needed to stop, so the vehicle flew through the goal instead
    #     # of braking.  The exploit is still covered: back_pitch_penalty handles
    #     # reverse tilt outside brake_radius, where braking is not the reason for
    #     # it.
    #     #
    #     # Gated on speed_deficit so forward tilt stops paying the instant the
    #     # vehicle is at or above the reference speed.  Without it this term was a
    #     # standing +2.0/step bribe at 1 m out to keep accelerating through the
    #     # brake zone -- the commented pursuit reward had no tilt term at all,
    #     # which is why it braked cleanly and this one did not.
    #     # PREVIOUS (revert to either of these):
    #     # tilt_reward = cfg.tilt_weight * tilt_fraction * gate * range_taper
    #     # tilt_reward = cfg.tilt_weight * torch.clamp(tilt_fraction, min=0.0) * gate * range_taper
    #     tilt_reward = (
    #         cfg.tilt_weight
    #         * torch.clamp(tilt_fraction, min=0.0)
    #         * gate
    #         * range_taper
    #         * speed_deficit
    #     )
    #     # Closes the pitch-back shortcut: reverse tilt is strictly negative
    #     # outside brake_radius, and free inside it so the vehicle can still stop.
    #     back_pitch_penalty = (
    #         cfg.back_pitch_weight
    #         * torch.clamp(-tilt_fraction, min=0.0)
    #         * (horizontal_distance > cfg.brake_radius).to(tilt_fraction.dtype)
    #     )
    #
    #     # --- translation ------------------------------------------------------
    #     # closing_speed / desired_speed are computed with the speed reference
    #     # above, since the pitch channel needs them first.
    #     #
    #     # Two tracking forms, blended by range:
    #     #
    #     #   OUTSIDE brake_radius -- scalar closing speed.  A velocity VECTOR
    #     #   setpoint there would demand a lateral strafe the roll-locked airframe
    #     #   cannot fly, and its exp() saturates flat exactly while the heading is
    #     #   still wrong.
    #     #
    #     #   INSIDE brake_radius -- the commented pursuit reward's full 3-vector
    #     #   form.  Closing speed alone is blind to lateral drift: sliding
    #     #   sideways across the goal leaves v_close ~= 0, which scores as PERFECT
    #     #   braking (4.80 of 5.00 whether drifting at 0.0 or 0.6 m/s).  That is
    #     #   why this reward neither braked nor held position while the commented
    #     #   one did.  The objection above does not apply here: the heading is
    #     #   already correct by this point and desired_speed has tapered to ~0, so
    #     #   the reference asks for no lateral motion at all.
    #     # Evaluated in the world frame: the error norm is rotation-invariant, so
    #     # this is identical to the commented reward's vehicle-frame version
    #     # without needing to build the yaw-only quaternion.
    #     desired_vel_w = desired_speed.unsqueeze(-1) * pos_error_w / (
    #         distance.unsqueeze(-1) + eps
    #     )
    #     vel_error_vec = root_lin_vel_w - desired_vel_w
    #     full_vector_tracking = torch.exp(
    #         -cfg.full_vector_sharpness * (vel_error_vec * vel_error_vec).sum(dim=-1)
    #     )
    #     scalar_tracking = torch.exp(
    #         -cfg.speed_tracking_sharpness * (closing_speed - desired_speed).square()
    #     )
    #     # range_taper is 1 beyond brake_radius and 0 at the goal.
    #     # PREVIOUS (revert to this): speed_reward = weight * gate * scalar_tracking
    #     speed_reward = (
    #         cfg.speed_tracking_weight
    #         * gate
    #         * (range_taper * scalar_tracking + (1.0 - range_taper) * full_vector_tracking)
    #     )
    #     # Dense distance-closed shaping, gated so that closing while pointed the
    #     # wrong way earns little.  Clamped per step against reset/push spikes.
    #     progress = torch.clamp(
    #         self._previous_distance - distance,
    #         -cfg.max_progress_per_step,
    #         cfg.max_progress_per_step,
    #     )
    #     progress_reward = cfg.progress_weight * gate * progress
    #     # Sideslip is pure loss on a roll-locked airframe.  Written as a PENALTY,
    #     # not a reward: w * exp(-k x^2) has the same gradient as
    #     # -w * (1 - exp(-k x^2)) but pays a constant w for holding still, and
    #     # that constant was a large part of a risk-free hover income of ~7/step.
    #     # PREVIOUS (revert to this):
    #     # sway_reward = cfg.cross_track_vel_weight * torch.exp(
    #     #     -cfg.cross_track_vel_sharpness * body_lin_vel[:, 1].square()
    #     # )
    #     sway_penalty = cfg.cross_track_vel_weight * (
    #         1.0
    #         - torch.exp(-cfg.cross_track_vel_sharpness * body_lin_vel[:, 1].square())
    #     )
    #
    #     # --- vertical, decoupled ----------------------------------------------
    #     desired_climb = torch.clamp(
    #         cfg.vertical_gain * pos_error_w[:, 2], -cfg.max_climb_rate, cfg.max_climb_rate
    #     )
    #     # Penalty form, same reasoning as sway_penalty: identical gradient, but a
    #     # level goal made this a free +2.0/step for holding altitude and doing
    #     # nothing else.
    #     # PREVIOUS (revert to this):
    #     # vertical_reward = cfg.vertical_weight * torch.exp(
    #     #     -cfg.vertical_sharpness * (root_lin_vel_w[:, 2] - desired_climb).square()
    #     # )
    #     vertical_penalty = cfg.vertical_weight * (
    #         1.0
    #         - torch.exp(-cfg.vertical_sharpness * (root_lin_vel_w[:, 2] - desired_climb).square())
    #     )
    #     # Direct altitude-POSITION penalty.  vertical_penalty tracks climb RATE
    #     # only, so it cannot see a steady altitude offset -- it is satisfied by
    #     # vz == desired_climb even when the vehicle sits well off altitude.  This
    #     # charges for the offset itself, so the climb the vehicle takes while
    #     # levelling out for a turn is corrected the moment it appears, whatever
    #     # the flight phase.
    #     altitude_penalty = cfg.altitude_weight * (
    #         1.0 - torch.exp(-cfg.altitude_sharpness * pos_error_w[:, 2].square())
    #     )
    #
    #     # --- terminal ---------------------------------------------------------
    #     goal_reward = cfg.goal_weight * torch.exp(-distance) + cfg.fine_goal_weight * torch.exp(
    #         -(distance / cfg.fine_goal_sigma).square()
    #     )
    #     # Rewards being STATIONARY near the goal.  (1 - range_taper) is 0 beyond
    #     # brake_radius and 1 at the goal, so this is inert during the cruise and
    #     # becomes the dominant terminal term.  Uses total speed, which is what
    #     # the scalar closing-speed term cannot see: a pass straight through the
    #     # goal has closing_speed ~= 0 at the crossing and scores as perfect.
    #     speed = body_lin_vel.norm(dim=-1)
    #     stop_reward = (
    #         cfg.stop_weight
    #         * (1.0 - range_taper)
    #         * torch.exp(-cfg.stop_sharpness * speed.square())
    #     )
    #     # Dwell now requires the speed gate as well, mirroring the commented
    #     # pursuit reward's near_goal condition.  Without it the counter ran while
    #     # the vehicle was flying THROUGH the goal, paying it to overshoot.
    #     # PREVIOUS (revert to this):
    #     # inside = distance < cfg.target_reached_distance
    #     inside = torch.logical_and(
    #         distance < cfg.target_reached_distance, speed < cfg.goal_speed_threshold
    #     )
    #     self._dwell_counter = torch.where(
    #         inside, self._dwell_counter + 1.0, torch.zeros_like(self._dwell_counter)
    #     )
    #     dwell_reward = cfg.dwell_weight * torch.clamp(
    #         self._dwell_counter / cfg.dwell_steps, 0.0, 1.0
    #     )
    #
    #     # --- straight-line corridor -------------------------------------------
    #     # Perpendicular deviation from the spawn->goal line.  All frame-difference
    #     # vectors, so the env origin cancels.  Saturates at -straight_line_weight,
    #     # so it biases toward the line without ever dominating.
    #     op_w = self._robot.data.root_pos_w - self._path_origin_w        # spawn -> current
    #     line_vec_w = pos_error_w + op_w                                 # spawn -> goal
    #     line_dir_w = line_vec_w / (line_vec_w.norm(dim=-1, keepdim=True) + eps)
    #     along_w = (op_w * line_dir_w).sum(dim=-1, keepdim=True)
    #     cross_track = (op_w - along_w * line_dir_w).norm(dim=-1)        # perp dist from line
    #     corridor_penalty = 1.0 - torch.exp(-((cross_track / cfg.path_corridor) ** 2))
    #     # ALIGNMENT-gated, not range-gated.  The point of gating this term is to
    #     # let the vehicle arc freely while it is still turning; "still turning"
    #     # is a heading condition, not a distance one.  Keyed to (1 - range_taper)
    #     # it was 0 for the entire cruise (range_taper is 1.0 beyond brake_radius)
    #     # and only switched on inside 1.5 m, so the straight-line objective was
    #     # effectively absent from the whole flight -- the vehicle was free to
    #     # wander off the line at any distance provided it was not yet close.
    #     # gate is ~0 while turning and ~1 once pointed at the goal, so the arc is
    #     # still free but the run-in is held to the line at every distance.
    #     # PREVIOUS (revert to either of these):
    #     # corridor_weight = cfg.straight_line_weight * (1.0 - range_taper)
    #     # corridor_weight = cfg.straight_line_weight * gate       # alignment-gated
    #     corridor_weight = cfg.straight_line_weight              # ungated
    #
    #     # --- regularisers ------------------------------------------------------
    #     # Collective channel only; channels 1 and 2 are deltas whose zero means
    #     # "hold attitude", so penalising their magnitude penalises the task.
    #     effort_penalty = cfg.effort_weight * self._raw_actions[:, 0].square()
    #     action_diff = self._actions - self._previous_actions
    #     # _actions[:, 3] is a WRAPPED yaw setpoint, so a heading crossing +/-pi
    #     # gives a ~2*pi raw difference and a ~20-point penalty spike out of
    #     # nothing -- which directly suppressed large heading corrections.
    #     # _actions[:, 1] (roll) is pinned to 0 and contributes nothing.
    #     yaw_setpoint_diff = wrap_to_pi(action_diff[:, 3])
    #     smoothness_penalty = (
    #         cfg.smoothness_thrust_weight * action_diff[:, 0].square()
    #         + 0.5 * action_diff[:, 2].square()
    #         + 0.5 * yaw_setpoint_diff.square()
    #     )
    #     # Split by axis: damping yaw rate as hard as roll/pitch rate puts an
    #     # internal speed limit on heading correction (see ang_rate_weight).
    #     # PREVIOUS (revert to this):
    #     # ang_rate_penalty = cfg.ang_rate_weight * (body_ang_vel * body_ang_vel).sum(dim=-1)
    #     ang_rate_penalty = cfg.ang_rate_weight * body_ang_vel[:, :2].square().sum(
    #         dim=-1
    #     ) + cfg.yaw_rate_weight * body_ang_vel[:, 2].square()
    #
    #     reward = (
    #         cfg.yaw_progress_weight * yaw_progress
    #         + cfg.yaw_direction_weight * yaw_direction
    #         + heading_reward
    #         + tilt_reward
    #         + speed_reward
    #         + progress_reward
    #         + goal_reward
    #         + stop_reward
    #         + dwell_reward
    #         - sway_penalty
    #         - vertical_penalty
    #         - altitude_penalty
    #         - back_pitch_penalty
    #         - corridor_weight * corridor_penalty
    #         - effort_penalty
    #         - smoothness_penalty
    #         - ang_rate_penalty
    #     )
    #     return torch.where(distance > cfg.crash_distance, torch.full_like(reward, -100.0), reward)
    # -----------------------------------------------------------------------

    def _raw_goal_yaw_error(
        self, position_error_w: torch.Tensor, current_yaw: torch.Tensor
    ) -> torch.Tensor:
        """Bearing error with no near-goal dead zone.

        The inherited ``_goal_yaw_error`` switches the error to zero inside
        ``target_reached_distance``.  The turn-and-go reward fades it linearly
        instead, so it needs the unswitched value.
        """
        desired_yaw = torch.atan2(position_error_w[:, 1], position_error_w[:, 0])
        return wrap_to_pi(desired_yaw - current_yaw)
