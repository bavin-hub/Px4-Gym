"""Starling 2 Max depth-image navigation with attitude-delta actions.

Same perception, obstacle course, curriculum and termination as
:mod:`aerial_isaac_lab.tasks.starling2max_navigation_env`, but the policy emits
attitude deltas instead of a velocity command:

* **action (3-D)** ``[delta_thrust, delta_pitch, delta_yaw]`` in ``[-1, 1]``,
  converted to an absolute ``[thrust, roll, pitch, yaw]`` setpoint for
  :class:`~aerial_isaac_lab.core.LeeAttitudeYawController`.  **Roll is pinned to
  0.0** -- there is no lateral acceleration, so avoidance is turn-then-go, the
  same constraint the velocity navigation policy has.
* **observation (81-D)** -- identical layout to the velocity navigation task.
  Slots 13:17 carry the previous *raw* deltas (3 channels plus a zero pad),
  which are bounded in ``[-1, 1]`` and are exactly what the reward penalises.
* **50 Hz policy** (``decimation = 20`` on a 1 kHz tick).  The depth camera stays
  at 10 Hz via ``update_period = 0.1`` and the frozen latent is held between
  renders, so the encoder runs at 10 Hz rather than 50 Hz.

Nothing here is shared with the velocity navigation task or the
position/velocity/attitude-delta waypoint tasks.  The reward lives in its own
module (:mod:`aerial_isaac_lab.tasks.depth_attitude_delta_reward`), the robot
spawn config is a task-local copy, and the controller/allocator/motor model and
all their gains are reused unchanged from :mod:`aerial_isaac_lab.core`.

Attitude limits differ from the waypoint attitude-delta task on purpose: that
task flies gently between waypoints (``max_roll_pitch = 2 deg``), which gives
only 0.34 m/s^2 of horizontal authority and cannot dodge an obstacle.

PORT NOTE (this repository only)
--------------------------------
This is a port of ``starling_aerial_isaac``'s task of the same name.  The task
retains the same 3-D action, 81-D observation, camera, obstacle field and
termination, with a locally modified 50 Hz policy and CRL-inspired reward, and
with exactly one substitution: the inner loop is
:class:`~aerial_isaac_lab.core.PX4AttitudeRateController` plus the PX4 mixer
chain, instead of :class:`~aerial_isaac_lab.core.LeeAttitudeYawController`
driving :meth:`PX4MultirotorAllocator.allocate_wrench`.  Three consequences:

* **The thrust command is passed straight through.**  The upstream task maps its
  collective into Aerial Gym's normalised domain (``collective / 0.13 - 1``,
  where 0 means hover) because that is what the Lee controller consumes.  The
  PX4 controller consumes the PX4 throttle magnitude directly, which is what
  ``collective`` already is, so that conversion is simply dropped.  The commanded
  active safe range is 0.77x -> 1.42x hover, with hover at ``a = +0.266``.
* **The wrench path is replaced by the PX4 mixer chain**
  (``allocate_normalized_control`` -> ``function_motors`` -> ``mixing_output``
  -> ``gazebo_motor_model``), matching every other task in this repository.
  Unlike the Lee path this models motor lag and per-actuator saturation, so
  collective and torque now genuinely compete for authority.
* **Aggregate mass/inertia is no longer computed.**  It existed solely as an
  input to the Lee controller; the PX4 loop is a normalised-output controller
  and does not need it.

The PX4 loops keep their native frequencies.  Upstream runs physics at 100 Hz
with ``decimation = 1``, which would clock the rate PID at 100 Hz -- ten times
slower than the cadence its gains were tuned for.  Rather than retune the
controller, the base tick moves to 1 kHz and the policy decimation to 20:

===================  ==========  ===============================================
loop                 rate        how
===================  ==========  ===============================================
physics + rate PID   1 kHz       ``sim.dt = 0.001``
attitude             250 Hz      ``attitude_interval = 4``
policy                50 Hz      ``decimation = 20`` -> ``step_dt = 0.02``
depth camera         10 Hz       ``update_period = 0.1``, latent held between
===================  ==========  ===============================================

The 16.1-second episode is now 805 policy steps; the camera cadence is unchanged.
Two knock-on adjustments follow from the finer physics tick:
``sim.render_interval`` becomes 100 to keep rendering every 0.1 s, and the
contact sensor's ``history_length`` tracks ``decimation`` so a collision in any
substep is still seen.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObjectCollection
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import (
    ContactSensor,
    ContactSensorCfg,
    MultiMeshRayCasterCamera,
    MultiMeshRayCasterCameraCfg,
)
from isaaclab.sensors.ray_caster.patterns import PinholeCameraPatternCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from aerial_isaac_lab.core import PX4AttitudeRateController
from aerial_isaac_lab.core.math import (
    euler_xyz_from_quat,
    quat_apply_inverse,
    quat_from_euler_xyz,
    wrap_to_pi,
)
from aerial_isaac_lab.core.motor_px4 import PX4MultirotorAllocator
from aerial_isaac_lab.models import STARLING2MAX_MODEL
from aerial_isaac_lab.models.model_cfg import MultirotorModelCfg
from aerial_isaac_lab.perception import (
    DceDepthEncoder,
    DceEncoderCfg,
    DepthRangeCfg,
    normalize_depth_image,
)
from aerial_isaac_lab.scene import (
    BoxToggler,
    ObstacleFieldCfg,
    ObstacleFieldRandomizer,
    build_obstacle_collection_cfg,
    build_wall_collection_cfg,
    create_group_prims,
    raycast_targets,
)

from .depth_attitude_delta_reward import compute_depth_attitude_delta_reward
from .depth_navigation_layout import (
    DepthNavigationCurriculumCfg,
    RESET_ANGULAR_VELOCITY,
    RESET_LINEAR_VELOCITY,
    TARGET_RATIO,
    make_depth_navigation_obstacle_cfg,
    make_depth_navigation_viewer_cfg,
    sample_depth_navigation_spawn,
    update_depth_chase_view,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
STARLING2MAX_ALLOCATION_PATH = str(_PROJECT_ROOT / "config" / "allocation_starling2max.yaml")

# CHANGE10: the real Starling 2 Max C29 forward ToF (PMD M0178), replacing Aerial
# Gym's BaseDepthCameraConfig (135x240, 87 deg HFOV, 0.2-10 m).
#
#   240 x 180, HFOV 86 deg, VFOV 106 deg, DFOV 138.4 deg, 0.2-6 m, up to 45 Hz
#
# Note the real sensor has NON-SQUARE pixels: 86/240 = 0.358 deg/px horizontally
# against 106/180 = 0.589 deg/px vertically. Isaac Lab derives vertical_aperture
# from the aspect ratio when it is left None, which forces square pixels and would
# give only 69.9 deg VFOV at this resolution -- so it must be set explicitly.
# To revert CHANGE10: height 135, HFOV 87.0, vertical_aperture None,
#                     max_distance / max_range 10.0.
DEPTH_IMAGE_HEIGHT = 180
DEPTH_IMAGE_WIDTH = 240
DEPTH_HORIZONTAL_FOV_DEG = 86.0
DEPTH_VERTICAL_FOV_DEG = 106.0
DEPTH_HORIZONTAL_APERTURE = 20.955
DEPTH_FOCAL_LENGTH = DEPTH_HORIZONTAL_APERTURE / (
    2.0 * math.tan(math.radians(DEPTH_HORIZONTAL_FOV_DEG) / 2.0)
)
DEPTH_VERTICAL_APERTURE = (
    2.0 * DEPTH_FOCAL_LENGTH * math.tan(math.radians(DEPTH_VERTICAL_FOV_DEG) / 2.0)
)
DEPTH_MAX_RANGE = 6.0
DEPTH_MIN_RANGE = 0.2

# The depth camera renders every DEPTH_UPDATE_PERIOD seconds while the policy
# steps at sim.dt, so the latent is reused for this many policy steps.
DEPTH_UPDATE_PERIOD = 0.1


def make_depth_attitude_delta_robot_cfg(model: MultirotorModelCfg) -> ArticulationCfg:
    """Robot spawn config for this task only.

    A task-local copy (the existing tasks each keep their own) that additionally
    sets ``activate_contact_sensors=True``, which must be applied at spawn time
    for the contact sensor to report anything.
    """
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=model.urdf_path,
            fix_base=False,
            root_link_name=model.root_body_name,
            merge_fixed_joints=False,
            joint_drive=None,
            make_instanceable=True,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=model.rigid_linear_damping,
                angular_damping=model.rigid_angular_damping,
                max_linear_velocity=100.0,
                # Isaac Gym config is 100 rad/s; Isaac Lab expects deg/s.
                max_angular_velocity=math.degrees(100.0),
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(joint_pos={}, joint_vel={}),
        actuators={},
    )


@configclass
class Starling2MaxDepthAttitudeDeltaEnvCfg(DirectRLEnvCfg):
    """Configuration for the Starling 2 Max depth attitude-delta navigation task."""

    model = STARLING2MAX_MODEL
    allocation_config_path = STARLING2MAX_ALLOCATION_PATH

    # ACTIVE: 50 Hz policy. The previous 100 Hz setting is retained immediately
    # below as a comment for reversion.
    #
    # CHANGE14 (this task only): extended to 16.1 s. At the previous 100 Hz rate
    # that was 1610 steps; at the active 50 Hz rate it is 805 steps. The active
    # 15 deg pitch cap gives about 2.63 m/s^2, while CHANGE2
    # spawns the drone 3.5-8.4 m from the goal. Worst-case budget at 8.4 m:
    # 180 deg turn at ~73 deg/s (2.5 s) + accelerate to 1 m/s (1.2 s) + cruise
    # (7.8 s) = 11.4 s, and ~15 s with a 40% detour allowance -- so 10.1 s could
    # only ever complete spawns inside about 5 m. Success also requires being
    # within 1 m of the goal *at timeout*, so arriving late scores nothing.
    # The velocity task took the same extension for the same reason (CHANGE9).
    # To revert CHANGE14: episode_length_s = 10.1.
    # PORT: the upstream task runs physics at 100 Hz with decimation = 1, which
    # would drag the PX4 rate PID down to 100 Hz.  The controller's frequencies
    # are NOT changed by this port: the rate loop keeps its native 1 kHz and the
    # attitude loop its 250 Hz, as in every other task in this repository, by
    # moving the base tick to 1 kHz. Policy decimation is now 20 for 50 Hz.
    #
    #   physics / rate loop : 1 kHz   (sim.dt = 0.001)
    #   attitude loop       : 250 Hz  (every 4th tick)
    #   policy              :  50 Hz  (decimation = 20 -> step_dt = 0.02)
    #
    # 16.1 seconds is now ceil(16.1 / 0.02) = 805 policy steps.
    rate_control_hz = 1000.0
    attitude_control_hz = 250.0
    # Previous policy rate (100 Hz): decimation = 10
    decimation = 20
    episode_length_s = 16.1

    # [delta_thrust, delta_pitch, delta_yaw]; roll is pinned to 0.
    action_space = 3
    # Same 81-D layout as the velocity navigation task.
    observation_space = 81
    state_space = 0
    seed = 1

    sim: SimulationCfg = SimulationCfg(
        # PORT: 1 kHz base tick so the PX4 rate loop runs at its native rate.
        dt=0.001,
        # Render at the depth-camera rate, not the policy rate.  Still every
        # 0.1 s (10 Hz), which is 100 ticks now that the tick is 1 ms.
        render_interval=100,
        gravity=(0.0, 0.0, -9.81),
        physx=sim_utils.PhysxCfg(
            solver_type=1,
            bounce_threshold_velocity=0.1,
            gpu_max_rigid_contact_count=2**24,
        ),
    )
    # MultiMeshRayCaster keeps per-environment mesh lists, so depth never bleeds
    # across environments; env_spacing only separates the physics.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=16, env_spacing=14.0, replicate_physics=True, clone_in_fabric=False
    )
    robot: ArticulationCfg = make_depth_attitude_delta_robot_cfg(STARLING2MAX_MODEL)
    obstacles: ObstacleFieldCfg = make_depth_navigation_obstacle_cfg()

    viewer: ViewerCfg = make_depth_navigation_viewer_cfg()
    follow_viewer = True

    depth_camera: MultiMeshRayCasterCameraCfg = MultiMeshRayCasterCameraCfg(
        prim_path="/World/envs/env_.*/Robot/vehicle",
        # 10 Hz depth on a 50 Hz policy. The latent is held between renders.
        update_period=DEPTH_UPDATE_PERIOD,
        max_distance=DEPTH_MAX_RANGE,
        # Folds ray misses (nan) and out-of-range hits onto the far plane, the
        # first of Aerial Gym's three range steps.
        depth_clipping_behavior="max",
        data_types=["distance_to_image_plane"],
        # Aerial Gym nominal_position in the body (FLU) frame. The "world"
        # convention is +X forward / +Z up, i.e. the body frame itself.
        offset=MultiMeshRayCasterCameraCfg.OffsetCfg(
            pos=(0.10, 0.0, 0.03), rot=(1.0, 0.0, 0.0, 0.0), convention="world"
        ),
        pattern_cfg=PinholeCameraPatternCfg(
            width=DEPTH_IMAGE_WIDTH,
            height=DEPTH_IMAGE_HEIGHT,
            horizontal_aperture=DEPTH_HORIZONTAL_APERTURE,
            # CHANGE10: explicit, because the real sensor's pixels are not square.
            vertical_aperture=DEPTH_VERTICAL_APERTURE,
            focal_length=DEPTH_FOCAL_LENGTH,
        ),
        mesh_prim_paths=raycast_targets(),
    )

    # history_length > 0 makes SensorBase refresh on every scene.update(), i.e.
    # every physics step.
    #
    # PORT: upstream runs decimation = 1, where a single slot already covers the
    # whole policy step. At decimation = 20 one slot would only sample the LAST
    # of the twenty substeps and quietly miss most collisions, so this tracks the
    # decimation -- the same idiom as starling2max_navigation_env.
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        update_period=0.0,
        history_length=decimation,
        track_air_time=False,
    )

    # CHANGE10: 6 m far plane, matching the real ToF.
    depth_range: DepthRangeCfg = DepthRangeCfg(
        max_range=DEPTH_MAX_RANGE, min_range=DEPTH_MIN_RANGE
    )
    dce_encoder: DceEncoderCfg = DceEncoderCfg()
    curriculum: DepthNavigationCurriculumCfg = DepthNavigationCurriculumCfg()

    # --- attitude-delta action contract ------------------------------------
    # CHANGE13: absolute collective thrust, matching the CTBR scaling, with a
    # task-level safe lower clamp:
    #
    #     collective = clamp(scale * (a + 1) * 0.5, min, max)
    #
    # Previously this was a delta on the hover constant
    # (``clamp(-0.13 + a * 0.05, -0.18, -0.08)``), which gave only +/-0.385
    # normalised (+/-3.8 m/s^2) and, unlike the pitch/yaw deltas, was a delta on a
    # *constant* rather than on a measurement -- so it carried the complexity of a
    # delta parameterisation with none of the benefit, in a needlessly narrow band.
    #
    # The lower bound is raised to 0.10 to prevent a zero-thrust/free-fall command.
    # The original scaling is unchanged, so hover remains at raw action +0.266;
    # actions below about -0.026 saturate at the new lower bound.
    #
    # To revert CHANGE13: uncomment the three constants below and restore the
    # delta form in _pre_physics_step (also commented there).
    px4_hover_thrust = -0.13
    collective_thrust_scale = 0.2054361567635904
    # Previous lower bound: collective_thrust_min = 0.0
    collective_thrust_min = 0.10
    collective_thrust_max = 0.18489254108723135
    # PORT: the controller's own clamp, in the sign/order the PX4 constructor
    # expects (min_thrust=abs(px4_max_thrust), max_thrust=abs(px4_min_thrust)).
    # Keep the controller's original broad band; the task-level mapping above is
    # responsible for enforcing the safer 0.10 lower bound.
    px4_min_thrust = -0.18489254108723135
    px4_max_thrust = -0.0
    # --- previous (delta-about-hover) scaling, kept for reference -------------
    # px4_min_thrust = -0.18
    # px4_max_thrust = -0.08
    # delta_thrust_scale = 0.05

    # Per-step pitch delta. Like the yaw delta below, it is applied to the
    # *measured* pitch each step, so it holds a constant setpoint error rather
    # than integrating open-loop.
    # Previous value: max_delta_pitch = math.radians(1.0)
    max_delta_pitch = math.radians(5.0)
    # Absolute pitch cap. The delta accumulates on the measured pitch, so this is
    # what bounds the tilt envelope. The active 15 deg cap provides about
    # g*tan(15 deg) = 2.63 m/s^2 of horizontal acceleration.
    # Previous value: max_pitch = math.radians(5.0)
    max_pitch = math.radians(15.0)
    # Per-step yaw delta.  Because it is re-applied to the *measured* yaw every
    # step it never integrates: the setpoint sits a constant max_delta_yaw ahead
    # of the vehicle, so this value acts as a steady-state attitude error and
    # therefore as a proportional yaw-rate command rather than a slew limit.
    # Measured at 10 deg: a held +1.0 command yielded ~27 deg/s achieved, well
    # under the velocity task's pi/3 = 60 deg/s cap.
    # CHANGE12: raised 10 -> 20 deg for more turning authority. With roll pinned
    # the drone only travels where it points, so yaw rate bounds how fast it can
    # redirect around an obstacle.
    # To revert CHANGE12: math.radians(10.0).
    max_delta_yaw = math.radians(20.0)
    # Roll is not commanded; the setpoint is held at zero.
    roll_setpoint = 0.0

    # Match the shared depth layout: centre goal, perimeter-shell spawn, and
    # level attitude facing the goal within a random +/-30 degree yaw offset.
    target_min_ratio = TARGET_RATIO
    target_max_ratio = TARGET_RATIO
    reset_linear_velocity = RESET_LINEAR_VELOCITY
    reset_angular_velocity = RESET_ANGULAR_VELOCITY
    success_distance = 1.0

    heading_deadzone_distance = 0.3
    """Horizontal distance inside which the heading-alignment reward is zeroed.

    Once the drone is essentially at the goal, pointing at it stops mattering and
    the bearing becomes ill-conditioned. Same idea as the velocity task's
    ``target_reached_distance`` gate on its yaw reward.
    """

    # EnvWithObstaclesCfg.env.collision_force_threshold. PhysX reports contact
    # magnitudes differently from Isaac Gym, so recalibrate by flying into a wall
    # and printing the sensor reading.
    collision_force_threshold = 0.05

    # CHANGE4: with the box parked there is no floor, so a drone that leaves the
    # virtual box would otherwise free-fall for the rest of the episode. Leaving
    # this radius counts as a crash: it takes the collision penalty and the
    # curriculum logs it as a crash, which discourages escaping into open space.
    # The box half-diagonal is 5.5*sqrt(3) = 9.53 m, so 12.0 leaves margin for a
    # legitimate overshoot near a corner.
    # To revert CHANGE4: set this to a large value such as 1.0e9.
    out_of_bounds_distance = 12.0

    observe_true_roll_pitch = True
    """Fill observation slots 4:6 with the true body roll/pitch.

    Worth keeping here: the pitch delta accumulates on the *measured* pitch and
    saturates at :attr:`max_pitch`, so the policy needs to observe its attitude
    to know when it is saturated.
    """

    debug_markers = True
    """Draw goal/body-frame gizmos. Only active when a GUI is attached."""

    # CRL-inspired single-goal reward. All values are intentionally on CRL's
    # small raw scale; the training config therefore uses reward scale 1.0.
    reward_parameters = {
        "progress_reward_multiplier": 1.0,
        "yaw_alignment_reward_magnitude": 0.05,
        "yaw_alignment_reward_exponent": 1.0,
        # CTBR-style approach braking and stationary dwell at the goal.
        "cruise_speed": 0.7,
        "brake_radius": 0.8,
        "velocity_tracking_magnitude": 0.5,
        "velocity_tracking_exponent": 2.0,
        "stop_reward_magnitude": 1.0,
        "stop_reward_exponent": 5.0,
        "dwell_distance": 0.2,
        "dwell_speed": 0.1,
        "dwell_reward": 5.0,
        "action_penalty_multiplier": 0.005,
        "action_diff_penalty_multiplier": 0.005,
        "vertical_error_penalty_multiplier": 0.01,
        "proximity_penalty_multiplier": 0.02,
        "proximity_softener": 0.2,
        "goal_reached_reward": 5.0,
        "collision_penalty": -4.0,
    }

    # Previous reward parameters (disabled; uncomment with the previous formula):
    # pos 5.0/1/3.5, very-close 5.0/2.0, progress 10.0 (retreat x2),
    # action-diff magnitudes 0.8/0.8/0.8 with exponents 5/3.333/3.33,
    # absolute-action magnitudes 1.5/0.1/1.5 with exponents 1/0.3/2,
    # yaw alignment 2.0/3.0, speed limit 1.0 with 4.0/2.0 penalty,
    # proximity 4.0/1.0, collision -20.0.


class Starling2MaxDepthAttitudeDeltaEnv(DirectRLEnv):
    """Depth navigation driven by attitude-delta commands at 50 Hz."""

    cfg: Starling2MaxDepthAttitudeDeltaEnvCfg

    def __init__(
        self,
        cfg: Starling2MaxDepthAttitudeDeltaEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)

        # ``_raw_actions`` is the clipped policy output (what the reward
        # penalises and what the observation reports); ``_setpoints`` is the
        # absolute [thrust, roll, pitch, yaw] command handed to the controller.
        self._raw_actions = torch.zeros((self.num_envs, 3), device=self.device)
        self._previous_raw_actions = torch.zeros_like(self._raw_actions)
        self._setpoints = torch.zeros((self.num_envs, 4), device=self.device)
        # One-time CRL-style gate bonus, adapted to entering the single goal.
        self._goal_bonus_awarded = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        self._target_pos_w = self.scene.env_origins.clone()
        self._pos_error_vehicle = torch.zeros((self.num_envs, 3), device=self.device)
        self._pos_error_vehicle_prev = torch.zeros_like(self._pos_error_vehicle)
        self._image_latents = torch.zeros(
            (self.num_envs, self.cfg.dce_encoder.latent_dims), device=self.device
        )
        # Tracks which environments have a fresh depth render, so the frozen
        # encoder runs at the camera's 10 Hz rather than the policy's 50 Hz.
        self._last_depth_frame = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        # CHANGE11: nearest surface in the depth image, refreshed on the camera's
        # cadence and held between renders like the latent. Starts at the far
        # plane so the barrier is inactive before the first render.
        self._min_obstacle_dist = torch.full(
            (self.num_envs,), self.cfg.depth_range.max_range, device=self.device
        )
        self._motor_forces_b = torch.zeros(
            (self.num_envs, self._robot.num_bodies, 3), device=self.device
        )
        self._motor_torques_b = torch.zeros_like(self._motor_forces_b)
        self._crashes = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        model = self.cfg.model
        self._root_body_id = self._robot.find_bodies(model.root_body_name, preserve_order=True)[0][0]
        motor_body_ids, motor_names = self._robot.find_bodies(
            list(model.motor_body_names), preserve_order=True
        )
        if tuple(motor_names) != model.motor_body_names:
            raise RuntimeError(
                f"Motor-link order changed: expected {model.motor_body_names}, got {tuple(motor_names)}"
            )
        self._motor_body_ids = torch.tensor(motor_body_ids, dtype=torch.long, device=self.device)

        # Reused unchanged, gains included.
        self._allocator = PX4MultirotorAllocator(
            self.num_envs,
            self.physics_dt,
            self.device,
            config_path=self.cfg.allocation_config_path,
        )
        # PORT: PX4 attitude -> body-rate -> normalized-control loop, replacing the
        # Lee geometric controller.  The rate loop runs every physics tick; the
        # attitude loop runs every attitude_interval-th tick.
        physics_hz = 1.0 / self.physics_dt
        if not math.isclose(physics_hz, self.cfg.rate_control_hz, rel_tol=0.0, abs_tol=1.0e-6):
            raise ValueError(
                f"Physics/rate loop must run at {self.cfg.rate_control_hz:g} Hz, got {physics_hz:g} Hz"
            )
        attitude_interval = round(physics_hz / self.cfg.attitude_control_hz)
        if not math.isclose(
            physics_hz / attitude_interval,
            self.cfg.attitude_control_hz,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        ):
            raise ValueError("attitude_control_hz must divide the physics rate exactly")
        self._controller = PX4AttitudeRateController(
            self.num_envs,
            self.device,
            hover_thrust=abs(self.cfg.px4_hover_thrust),
            min_thrust=abs(self.cfg.px4_max_thrust),
            max_thrust=abs(self.cfg.px4_min_thrust),
            attitude_interval=attitude_interval,
        )

        self._linear_drag = torch.tensor(model.aerodynamic_linear_damping, device=self.device)
        self._quadratic_drag = torch.tensor(model.aerodynamic_quadratic_damping, device=self.device)
        self._angular_drag = torch.tensor(
            model.aerodynamic_angular_linear_damping, device=self.device
        )
        self._angular_quadratic_drag = torch.tensor(
            model.aerodynamic_angular_quadratic_damping, device=self.device
        )

        # Frozen Deep Collision Encoder: lives on the environment, so rl_games
        # never sees its parameters and there is no gradient path into it.
        self._dce_encoder = DceDepthEncoder(self.cfg.dce_encoder, device=self.device)

        self._obstacle_randomizer = ObstacleFieldRandomizer(
            self.cfg.obstacles, self._obstacles, self.scene.env_origins, self.device
        )
        # CHANGE4
        self._box_toggler = BoxToggler(
            self.cfg.obstacles, self._walls, self.scene.env_origins, self.device
        )

        self._curriculum_level = self.cfg.curriculum.min_level
        self._curriculum_progress_fraction = 0.0
        self._success_aggregate = 0
        self._crash_aggregate = 0
        self._timeout_aggregate = 0

        self._setup_markers()

    """
    Scene.
    """

    def _setup_scene(self) -> None:
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        # Grouping Xforms must exist before anything is spawned beneath them.
        create_group_prims()

        self._obstacles = RigidObjectCollection(build_obstacle_collection_cfg(self.cfg.obstacles))
        self.scene.rigid_object_collections["obstacles"] = self._obstacles

        # CHANGE4: boundary walls are a kinematic collection now, so BoxToggler
        # can park them per environment at reset.
        self._walls = RigidObjectCollection(build_wall_collection_cfg(self.cfg.obstacles))
        self.scene.rigid_object_collections["walls"] = self._walls

        self._depth_camera = MultiMeshRayCasterCamera(self.cfg.depth_camera)
        self.scene.sensors["depth_camera"] = self._depth_camera
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions()
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _setup_markers(self) -> None:
        """Viewport-only gizmos; never participate in physics or observations."""
        self._markers_enabled = self.cfg.debug_markers and self.sim.has_gui()
        if not self._markers_enabled:
            return
        frame_marker_cfg = FRAME_MARKER_CFG.copy()
        frame_marker_cfg.prim_path = "/World/Visuals/Starling2MaxDepthAttDeltaBodyFrames"
        frame_marker_cfg.markers["frame"].scale = (0.30, 0.30, 0.30)
        self._body_frame_gizmo = VisualizationMarkers(frame_marker_cfg)
        goal_marker_cfg = VisualizationMarkersCfg(
            prim_path="/World/Visuals/DepthAttDeltaGoalPositions",
            markers={
                "goal": sim_utils.SphereCfg(
                    radius=0.12,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
                )
            },
        )
        self._goal_gizmo = VisualizationMarkers(goal_marker_cfg)
        self._gizmo_indices = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

    """
    Stepping.
    """

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Convert ``[delta_thrust, delta_pitch, delta_yaw]`` into an attitude setpoint.

        The deltas are applied to the *measured* attitude each step, so the
        setpoint tracks the vehicle rather than integrating open-loop. Pitch is
        clamped to ``max_pitch``; yaw wraps and is deliberately unbounded; roll is
        pinned to ``roll_setpoint`` (0.0).
        """
        self._previous_raw_actions[:] = self._raw_actions
        clipped = actions.clamp(-1.0, 1.0)
        self._raw_actions[:] = clipped

        _, pitch, yaw = euler_xyz_from_quat(self._robot.data.root_quat_w)

        # CHANGE13: absolute collective thrust in the PX4 throttle domain. Keep
        # the original CTBR scaling and apply the task-level safe lower clamp.
        #
        # PORT: this value is handed to the controller as-is.  The upstream task
        # follows this with `(collective / 0.13) - 1.0` to reach Aerial Gym's
        # normalised domain, because that is what the Lee controller consumes;
        # the PX4 controller consumes the throttle magnitude itself, so that
        # conversion is dropped.  The commanded thrust is identical either way.
        self._setpoints[:, 0] = torch.clamp(
            self.cfg.collective_thrust_scale * (clipped[:, 0] + 1.0) * 0.5,
            self.cfg.collective_thrust_min,
            self.cfg.collective_thrust_max,
        )
        # --- previous (delta-about-hover) scaling, kept for reference ---------
        # Gave a symmetric +/-0.385 band with hover at a = 0, but only 0.62x-1.38x
        # hover, so the policy could never cut thrust to arrest a descent.
        # delta_thrust = clipped[:, 0] * self.cfg.delta_thrust_scale
        # self._setpoints[:, 0] = -torch.clamp(
        #     self.cfg.px4_hover_thrust + delta_thrust,
        #     self.cfg.px4_min_thrust,
        #     self.cfg.px4_max_thrust,
        # )
        self._setpoints[:, 1] = self.cfg.roll_setpoint
        self._setpoints[:, 2] = torch.clamp(
            pitch + clipped[:, 1] * self.cfg.max_delta_pitch,
            -self.cfg.max_pitch,
            self.cfg.max_pitch,
        )
        self._setpoints[:, 3] = wrap_to_pi(yaw + clipped[:, 2] * self.cfg.max_delta_yaw)

    def _apply_action(self) -> None:
        """PX4 attitude/rate control -> PX4 mixer -> motor model, at the physics rate.

        PORT: the upstream task runs the Lee controller to a body wrench and hands
        that to :meth:`PX4MultirotorAllocator.allocate_wrench`.  Here the PX4
        controller returns *normalized* ``[roll, pitch, yaw, thrust]`` control,
        which goes through the same four-stage mixer chain every other task in
        this repository uses.  That chain models motor lag and per-actuator
        saturation, so collective and torque compete for authority as they do on
        the real vehicle.
        """
        root_quat_wxyz = self._robot.data.root_quat_w
        root_ang_vel_b = quat_apply_inverse(root_quat_wxyz, self._robot.data.root_ang_vel_w)
        control = self._controller.compute(
            self._setpoints,
            root_quat_wxyz,
            root_ang_vel_b,
            self.physics_dt,
        )
        body_lin_vel = quat_apply_inverse(root_quat_wxyz, self._robot.data.root_lin_vel_w)
        actuator_sp = self._allocator.allocate_normalized_control(control)
        motor_values = self._allocator.function_motors(actuator_sp)
        motor_speed_command = self._allocator.mixing_output(motor_values)
        motor_forces_b, motor_torques_b = self._allocator.gazebo_motor_model(
            motor_speed_command, body_velocity_b=body_lin_vel
        )
        self._motor_forces_b.zero_()
        self._motor_torques_b.zero_()
        self._motor_forces_b[:, self._motor_body_ids] = motor_forces_b
        self._motor_torques_b[:, self._motor_body_ids] = motor_torques_b

        # Starling 2 Max's aerodynamic coefficients are zero, so this is a no-op
        # today; the hook is kept generic for parity with the other tasks.
        drag_force = -self._linear_drag * body_lin_vel - self._quadratic_drag * body_lin_vel.norm(
            dim=-1, keepdim=True
        ) * body_lin_vel
        drag_torque = (
            -self._angular_drag * root_ang_vel_b
            - self._angular_quadratic_drag * root_ang_vel_b.abs() * root_ang_vel_b
        )
        self._motor_forces_b[:, self._root_body_id] += drag_force
        self._motor_torques_b[:, self._root_body_id] += drag_torque
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=torch.arange(self._robot.num_bodies, device=self.device),
            forces=self._motor_forces_b,
            torques=self._motor_torques_b,
        )

    """
    Observations.
    """

    def _refresh_image_latents(self) -> None:
        """Re-encode depth only for environments that produced a new frame.

        The camera is configured at ``update_period = 0.1`` while the policy runs
        at 50 Hz, so ``RayCasterCamera.frame`` advances once every five policy
        steps.  Accessing ``.data`` first triggers the lazy ray-cast, after which
        ``frame`` reflects whether this environment actually re-rendered.  A reset
        zeroes ``frame``, which also registers as a change and forces a re-encode.
        """
        depth = self._depth_camera.data.output["distance_to_image_plane"].squeeze(-1)
        frame = self._depth_camera.frame
        fresh = (frame != self._last_depth_frame).nonzero(as_tuple=False).squeeze(-1)
        if fresh.numel() == 0:
            return
        normalized = normalize_depth_image(depth[fresh], self.cfg.depth_range)
        self._image_latents[fresh] = self._dce_encoder.encode(normalized)

        # CHANGE11: nearest surface in metres, on the same 10 Hz cadence. The near
        # sentinel (-1) means "too close to measure", so it is folded onto the far
        # plane rather than read as a negative distance -- matching Aerial Gym's
        # post_image_reward_addition.
        metres = normalized * self.cfg.depth_range.max_range
        metres = torch.where(
            metres < 0.0, torch.full_like(metres, self.cfg.depth_range.max_range), metres
        )
        self._min_obstacle_dist[fresh] = metres.amin(dim=(1, 2))

        self._last_depth_frame[fresh] = frame[fresh]

    def _get_observations(self) -> dict[str, torch.Tensor]:
        self._refresh_image_latents()

        quat_wxyz = self._robot.data.root_quat_w
        roll, pitch, yaw = euler_xyz_from_quat(quat_wxyz)
        vehicle_quat = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)

        # Goal vector in the vehicle (yaw-only) frame, as Aerial Gym does.
        vec_to_target = quat_apply_inverse(
            vehicle_quat, self._target_pos_w - self._robot.data.root_pos_w
        )
        dist_to_target = vec_to_target.norm(dim=-1)

        body_lin_vel = quat_apply_inverse(quat_wxyz, self._robot.data.root_lin_vel_w)
        body_ang_vel = quat_apply_inverse(quat_wxyz, self._robot.data.root_ang_vel_w)

        obs = torch.zeros((self.num_envs, self.cfg.observation_space), device=self.device)
        obs[:, 0:3] = vec_to_target / dist_to_target.clamp_min(1.0e-6).unsqueeze(1)
        obs[:, 3] = dist_to_target / 5.0
        if self.cfg.observe_true_roll_pitch:
            obs[:, 4] = wrap_to_pi(roll)
            obs[:, 5] = wrap_to_pi(pitch)
        obs[:, 6] = 0.0
        obs[:, 7:10] = body_lin_vel
        obs[:, 10:13] = body_ang_vel
        # Raw deltas in [-1, 1]; slot 16 stays zero so the layout matches the
        # velocity navigation task's four action channels.
        obs[:, 13:16] = self._raw_actions
        obs[:, 16] = 0.0
        obs[:, 17:81] = self._image_latents

        update_depth_chase_view(
            self.sim, self._robot, self.cfg.viewer, self.cfg.follow_viewer
        )
        self._visualize()
        return {"policy": obs}

    def _visualize(self) -> None:
        if not self._markers_enabled:
            return
        self._body_frame_gizmo.visualize(
            translations=self._robot.data.root_pos_w,
            orientations=self._robot.data.root_quat_w,
            marker_indices=self._gizmo_indices,
        )
        self._goal_gizmo.visualize(
            translations=self._target_pos_w, marker_indices=self._gizmo_indices
        )

    """
    Reward and termination.
    """

    def _detect_crashes(self) -> torch.Tensor:
        """Any body contact above the force threshold during this policy step."""
        forces = self._contact_sensor.data.net_forces_w_history
        # (N, T, B, 3) -> peak force magnitude over history and bodies.
        peak_force = forces.norm(dim=-1).amax(dim=(1, 2))
        return peak_force > self.cfg.collision_force_threshold

    def _goal_yaw_error(self, position_error_w: torch.Tensor, current_yaw: torch.Tensor) -> torch.Tensor:
        """Wrapped heading error to the goal bearing, zeroed near the goal.

        Ported from the velocity task's ``_goal_yaw_error``.
        """
        desired_yaw = torch.atan2(position_error_w[:, 1], position_error_w[:, 0])
        yaw_error = wrap_to_pi(desired_yaw - current_yaw)
        return torch.where(
            position_error_w[:, :2].norm(dim=-1) <= self.cfg.heading_deadzone_distance,
            torch.zeros_like(yaw_error),
            yaw_error,
        )

    def _get_rewards(self) -> torch.Tensor:
        # DirectRLEnv.step calls _get_dones() before _get_rewards(), so
        # self._crashes is already current for this policy step.
        _, _, yaw = euler_xyz_from_quat(self._robot.data.root_quat_w)
        vehicle_quat = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
        position_error_w = self._target_pos_w - self._robot.data.root_pos_w
        self._pos_error_vehicle_prev[:] = self._pos_error_vehicle
        self._pos_error_vehicle[:] = quat_apply_inverse(vehicle_quat, position_error_w)

        yaw_error = self._goal_yaw_error(position_error_w, yaw)
        velocity_vehicle = quat_apply_inverse(vehicle_quat, self._robot.data.root_lin_vel_w)
        speed = velocity_vehicle.norm(dim=-1)
        inside_goal = position_error_w.norm(dim=-1) < self.cfg.success_distance
        reached_goal = inside_goal & ~self._goal_bonus_awarded
        self._goal_bonus_awarded |= inside_goal

        reward, _ = compute_depth_attitude_delta_reward(
            self._pos_error_vehicle,
            self._pos_error_vehicle_prev,
            self._crashes.float(),
            self._raw_actions,
            self._previous_raw_actions,
            yaw_error,
            speed,
            velocity_vehicle,
            self._min_obstacle_dist,
            reached_goal.float(),
            self.cfg.reward_parameters,
        )
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # CHANGE4: leaving the virtual box counts as a crash. Necessary because a
        # parked box has no floor to stop a descent.
        out_of_bounds = (
            self._robot.data.root_pos_w - self.scene.env_origins
        ).norm(dim=-1) > self.cfg.out_of_bounds_distance
        self._crashes[:] = self._detect_crashes() | out_of_bounds
        timed_out = self.episode_length_buf >= self.max_episode_length - 1
        self._update_curriculum(self._crashes, timed_out)
        return self._crashes.clone(), timed_out

    def _update_curriculum(self, crashes: torch.Tensor, timed_out: torch.Tensor) -> None:
        """Port of ``navigation_task.check_and_update_curriculum_level``.

        Success = timed out within ``success_distance`` of the goal without
        crashing; timeout = timed out otherwise. Crashes are counted separately
        and never double-counted as timeouts.
        """
        curriculum = self.cfg.curriculum
        distance = (self._target_pos_w - self._robot.data.root_pos_w).norm(dim=-1)
        successes = timed_out & (distance < self.cfg.success_distance) & ~crashes
        timeouts = timed_out & ~successes & ~crashes

        self._success_aggregate += int(successes.sum())
        self._crash_aggregate += int(crashes.sum())
        self._timeout_aggregate += int(timeouts.sum())

        instances = self._success_aggregate + self._crash_aggregate + self._timeout_aggregate
        if instances < curriculum.check_after_log_instances:
            return

        success_rate = self._success_aggregate / instances
        if success_rate > curriculum.success_rate_for_increase:
            self._curriculum_level += curriculum.increase_step
        elif success_rate < curriculum.success_rate_for_decrease:
            self._curriculum_level -= curriculum.decrease_step
        self._curriculum_level = min(
            max(self._curriculum_level, curriculum.min_level), curriculum.max_level
        )
        self._curriculum_progress_fraction = (self._curriculum_level - curriculum.min_level) / (
            curriculum.max_level - curriculum.min_level
        )
        self.extras["log"] = {
            "curriculum/level": float(self._curriculum_level),
            "curriculum/progress_fraction": self._curriculum_progress_fraction,
            "curriculum/success_rate": success_rate,
            "curriculum/crash_rate": self._crash_aggregate / instances,
            "curriculum/timeout_rate": self._timeout_aggregate / instances,
        }
        self._success_aggregate = 0
        self._crash_aggregate = 0
        self._timeout_aggregate = 0

    """
    Reset.
    """

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)
        self._robot.reset(env_ids)
        self._allocator.reset(env_ids)
        # PORT: the PX4 loop carries rate-integrator and derivative state; the
        # Lee controller was stateless, so the upstream task has nothing here.
        self._controller.reset(env_ids)
        self._raw_actions[env_ids] = 0.0
        self._previous_raw_actions[env_ids] = 0.0
        self._setpoints[env_ids] = 0.0
        self._goal_bonus_awarded[env_ids] = False

        count = len(env_ids)
        bounds_min = self._robot.data.root_pos_w.new_tensor(self.cfg.obstacles.bounds_min)
        bounds_max = self._robot.data.root_pos_w.new_tensor(self.cfg.obstacles.bounds_max)
        span = bounds_max - bounds_min

        # CHANGE2: the goal is resolved first, because it is the cube centre and
        # the spawn sampler rejects against it.
        target_ratio_min = bounds_min.new_tensor(self.cfg.target_min_ratio)
        target_ratio_max = bounds_min.new_tensor(self.cfg.target_max_ratio)
        target_ratio = target_ratio_min + torch.rand((count, 3), device=self.device) * (
            target_ratio_max - target_ratio_min
        )
        target_local = bounds_min + span * target_ratio
        self._target_pos_w[env_ids] = target_local + self.scene.env_origins[env_ids]

        # Match the shared dense-core layout: start in the obstacle-free shell
        # and face the centre goal with a bounded random yaw offset.
        pos_local, spawn_yaw = sample_depth_navigation_spawn(
            self.cfg.obstacles,
            count,
            target_local,
            self.device,
        )

        # Spawn level so the pitch delta starts from a known attitude.
        euler = torch.zeros((count, 3), device=self.device)
        euler[:, 2] = spawn_yaw
        pose_w = torch.cat(
            (
                pos_local + self.scene.env_origins[env_ids],
                quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2]),
            ),
            dim=-1,
        )
        velocity_w = torch.empty((count, 6), device=self.device)
        velocity_w[:, :3].uniform_(-self.cfg.reset_linear_velocity, self.cfg.reset_linear_velocity)
        velocity_w[:, 3:].uniform_(
            -self.cfg.reset_angular_velocity, self.cfg.reset_angular_velocity
        )
        self._robot.write_root_pose_to_sim(pose_w, env_ids)
        self._robot.write_root_velocity_to_sim(velocity_w, env_ids)

        # Reset the getting-closer baseline so the first post-reset step does not
        # see a spurious distance delta across the teleport.
        _, _, yaw = euler_xyz_from_quat(self._robot.data.root_quat_w[env_ids])
        vehicle_quat = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
        error = quat_apply_inverse(vehicle_quat, target_local - pos_local)
        self._pos_error_vehicle[env_ids] = error
        self._pos_error_vehicle_prev[env_ids] = error

        # Obstacles stay inside the central core and clear of goal and spawn.
        self._obstacle_randomizer.randomize(
            env_ids,
            self._curriculum_level,
            spawn_pos_local=pos_local,
            goal_pos_local=target_local,
        )

        # CHANGE4: sample the box on/off for these environments.
        self._box_toggler.randomize(env_ids, self.cfg.obstacles.box_probability)

    @property
    def box_present(self) -> torch.Tensor:
        """``(num_envs,)`` mask of environments whose boundary walls are in place."""
        return self._box_toggler.present

    @property
    def curriculum_level(self) -> int:
        """Current number of active obstacles."""
        return self._curriculum_level

    @property
    def attitude_setpoints(self) -> torch.Tensor:
        """Latest ``[thrust, roll, pitch, yaw]`` command, ``(num_envs, 4)``."""
        return self._setpoints

    @property
    def depth_image(self) -> torch.Tensor:
        """Latest normalised depth image, ``(num_envs, H, W)`` in ``[-1] U [0, 1]``."""
        depth = self._depth_camera.data.output["distance_to_image_plane"].squeeze(-1)
        return normalize_depth_image(depth, self.cfg.depth_range)

    @property
    def min_obstacle_dist(self) -> torch.Tensor:
        """Nearest surface in the depth image, metres, ``(num_envs,)``.  CHANGE11."""
        return self._min_obstacle_dist

    @property
    def image_latents(self) -> torch.Tensor:
        """Latest frozen-encoder latents, ``(num_envs, 64)``."""
        return self._image_latents
