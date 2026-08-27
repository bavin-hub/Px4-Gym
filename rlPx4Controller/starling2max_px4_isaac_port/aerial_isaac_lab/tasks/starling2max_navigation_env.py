"""Starling 2 Max depth-image navigation task.

Port of Aerial Gym's ``navigation_task`` / ``dce_navigation_task`` -- the
Deep Collision Encoder policy from *Reinforcement Learning for Collision-free
Flight Exploiting Deep Collision Encoding* (Kulkarni & Alexis, arXiv:2402.03947)
-- onto Isaac Lab physics with the Starling 2 Max airframe.

What is preserved from Aerial Gym:

* the **81-D observation**: goal unit vector (3) + normalised distance (1) +
  roll/pitch (2) + reserved slot (1) + body linear velocity (3) + body angular
  velocity (3) + previous velocity command (4) + frozen depth latent (64);
* the **3-D action** mapped to a ``[vx, 0, vz]`` body-FLU velocity command plus
  measured-yaw-relative delta yaw (max speed 1.2 m/s, inclination <= pi/4,
  yaw delta <= 20 degrees);
* the **reward**, verbatim from ``navigation_task.compute_reward``, including the
  curriculum-scaled multiplier and the ``-100`` collision penalty;
* the **obstacle course** and its 15 -> 50 obstacle-count curriculum;
* the **frozen Deep Collision Encoder**, run under ``no_grad`` inside the
  environment so no gradient path to it exists.

What is Isaac-Lab-native here:

* depth comes from :class:`~isaaclab.sensors.MultiMeshRayCasterCamera` (Warp
  ray-casting, per-environment mesh sets) instead of Aerial Gym's Warp kernels;
* collisions come from a :class:`~isaaclab.sensors.ContactSensor` instead of
  Isaac Gym net contact forces;
* obstacles are a :class:`~isaaclab.assets.RigidObjectCollection` pool with
  curriculum-gated pose randomisation instead of Aerial Gym's asset manager.

The navigation pipeline is unchanged above the controller boundary.  The
policy, reward and depth render run at 10 Hz while a Torch-native PX4 cascade
runs the velocity loop at 50 Hz, attitude at 250 Hz, and rate/motor/physics at
1 kHz.  Nothing in this module is shared with the existing waypoint tasks.
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
from isaaclab.sensors import ContactSensor, ContactSensorCfg, MultiMeshRayCasterCamera, MultiMeshRayCasterCameraCfg
from isaaclab.sensors.ray_caster.patterns import PinholeCameraPatternCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from aerial_isaac_lab.core import PX4AttitudeRateController, PX4VelocityController
from aerial_isaac_lab.core.math import (
    euler_xyz_from_quat,
    matrix_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
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

from .depth_navigation_layout import (
    RESET_ANGULAR_VELOCITY,
    RESET_LINEAR_VELOCITY,
    TARGET_RATIO,
    DepthNavigationCurriculumCfg,
    make_depth_navigation_obstacle_cfg,
    make_depth_navigation_viewer_cfg,
    sample_depth_navigation_spawn,
    update_depth_chase_view,
)
from .depth_attitude_delta_reward import compute_depth_attitude_delta_reward
from .navigation_reward import compute_navigation_reward

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


def make_navigation_robot_cfg(model: MultirotorModelCfg) -> ArticulationCfg:
    """Robot spawn config for the navigation task.

    A task-local copy of the velocity task's factory -- the existing tasks each
    keep their own copy -- differing only by ``activate_contact_sensors=True``,
    which must be set at spawn time for the contact sensor to report anything.
    Keeping it local means the position/velocity/attitude-delta tasks do not pay
    for contact reporting.
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
            # Required for the ContactSensor below; not set by the other tasks.
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
        # These robots have only fixed or unmanaged rotor joints. Empty
        # mappings prevent Isaac Lab's default joint-state regex from applying.
        init_state=ArticulationCfg.InitialStateCfg(joint_pos={}, joint_vel={}),
        actuators={},
    )


@configclass
class Starling2MaxNavigationEnvCfg(DirectRLEnvCfg):
    """Configuration for the Starling 2 Max depth navigation task."""

    model = STARLING2MAX_MODEL
    allocation_config_path = STARLING2MAX_ALLOCATION_PATH

    # PX4 multi-rate cascade.  Physics and the rate loop share the 1 kHz base
    # tick; attitude refreshes every 4 ticks and velocity every 20 ticks.
    rate_control_hz = 1000.0
    attitude_control_hz = 250.0
    velocity_control_hz = 50.0
    px4_hover_thrust = 0.13
    px4_min_thrust = 0.04
    px4_max_thrust = 0.60
    velocity_gain = (1.5, 1.5, 1.5)
    max_velocity_acceleration = 4.0

    # 10 Hz policy on top of the 1 kHz PX4 rate loop.  Aerial Gym's
    # episode_len_steps = 100 truncates after 101 policy steps, which
    # ceil(10.1 / 0.1) = 101 reproduced.
    #
    # CHANGE9: extended to 16.1 s (161 policy steps) because CHANGE9 drops the
    # speed cap to 1.2 m/s. Worst-case budget: a 180 deg turn at the pi/3 yaw-rate
    # cap (3.0 s) + 8.4 m transit at 1.2 m/s (7.0 s) + ~40% detour overhead
    # (2.8 s) = 12.8 s, so 10.1 s could not be completed from a far spawn facing
    # away from the goal.
    # To revert CHANGE9: episode_length_s = 10.1.
    decimation = 100
    episode_length_s = 16.1

    # 3 actions (forward speed, inclination, delta yaw) -> velocity + yaw setpoint.
    action_space = 3
    # 13 state + 4 previous action + 64 latent.
    observation_space = 81
    state_space = 0
    seed = 1

    sim: SimulationCfg = SimulationCfg(
        dt=0.001,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.81),
        physx=sim_utils.PhysxCfg(
            solver_type=1,
            bounce_threshold_velocity=0.1,
            gpu_max_rigid_contact_count=2**24,
        ),
    )
    # env_spacing only has to keep the *physics* of neighbouring environments
    # apart: MultiMeshRayCaster keeps a per-environment mesh list, so depth never
    # bleeds across environments regardless of spacing.  The obstacle box spans
    # 11 m in x, so 14 m leaves a comfortable margin.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=16, env_spacing=14.0, replicate_physics=True, clone_in_fabric=False
    )
    robot: ArticulationCfg = make_navigation_robot_cfg(STARLING2MAX_MODEL)
    obstacles: ObstacleFieldCfg = make_depth_navigation_obstacle_cfg()

    viewer: ViewerCfg = make_depth_navigation_viewer_cfg()
    follow_viewer = True

    depth_camera: MultiMeshRayCasterCameraCfg = MultiMeshRayCasterCameraCfg(
        prim_path="/World/envs/env_.*/Robot/vehicle",
        # One ray-cast per policy step. Isaac Lab calls scene.update() once per
        # *physics* step, so without this the depth image would be recomputed 100x
        # per policy step.
        update_period=0.1,
        max_distance=DEPTH_MAX_RANGE,
        # Maps ray misses (nan) and out-of-range hits onto the far plane, which is
        # the first of Aerial Gym's three range steps. The near sentinel and the
        # normalisation are applied by normalize_depth_image().
        depth_clipping_behavior="max",
        data_types=["distance_to_image_plane"],
        # Aerial Gym BaseDepthCameraConfig.nominal_position, in the robot body
        # (FLU) frame. The "world" convention is +X forward / +Z up, i.e. the body
        # frame itself, so no extra frame rotation is needed.
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

    # history_length > 0 makes SensorBase refresh eagerly on every scene.update(),
    # i.e. every physics step. With one slot per decimated substep we can ask
    # "did this robot touch anything at any point during the policy step?" rather
    # than only sampling the final substep.
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

    # Shared across all depth tasks: centre goal, perimeter-shell spawn, and
    # goal-bearing yaw with +/-30 degrees of random offset.
    target_min_ratio = TARGET_RATIO
    target_max_ratio = TARGET_RATIO
    reset_linear_velocity = RESET_LINEAR_VELOCITY
    reset_angular_velocity = RESET_ANGULAR_VELOCITY
    success_distance = 1.0
    """Distance below which a timed-out episode counts as a success."""

    # navigation_task_config.action_transformation_function
    # Magnitude of the vehicle-frame (yaw-only) velocity setpoint, so it caps the
    # commanded world speed. The Lee controller is proportional, so actual speed
    # can still overshoot this.
    # CHANGE9: 1.2 m/s for slower navigation, down from Aerial Gym's 2.0. Paired
    # with the longer episode below, which the slower transit requires.
    # NOTE: gamma stays at 0.98 by request. At 10 Hz that is a 50-step (5 s)
    # effective horizon, about 6 m at 1.2 m/s, while spawns reach 8.4 m from the
    # goal -- so the far end of the spawn range starts outside the value horizon.
    # Raise gamma to ~0.99 if early learning stalls on distant spawns.
    # To revert CHANGE9: action_max_speed = 2.0, episode_length_s = 10.1.
    action_max_speed = 1.2
    action_max_inclination_angle = math.pi / 4.0
    max_delta_yaw = math.radians(20.0)

    # EnvWithObstaclesCfg.env.collision_force_threshold. Isaac Gym and PhysX
    # report contact magnitudes differently, so expect to recalibrate this by
    # flying into a wall and printing the sensor reading.
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

    Aerial Gym's ``dce_navigation_task`` reads these from the *vehicle* (yaw-only)
    orientation, so in the original they are structurally zero.  Since the policy
    is retrained here, feeding the real attitude is strictly more informative and
    keeps the 81-D layout unchanged.  Set to ``False`` to reproduce the original
    contract bit-for-bit (required if you ever load the paper's checkpoint).
    """

    debug_markers = True
    """Draw goal/body-frame gizmos. Only active when a GUI is attached."""

    heading_deadzone_distance = 0.3

    # --- SHARED DEPTH REWARD PARAMETERS (disabled; retained for rollback) --
    # reward_parameters = {
    #     "progress_reward_multiplier": 1.0,
    #     "yaw_alignment_reward_magnitude": 0.05,
    #     "yaw_alignment_reward_exponent": 1.0,
    #     "cruise_speed": 0.7,
    #     "brake_radius": 0.8,
    #     "velocity_tracking_magnitude": 0.5,
    #     "velocity_tracking_exponent": 2.0,
    #     "stop_reward_magnitude": 1.0,
    #     "stop_reward_exponent": 5.0,
    #     "dwell_distance": 0.2,
    #     "dwell_speed": 0.1,
    #     "dwell_reward": 5.0,
    #     "action_penalty_multiplier": 0.005,
    #     "action_diff_penalty_multiplier": 0.005,
    #     "vertical_error_penalty_multiplier": 0.01,
    #     "proximity_penalty_multiplier": 0.02,
    #     "proximity_softener": 0.2,
    #     "goal_reached_reward": 5.0,
    #     "collision_penalty": -4.0,
    # }

    # Legacy depth-velocity reward ported from starling_aerial_isaac.
    reward_parameters = {
        "pos_reward_magnitude": 5.0,
        "pos_reward_exponent": 1.0 / 3.5,
        "very_close_to_goal_reward_magnitude": 5.0,
        "very_close_to_goal_reward_exponent": 2.0,
        "getting_closer_reward_multiplier": 10.0,
        "cruise_speed": 0.7,
        "brake_radius": 0.4,
        "velocity_tracking_magnitude": 0.5,
        "velocity_tracking_exponent": 2.0,
        "x_action_diff_penalty_magnitude": 0.8,
        "x_action_diff_penalty_exponent": 3.333,
        "z_action_diff_penalty_magnitude": 0.8,
        "z_action_diff_penalty_exponent": 5.0,
        "delta_yaw_action_diff_penalty_magnitude": 0.8,
        "delta_yaw_action_diff_penalty_exponent": 3.33,
        "x_absolute_action_penalty_magnitude": 0.1,
        "x_absolute_action_penalty_exponent": 0.3,
        "z_absolute_action_penalty_magnitude": 1.5,
        "z_absolute_action_penalty_exponent": 1.0,
        "delta_yaw_absolute_action_penalty_magnitude": 1.5,
        "delta_yaw_absolute_action_penalty_exponent": 2.0,
        "collision_penalty": -100.0,
    }


class Starling2MaxNavigationEnv(DirectRLEnv):
    """Direct-RL port of the Aerial Gym Deep Collision Encoder navigation task."""

    cfg: Starling2MaxNavigationEnvCfg

    def __init__(
        self, cfg: Starling2MaxNavigationEnvCfg, render_mode: str | None = None, **kwargs
    ):
        super().__init__(cfg, render_mode, **kwargs)

        # ``_actions`` holds [vx, 0, vz, raw_delta_yaw]. The fourth channel stays
        # normalized for observation/reward while ``_yaw_setpoints`` stores the
        # absolute measured-yaw-relative heading command.
        self._actions = torch.zeros((self.num_envs, 4), device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._reward_actions = torch.zeros((self.num_envs, 3), device=self.device)
        self._previous_reward_actions = torch.zeros_like(self._reward_actions)
        self._yaw_setpoints = torch.zeros(self.num_envs, device=self.device)
        self._target_pos_w = self.scene.env_origins.clone()
        self._pos_error_vehicle = torch.zeros((self.num_envs, 3), device=self.device)
        self._pos_error_vehicle_prev = torch.zeros_like(self._pos_error_vehicle)
        self._image_latents = torch.zeros(
            (self.num_envs, self.cfg.dce_encoder.latent_dims), device=self.device
        )
        self._motor_forces_b = torch.zeros(
            (self.num_envs, self._robot.num_bodies, 3), device=self.device
        )
        self._motor_torques_b = torch.zeros_like(self._motor_forces_b)
        self._crashes = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._min_obstacle_dist = torch.full(
            (self.num_envs,), self.cfg.depth_range.max_range, device=self.device
        )
        self._goal_bonus_awarded = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

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

        physics_hz = 1.0 / self.physics_dt
        if not math.isclose(physics_hz, self.cfg.rate_control_hz, rel_tol=0.0, abs_tol=1.0e-6):
            raise ValueError(
                f"Physics/rate loop must run at {self.cfg.rate_control_hz:g} Hz, got {physics_hz:g} Hz"
            )
        self._velocity_interval = round(physics_hz / self.cfg.velocity_control_hz)
        attitude_interval = round(physics_hz / self.cfg.attitude_control_hz)
        if not math.isclose(
            physics_hz / self._velocity_interval,
            self.cfg.velocity_control_hz,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        ):
            raise ValueError("velocity_control_hz must divide the physics rate exactly")
        if not math.isclose(
            physics_hz / attitude_interval,
            self.cfg.attitude_control_hz,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        ):
            raise ValueError("attitude_control_hz must divide the physics rate exactly")

        # The allocator and motor model are the same ones used by the existing
        # Python PX4 attitude task.  The new velocity stage is a batched Torch
        # port of SimplePositionController's CTRL_VEL_ONLY equations.
        self._allocator = PX4MultirotorAllocator(
            self.num_envs,
            self.physics_dt,
            self.device,
            config_path=self.cfg.allocation_config_path,
        )
        self._velocity_controller = PX4VelocityController(
            self.num_envs,
            self.device,
            hover_thrust=self.cfg.px4_hover_thrust,
            velocity_gain=self.cfg.velocity_gain,
            max_acceleration=self.cfg.max_velocity_acceleration,
        )
        self._controller = PX4AttitudeRateController(
            self.num_envs,
            self.device,
            hover_thrust=self.cfg.px4_hover_thrust,
            min_thrust=self.cfg.px4_min_thrust,
            max_thrust=self.cfg.px4_max_thrust,
            attitude_interval=attitude_interval,
        )
        self._attitude_thrust_sp = torch.zeros((self.num_envs, 4), device=self.device)
        self._attitude_thrust_sp[:, 0] = self.cfg.px4_hover_thrust
        self._control_tick = 0

        self._linear_drag = torch.tensor(model.aerodynamic_linear_damping, device=self.device)
        self._quadratic_drag = torch.tensor(model.aerodynamic_quadratic_damping, device=self.device)
        self._angular_drag = torch.tensor(
            model.aerodynamic_angular_linear_damping, device=self.device
        )
        self._angular_quadratic_drag = torch.tensor(
            model.aerodynamic_angular_quadratic_damping, device=self.device
        )

        # Frozen Deep Collision Encoder. It lives on the environment, not in the
        # policy network, so rl_games never sees these parameters.
        self._dce_encoder = DceDepthEncoder(self.cfg.dce_encoder, device=self.device)

        self._obstacle_randomizer = ObstacleFieldRandomizer(
            self.cfg.obstacles, self._obstacles, self.scene.env_origins, self.device
        )
        # CHANGE4
        self._box_toggler = BoxToggler(
            self.cfg.obstacles, self._walls, self.scene.env_origins, self.device
        )

        # Curriculum state, mirroring navigation_task.check_and_update_curriculum_level.
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
        frame_marker_cfg.prim_path = "/World/Visuals/Starling2MaxNavBodyFrames"
        frame_marker_cfg.markers["frame"].scale = (0.30, 0.30, 0.30)
        self._body_frame_gizmo = VisualizationMarkers(frame_marker_cfg)
        goal_marker_cfg = VisualizationMarkersCfg(
            prim_path="/World/Visuals/NavGoalPositions",
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
        self._previous_actions[:] = self._actions
        self._previous_reward_actions[:] = self._reward_actions
        clamped = actions.clamp(-1.0, 1.0)
        # Convert the forward channel from [-1, 1] to an effort fraction in
        # [0, 1]; inclination and delta yaw remain centred at zero.
        self._reward_actions[:, 0] = (clamped[:, 0] + 1.0) * 0.5
        self._reward_actions[:, 1:] = clamped[:, 1:]
        self._actions[:] = self._transform_actions(clamped)

        _, _, yaw = euler_xyz_from_quat(self._robot.data.root_quat_w)
        self._yaw_setpoints[:] = wrap_to_pi(
            yaw + clamped[:, 2] * self.cfg.max_delta_yaw
        )

    def _transform_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Port of ``navigation_task_config.action_transformation_function``.

        The 3-D policy output becomes ``[vx, 0, vz, raw_delta_yaw]``: channel 0
        is a forward speed offset into
        ``[0, 2]``, channel 1 tilts the velocity vector in the x-z plane, and
        channel 2 remains normalized for measured-yaw-relative integration.
        There is no lateral velocity command.
        """
        clamped = actions.clamp(-1.0, 1.0).clone()
        clamped[:, 0] += 1.0

        processed = torch.zeros((clamped.shape[0], 4), device=self.device)
        processed[:, 0] = (
            clamped[:, 0]
            * torch.cos(self.cfg.action_max_inclination_angle * clamped[:, 1])
            * self.cfg.action_max_speed
            / 2.0
        )
        processed[:, 1] = 0.0
        processed[:, 2] = (
            clamped[:, 0]
            * torch.sin(self.cfg.action_max_inclination_angle * clamped[:, 1])
            * self.cfg.action_max_speed
            / 2.0
        )
        processed[:, 3] = clamped[:, 2]
        return processed

    def _apply_action(self) -> None:
        """Run velocity/attitude/rate at 50/250/1000 Hz, respectively."""
        root_quat_wxyz = self._robot.data.root_quat_w
        root_lin_vel_w = self._robot.data.root_lin_vel_w
        root_ang_vel_b = quat_apply_inverse(root_quat_wxyz, self._robot.data.root_ang_vel_w)

        # The 10 Hz policy command is held.  Recompute its velocity-to-attitude
        # conversion every 20 physics ticks (50 Hz) and hold roll/pitch/thrust in
        # between. The policy's delta yaw has already been integrated against
        # measured yaw into an absolute setpoint for the attitude loop.
        if self._control_tick % self._velocity_interval == 0:
            self._attitude_thrust_sp[:] = self._velocity_controller.compute(
                self._actions, root_quat_wxyz, root_lin_vel_w
            )
        self._attitude_thrust_sp[:, 3] = self._yaw_setpoints
        self._controller.yawspeed_setpoint.zero_()

        control = self._controller.compute(
            self._attitude_thrust_sp,
            root_quat_wxyz,
            root_ang_vel_b,
            self.physics_dt,
        )
        self._control_tick += 1

        body_lin_vel = quat_apply_inverse(root_quat_wxyz, root_lin_vel_w)
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
        # today; the hook is kept generic to match the velocity task.
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

    def _encode_depth(self) -> torch.Tensor:
        """Render depth, apply Aerial Gym range semantics, and encode it.

        The sensor is configured with ``update_period = 0.1``, so accessing
        ``.data`` here triggers exactly one ray-cast per policy step.
        """
        depth = self._depth_camera.data.output["distance_to_image_plane"]
        # (num_envs, H, W, 1) -> (num_envs, H, W)
        depth = depth.squeeze(-1)
        normalized = normalize_depth_image(depth, self.cfg.depth_range)
        metres = normalized * self.cfg.depth_range.max_range
        metres = torch.where(
            metres < 0.0, torch.full_like(metres, self.cfg.depth_range.max_range), metres
        )
        self._min_obstacle_dist[:] = metres.amin(dim=(1, 2))
        return self._dce_encoder.encode(normalized)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        self._image_latents[:] = self._encode_depth()

        quat_wxyz = self._robot.data.root_quat_w
        roll, pitch, yaw = euler_xyz_from_quat(quat_wxyz)
        vehicle_quat = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)

        # Goal vector expressed in the vehicle (yaw-only) frame, as Aerial Gym does.
        vec_to_target = quat_apply_inverse(vehicle_quat, self._target_pos_w - self._robot.data.root_pos_w)
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
        obs[:, 13:17] = self._actions
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

    def _goal_yaw_error(
        self, position_error_w: torch.Tensor, current_yaw: torch.Tensor
    ) -> torch.Tensor:
        """Wrapped goal-bearing error, zeroed inside the arrival deadzone."""
        desired_yaw = torch.atan2(position_error_w[:, 1], position_error_w[:, 0])
        yaw_error = wrap_to_pi(desired_yaw - current_yaw)
        return torch.where(
            position_error_w[:, :2].norm(dim=-1) <= self.cfg.heading_deadzone_distance,
            torch.zeros_like(yaw_error),
            yaw_error,
        )

    # --- SHARED DEPTH REWARD (disabled; retained for rollback) -------------
    # def _get_rewards(self) -> torch.Tensor:
    #     _, _, yaw = euler_xyz_from_quat(self._robot.data.root_quat_w)
    #     vehicle_quat = quat_from_euler_xyz(
    #         torch.zeros_like(yaw), torch.zeros_like(yaw), yaw
    #     )
    #     position_error_w = self._target_pos_w - self._robot.data.root_pos_w
    #     self._pos_error_vehicle_prev[:] = self._pos_error_vehicle
    #     self._pos_error_vehicle[:] = quat_apply_inverse(vehicle_quat, position_error_w)
    #
    #     yaw_error = self._goal_yaw_error(position_error_w, yaw)
    #     velocity_vehicle = quat_apply_inverse(vehicle_quat, self._robot.data.root_lin_vel_w)
    #     speed = velocity_vehicle.norm(dim=-1)
    #     inside_goal = position_error_w.norm(dim=-1) < self.cfg.success_distance
    #     reached_goal = inside_goal & ~self._goal_bonus_awarded
    #     self._goal_bonus_awarded |= inside_goal
    #
    #     reward, _ = compute_depth_attitude_delta_reward(
    #         self._pos_error_vehicle,
    #         self._pos_error_vehicle_prev,
    #         self._crashes.float(),
    #         self._reward_actions,
    #         self._previous_reward_actions,
    #         yaw_error,
    #         speed,
    #         velocity_vehicle,
    #         self._min_obstacle_dist,
    #         reached_goal.float(),
    #         self.cfg.reward_parameters,
    #     )
    #     return reward

    def _get_rewards(self) -> torch.Tensor:
        """Legacy depth-velocity reward ported from starling_aerial_isaac."""
        _, _, yaw = euler_xyz_from_quat(self._robot.data.root_quat_w)
        vehicle_quat = quat_from_euler_xyz(
            torch.zeros_like(yaw), torch.zeros_like(yaw), yaw
        )
        self._pos_error_vehicle_prev[:] = self._pos_error_vehicle
        self._pos_error_vehicle[:] = quat_apply_inverse(
            vehicle_quat, self._target_pos_w - self._robot.data.root_pos_w
        )
        velocity_vehicle = quat_apply_inverse(
            vehicle_quat, self._robot.data.root_lin_vel_w
        )
        reward, _ = compute_navigation_reward(
            self._pos_error_vehicle,
            self._pos_error_vehicle_prev,
            self._crashes.float(),
            self._actions,
            self._previous_actions,
            velocity_vehicle,
            self._curriculum_progress_fraction,
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
        crashing; timeout = timed out otherwise.  Crashes are counted separately
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
        self._controller.reset(env_ids)
        self._controller.yawspeed_setpoint[env_ids] = 0.0
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._reward_actions[env_ids] = 0.0
        self._previous_reward_actions[env_ids] = 0.0
        self._min_obstacle_dist[env_ids] = self.cfg.depth_range.max_range
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

        # Shared dense-core layout: spawn in the obstacle-free perimeter shell
        # and point toward the centre goal with a bounded random yaw offset.
        pos_local, spawn_yaw = sample_depth_navigation_spawn(
            self.cfg.obstacles,
            count,
            target_local,
            self.device,
        )

        euler = torch.zeros((count, 3), device=self.device)
        euler[:, 2] = spawn_yaw
        self._yaw_setpoints[env_ids] = spawn_yaw
        pose_w = torch.cat(
            (
                pos_local + self.scene.env_origins[env_ids],
                quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2]),
            ),
            dim=-1,
        )
        velocity_w = torch.empty((count, 6), device=self.device)
        velocity_w[:, :3].uniform_(-self.cfg.reset_linear_velocity, self.cfg.reset_linear_velocity)
        velocity_w[:, 3:].uniform_(-self.cfg.reset_angular_velocity, self.cfg.reset_angular_velocity)
        self._robot.write_root_pose_to_sim(pose_w, env_ids)
        self._robot.write_root_velocity_to_sim(velocity_w, env_ids)
        self._attitude_thrust_sp[env_ids, 0] = self.cfg.px4_hover_thrust
        self._attitude_thrust_sp[env_ids, 1:3] = 0.0
        self._attitude_thrust_sp[env_ids, 3] = euler[:, 2]

        # Reset the getting-closer baseline so the first post-reset step does not
        # see a spurious distance delta across the teleport.
        _, _, yaw = euler_xyz_from_quat(self._robot.data.root_quat_w[env_ids])
        vehicle_quat = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
        error = quat_apply_inverse(vehicle_quat, target_local - pos_local)
        self._pos_error_vehicle[env_ids] = error
        self._pos_error_vehicle_prev[env_ids] = error

        # CHANGE2: obstacles keep clear of both the goal and the spawn point.
        self._obstacle_randomizer.randomize(
            env_ids,
            self._curriculum_level,
            spawn_pos_local=pos_local,
            goal_pos_local=target_local,
        )

        # CHANGE4: sample the box on/off for these environments.
        self._box_toggler.randomize(env_ids, self.cfg.obstacles.box_probability)

    """
    Helpers.
    """

    def _compute_aggregate_mass_inertia(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute a root-frame aggregate inertia using the parallel-axis theorem."""
        # Isaac Lab stores imported physical defaults on CPU even when the
        # articulation state buffers are on CUDA.
        masses = self._robot.data.default_mass.to(self.device)
        inertias = self._robot.data.default_inertia.to(self.device)
        if inertias.shape[-1] == 9:
            inertias = inertias.reshape(*inertias.shape[:-1], 3, 3)
        root_pos = self._robot.data.root_pos_w
        root_quat = self._robot.data.root_quat_w
        root_quat_per_body = root_quat[:, None, :].expand_as(self._robot.data.body_link_quat_w)
        body_pos = quat_apply_inverse(
            root_quat_per_body, self._robot.data.body_link_pos_w - root_pos[:, None, :]
        )
        link_quat_b = quat_mul(
            quat_inv(root_quat)[:, None, :].expand_as(self._robot.data.body_link_quat_w),
            self._robot.data.body_link_quat_w,
        )
        body_com_pos = body_pos + quat_apply(link_quat_b, self._robot.data.body_com_pos_b)
        body_com_quat = quat_mul(link_quat_b, self._robot.data.body_com_quat_b)
        rotation = matrix_from_quat(body_com_quat)
        inertia_root = rotation @ inertias @ rotation.transpose(-2, -1)
        total_mass = masses.sum(dim=1)
        com = (masses.unsqueeze(-1) * body_com_pos).sum(dim=1) / total_mass.unsqueeze(-1)
        offset = body_com_pos - com[:, None, :]
        identity = torch.eye(3, device=self.device).expand(
            self.num_envs, self._robot.num_bodies, -1, -1
        )
        parallel_axis = masses[..., None, None] * (
            offset.square().sum(dim=-1)[..., None, None] * identity
            - offset[..., :, None] * offset[..., None, :]
        )
        return total_mass, (inertia_root + parallel_axis).sum(dim=1)

    @property
    def box_present(self) -> torch.Tensor:
        """``(num_envs,)`` mask of environments whose boundary walls are in place."""
        return self._box_toggler.present

    @property
    def curriculum_level(self) -> int:
        """Current number of active obstacles."""
        return self._curriculum_level

    @property
    def depth_image(self) -> torch.Tensor:
        """Latest normalised depth image, ``(num_envs, H, W)`` in ``[-1] U [0, 1]``."""
        depth = self._depth_camera.data.output["distance_to_image_plane"].squeeze(-1)
        return normalize_depth_image(depth, self.cfg.depth_range)

    @property
    def image_latents(self) -> torch.Tensor:
        """Latest frozen-encoder latents, ``(num_envs, 64)``."""
        return self._image_latents

    @property
    def yaw_setpoints(self) -> torch.Tensor:
        """Latest absolute measured-yaw-relative heading targets, ``(num_envs,)``."""
        return self._yaw_setpoints
