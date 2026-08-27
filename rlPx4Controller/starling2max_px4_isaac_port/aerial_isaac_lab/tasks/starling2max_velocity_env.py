"""Starling 2 Max velocity-setpoint task using Aerial Gym dynamics on Isaac Lab physics.

Isaac Lab is used only for scene cloning, rigid-body integration, and state
access. Motor dynamics, allocation, Lee velocity control, reward, action
scaling, reset ranges, and policy observation semantics are ported from Aerial
Gym.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from aerial_isaac_lab.core import LeeVelocityController
from aerial_isaac_lab.core.math import (
    euler_xyz_from_quat,
    matrix_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    quat_wxyz_to_xyzw,
    wrap_to_pi,
)
from aerial_isaac_lab.core.motor_px4 import PX4MultirotorAllocator
from aerial_isaac_lab.models.model_cfg import MultirotorModelCfg
from aerial_isaac_lab.models import STARLING2MAX_MODEL


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
STARLING2MAX_ALLOCATION_PATH = str(_PROJECT_ROOT / "config" / "allocation_starling2max.yaml")


def make_multirotor_robot_cfg(model: MultirotorModelCfg) -> ArticulationCfg:
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=model.urdf_path,
            fix_base=False,
            root_link_name=model.root_body_name,
            merge_fixed_joints=False,
            joint_drive=None,
            make_instanceable=True,
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
class Starling2MaxVelocityEnvCfg(DirectRLEnvCfg):
    """Configuration retaining the Aerial Gym Starling 2 Max + Lee velocity constants."""

    model = STARLING2MAX_MODEL
    allocation_config_path = STARLING2MAX_ALLOCATION_PATH

    # Legacy task terminates only when sim_steps > 800, i.e. after 801 steps.
    episode_length_s = 8.01
    decimation = 1
    action_space = 4
    observation_space = 18
    state_space = 0
    seed = 1
    sim: SimulationCfg = SimulationCfg(
        dt=0.01,
        render_interval=1,
        gravity=(0.0, 0.0, -9.81),
        physx=sim_utils.PhysxCfg(
            solver_type=1,
            bounce_threshold_velocity=0.1,
            gpu_max_rigid_contact_count=2**24,
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        # Fabric cloning can fail for this imported URDF on Isaac Sim 5.1 and
        # has no USD-cloning fallback.  Regular USD cloning preserves the
        # same replicated PhysX scene and reliably creates every environment.
        num_envs=16, env_spacing=1.0, replicate_physics=True, clone_in_fabric=False
    )
    robot: ArticulationCfg = make_multirotor_robot_cfg(STARLING2MAX_MODEL)
    # Aerial Gym EmptyEnv5m sampling, in environment-local metres.  The
    # environment is a cube of half-extent 5/sqrt(3) m about each origin
    # (env_bounds_min/max = -/+ this value, matching empty_env_5m env_spacing).
    # The drone spawns uniformly over the full box (Starling 2 Max init ratio 0..1) and
    # the target is sampled with Aerial Gym's ratio range, kept at least
    # minimum_target_distance away, and resampled whenever the drone arrives.
    env_bounds_extent = 5.0 / math.sqrt(3.0)
    target_min_ratio = (0.05, 0.05, 0.10)
    target_max_ratio = (0.95, 0.95, 0.95)
    minimum_target_distance = 0.75
    target_sampling_attempts = 6
    target_reached_distance = 0.20
    resample_target_on_reach = True
    reset_roll_pitch = math.pi / 6.0
    reset_linear_velocity = 0.5
    reset_angular_velocity = 0.2
    crash_distance = 10.0
    observation_position_noise_std = 0.03
    observation_euler_noise_std = 0.02
    observation_linear_velocity_noise_std = 0.02
    observation_angular_velocity_noise_std = 0.02
    # Aerial Gym process_actions_for_task scaling applied to the clipped policy
    # output before it reaches the velocity controller, the reward penalties,
    # and the previous-action observation slot. Linear command scale is the
    # identity (1.0 m/s); yaw-rate command is scaled to +/- pi/5 rad/s.
    action_max_velocity = 1.0
    action_max_yaw_rate = math.pi / 5.0


class Starling2MaxVelocityEnv(DirectRLEnv):
    """Direct-RL port of the Aerial Gym sim2real velocity-setpoint task."""

    cfg: Starling2MaxVelocityEnvCfg

    def __init__(self, cfg: Starling2MaxVelocityEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # Aerial Gym velocity-controller action contract: the policy emits
        # [vx, vy, vz, yaw_rate] in [-1, 1], scaled in _pre_physics_step to a
        # body-FLU velocity (m/s) and yaw rate (rad/s). _actions holds the
        # scaled command, matching Aerial Gym's robot_actions tensor.
        self._actions = torch.zeros((self.num_envs, 4), device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._target_pos_w = self.scene.env_origins.clone()
        self._previous_distance = torch.zeros(self.num_envs, device=self.device)
        self._motor_forces_b = torch.zeros((self.num_envs, self._robot.num_bodies, 3), device=self.device)
        self._motor_torques_b = torch.zeros_like(self._motor_forces_b)

        model = self.cfg.model
        self._root_body_id = self._robot.find_bodies(
            model.root_body_name, preserve_order=True
        )[0][0]
        motor_body_ids, motor_names = self._robot.find_bodies(
            list(model.motor_body_names), preserve_order=True
        )
        if tuple(motor_names) != model.motor_body_names:
            raise RuntimeError(
                f"Motor-link order changed: expected {model.motor_body_names}, got {tuple(motor_names)}"
            )
        self._motor_body_ids = torch.tensor(motor_body_ids, dtype=torch.long, device=self.device)
        self._allocator = PX4MultirotorAllocator(
            self.num_envs,
            self.physics_dt,
            self.device,
            config_path=self.cfg.allocation_config_path,
        )
        self._controller = LeeVelocityController(self.num_envs, self.device)
        self._mass, self._inertia_b = self._compute_aggregate_mass_inertia()

        # Viewport-only body-frame gizmo.  It is a visual point-instancer,
        # never participates in physics or observations.
        frame_marker_cfg = FRAME_MARKER_CFG.copy()
        frame_marker_cfg.prim_path = "/World/Visuals/Starling2MaxBodyFrames"
        frame_marker_cfg.markers["frame"].scale = (0.30, 0.30, 0.30)
        self._body_frame_gizmo = VisualizationMarkers(frame_marker_cfg)
        self._body_frame_gizmo_indices = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # Viewport-only goal-position gizmo: a yellow sphere at the target,
        # mirroring Aerial Gym's target visualization (radius 0.12 m, yellow).
        # Like the body gizmo it is purely visual and never affects physics.
        goal_marker_cfg = VisualizationMarkersCfg(
            prim_path="/World/Visuals/GoalPositions",
            markers={
                "goal": sim_utils.SphereCfg(
                    radius=0.12,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
                )
            },
        )
        self._goal_gizmo = VisualizationMarkers(goal_marker_cfg)
        self._goal_gizmo_indices = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        self._linear_drag = torch.tensor(model.aerodynamic_linear_damping, device=self.device)
        self._quadratic_drag = torch.tensor(model.aerodynamic_quadratic_damping, device=self.device)
        self._angular_drag = torch.tensor(model.aerodynamic_angular_linear_damping, device=self.device)
        self._angular_quadratic_drag = torch.tensor(
            model.aerodynamic_angular_quadratic_damping, device=self.device
        )

    def _setup_scene(self) -> None:
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions()
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._previous_actions[:] = self._actions
        # Mirror Aerial Gym process_actions_for_task: clip to [-1, 1], then
        # scale into a body-FLU velocity command and yaw-rate command. The
        # scaled action is what feeds the controller, the reward penalties,
        # and the previous-action observation slot.
        scaled_actions = actions.clamp(-1.0, 1.0).clone()
        scaled_actions[:, 0:3] *= self.cfg.action_max_velocity
        scaled_actions[:, 3] *= self.cfg.action_max_yaw_rate
        self._actions[:] = scaled_actions
        self._previous_distance[:] = self._position_error_w().norm(dim=-1)

    def _apply_action(self) -> None:
        root_quat_wxyz = self._robot.data.root_quat_w
        root_lin_vel_w = self._robot.data.root_lin_vel_w
        root_ang_vel_b = quat_apply_inverse(root_quat_wxyz, self._robot.data.root_ang_vel_w)
        wrench_b = self._controller.compute(
            self._actions,
            self._robot.data.root_pos_w,
            root_quat_wxyz,
            root_lin_vel_w,
            root_ang_vel_b,
            self._mass,
            self._inertia_b,
        )
        body_lin_vel = quat_apply_inverse(root_quat_wxyz, root_lin_vel_w)
        motor_forces_b, motor_torques_b = self._allocator.allocate_wrench(
            wrench_b, body_velocity_b=body_lin_vel
        )
        self._motor_forces_b.zero_()
        self._motor_torques_b.zero_()
        self._motor_forces_b[:, self._motor_body_ids] = motor_forces_b
        self._motor_torques_b[:, self._motor_body_ids] = motor_torques_b

        # Keep the Aerial Gym aerodynamic-drag hook generic. Starling 2 Max's configured
        # coefficients are zero, so this is numerically identical today.
        drag_force = -self._linear_drag * body_lin_vel - self._quadratic_drag * body_lin_vel.norm(
            dim=-1, keepdim=True
        ) * body_lin_vel
        drag_torque = -self._angular_drag * root_ang_vel_b - self._angular_quadratic_drag * root_ang_vel_b.abs() * root_ang_vel_b
        self._motor_forces_b[:, self._root_body_id] += drag_force
        self._motor_torques_b[:, self._root_body_id] += drag_torque
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=torch.arange(self._robot.num_bodies, device=self.device),
            forces=self._motor_forces_b,
            torques=self._motor_torques_b,
        )

    def _get_observations(self) -> dict[str, torch.Tensor]:
        self._body_frame_gizmo.visualize(
            translations=self._robot.data.root_pos_w,
            orientations=self._robot.data.root_quat_w,
            marker_indices=self._body_frame_gizmo_indices,
        )
        # Aerial Gym resamples a new target once the drone reaches the current
        # one (after the reward, before the observation), giving the multi-leg
        # waypoint chase the deployment relies on.
        self._resample_reached_targets()
        self._goal_gizmo.visualize(
            translations=self._target_pos_w,
            marker_indices=self._goal_gizmo_indices,
        )
        pos_error_w = self._position_error_w()
        quat_wxyz = self._robot.data.root_quat_w
        roll, pitch, yaw = euler_xyz_from_quat(quat_wxyz)
        noisy_quat_wxyz = quat_from_euler_xyz(
            roll + torch.randn_like(roll) * self.cfg.observation_euler_noise_std,
            pitch + torch.randn_like(pitch) * self.cfg.observation_euler_noise_std,
            yaw + torch.randn_like(yaw) * self.cfg.observation_euler_noise_std,
        )
        body_lin_vel = quat_apply_inverse(quat_wxyz, self._robot.data.root_lin_vel_w)
        body_ang_vel = quat_apply_inverse(quat_wxyz, self._robot.data.root_ang_vel_w)
        # Intentional observation-only extension requested for the velocity
        # policy: wrapped bearing error to the position target.
        yaw_error = self._goal_yaw_error(pos_error_w, yaw)
        obs = torch.cat(
            (
                pos_error_w + torch.randn_like(pos_error_w) * self.cfg.observation_position_noise_std,
                quat_wxyz_to_xyzw(noisy_quat_wxyz),
                body_lin_vel + torch.randn_like(body_lin_vel) * self.cfg.observation_linear_velocity_noise_std,
                body_ang_vel + torch.randn_like(body_ang_vel) * self.cfg.observation_angular_velocity_noise_std,
                # Aerial Gym observes obs["robot_actions"], i.e. the scaled
                # action applied this step (most-recent), not the prior one.
                self._actions,
                (yaw_error / math.pi).unsqueeze(-1),
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # Formula ported from Aerial Gym's position_setpoint_task_sim2real.
        pos_error_w = self._position_error_w()
        distance = pos_error_w.norm(dim=-1)
        body_lin_vel = quat_apply_inverse(self._robot.data.root_quat_w, self._robot.data.root_lin_vel_w)
        _, _, yaw = euler_xyz_from_quat(self._robot.data.root_quat_w)
        # Aerial Gym rewards aligning heading with the bearing to the target,
        # zeroed within target_reached_distance horizontally.
        yaw_error = self._goal_yaw_error(pos_error_w, yaw)
        pos_reward = (
            self._exp(distance, 2.0, 1.0)
            + self._exp(distance, 3.0, 10.0)
            + self._abs_exp(distance, 3.0, 50.0)
        )
        speed_reward = self._exp(body_lin_vel.norm(dim=-1), 1.0, 3.0)
        action_penalty = self._abs_exp_penalty(self._actions, 0.2, 4.0).sum(dim=-1)
        action_difference_penalty = self._abs_exp_penalty(
            self._actions - self._previous_actions, 0.3, 6.0
        ).sum(dim=-1)
        closer_reward = 400.0 * (self._previous_distance - distance)
        yaw_reward = self._abs_exp(yaw_error, 2.0, 3.0)
        reward = (
            pos_reward
            + (20.0 - distance) / 40.0
            + pos_reward * (speed_reward + action_penalty + closer_reward / 10.0)
            + action_penalty
            + action_difference_penalty
            + closer_reward
            + yaw_reward
        )
        return torch.where(distance > self.cfg.crash_distance, torch.full_like(reward, -50.0), reward)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = self._position_error_w().norm(dim=-1) > self.cfg.crash_distance
        timed_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, timed_out

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)
        self._robot.reset(env_ids)
        self._allocator.reset(env_ids)
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0

        count = len(env_ids)
        extent = self.cfg.env_bounds_extent
        # Aerial Gym Starling 2 Max init_config: position ratio 0..1 over the env box,
        # i.e. uniform in [-extent, +extent] per axis.
        pos_local = torch.empty((count, 3), device=self.device).uniform_(-extent, extent)
        euler = torch.empty((count, 3), device=self.device)
        euler[:, :2].uniform_(-self.cfg.reset_roll_pitch, self.cfg.reset_roll_pitch)
        euler[:, 2].uniform_(-math.pi, math.pi)
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

        # Sample an independent target inside the env box, >= minimum_target
        # distance from the spawn (Aerial Gym _sample_target_positions).
        target_local = self._sample_local_targets(pos_local)
        self._target_pos_w[env_ids] = target_local + self.scene.env_origins[env_ids]
        self._previous_distance[env_ids] = (target_local - pos_local).norm(dim=-1)

    def _position_error_w(self) -> torch.Tensor:
        return self._target_pos_w - self._robot.data.root_pos_w

    def _goal_yaw_error(self, position_error_w: torch.Tensor, current_yaw: torch.Tensor) -> torch.Tensor:
        desired_yaw = torch.atan2(position_error_w[:, 1], position_error_w[:, 0])
        yaw_error = wrap_to_pi(desired_yaw - current_yaw)
        return torch.where(
            position_error_w[:, :2].norm(dim=-1) <= self.cfg.target_reached_distance,
            torch.zeros_like(yaw_error),
            yaw_error,
        )

    def _resample_reached_targets(self) -> None:
        """Resample targets for environments that have reached the current one."""
        if not self.cfg.resample_target_on_reach:
            return
        distance = self._position_error_w().norm(dim=-1)
        reached = (distance < self.cfg.target_reached_distance).nonzero(as_tuple=False).squeeze(-1)
        if reached.numel() == 0:
            return
        robot_local = self._robot.data.root_pos_w[reached] - self.scene.env_origins[reached]
        self._target_pos_w[reached] = (
            self._sample_local_targets(robot_local) + self.scene.env_origins[reached]
        )

    def _sample_local_targets(self, robot_pos_local: torch.Tensor) -> torch.Tensor:
        """Port of Aerial Gym ``_sample_target_positions`` in env-local metres.

        Ratio-interpolate a target inside the env box, keep it at least
        ``minimum_target_distance`` from the drone across a few resampling
        attempts, then fall back to the farther extreme corner if the drone and
        target are still too close.
        """
        count = robot_pos_local.shape[0]
        extent = self.cfg.env_bounds_extent
        bounds_min = robot_pos_local.new_full((count, 3), -extent)
        bounds_max = robot_pos_local.new_full((count, 3), extent)
        ratio_min = robot_pos_local.new_tensor(self.cfg.target_min_ratio).expand(count, -1)
        ratio_max = robot_pos_local.new_tensor(self.cfg.target_max_ratio).expand(count, -1)

        def interpolate(ratio: torch.Tensor) -> torch.Tensor:
            return bounds_min + (bounds_max - bounds_min) * ratio

        def random_target() -> torch.Tensor:
            ratio = ratio_min + torch.rand((count, 3), device=self.device) * (ratio_max - ratio_min)
            return interpolate(ratio)

        target = random_target()
        for _ in range(self.cfg.target_sampling_attempts):
            too_close = (target - robot_pos_local).norm(dim=-1) < self.cfg.minimum_target_distance
            target = torch.where(too_close.unsqueeze(-1), random_target(), target)

        too_close = (target - robot_pos_local).norm(dim=-1) < self.cfg.minimum_target_distance
        min_corner = interpolate(ratio_min)
        max_corner = interpolate(ratio_max)
        use_max_corner = (max_corner - robot_pos_local).norm(dim=-1) > (
            min_corner - robot_pos_local
        ).norm(dim=-1)
        fallback = torch.where(use_max_corner.unsqueeze(-1), max_corner, min_corner)
        return torch.where(too_close.unsqueeze(-1), fallback, target)

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
        identity = torch.eye(3, device=self.device).expand(self.num_envs, self._robot.num_bodies, -1, -1)
        parallel_axis = masses[..., None, None] * (
            offset.square().sum(dim=-1)[..., None, None] * identity - offset[..., :, None] * offset[..., None, :]
        )
        return total_mass, (inertia_root + parallel_axis).sum(dim=1)

    @staticmethod
    def _exp(value: torch.Tensor, gain: float, exponent: float) -> torch.Tensor:
        return gain * torch.exp(-exponent * value.square())

    @staticmethod
    def _abs_exp(value: torch.Tensor, gain: float, exponent: float) -> torch.Tensor:
        return gain * torch.exp(-exponent * value.abs())

    @staticmethod
    def _abs_exp_penalty(value: torch.Tensor, gain: float, exponent: float) -> torch.Tensor:
        return gain * (torch.exp(-exponent * value.abs()) - 1.0)
