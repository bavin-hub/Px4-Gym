"""SITL deployment node for the Starling 2 Max depth-velocity policy."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import yaml
from rl_games.algos_torch import model_builder

import rclpy
from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleGlobalPosition,
    VehicleOdometry,
    VehicleStatus,
)
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Image


def _default_training_root() -> Path:
    """Find the checked-out Isaac task package when using --symlink-install."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "rlPx4Controller" / "starling2max_px4_isaac_port"
        if candidate.is_dir():
            return candidate
    return Path.cwd() / "rlPx4Controller" / "starling2max_px4_isaac_port"


def _require_file(value: str, parameter: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not value or not path.is_file():
        raise FileNotFoundError(f"ROS parameter {parameter!r} is not a file: {value!r}")
    return path

# The training environment uses dt=0.01 and decimation=10: depth encoding,
# policy inference, and policy command updates therefore all run at 10 Hz.
CONTROL_FREQUENCY_HZ = 10.0
CONTROL_PERIOD_SECONDS = 1.0 / CONTROL_FREQUENCY_HZ

MIN_HEIGHT_M = 1.0
MAX_HEIGHT_M = 4.0

OBS_DIM = 81
ACTION_DIM = 3
LATENT_DIM = 64

# Starling 2 Max C29 forward ToF (PMD M0178), matching the training env's
# CHANGE10: 240x180, HFOV 86 deg, VFOV 106 deg, 0.2-6 m. The max range is what
# normalize_depth_image divides by, so it has to track the training value or
# every pixel is scaled wrong.
DEPTH_MIN_METERS = 0.2
DEPTH_MAX_METERS = 6.0
ENCODER_IMAGE_SIZE = (270, 480)

# Exact Starling2MaxNavigationEnvCfg action transformation values.
ACTION_CLIP = 1.0
ACTION_MAX_SPEED = 1.2
ACTION_MAX_INCLINATION = math.pi / 4.0
ACTION_MAX_DELTA_YAW = math.radians(20.0)


def load_model(
    device: torch.device,
    policy_config_path: Path,
    policy_checkpoint_path: Path,
) -> nn.Module:
    """Build the training GRU policy and restore its rl_games checkpoint."""
    with policy_config_path.open(encoding="utf-8") as stream:
        params = yaml.safe_load(stream)["params"]

    network = model_builder.ModelBuilder().load(params)
    config = params["config"]
    model = network.build(
        {
            "actions_num": ACTION_DIM,
            "input_shape": (OBS_DIM,),
            "num_seqs": 1,
            "value_size": 1,
            "normalize_value": config.get("normalize_value", False),
            "normalize_input": config.get("normalize_input", False),
        }
    )

    checkpoint = torch.load(
        policy_checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    weights = {
        key.removeprefix("_orig_mod."): value
        for key, value in checkpoint["model"].items()
    }
    model.load_state_dict(weights)
    model.to(device)
    model.eval()
    return model


class OffboardControl(Node):
    """Evaluate the depth policy and send velocity setpoints to PX4."""

    def __init__(self) -> None:
        super().__init__("depth_velocity_sp")

        training_root = Path(
            self.declare_parameter(
                "training_root", str(_default_training_root())
            ).value
        ).expanduser().resolve()
        default_config = training_root / "config" / "rl_games_starling2max_navigation.yaml"
        default_encoder = (
            training_root
            / "aerial_isaac_lab"
            / "perception"
            / "weights"
            / "ICRA_test_set_more_sim_data_kld_beta_3_LD_64_epoch_49.pth"
        )
        policy_config_path = _require_file(
            self.declare_parameter("policy_config", str(default_config)).value,
            "policy_config",
        )
        policy_checkpoint_path = _require_file(
            self.declare_parameter("checkpoint", "").value,
            "checkpoint",
        )
        depth_encoder_path = _require_file(
            self.declare_parameter("encoder_checkpoint", str(default_encoder)).value,
            "encoder_checkpoint",
        )
        self.depth_topic = self.declare_parameter(
            "depth_topic", "/starling/raw_depth"
        ).value
        default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        device_name = self.declare_parameter("device", default_device).value
        self.device = torch.device(device_name)
        self.model = load_model(
            self.device,
            policy_config_path,
            policy_checkpoint_path,
        )
        self.rnn_states = tuple(
            state.to(self.device)
            for state in self.model.get_default_rnn_state()
        )

        # Import perception from the selected training tree so deployment and
        # training always use the same encoder implementation.
        sys.path.insert(0, str(training_root))
        from aerial_isaac_lab.perception import (  # noqa: PLC0415
            DceDepthEncoder,
            DceEncoderCfg,
            DepthRangeCfg,
            normalize_depth_image,
        )

        self._normalize_depth_image = normalize_depth_image

        # These are the defaults used by Starling2MaxNavigationEnvCfg.  In
        # particular, training samples the DCE posterior rather than taking its
        # mean, so return_sampled_latent intentionally remains True here.
        encoder_config = DceEncoderCfg(
            latent_dims=LATENT_DIM,
            model_file=str(depth_encoder_path),
            image_res=ENCODER_IMAGE_SIZE,
            interpolation_mode="nearest",
            return_sampled_latent=True,
        )
        self.depth_range = DepthRangeCfg(
            min_range=DEPTH_MIN_METERS,
            max_range=DEPTH_MAX_METERS,
        )
        self.depth_encoder = DceDepthEncoder(
            encoder_config,
            device=self.device,
        )
        self.get_logger().info(
            f"Policy and depth encoder loaded on {self.device}; "
            f"control rate is {CONTROL_FREQUENCY_HZ:.0f} Hz"
        )

        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        depth_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode,
            "/fmu/in/offboard_control_mode",
            px4_qos,
        )
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint,
            "/fmu/in/trajectory_setpoint",
            px4_qos,
        )
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand,
            "/fmu/in/vehicle_command",
            px4_qos,
        )

        self.vehicle_odometry_subscriber = self.create_subscription(
            VehicleOdometry,
            "/fmu/out/vehicle_odometry",
            self.vehicle_odometry_callback,
            px4_qos,
        )
        self.vehicle_global_position_subscriber = self.create_subscription(
            VehicleGlobalPosition,
            "/fmu/out/vehicle_global_position",
            self.vehicle_global_position_callback,
            px4_qos,
        )
        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus,
            "/fmu/out/vehicle_status",
            self.vehicle_status_callback,
            px4_qos,
        )
        self.depth_subscriber = self.create_subscription(
            Image,
            self.depth_topic,
            self.depth_callback,
            depth_qos,
        )

        self.vehicle_odometry = VehicleOdometry()
        self.vehicle_global_position = VehicleGlobalPosition()
        self.vehicle_status = VehicleStatus()
        self.takeoff_height = -3.0

        # Ten metres straight ahead of the default yaw-zero spawn (+NED X).
        self.waypoints = [[8.0, 0.0, self.takeoff_height]]
        self.curr_wp = self.waypoints[0]
        self.wp_idx = 0
        self.nn_eval = True

        # Training places the previous *processed* command
        # [vx_flu, vy_flu, vz_flu, raw_delta_yaw] in observation slots 13:17.
        self.previous_command = torch.zeros(
            4, dtype=torch.float32, device=self.device
        )
        self.latest_depth = torch.empty(
            0, dtype=torch.float32, device=self.device
        )
        self.depth_received = False
        self.depth_error_reported = False

        self.home_lat = None
        self.home_lon = None
        self.home_alt = None
        self.home_lock = False

        self.timer = self.create_timer(
            CONTROL_PERIOD_SECONDS,
            self.timer_callback,
        )

    @staticmethod
    def quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Multiply quaternions stored as [w, x, y, z]."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return torch.stack(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ]
        )

    @staticmethod
    def inverse_rotate(q: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        """Rotate a world-frame vector into the quaternion's body frame."""
        q_vector = q[1:4]
        return (
            vector
            - 2.0 * q[0] * torch.cross(q_vector, vector, dim=0)
            + 2.0
            * torch.cross(
                q_vector,
                torch.cross(q_vector, vector, dim=0),
                dim=0,
            )
        )

    def vehicle_state_in_policy_frames(self):
        """Return state represented in Isaac's world-NWU/body-FLU frames."""
        odom = self.vehicle_odometry
        if odom.pose_frame != VehicleOdometry.POSE_FRAME_NED:
            raise ValueError(f"Unsupported pose frame: {odom.pose_frame}")

        position_ned = torch.tensor(
            odom.position, dtype=torch.float32, device=self.device
        )
        q_ned_frd = torch.tensor(
            odom.q, dtype=torch.float32, device=self.device
        )
        q_ned_frd /= torch.linalg.vector_norm(q_ned_frd).clamp_min(1.0e-8)

        # Left and right 180-degree X rotations change NED/FRD to NWU/FLU.
        frame_flip = torch.tensor(
            [0.0, 1.0, 0.0, 0.0],
            dtype=torch.float32,
            device=self.device,
        )
        q_nwu_flu = self.quat_multiply(
            self.quat_multiply(frame_flip, q_ned_frd),
            frame_flip,
        )
        q_nwu_flu /= torch.linalg.vector_norm(q_nwu_flu).clamp_min(1.0e-8)
        if q_nwu_flu[0] < 0.0:
            q_nwu_flu = -q_nwu_flu

        qw, qx, qy, qz = q_nwu_flu
        roll = torch.atan2(
            2.0 * (qw * qx + qy * qz),
            1.0 - 2.0 * (qx * qx + qy * qy),
        )
        pitch = torch.asin(
            torch.clamp(2.0 * (qw * qy - qz * qx), -1.0, 1.0)
        )
        yaw = torch.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )

        velocity = torch.tensor(
            odom.velocity, dtype=torch.float32, device=self.device
        )
        if odom.velocity_frame == VehicleOdometry.VELOCITY_FRAME_NED:
            velocity_frd = self.inverse_rotate(q_ned_frd, velocity)
        elif odom.velocity_frame == VehicleOdometry.VELOCITY_FRAME_BODY_FRD:
            velocity_frd = velocity
        else:
            raise ValueError(
                f"Unsupported velocity frame: {odom.velocity_frame}"
            )
        body_linear_velocity = torch.stack(
            [velocity_frd[0], -velocity_frd[1], -velocity_frd[2]]
        )

        angular_frd = torch.tensor(
            odom.angular_velocity,
            dtype=torch.float32,
            device=self.device,
        )
        body_angular_velocity = torch.stack(
            [angular_frd[0], -angular_frd[1], -angular_frd[2]]
        )
        return (
            position_ned,
            q_ned_frd,
            roll,
            pitch,
            yaw,
            body_linear_velocity,
            body_angular_velocity,
        )

    def create_observation(self, curr_wp) -> torch.Tensor:
        """Build the training pipeline's exact 81-D policy observation."""
        (
            position_ned,
            _q_ned_frd,
            roll,
            pitch,
            yaw,
            body_linear_velocity,
            body_angular_velocity,
        ) = self.vehicle_state_in_policy_frames()

        waypoint_ned = torch.tensor(
            curr_wp, dtype=torch.float32, device=self.device
        )
        error_ned = waypoint_ned - position_ned
        error_nwu = torch.stack(
            [error_ned[0], -error_ned[1], -error_ned[2]]
        )

        # The training goal vector is rotated by yaw only, not full attitude.
        cosine_yaw = torch.cos(yaw)
        sine_yaw = torch.sin(yaw)
        error_vehicle = torch.stack(
            [
                cosine_yaw * error_nwu[0] + sine_yaw * error_nwu[1],
                -sine_yaw * error_nwu[0] + cosine_yaw * error_nwu[1],
                error_nwu[2],
            ]
        )
        goal_distance = torch.linalg.vector_norm(error_vehicle)
        goal_direction = error_vehicle / goal_distance.clamp_min(1.0e-6)

        # Training renders, normalizes, and encodes depth once per policy step.
        normalized_depth = self._normalize_depth_image(
            self.latest_depth.unsqueeze(0),
            self.depth_range,
        )
        depth_latent = self.depth_encoder.encode(normalized_depth)[0]

        observation = torch.zeros(
            OBS_DIM, dtype=torch.float32, device=self.device
        )
        observation[0:3] = goal_direction
        observation[3] = goal_distance / 5.0
        observation[4] = torch.atan2(torch.sin(roll), torch.cos(roll))
        observation[5] = torch.atan2(torch.sin(pitch), torch.cos(pitch))
        observation[6] = 0.0
        observation[7:10] = body_linear_velocity
        observation[10:13] = body_angular_velocity
        observation[13:17] = self.previous_command
        observation[17:81] = depth_latent
        return observation.unsqueeze(0)

    @staticmethod
    def transform_actions(actions: torch.Tensor) -> torch.Tensor:
        """Apply the training action clip and body-FLU command scaling."""
        clamped = actions.clamp(-ACTION_CLIP, ACTION_CLIP).clone()
        clamped[:, 0] += 1.0

        processed = actions.new_zeros((clamped.shape[0], 4))
        processed[:, 0] = (
            clamped[:, 0]
            * torch.cos(ACTION_MAX_INCLINATION * clamped[:, 1])
            * ACTION_MAX_SPEED
            / 2.0
        )
        processed[:, 1] = 0.0
        processed[:, 2] = (
            clamped[:, 0]
            * torch.sin(ACTION_MAX_INCLINATION * clamped[:, 1])
            * ACTION_MAX_SPEED
            / 2.0
        )
        processed[:, 3] = clamped[:, 2]
        return processed

    def body_flu_to_ned(
        self, command: torch.Tensor
    ) -> tuple[float, float, float, float]:
        """Convert the training command into a PX4 NED velocity setpoint."""
        q = torch.tensor(
            self.vehicle_odometry.q,
            dtype=command.dtype,
            device=command.device,
        )
        q /= torch.linalg.vector_norm(q).clamp_min(1.0e-8)
        qw, qx, qy, qz = q
        yaw_ned = torch.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )

        vx_flu, vy_flu, vz_flu, raw_delta_yaw = command
        cosine_yaw = torch.cos(yaw_ned)
        sine_yaw = torch.sin(yaw_ned)

        # The training Lee controller rotates velocity with vehicle yaw only.
        velocity_north = cosine_yaw * vx_flu + sine_yaw * vy_flu
        velocity_east = sine_yaw * vx_flu - cosine_yaw * vy_flu
        velocity_down = -vz_flu
        # Training adds the scaled delta to measured NWU yaw. NED yaw has the
        # opposite sign, so the equivalent absolute PX4 setpoint subtracts it.
        yaw_setpoint_ned = yaw_ned - raw_delta_yaw * ACTION_MAX_DELTA_YAW
        yaw_setpoint_ned = torch.atan2(
            torch.sin(yaw_setpoint_ned), torch.cos(yaw_setpoint_ned)
        )
        values = torch.stack(
            [
                velocity_north,
                velocity_east,
                velocity_down,
                yaw_setpoint_ned,
            ]
        )
        return tuple(float(value) for value in values.detach().cpu())

    def policy_eval(self, curr_wp) -> tuple[float, float, float, float]:
        observation = self.create_observation(curr_wp)
        with torch.no_grad():
            result = self.model(
                {
                    "is_train": False,
                    "prev_actions": None,
                    "obs": observation,
                    "rnn_states": self.rnn_states,
                }
            )
        self.rnn_states = result["rnn_states"]
        processed = self.transform_actions(result["mus"])[0]

        # This processed 4-D command, not the raw actor output or the converted
        # NED setpoint, is what the next training observation contains.
        self.previous_command = processed.detach().clone()
        return self.body_flu_to_ned(processed)

    @staticmethod
    def image_to_meters(message: Image) -> np.ndarray:
        """Decode single-channel ROS depth encodings without cv_bridge."""
        encoding = message.encoding.lower()
        if encoding == "32fc1":
            scalar_type = np.dtype(">f4" if message.is_bigendian else "<f4")
            scale = 1.0
        elif encoding == "64fc1":
            scalar_type = np.dtype(">f8" if message.is_bigendian else "<f8")
            scale = 1.0
        elif encoding in {"16uc1", "mono16"}:
            scalar_type = np.dtype(">u2" if message.is_bigendian else "<u2")
            scale = 0.001
        else:
            raise ValueError(
                f"Unsupported depth encoding {message.encoding!r}; expected "
                "32FC1, 64FC1, 16UC1, or mono16"
            )

        if message.step % scalar_type.itemsize != 0:
            raise ValueError(
                f"Depth row step {message.step} is not aligned to "
                f"{scalar_type.itemsize}-byte pixels"
            )
        row_width = message.step // scalar_type.itemsize
        if row_width < message.width:
            raise ValueError("Depth image row step is smaller than its width")

        expected_size = message.height * message.step
        if len(message.data) < expected_size:
            raise ValueError(
                f"Depth payload has {len(message.data)} bytes; "
                f"expected {expected_size}"
            )
        image = np.frombuffer(
            message.data,
            dtype=scalar_type,
            count=message.height * row_width,
        )
        image = image.reshape(message.height, row_width)[:, : message.width]
        return image.astype(np.float32, copy=True) * scale

    def depth_callback(self, message: Image) -> None:
        """Hold the newest metric depth for the next 10 Hz policy step."""
        try:
            depth = self.image_to_meters(message)
            self.latest_depth = torch.from_numpy(depth).to(self.device)
            if not self.depth_received:
                self.get_logger().info(
                    f"Receiving depth on {self.depth_topic}: "
                    f"{message.width}x{message.height} {message.encoding}"
                )
            self.depth_received = True
            self.depth_error_reported = False
        except (RuntimeError, ValueError) as error:
            if not self.depth_error_reported:
                self.get_logger().error(
                    f"Failed to decode depth image: {error}"
                )
                self.depth_error_reported = True

    def update_waypoint(self, curr_wp) -> None:
        threshold = 1.0
        vehicle_position = self.vehicle_odometry.position
        distance = sum(
            (curr_wp[index] - vehicle_position[index]) ** 2
            for index in range(3)
        ) ** 0.5
        if distance <= threshold:
            self.wp_idx = (self.wp_idx + 1) % len(self.waypoints)
            self.curr_wp = self.waypoints[self.wp_idx]
        print(f"distance to waypoint {self.wp_idx}: {distance:.3f}")

    def vehicle_odometry_callback(self, message: VehicleOdometry) -> None:
        self.vehicle_odometry = message

    def vehicle_global_position_callback(
        self, message: VehicleGlobalPosition
    ) -> None:
        self.vehicle_global_position = message
        if message.lat > 0.0 and not self.home_lock:
            self.home_lat = message.lat
            self.home_lon = message.lon
            self.home_alt = message.alt
            self.home_lock = True

    def vehicle_status_callback(self, message: VehicleStatus) -> None:
        self.vehicle_status = message

    def arm(self) -> None:
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            param1=1.0,
        )
        self.get_logger().info("Arm command sent")

    def takeoff(self) -> None:
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_NAV_TAKEOFF,
            param5=self.home_lat,
            param6=self.home_lon,
            param7=self.home_alt - self.takeoff_height,
        )
        self.get_logger().info("Takeoff command sent")

    def disarm(self) -> None:
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            param1=0.0,
        )
        self.get_logger().info("Disarm command sent")

    def engage_offboard_mode(self) -> None:
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
            param1=1.0,
            param2=6.0,
        )
        self.get_logger().info("Switching to offboard mode")

    def land(self) -> None:
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info("Switching to land mode")

    def publish_offboard_control_heartbeat_signal(self) -> None:
        msg = OffboardControlMode()
        msg.position = False
        msg.velocity = True
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)

    def publish_velocity_setpoint(
        self,
        velocity_north: float,
        velocity_east: float,
        velocity_down: float,
        yaw_setpoint_ned: float,
    ) -> None:
        """Publish world-NED velocity and absolute NED yaw commands to PX4."""
        velocity_down = self.apply_vertical_geofence(velocity_down)
        nan = float("nan")
        msg = TrajectorySetpoint()
        msg.position = [nan, nan, nan]
        msg.velocity = [velocity_north, velocity_east, velocity_down]
        msg.acceleration = [nan, nan, nan]
        msg.jerk = [nan, nan, nan]
        msg.yaw = yaw_setpoint_ned
        msg.yawspeed = nan
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)

    def apply_vertical_geofence(self, velocity_down: float) -> float:
        """Block commands that would move outside the configured height range."""
        height = -float(self.vehicle_odometry.position[2])
        if height >= MAX_HEIGHT_M:
            return max(velocity_down, 0.0)
        if height <= MIN_HEIGHT_M:
            return min(velocity_down, 0.0)
        return velocity_down

    def publish_vehicle_command(self, command, **params) -> None:
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = params.get("param1", 0.0)
        msg.param2 = params.get("param2", 0.0)
        msg.param3 = params.get("param3", 0.0)
        msg.param4 = params.get("param4", 0.0)
        msg.param5 = params.get("param5", 0.0)
        msg.param6 = params.get("param6", 0.0)
        msg.param7 = params.get("param7", 0.0)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_publisher.publish(msg)

    def timer_callback(self) -> None:
        """Run the offboard state machine and policy at exactly 10 Hz."""
        self.publish_offboard_control_heartbeat_signal()

        if not self.home_lock:
            return

        if (
            self.vehicle_status.arming_state
            != VehicleStatus.ARMING_STATE_ARMED
        ):
            self.arm()
        elif self.vehicle_status.takeoff_time == 0:
            self.takeoff()
        elif (
            self.vehicle_odometry.position[-1]
            <= self.takeoff_height + 1.0
            and self.vehicle_status.nav_state
            != VehicleStatus.NAVIGATION_STATE_OFFBOARD
        ):
            if self.depth_received:
                self.engage_offboard_mode()
        elif (
            self.vehicle_status.nav_state
            == VehicleStatus.NAVIGATION_STATE_OFFBOARD
        ):
            if self.nn_eval and self.depth_received:
                velocity = self.policy_eval(self.curr_wp)
            else:
                velocity = (0.0, 0.0, 0.0, 0.0)
            self.publish_velocity_setpoint(*velocity)
            self.update_waypoint(self.curr_wp)


def main(args=None) -> None:
    """Start the depth-velocity policy offboard-control node."""
    print("Starting depth-velocity policy offboard control node...")
    rclpy.init(args=args)
    offboard_control = OffboardControl()
    try:
        rclpy.spin(offboard_control)
    finally:
        offboard_control.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
