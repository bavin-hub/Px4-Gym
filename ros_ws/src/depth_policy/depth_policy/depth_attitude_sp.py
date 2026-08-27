"""SITL deployment node for the Starling 2 Max depth-attitude policy."""

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
    HoverThrustEstimate,
    OffboardControlMode,
    VehicleAttitudeSetpoint,
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


def load_model(
    device: torch.device,
    policy_config_path: Path,
    policy_checkpoint_path: Path,
) -> nn.Module:
    """Build the rl_games GRU policy and restore its checkpoint."""
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
    """Node for controlling a vehicle in offboard mode."""

    def __init__(self) -> None:
        super().__init__("depth_attitude_sp")

        training_root = Path(
            self.declare_parameter(
                "training_root", str(_default_training_root())
            ).value
        ).expanduser().resolve()
        default_config = (
            training_root
            / "config"
            / "rl_games_starling2max_depth_attitude_delta.yaml"
        )
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

        sys.path.insert(0, str(training_root))
        from aerial_isaac_lab.perception import (  # noqa: PLC0415
            DceDepthEncoder,
            DceEncoderCfg,
            DepthRangeCfg,
            normalize_depth_image,
        )

        self._normalize_depth_image = normalize_depth_image
        depth_encoder_config = DceEncoderCfg(
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
            depth_encoder_config,
            device=self.device,
        )
        self.rnn_states = tuple(
            state.to(self.device)
            for state in self.model.get_default_rnn_state()
        )
        self.get_logger().info(
            f"Policy and depth encoder loaded on {self.device}"
        )

        # Configure QoS profile for PX4 publishing and subscribing.
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        # Camera publishers are normally volatile. Requesting transient-local
        # here would make the subscription incompatible with them.
        depth_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Create publishers.
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, "/fmu/in/offboard_control_mode", qos_profile
        )
        self.vehicle_attitude_setpoint_publisher = self.create_publisher(
            VehicleAttitudeSetpoint,
            "/fmu/in/vehicle_attitude_setpoint",
            qos_profile,
        )
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, "/fmu/in/vehicle_command", qos_profile
        )

        # Create subscribers.
        self.vehicle_odometry_subscriber = self.create_subscription(
            VehicleOdometry,
            "/fmu/out/vehicle_odometry",
            self.vehicle_odometry_callback,
            qos_profile,
        )
        self.vehicle_global_position_subscriber = self.create_subscription(
            VehicleGlobalPosition,
            "/fmu/out/vehicle_global_position",
            self.vehicle_global_position_callback,
            qos_profile,
        )
        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus,
            "/fmu/out/vehicle_status",
            self.vehicle_status_callback,
            qos_profile,
        )
        self.vehicle_hover_thrust_subscriber = self.create_subscription(
            HoverThrustEstimate,
            "/fmu/out/hover_thrust_estimate",
            self.vehicle_hover_callback,
            qos_profile,
        )
        self.depth_subscriber = self.create_subscription(
            Image,
            self.depth_topic,
            self.depth_callback,
            depth_qos_profile,
        )

        # Initialize variables.
        self.vehicle_odometry = VehicleOdometry()
        self.vehicle_global_position = VehicleGlobalPosition()
        self.vehicle_status = VehicleStatus()
        self.vehicle_hover_thrust = HoverThrustEstimate()
        self.takeoff_height = -3.0

        self.waypoints = [[0.0, 5.0, self.takeoff_height]]
        # self.waypoints = [
        #     [4.0, 2.0, self.takeoff_height],
        #     [-4.0, 2.0, self.takeoff_height],
        #     [-4.0, -2.0, self.takeoff_height],
        #     [4.0, -2.0, self.takeoff_height],
        # ]
        # Observation slots 13:16 contain the previous clipped raw policy
        # deltas: [delta_thrust, delta_pitch, delta_yaw].
        self.prev_action = torch.zeros(
            3, dtype=torch.float32, device=self.device
        )
        self.current_euler_flu = torch.zeros(
            3, dtype=torch.float32, device=self.device
        )
        self.depth_latent = torch.zeros(
            LATENT_DIM, dtype=torch.float32, device=self.device
        )
        self.depth_received = False
        self.depth_error_reported = False
        self.curr_wp = self.waypoints[0]
        self.wp_idx = 0
        self.nn_eval = True

        # Home position.
        self.home_lat, self.home_lon, self.home_alt = None, None, None
        self.home_lock = False

        # Must match the training task's policy rate exactly: the GRU hidden state
        # and the per-step pitch/yaw deltas are both per-step quantities, so
        # running faster than trained accumulates attitude twice as fast.
        # The task is 50 Hz (decimation = 20 on its 1 kHz physics tick).
        # Previously 0.01 (100 Hz), matching the task before its 50 Hz change.
        self.timer = self.create_timer(0.02, self.timer_callback)

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

    def create_observation(self, curr_wp) -> torch.Tensor:
        """Build the exact 81-D depth-navigation policy observation."""
        odom = self.vehicle_odometry
        if odom.pose_frame != VehicleOdometry.POSE_FRAME_NED:
            raise ValueError(f"Unsupported pose frame: {odom.pose_frame}")

        position_ned = torch.tensor(
            odom.position, dtype=torch.float32, device=self.device
        )
        waypoint_ned = torch.tensor(
            curr_wp, dtype=torch.float32, device=self.device
        )
        error_ned = waypoint_ned - position_ned
        position_error_nwu = torch.stack(
            [error_ned[0], -error_ned[1], -error_ned[2]]
        )

        q_ned_frd = torch.tensor(
            odom.q, dtype=torch.float32, device=self.device
        )
        q_ned_frd /= torch.linalg.vector_norm(q_ned_frd).clamp_min(1.0e-8)
        frame_flip = torch.tensor(
            [0.0, 1.0, 0.0, 0.0],
            dtype=torch.float32,
            device=self.device,
        )
        q_nwu_flu = self.quat_multiply(
            self.quat_multiply(frame_flip, q_ned_frd), frame_flip
        )
        q_nwu_flu /= torch.linalg.vector_norm(q_nwu_flu).clamp_min(1.0e-8)
        if q_nwu_flu[0] < 0:
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
        self.current_euler_flu = torch.stack([roll, pitch, yaw])

        cosine_yaw = torch.cos(yaw)
        sine_yaw = torch.sin(yaw)
        error_vehicle = torch.stack(
            [
                cosine_yaw * position_error_nwu[0]
                + sine_yaw * position_error_nwu[1],
                -sine_yaw * position_error_nwu[0]
                + cosine_yaw * position_error_nwu[1],
                position_error_nwu[2],
            ]
        )
        goal_distance = torch.linalg.vector_norm(error_vehicle)
        goal_direction = error_vehicle / goal_distance.clamp_min(1.0e-6)

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
        observation[13:16] = self.prev_action
        observation[16] = 0.0
        observation[17:81] = self.depth_latent
        return observation.unsqueeze(0)

    def scale_and_clip(self, action: torch.Tensor) -> torch.Tensor:
        """Convert raw deltas into a PX4 attitude/thrust setpoint."""
        clipped_action = torch.clamp(action, -1.0, 1.0).clone()

        # Every constant below mirrors Starling2MaxDepthAttitudeDeltaEnvCfg. They
        # are duplicated here rather than imported because importing the task cfg
        # would pull in isaaclab; keep them in step whenever the task changes.
        px4_hover_thrust = -0.13
        collective_thrust_scale = 0.2054361567635904
        collective_thrust_min = 0.10
        collective_thrust_max = 0.18489254108723135
        max_delta_pitch = math.radians(2.0)
        max_pitch = math.radians(8.0)
        max_delta_yaw = math.radians(20.0)

        # Absolute collective thrust, replacing the old delta-about-hover form
        #     px4_thrust = clamp(-0.13 + a*0.05, -0.18, -0.08)
        # which is NOT the same mapping: it centred hover at a = 0 and spanned
        # 0.62x-1.38x hover, whereas the task centres hover at a = +0.266 and
        # spans 0.77x-1.42x. Running the old form against a policy trained on the
        # new one makes a = 0 command hover where the policy intends 0.79x hover.
        #
        # ``collective`` is a POSITIVE PX4 throttle magnitude; thrust_body is
        # NED-down, so it is negated on the way out (see the return below).
        collective = torch.clamp(
            collective_thrust_scale * (clipped_action[:, 0] + 1.0) * 0.5,
            collective_thrust_min,
            collective_thrust_max,
        )
        current_euler = self.current_euler_flu.to(
            device=action.device, dtype=action.dtype
        )
        roll_command = torch.zeros_like(clipped_action[:, 0])
        pitch_command = torch.clamp(
            current_euler[1] + clipped_action[:, 1] * max_delta_pitch,
            -max_pitch,
            max_pitch,
        )
        yaw_command = torch.atan2(
            torch.sin(current_euler[2] + clipped_action[:, 2] * max_delta_yaw),
            torch.cos(current_euler[2] + clipped_action[:, 2] * max_delta_yaw),
        )

        # Training observes the clipped raw deltas, not processed setpoints.
        self.prev_action = clipped_action[0].detach().clone()

        # Convert FLU/NWU commands to the FRD/NED convention expected by PX4.
        # ``-collective`` because thrust_body's z axis points DOWN, so lift is
        # negative. The old px4_thrust was already signed; collective is not.
        return torch.stack(
            [-collective, roll_command, -pitch_command, -yaw_command],
            dim=1,
        )

    def policy_eval(self, curr_wp):
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
        action = self.scale_and_clip(result["mus"])[0]
        thrust, roll, pitch, yaw = action.detach().cpu().tolist()
        return float(thrust), float(roll), float(pitch), float(yaw)

    @staticmethod
    def image_to_meters(message: Image) -> np.ndarray:
        """Decode common ROS depth encodings without requiring cv_bridge."""
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
            actual_size = len(message.data)
            raise ValueError(
                f"Depth payload has {actual_size} bytes; "
                f"expected {expected_size}"
            )
        image = np.frombuffer(
            message.data,
            dtype=scalar_type,
            count=message.height * row_width,
        )
        image = image.reshape(message.height, row_width)[:, :message.width]
        return image.astype(np.float32, copy=True) * scale

    def depth_callback(self, message: Image) -> None:
        """Encode a new frame and hold its latent until the next one."""
        try:
            depth = self.image_to_meters(message)
            depth_tensor = torch.from_numpy(depth).to(self.device)
            normalized_depth = self._normalize_depth_image(
                depth_tensor.unsqueeze(0),
                self.depth_range,
            )
            self.depth_latent = self.depth_encoder.encode(normalized_depth)[0]
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
                    f"Failed to process depth image: {error}"
                )
                self.depth_error_reported = True

    def update_waypoint(self, curr_wp):
        thresh = 1.2
        vehicle_position = self.vehicle_odometry.position

        # Distance between vehicle and waypoint.
        distance = sum(
            (curr_wp[i] - vehicle_position[i]) ** 2 for i in range(3)
        ) ** 0.5

        if distance <= thresh:
            self.wp_idx = (self.wp_idx + 1) % len(self.waypoints)
            self.curr_wp = self.waypoints[self.wp_idx]

        print("distance : ", distance)

    def vehicle_hover_callback(self, vehicle_hover_thrust):
        """Store the latest PX4 hover-thrust estimate."""
        self.vehicle_hover_thrust = vehicle_hover_thrust

    def vehicle_odometry_callback(self, vehicle_odometry):
        """Store the latest PX4 vehicle odometry."""
        self.vehicle_odometry = vehicle_odometry

    def vehicle_global_position_callback(self, vehicle_global_position):
        """Store global position and lock the first valid home position."""
        self.vehicle_global_position = vehicle_global_position

        if self.vehicle_global_position.lat > 0 and self.home_lock is False:
            self.home_lat = self.vehicle_global_position.lat
            self.home_lon = self.vehicle_global_position.lon
            self.home_alt = self.vehicle_global_position.alt
            self.home_lock = True

    def vehicle_status_callback(self, vehicle_status):
        """Store the latest PX4 vehicle status."""
        self.vehicle_status = vehicle_status

    def arm(self):
        """Send an arm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0
        )
        self.get_logger().info("Arm command sent")

    def takeoff_yaw_ned(self) -> float:
        """NED yaw that points the nose at the first waypoint.

        The task samples the spawn heading within +/-30 deg of the bearing to the
        goal, so the policy has never seen a goal far off its nose -- and because
        roll is pinned to zero it can only translate where it points. Handing over
        at PX4's default takeoff yaw (param4 = 0.0, i.e. due North) with an
        eastward waypoint would start it 90 deg out of distribution.

        NED yaw: 0 = North, +pi/2 = East.
        """
        position = self.vehicle_odometry.position
        return float(
            math.atan2(
                self.curr_wp[1] - position[1],
                self.curr_wp[0] - position[0],
            )
        )

    def takeoff(self):
        """Send a takeoff command using the locked home position."""
        yaw_ned = self.takeoff_yaw_ned()
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_NAV_TAKEOFF,
            # param4 is the NED yaw to hold on arrival. Left unset it defaults to
            # 0.0 (North); aim it at the waypoint instead. See takeoff_yaw_ned.
            param4=yaw_ned,
            param5=self.home_lat,
            param6=self.home_lon,
            param7=self.home_alt - self.takeoff_height,
        )
        self.get_logger().info(
            f"Takeoff command sent (yaw {math.degrees(yaw_ned):+.1f} deg NED, "
            f"toward waypoint {self.curr_wp})"
        )

    def disarm(self):
        """Send a disarm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0
        )
        self.get_logger().info("Disarm command sent")

    def engage_offboard_mode(self):
        """Switch to offboard mode."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0
        )
        self.get_logger().info("Switching to offboard mode")

    def land(self):
        """Switch to land mode."""
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info("Switching to land mode")

    def publish_offboard_control_heartbeat_signal(self):
        """Publish the offboard control mode."""
        msg = OffboardControlMode()
        msg.position = False
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = True
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)

    @staticmethod
    def euler_to_quaternion(roll, pitch, yaw):
        """Convert XYZ Euler angles to a [w, x, y, z] quaternion."""
        cosine_yaw = np.cos(yaw * 0.5)
        sine_yaw = np.sin(yaw * 0.5)
        cosine_pitch = np.cos(pitch * 0.5)
        sine_pitch = np.sin(pitch * 0.5)
        cosine_roll = np.cos(roll * 0.5)
        sine_roll = np.sin(roll * 0.5)
        w = (
            cosine_roll * cosine_pitch * cosine_yaw
            + sine_roll * sine_pitch * sine_yaw
        )
        x = (
            sine_roll * cosine_pitch * cosine_yaw
            - cosine_roll * sine_pitch * sine_yaw
        )
        y = (
            cosine_roll * sine_pitch * cosine_yaw
            + sine_roll * cosine_pitch * sine_yaw
        )
        z = (
            cosine_roll * cosine_pitch * sine_yaw
            - sine_roll * sine_pitch * cosine_yaw
        )
        return [w, x, y, z]

    def publish_attitude_setpoint(
        self, t: float, r: float, p: float, y: float
    ):
        """Publish an attitude setpoint from the policy output."""
        msg = VehicleAttitudeSetpoint()
        w, qx, qy, qz = self.euler_to_quaternion(r, p, y)
        print("applied thrust : ", t)
        msg.q_d = [w, qx, qy, qz]
        msg.thrust_body = [0.0, 0.0, t]
        print("\n\n")
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_attitude_setpoint_publisher.publish(msg)

    def publish_vehicle_command(self, command, **params) -> None:
        """Publish a vehicle command."""
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
        """Run the reference offboard-control timer state machine."""
        self.publish_offboard_control_heartbeat_signal()

        if self.home_lock is True:
            if (
                self.vehicle_status.arming_state
                != VehicleStatus.ARMING_STATE_ARMED
            ):
                self.arm()

            elif (
                self.vehicle_status.takeoff_time == 0
                and self.vehicle_status.arming_state
                == VehicleStatus.ARMING_STATE_ARMED
            ):
                self.takeoff()

            elif (
                self.vehicle_odometry.position[-1]
                <= (self.takeoff_height + 1.0)
                and self.vehicle_status.nav_state
                != VehicleStatus.NAVIGATION_STATE_OFFBOARD
            ):
                print("i am here")
                if (
                    self.vehicle_hover_thrust.hover_thrust is not None
                    and self.vehicle_hover_thrust.hover_thrust > 0.0
                ):
                    print(
                        "hover thrust : ",
                        self.vehicle_hover_thrust.hover_thrust,
                    )
                    self.engage_offboard_mode()
                # else:

            elif (
                self.vehicle_status.nav_state
                == VehicleStatus.NAVIGATION_STATE_OFFBOARD
            ):

                if self.nn_eval is True:
                    t, r, p, y = self.policy_eval(self.curr_wp)
                else:
                    t, r, p, y = 0.0, 0.0, 0.0, 0.0

                self.publish_attitude_setpoint(t, r, p, y)

                self.update_waypoint(self.curr_wp)


def main(args=None) -> None:
    """Start the depth policy offboard-control node."""
    print("Starting depth-policy offboard control node...")
    rclpy.init(args=args)
    offboard_control = OffboardControl()
    rclpy.spin(offboard_control)
    offboard_control.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(error)
