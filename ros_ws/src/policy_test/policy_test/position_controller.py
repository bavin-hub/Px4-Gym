import torch
from math import pi, sqrt, atan2, hypot

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, \
                            VehicleLocalPosition, VehicleStatus, VehicleOdometry, \
                            VehicleGlobalPosition

from policy_test.model_loader import load_policy_model


# /usr/bin/python3 -m pip install --user torch \
#   --index-url https://download.pytorch.org/whl/cu128
# /usr/bin/python3 -m pip install --user rl-games PyYAML


class OffboardControl(Node):
    """Node for controlling a vehicle in offboard mode with the position policy."""

    def __init__(self) -> None:
        super().__init__('offboard_control_takeoff_and_land')

        self.device = self.declare_parameter("device", "cuda:0").value
        self.policy_config = self.declare_parameter("policy_config", "").value
        self.checkpoint = self.declare_parameter("checkpoint", "").value
        self.model = load_policy_model(
            policy_config=self.policy_config,
            checkpoint=self.checkpoint,
            device=self.device,
            actions_num=4,
            input_shape=18,
        )
        self.get_logger().info("Policy model loaded successfully")

        # Configure QoS profile for publishing and subscribing
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Create publishers
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_profile)

        # Create subscribers
        self.vehicle_odometry_subscriber = self.create_subscription(
            VehicleOdometry, '/fmu/out/vehicle_odometry', self.vehicle_odometry_callback, qos_profile)
        self.vehicle_global_position_subscriber = self.create_subscription(
            VehicleGlobalPosition, '/fmu/out/vehicle_global_position', self.vehicle_global_position_callback, qos_profile)
        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status', self.vehicle_status_callback, qos_profile)

        # Initialize variables
        self.offboard_setpoint_counter = 0
        self.vehicle_local_position = VehicleLocalPosition()
        self.vehicle_odometry = VehicleOdometry()
        self.vehicle_global_position = VehicleGlobalPosition()
        self.vehicle_status = VehicleStatus()
        self.takeoff_height = -5.0

        # [0.0, 0.0, self.takeoff_height],
        self.waypoints = [[6.0, 6.0, self.takeoff_height],
                          [-6.0, 6.0, self.takeoff_height],
                          [-6.0, -6.0, self.takeoff_height],
                          [6.0, -6.0, self.takeoff_height]]

        # Training action contract (Starling2MaxPositionEnvCfg):
        # channels 0:3 -> env-local position setpoint scaled to +/- 5/sqrt(3) m,
        # channel 3 -> world yaw setpoint scaled to +/- pi rad.
        self.action_position_scale = 5.0 / sqrt(3.0)
        self.action_yaw_scale = pi

        self.prev_action = [0.0, 0.0, 0.0, 0.0]
        self.curr_wp = self.waypoints[0]
        self.wp_idx = 0
        self.nn_eval = True

        # home pos
        self.home_lat, self.home_lon, self.alt = None, None, None
        self.home_lock = False

        # Create a timer to publish control commands
        self.timer = self.create_timer(0.01, self.timer_callback)

    def create_observation(self, curr_wp):
        """Convert PX4 NED/FRD odometry into the policy's NWU/FLU inputs."""
        device = self.device

        def quat_multiply(q1, q2):
            """Multiply quaternions stored as [w, x, y, z]."""
            w1, x1, y1, z1 = q1
            w2, x2, y2, z2 = q2
            return torch.stack([
                w1*w2 - x1*x2 - y1*y2 - z1*z2,
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2,
            ])

        def inverse_rotate(q, vector):
            """Rotate a world-NED vector into body FRD using q inverse."""
            q_vec = q[1:4]
            return (
                vector
                - 2.0 * q[0] * torch.cross(q_vec, vector, dim=0)
                + 2.0 * torch.cross(
                    q_vec, torch.cross(q_vec, vector, dim=0), dim=0
                )
            )

        odom = self.vehicle_odometry

        if odom.pose_frame != VehicleOdometry.POSE_FRAME_NED:
            raise ValueError(f"Unsupported pose frame: {odom.pose_frame}")

        position_ned = torch.tensor(
            odom.position, dtype=torch.float32, device=device
        )

        waypoint_ned = torch.tensor(
            curr_wp, dtype=torch.float32, device=device
        )

        # Position error expected by the policy:
        # PX4 world NED [north, east, down] -> Isaac world NWU [north, west, up].
        error_ned = waypoint_ned - position_ned
        position_error = torch.stack([
            error_ned[0],
            -error_ned[1],
            -error_ned[2],
        ])

        q_ned_frd = torch.tensor(
            odom.q, dtype=torch.float32, device=device
        )
        q_ned_frd /= torch.linalg.vector_norm(q_ned_frd).clamp_min(1e-8)

        # PX4 q maps body FRD into world NED. Apply 180-degree X rotations on
        # both sides so the resulting q maps body FLU into world NWU.
        q_nwu_ned = torch.tensor(
            [0.0, 1.0, 0.0, 0.0],
            dtype=torch.float32,
            device=device,
        )
        q_frd_flu = torch.tensor(
            [0.0, 1.0, 0.0, 0.0],
            dtype=torch.float32,
            device=device,
        )

        q_nwu_flu = quat_multiply(
            quat_multiply(q_nwu_ned, q_ned_frd),
            q_frd_flu,
        )
        q_nwu_flu /= torch.linalg.vector_norm(q_nwu_flu).clamp_min(1e-8)

        if q_nwu_flu[0] < 0:
            q_nwu_flu = -q_nwu_flu

        # Isaac Gym stores quaternion observations as [qx, qy, qz, qw].
        orientation = q_nwu_flu[[1, 2, 3, 0]]


        velocity = torch.tensor(
            odom.velocity, dtype=torch.float32, device=device
        )

        if odom.velocity_frame == VehicleOdometry.VELOCITY_FRAME_NED:
            velocity_frd = inverse_rotate(q_ned_frd, velocity)
        elif odom.velocity_frame == VehicleOdometry.VELOCITY_FRAME_BODY_FRD:
            velocity_frd = velocity
        else:
            raise ValueError(f"Unsupported velocity frame: {odom.velocity_frame}")


        # The policy uses body FLU velocity. Convert PX4 body FRD by negating
        # the right and down components.
        body_linear_velocity = torch.stack([
            velocity_frd[0],
            -velocity_frd[1],
            -velocity_frd[2],
        ])

        angular_frd = torch.tensor(
            odom.angular_velocity,
            dtype=torch.float32,
            device=device,
        )
        body_angular_velocity = torch.stack([
            angular_frd[0],
            -angular_frd[1],
            -angular_frd[2],
        ])


        # Training observes the previous normalized [-1, 1] action
        # (obs["robot_actions"]), i.e. the clipped network output before scaling.
        if self.prev_action is None:
            previous_action = torch.zeros(
                4, dtype=torch.float32, device=device
            )
        else:
            previous_action = torch.as_tensor(
                self.prev_action, dtype=torch.float32, device=device
            ).flatten()

        # Compute wrapped waypoint-bearing error in NWU and normalize it to
        # [-1, 1]. Heading is ignored very close to the horizontal target.
        qw, qx, qy, qz = q_nwu_flu
        current_yaw = torch.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )
        desired_yaw = torch.atan2(position_error[1], position_error[0])
        yaw_error = torch.atan2(
            torch.sin(desired_yaw - current_yaw),
            torch.cos(desired_yaw - current_yaw),
        )
        if torch.linalg.vector_norm(position_error[:2]) <= 0.2:
            yaw_error = torch.zeros_like(yaw_error)
        normalized_yaw_error = (yaw_error / torch.pi).reshape(1)

        # Exact input order used during training (18 values).
        observation = torch.cat([
            position_error,           # [0:3]   world NWU waypoint error
            orientation,              # [3:7]   body FLU -> world NWU quaternion
            body_linear_velocity,     # [7:10]  body FLU linear velocity
            body_angular_velocity,    # [10:13] body FLU angular velocity
            previous_action,          # [13:17] previous normalized [x, y, z, yaw] action
            normalized_yaw_error,     # [17]    wrapped heading error / pi
        ])

        return observation.unsqueeze(0)

    def scale_and_clip(self, action):
        # Match training: the policy emits a normalized [-1, 1] action.  Channels
        # 0:3 scale to a metric displacement applied RELATIVE to the current
        # position; channel 3 scales to an absolute world-NWU yaw setpoint.
        normalized = torch.clamp(action, -1.0, 1.0)[0].detach().cpu().tolist()

        # NWU relative displacement = normalized position channels * scale.
        disp_nwu = [normalized[i] * self.action_position_scale for i in range(3)]
        position_ned = self.vehicle_odometry.position  # [north, east, down]

        # NWU displacement -> NED, added to the current NED position.
        setpoint_north = float(position_ned[0] + disp_nwu[0])
        setpoint_east = float(position_ned[1] - disp_nwu[1])
        setpoint_down = float(position_ned[2] - disp_nwu[2])

        # Yaw from the commanded NED horizontal direction instead of the policy.
        # NED displacement: north = disp_nwu[0], east = -disp_nwu[1].
        disp_north = disp_nwu[0]
        disp_east = -disp_nwu[1]
        if hypot(disp_north, disp_east) > 0.1:
            yaw_ned = float(atan2(disp_east, disp_north))
            self._last_yaw = yaw_ned
        else:
            # Hold the last heading when the horizontal command is too small.
            yaw_ned = float(getattr(self, "_last_yaw", 0.0))

        # Policy yaw (disabled): absolute world-NWU yaw setpoint -> PX4 NED yaw.
        # yaw_ned = float(-(normalized[3] * self.action_yaw_scale))

        # Store the NORMALIZED action for the next observation's previous-action slot.
        self.prev_action = normalized

        return setpoint_north, setpoint_east, setpoint_down, yaw_ned

    def policy_eval(self, curr_wp):
        observation = self.create_observation(curr_wp)
        with torch.no_grad():
            result = self.model({
                                    "is_train": False,
                                    "prev_actions": None,
                                    "obs": observation,
                                    "rnn_states": None,
                                })
        action = result["mus"]
        north, east, down, yaw = self.scale_and_clip(action)

        return float(north), float(east), float(down), float(yaw)

    def update_waypoint(self, curr_wp):
        thresh = 0.2
        vehicle_position = self.vehicle_odometry.position

        # distance between vehicle and waypoint
        distance = sum(
                    (curr_wp[i] - vehicle_position[i]) ** 2
                    for i in range(3)
                ) ** 0.5

        if distance <= thresh:
            self.wp_idx = (self.wp_idx + 1) % len(self.waypoints)
            self.curr_wp = self.waypoints[self.wp_idx]

        print(f"dist to wp {self.wp_idx}: {distance}")
        print("\n\n")


    def vehicle_odometry_callback(self, vehicle_odometry):
        """Callback function for vehicle_odometry topic subscriber."""
        self.vehicle_odometry = vehicle_odometry

    def vehicle_global_position_callback(self, vehicle_global_position):
        """Callback function for vehicle_global_position topic subscriber."""
        self.vehicle_global_position = vehicle_global_position

        if self.vehicle_global_position.lat > 0 and self.home_lock is False:
            self.home_lat = self.vehicle_global_position.lat
            self.home_lon = self.vehicle_global_position.lon
            self.home_alt = self.vehicle_global_position.alt
            self.home_lock = True

    def vehicle_status_callback(self, vehicle_status):
        """Callback function for vehicle_status topic subscriber."""
        self.vehicle_status = vehicle_status

    def arm(self):
        """Send an arm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info('Arm command sent')

    def takeoff(self):
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_NAV_TAKEOFF,
            param5=self.home_lat,
            param6=self.home_lon,
            param7=self.home_alt - self.takeoff_height)
        self.get_logger().info('Takeoff command sent')


    def disarm(self):
        """Send a disarm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
        self.get_logger().info('Disarm command sent')

    def engage_offboard_mode(self):
        """Switch to offboard mode."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info("Switching to offboard mode")

    def land(self):
        """Switch to land mode."""
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info("Switching to land mode")

    def publish_offboard_control_heartbeat_signal(self):
        """Publish the offboard control mode."""
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)

    def publish_position_setpoint(self, north: float, east: float, down: float, yaw: float):
        """Publish the trajectory setpoint."""
        msg = TrajectorySetpoint()
        print(north, east, down)
        msg.position = [north, east, down]
        msg.velocity = [float('nan'), float('nan'), float('nan')]
        msg.yaw = yaw
        msg.yawspeed = float('nan')
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)

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
        """Callback function for the timer."""
        self.publish_offboard_control_heartbeat_signal()

        if self.home_lock is True:
            if self.vehicle_status.arming_state != VehicleStatus.ARMING_STATE_ARMED:
                self.arm()

            elif self.vehicle_status.takeoff_time == 0 and self.vehicle_status.arming_state == VehicleStatus.ARMING_STATE_ARMED:
                self.takeoff()

            elif self.vehicle_odometry.position[-1] <= (self.takeoff_height + 1.0) and self.vehicle_status.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
                print("i am here")
                self.engage_offboard_mode()

            elif self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:

                if self.nn_eval is True:

                    north, east, down, yaw = self.policy_eval(self.curr_wp)
                else:
                    north, east, down, yaw = 0.0, 0.0, self.takeoff_height, 0.0

                self.publish_position_setpoint(north, east, down, yaw)

                self.update_waypoint(self.curr_wp)

        if self.offboard_setpoint_counter < 11:
            self.offboard_setpoint_counter += 1


def main(args=None) -> None:
    print('Starting offboard control node...')
    rclpy.init(args=args)
    offboard_control = OffboardControl()
    rclpy.spin(offboard_control)
    offboard_control.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
