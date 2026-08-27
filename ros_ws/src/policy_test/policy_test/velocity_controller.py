import torch
from math import pi

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
    """Node for controlling a vehicle in offboard mode."""

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
        # self.vehicle_local_position_subscriber = self.create_subscription(
            # VehicleLocalPosition, '/fmu/out/vehicle_local_position', self.vehicle_local_position_callback, qos_profile)
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
        self.waypoints = [[3.0, 3.0, self.takeoff_height],
                          [-3.0, 3.0, self.takeoff_height],
                          [-3.0, -3.0, self.takeoff_height],
                          [3.0, -3.0, self.takeoff_height]]
        self.max_vel = 1.0
        self.max_yaw_rate = (pi/3)
        self.prev_action = [0.0, 0.0, 0.0, 0.0]
        self.curr_wp = self.waypoints[0]
        self.wp_idx = 0
        self.nn_eval = True

        # home pos
        self.home_lat, self.home_lon, self.alt = None, None, None
        self.home_lock = False

        # Create a timer to publish control commands
        self.timer = self.create_timer(0.01, self.timer_callback)

    def frd_to_ned_transform(self, vx, vy, vz, yaw_rate):
        q = torch.as_tensor(
            self.vehicle_odometry.q,
            dtype=vx.dtype,
            device=vx.device,
        )
        qw, qx, qy, qz = q
        yaw = torch.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )

        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)

        velocity_north = cos_yaw * vx + sin_yaw * vy
        velocity_east = sin_yaw * vx - cos_yaw * vy
        velocity_down = -vz
        yawspeed = -yaw_rate

        return velocity_north, velocity_east, velocity_down, yawspeed

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


        # Training observes the previous scaled body-FLU velocity command and
        # yaw-rate command, not the raw network output.
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
            previous_action,          # [13:17] previous vx, vy, vz, yaw-rate
            normalized_yaw_error,     # [17]    wrapped heading error / pi
            
        ])
        # normalized_yaw_error,     # [17]    wrapped heading error / pi

        return observation.unsqueeze(0)

    
    
    def scale_and_clip(self, action):
        # Match training: clip the raw actor output, then scale it into a
        # body-FLU velocity command and FLU yaw-rate command.
        body_flu_command = torch.clamp(action, -1.0, 1.0).clone()
        body_flu_command[:, 0:3] *= 1.0
        body_flu_command[:, 3] *= torch.pi / 5.0

        # Observation indices [13:17] use the previous scaled command before
        # conversion to PX4's world NED frame.
        self.prev_action = body_flu_command[0].detach().cpu().tolist()

        # PX4 TrajectorySetpoint expects world-NED velocity and NED yaw rate.
        velocity_north, velocity_east, velocity_down, yawspeed = (
            self.frd_to_ned_transform(
                body_flu_command[:, 0],
                body_flu_command[:, 1],
                body_flu_command[:, 2],
                body_flu_command[:, 3],
            )
        )

        # Use the horizontal NED velocity direction as the PX4 yaw setpoint.
        # Retain the last valid yaw when horizontal speed is too small.
        horizontal_speed = torch.hypot(velocity_north, velocity_east)
        velocity_yaw = torch.atan2(velocity_east, velocity_north)

        q = torch.as_tensor(
            self.vehicle_odometry.q,
            dtype=velocity_yaw.dtype,
            device=velocity_yaw.device,
        )
        qw, qx, qy, qz = q
        current_yaw = torch.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )
        fallback_yaw = torch.full_like(
            velocity_yaw,
            getattr(self, "_last_velocity_yaw", current_yaw.item()),
        )
        yaw = torch.where(horizontal_speed > 0.1, velocity_yaw, fallback_yaw)
        self._last_velocity_yaw = yaw[0].detach().cpu().item()

        return torch.stack(
            [velocity_north, velocity_east, velocity_down, yawspeed, yaw],
            dim=1,
        )

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
        action = self.scale_and_clip(action)[0]
        action = action.detach().cpu().tolist()

        return float(action[0]), float(action[1]), float(action[2]), float(action[3]), float(action[4])

    def update_waypoint(self, curr_wp):
        thresh = 0.4
        vehicle_position = self.vehicle_odometry.position

        # distance between vehicle and waypoint
        distance = sum(
                    (curr_wp[i] - vehicle_position[i]) ** 2
                    for i in range(3)
                ) ** 0.5
        
        if distance <= thresh:
            self.wp_idx = (self.wp_idx + 1) % len(self.waypoints)
            self.curr_wp = self.waypoints[self.wp_idx]
            # if self.wp_idx == len(self.waypoints)-1:
            #     for wp in self.waypoints:
            #         new_wp = [val*2 for val in wp]
            #         self.waypoints.insert(0, new_wp)

        print(f"dist to wp {self.wp_idx}: {distance}")
        


    # def vehicle_local_position_callback(self, vehicle_local_position):
    #     """Callback function for vehicle_local_position topic subscriber."""
    #     self.vehicle_local_position = vehicle_local_position

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
        msg.position = False
        msg.velocity = True
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)

    # def publish_position_setpoint(self, x: float, y: float, z: float):
    #     """Publish the trajectory setpoint."""
    #     msg = TrajectorySetpoint()
    #     msg.position = [x, y, z]
    #     msg.yaw = 1.57079  # (90 degree)
    #     msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
    #     self.trajectory_setpoint_publisher.publish(msg)
    #     self.get_logger().info(f"Publishing position setpoints {[x, y, z]}")

    def publish_position_setpoint(self, x: float, y: float, z: float, yawspeed: float, yaw: float):
        """Publish the trajectory setpoint."""
        msg = TrajectorySetpoint()
        # msg.position = [x, y, z]
        msg.position = [float('nan'), float('nan'), float('nan')]
        msg.velocity = [x, y, z]
        msg.yaw = yaw
        msg.yawspeed = float('nan')
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)
        # self.get_logger().info(f"Publishing position setpoints {[x, y, z]}")

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
                    vx, vy, vz, yawspeed, yaw = self.policy_eval(self.curr_wp)
                else:
                    vx, vy, vz, yawspeed, yaw = 0.0, 0.0, 0.0, 0.0, 0.0

                self.publish_position_setpoint(vx, vy, vz, yawspeed, yaw)
                
                # print(self.curr_wp)
                self.update_waypoint(self.curr_wp)
            
        
        # if self.offboard_setpoint_counter == 10:
        #     self.engage_offboard_mode()
        #     self.arm()

        # if self.vehicle_local_position.z > self.takeoff_height and self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
        #     self.publish_position_setpoint(50.0, 50.0, self.takeoff_height)

        # elif self.vehicle_local_position.z <= self.takeoff_height:
        #     self.land()
        #     exit(0)

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
