import torch
from math import pi

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, \
                            VehicleLocalPosition, VehicleStatus, VehicleOdometry, \
                            VehicleGlobalPosition, ActuatorMotors

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
            input_shape=15,
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
        self.actuator_motors_pub = self.create_publisher(
            ActuatorMotors, '/fmu/in/actuator_motors', qos_profile)

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
        self.takeoff_height = -3.0

        # self.waypoints = [[5.0, 5.0, self.takeoff_height],
        #                   [-5.0, 5.0, self.takeoff_height],
        #                   [-5.0, -5.0, self.takeoff_height],
        #                   [5.0, -5.0, self.takeoff_height]]
        self.waypoints = [[0.0, 0.0, self.takeoff_height]]

        # self.waypoints = [[0.0, 0.0, self.takeoff_height],
        #                   [5.0, 5.0, self.takeoff_height],
        #                   [-5.0, 5.0, self.takeoff_height],
        #                   [-5.0, -5.0, self.takeoff_height],
        #                   [5.0, -5.0, self.takeoff_height]]
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

        def rotate_vector(q, vector):
            """Rotate a vector using q, where q maps body FRD into world NED."""
            q_vec = q[1:4]
            return (
                vector
                + 2.0 * q[0] * torch.cross(q_vec, vector, dim=0)
                + 2.0 * torch.cross(
                    q_vec, torch.cross(q_vec, vector, dim=0), dim=0
                )
            )

        def quaternion_to_rotation_6d(q):
            """Return the first two rows of the rotation matrix, as in training."""
            w, x, y, z = q
            rotation_matrix = torch.stack([
                1.0 - 2.0 * (y*y + z*z),
                2.0 * (x*y - w*z),
                2.0 * (x*z + w*y),
                2.0 * (x*y + w*z),
                1.0 - 2.0 * (x*x + z*z),
                2.0 * (y*z - w*x),
                2.0 * (x*z - w*y),
                2.0 * (y*z + w*x),
                1.0 - 2.0 * (x*x + y*y),
            ]).reshape(3, 3)
            return rotation_matrix[:2, :].reshape(6)

        odom = self.vehicle_odometry

        if odom.pose_frame != VehicleOdometry.POSE_FRAME_NED:
            raise ValueError(f"Unsupported pose frame: {odom.pose_frame}")

        position_ned = torch.as_tensor(
            odom.position, dtype=torch.float32, device=device)
        waypoint_ned = torch.as_tensor(
            curr_wp, dtype=torch.float32, device=device)

        # Policy inputs use Isaac's NWU world axes:
        # PX4 NED [north, east, down] -> Isaac NWU [north, west, up].
        error_ned = waypoint_ned - position_ned
        position_error = torch.stack([
            error_ned[0],
            -error_ned[1],
            -error_ned[2],
        ])

        # PX4 stores q as [w, x, y, z], rotating body FRD into world NED.
        q_ned_frd = torch.as_tensor(
            odom.q, dtype=torch.float32, device=device)
        q_ned_frd /= torch.linalg.vector_norm(q_ned_frd).clamp_min(1e-8)

        # A 180-degree rotation about X performs both NED -> NWU and
        # FLU -> FRD axis conversion. The result maps body FLU to world NWU.
        q_x_180 = torch.tensor(
            [0.0, 1.0, 0.0, 0.0],
            dtype=torch.float32,
            device=device,
        )
        q_nwu_flu = quat_multiply(
            quat_multiply(q_x_180, q_ned_frd),
            q_x_180,
        )
        q_nwu_flu /= torch.linalg.vector_norm(q_nwu_flu).clamp_min(1e-8)

        # q and -q describe the same attitude; training used a consistent sign.
        if q_nwu_flu[0] < 0:
            q_nwu_flu = -q_nwu_flu

        orientation_6d = quaternion_to_rotation_6d(q_nwu_flu)

        velocity = torch.as_tensor(
            odom.velocity, dtype=torch.float32, device=device)
        if odom.velocity_frame == VehicleOdometry.VELOCITY_FRAME_NED:
            velocity_ned = velocity
        elif odom.velocity_frame == VehicleOdometry.VELOCITY_FRAME_BODY_FRD:
            velocity_ned = rotate_vector(q_ned_frd, velocity)
        else:
            raise ValueError(f"Unsupported velocity frame: {odom.velocity_frame}")

        # Training used linear velocity in the Isaac NWU world frame.
        world_linear_velocity = torch.stack([
            velocity_ned[0],
            -velocity_ned[1],
            -velocity_ned[2],
        ])

        # PX4 angular velocity is body-fixed FRD. Convert it to body FLU.
        angular_frd = torch.as_tensor(
            odom.angular_velocity,
            dtype=torch.float32,
            device=device,
        )
        body_angular_velocity = torch.stack([
            angular_frd[0],
            -angular_frd[1],
            -angular_frd[2],
        ])

        # Exact 15-value order used by position_setpoint_task_sim2real_px4.
        observation = torch.cat([
            position_error,           # [0:3]   world NWU position error
            orientation_6d,           # [3:9]   body FLU -> world NWU attitude
            world_linear_velocity,    # [9:12]  world NWU linear velocity
            body_angular_velocity,    # [12:15] body FLU angular velocity
        ])

        return observation.unsqueeze(0)

    
    
    def scale_and_clip(self, action):
        # Training maps each normalized policy action from [-1, 1] to
        # [0, max motor thrust]. PX4 ActuatorMotors expects [0, 1] for
        # non-reversible motors, so normalize the same mapping by max thrust.
        actuator_inputs = (torch.clamp(action, -1.0, 1.0).clone() + 1.0) * 0.5

        # Order already matches PX4 quad-X:
        # front-right, rear-left, front-left, rear-right.
        return actuator_inputs

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

        return float(action[0]), float(action[1]), float(action[2]), float(action[3])

    def plan_intermediate_waypoints(self, start, goal, spacing=1.0):
        """Generate 3D NED waypoints with segments no longer than spacing."""
        start = torch.as_tensor(start, dtype=torch.float32)
        goal = torch.as_tensor(goal, dtype=torch.float32)
        displacement = goal - start
        distance = torch.linalg.vector_norm(displacement).item()

        if distance == 0.0:
            return goal.tolist()

        num_segments = max(1, int(torch.ceil(torch.tensor(distance / spacing)).item()))
        return [
            (start + displacement * (step / num_segments)).tolist()
            for step in range(1, num_segments + 1)
        ][0]

    def update_waypoint(self, curr_wp):
        thresh = 1.5
        vehicle_position = self.vehicle_odometry.position

        # distance between vehicle and waypoint
        distance = sum(
                    (curr_wp[i] - vehicle_position[i]) ** 2
                    for i in range(3)
                ) ** 0.5
        
        if distance <= thresh:
            self.wp_idx = (self.wp_idx + 1) % len(self.waypoints)
            self.curr_wp = self.waypoints[self.wp_idx]

        # print(distance, vehicle_position[0], vehicle_position[1])
        


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
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.actuator = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)

    def publish_actuator_setpoint(self, fr: float, bl: float, fl: float, br: float):
        msg = ActuatorMotors()
        now_us = int(self.get_clock().now().nanoseconds / 1000)
        msg.timestamp = now_us
        msg.timestamp_sample = now_us
        msg.reversible_flags = 0
        msg.control = [fr, bl, fl, br] + [float("nan")] * 8
        self.actuator_motors_pub.publish(msg)



    # def publish_position_setpoint(self, x: float, y: float, z: float):
    #     """Publish the trajectory setpoint."""
    #     msg = TrajectorySetpoint()
    #     msg.position = [x, y, z]
    #     msg.yaw = 1.57079  # (90 degree)
    #     msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
    #     self.trajectory_setpoint_publisher.publish(msg)
    #     self.get_logger().info(f"Publishing position setpoints {[x, y, z]}")

    def publish_position_setpoint(self, x: float, y: float, z: float, yawspeed: float):
        """Publish the trajectory setpoint."""
        msg = TrajectorySetpoint()
        # msg.position = [x, y, z]
        msg.position = [float('nan'), float('nan'), float('nan')]
        msg.velocity = [x, y, z]
        msg.yaw = float('nan')
        msg.yawspeed = yawspeed
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

        # print("in timer")
        if self.home_lock is True:
            # print("home is locked")
            if self.offboard_setpoint_counter == 10:
                print("engaing offboard mode and arming")
                self.engage_offboard_mode()
                self.arm()

            # self.vehicle_odometry.position[-1] > self.takeoff_height and 
            if self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD and self.vehicle_status.arming_state == VehicleStatus.ARMING_STATE_ARMED:
                # self.publish_position_setpoint(50.0, 50.0, self.takeoff_height)
                # print("actuator control")
                intermediate_wp = self.plan_intermediate_waypoints(self.vehicle_odometry.position, 
                                                                   self.curr_wp)

                # use actuator control
                fr, bl, fl, br = self.policy_eval(self.curr_wp)
                self.publish_actuator_setpoint(fr, bl, fl, br)
                # print("model running on gpu is the issue")


                
                # self.update_waypoint(self.curr_wp)

            # elif self.vehicle_odometry.position[-1] <= self.takeoff_height:
            #     self.land()
            #     exit(0)

            
        
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
