"""SITL deployment node for the current Starling attitude-delta policy."""

from __future__ import annotations

import math

import numpy as np
import rclpy
import torch
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
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from policy_test.model_loader import load_policy_model
from policy_test.starling_observation import StarlingHistoryObservation


class AttitudeDeltaController(Node):
    """Run the 100 Hz, 98-D policy and publish PX4 attitude setpoints."""

    def __init__(self) -> None:
        super().__init__("starling_attitude_delta_controller")
        default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(
            self.declare_parameter("device", default_device).value
        )
        self.model = load_policy_model(
            policy_config=self.declare_parameter("policy_config", "").value,
            checkpoint=self.declare_parameter("checkpoint", "").value,
            device=str(self.device),
            actions_num=4,
            input_shape=98,
        )
        self.takeoff_height = float(self.declare_parameter("takeoff_height", -3.0).value)
        self.waypoint = [
            float(self.declare_parameter("waypoint_north", 0.0).value),
            float(self.declare_parameter("waypoint_east", 0.0).value),
            float(self.declare_parameter("waypoint_down", self.takeoff_height).value),
        ]
        self.nominal_hover_thrust = float(
            self.declare_parameter("hover_thrust", 0.13).value
        )

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.offboard_pub = self.create_publisher(
            OffboardControlMode, "/fmu/in/offboard_control_mode", qos
        )
        self.attitude_pub = self.create_publisher(
            VehicleAttitudeSetpoint, "/fmu/in/vehicle_attitude_setpoint", qos
        )
        self.command_pub = self.create_publisher(
            VehicleCommand, "/fmu/in/vehicle_command", qos
        )
        self.create_subscription(
            VehicleOdometry, "/fmu/out/vehicle_odometry", self._odometry_callback, qos
        )
        self.create_subscription(
            VehicleGlobalPosition,
            "/fmu/out/vehicle_global_position",
            self._global_position_callback,
            qos,
        )
        self.create_subscription(
            VehicleStatus, "/fmu/out/vehicle_status", self._status_callback, qos
        )
        self.create_subscription(
            HoverThrustEstimate,
            "/fmu/out/hover_thrust_estimate",
            self._hover_callback,
            qos,
        )

        self.odometry = VehicleOdometry()
        self.odometry_received = False
        self.status = VehicleStatus()
        self.hover_estimate = HoverThrustEstimate()
        self.home: tuple[float, float, float] | None = None
        self.observation = StarlingHistoryObservation(self.device)
        self.previous_action = torch.tensor(
            [self.nominal_hover_thrust, 0.0, 0.0, 0.0],
            dtype=torch.float32,
            device=self.device,
        )
        self.timer = self.create_timer(0.01, self._timer_callback)

    def _odometry_callback(self, message: VehicleOdometry) -> None:
        self.odometry = message
        self.odometry_received = True

    def _global_position_callback(self, message: VehicleGlobalPosition) -> None:
        values = (float(message.lat), float(message.lon), float(message.alt))
        if self.home is None and all(np.isfinite(value) for value in values):
            self.home = values

    def _status_callback(self, message: VehicleStatus) -> None:
        self.status = message

    def _hover_callback(self, message: HoverThrustEstimate) -> None:
        self.hover_estimate = message

    def _publish_command(self, command: int, **params: float) -> None:
        message = VehicleCommand()
        message.command = command
        for index in range(1, 8):
            setattr(message, f"param{index}", float(params.get(f"param{index}", 0.0)))
        message.target_system = 1
        message.target_component = 1
        message.source_system = 1
        message.source_component = 1
        message.from_external = True
        message.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.command_pub.publish(message)

    def _publish_heartbeat(self) -> None:
        message = OffboardControlMode()
        message.attitude = True
        message.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_pub.publish(message)

    @staticmethod
    def _euler_to_quaternion(roll: float, pitch: float, yaw: float) -> list[float]:
        cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
        cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
        cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
        return [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]

    def _policy_command(self) -> tuple[float, float, float, float]:
        observation, current_rpy_flu = self.observation.build(
            self.odometry, self.waypoint, self.previous_action
        )
        with torch.no_grad():
            action = self.model(
                {
                    "is_train": False,
                    "prev_actions": None,
                    "obs": observation,
                    "rnn_states": None,
                }
            )["mus"][0]
        action = torch.clamp(action, -1.0, 1.0)

        hover = float(self.hover_estimate.hover_thrust)
        if not math.isfinite(hover) or hover <= 0.0:
            hover = self.nominal_hover_thrust
        px4_thrust = torch.clamp(
            torch.tensor(-hover, device=self.device) + action[0] * 0.03,
            -0.16,
            -0.10,
        )
        max_delta_roll_pitch = math.radians(0.5)
        roll_flu = torch.clamp(
            current_rpy_flu[0] + action[1] * max_delta_roll_pitch,
            -math.radians(2.0),
            math.radians(2.0),
        )
        pitch_flu = torch.clamp(
            current_rpy_flu[1] + action[2] * max_delta_roll_pitch,
            -math.radians(2.0),
            math.radians(2.0),
        )
        yaw_flu = torch.atan2(
            torch.sin(current_rpy_flu[2] + action[3] * math.radians(10.0)),
            torch.cos(current_rpy_flu[2] + action[3] * math.radians(10.0)),
        )
        collective = -px4_thrust
        self.previous_action = torch.stack((collective, roll_flu, pitch_flu, yaw_flu))

        # FLU/NWU Euler commands become FRD/NED by flipping pitch and yaw.
        return (
            float(px4_thrust),
            float(roll_flu),
            float(-pitch_flu),
            float(-yaw_flu),
        )

    def _publish_attitude(
        self, thrust: float, roll: float, pitch: float, yaw: float
    ) -> None:
        message = VehicleAttitudeSetpoint()
        message.q_d = self._euler_to_quaternion(roll, pitch, yaw)
        message.thrust_body = [0.0, 0.0, thrust]
        message.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.attitude_pub.publish(message)

    def _timer_callback(self) -> None:
        self._publish_heartbeat()
        if self.home is None or not self.odometry_received:
            return
        if self.status.arming_state != VehicleStatus.ARMING_STATE_ARMED:
            self._publish_command(
                VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0
            )
            return
        if self.odometry.position[2] > self.takeoff_height + 0.5:
            lat, lon, alt = self.home
            self._publish_command(
                VehicleCommand.VEHICLE_CMD_NAV_TAKEOFF,
                param5=lat,
                param6=lon,
                param7=alt - self.takeoff_height,
            )
            return
        if self.status.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            self._publish_command(
                VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0
            )
            return
        self._publish_attitude(*self._policy_command())


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AttitudeDeltaController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
