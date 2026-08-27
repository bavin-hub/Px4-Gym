"""SITL deployment node for the Starling 2 Max direct CTBR policy."""

from __future__ import annotations

import math

import numpy as np
import rclpy
import torch
from px4_msgs.msg import (
    OffboardControlMode,
    VehicleCommand,
    VehicleGlobalPosition,
    VehicleOdometry,
    VehicleRatesSetpoint,
    VehicleStatus,
)
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from policy_test.model_loader import load_policy_model
from policy_test.starling_observation import StarlingHistoryObservation


class BodyRatesController(Node):
    """Run a 100 Hz CTBR policy and publish PX4 body-rate setpoints."""

    def __init__(self) -> None:
        super().__init__("starling_body_rates_controller")
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
        self.hover_collective = float(self.declare_parameter("hover_thrust", 0.13).value)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.offboard_pub = self.create_publisher(
            OffboardControlMode, "/fmu/in/offboard_control_mode", qos
        )
        self.rates_pub = self.create_publisher(
            VehicleRatesSetpoint, "/fmu/in/vehicle_rates_setpoint", qos
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

        self.odometry = VehicleOdometry()
        self.odometry_received = False
        self.status = VehicleStatus()
        self.home: tuple[float, float, float] | None = None
        self.observation = StarlingHistoryObservation(self.device)
        self.previous_action = torch.tensor(
            [0.0, 0.0, 0.0, self.hover_collective],
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
        message.body_rate = True
        message.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_pub.publish(message)

    def _policy_command(self) -> tuple[float, float, float, float]:
        observation, _ = self.observation.build(
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
            )["mus"]
        action = torch.tanh(action[0])
        rate_scale = torch.tensor(
            [math.pi, math.pi, math.pi], dtype=action.dtype, device=self.device
        )
        rate_limit = torch.deg2rad(
            torch.tensor([130.0, 130.0, 150.0], device=self.device)
        )
        rates_flu = torch.clamp(action[:3] * rate_scale, -rate_limit, rate_limit)
        collective_scale = self.hover_collective / 0.6328
        collective = torch.clamp(
            collective_scale * (action[3] + 1.0) * 0.5,
            0.0,
            0.9 * collective_scale,
        )
        self.previous_action = torch.cat((rates_flu, collective.reshape(1)))
        return (
            float(rates_flu[0]),
            float(-rates_flu[1]),
            float(-rates_flu[2]),
            float(-collective),
        )

    def _publish_rates(self, roll: float, pitch: float, yaw: float, thrust: float) -> None:
        message = VehicleRatesSetpoint()
        message.roll = roll
        message.pitch = pitch
        message.yaw = yaw
        message.thrust_body = [0.0, 0.0, thrust]
        message.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.rates_pub.publish(message)

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
        self._publish_rates(*self._policy_command())


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BodyRatesController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
