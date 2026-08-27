"""Observation conversion shared by Starling attitude and CTBR deployments."""

from __future__ import annotations

import torch
from px4_msgs.msg import VehicleOdometry


class StarlingHistoryObservation:
    """Build the current task's 98-D NWU/FLU observation from PX4 odometry."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._history = torch.zeros((41, 10), dtype=torch.float32, device=device)
        self._history_taps = torch.arange(5, 41, 5, device=device)

    @staticmethod
    def _quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return torch.stack(
            (
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            )
        )

    @staticmethod
    def _inverse_rotate(q: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        q_vec = q[1:4]
        return (
            vector
            - 2.0 * q[0] * torch.cross(q_vec, vector, dim=0)
            + 2.0 * torch.cross(q_vec, torch.cross(q_vec, vector, dim=0), dim=0)
        )

    def reset(self) -> None:
        self._history.zero_()

    def build(
        self,
        odometry: VehicleOdometry,
        waypoint_ned: list[float],
        previous_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(observation[1,98], current_rpy_flu[3])``."""
        if odometry.pose_frame != VehicleOdometry.POSE_FRAME_NED:
            raise ValueError(f"Unsupported pose frame: {odometry.pose_frame}")

        position_ned = torch.as_tensor(
            odometry.position, dtype=torch.float32, device=self.device
        )
        waypoint = torch.as_tensor(
            waypoint_ned, dtype=torch.float32, device=self.device
        )
        error_ned = waypoint - position_ned
        position_error_nwu = torch.stack(
            (error_ned[0], -error_ned[1], -error_ned[2])
        )

        q_ned_frd = torch.as_tensor(
            odometry.q, dtype=torch.float32, device=self.device
        )
        q_ned_frd = q_ned_frd / torch.linalg.vector_norm(q_ned_frd).clamp_min(1.0e-8)
        x_180 = torch.tensor(
            [0.0, 1.0, 0.0, 0.0], dtype=torch.float32, device=self.device
        )
        q_nwu_flu = self._quat_multiply(
            self._quat_multiply(x_180, q_ned_frd), x_180
        )
        q_nwu_flu = q_nwu_flu / torch.linalg.vector_norm(q_nwu_flu).clamp_min(1.0e-8)
        if q_nwu_flu[0] < 0.0:
            q_nwu_flu = -q_nwu_flu
        orientation_xyzw = q_nwu_flu[[1, 2, 3, 0]]

        velocity = torch.as_tensor(
            odometry.velocity, dtype=torch.float32, device=self.device
        )
        if odometry.velocity_frame == VehicleOdometry.VELOCITY_FRAME_NED:
            velocity_frd = self._inverse_rotate(q_ned_frd, velocity)
        elif odometry.velocity_frame == VehicleOdometry.VELOCITY_FRAME_BODY_FRD:
            velocity_frd = velocity
        else:
            raise ValueError(f"Unsupported velocity frame: {odometry.velocity_frame}")
        linear_velocity_flu = torch.stack(
            (velocity_frd[0], -velocity_frd[1], -velocity_frd[2])
        )

        angular_velocity_frd = torch.as_tensor(
            odometry.angular_velocity, dtype=torch.float32, device=self.device
        )
        angular_velocity_flu = torch.stack(
            (
                angular_velocity_frd[0],
                -angular_velocity_frd[1],
                -angular_velocity_frd[2],
            )
        )

        qw, qx, qy, qz = q_nwu_flu
        roll = torch.atan2(
            2.0 * (qw * qx + qy * qz),
            1.0 - 2.0 * (qx * qx + qy * qy),
        )
        pitch = torch.asin(torch.clamp(2.0 * (qw * qy - qz * qx), -1.0, 1.0))
        yaw = torch.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )
        current_rpy_flu = torch.stack((roll, pitch, yaw))

        # The current attitude and CTBR tasks replace world position error with
        # a unit goal direction in the vehicle's yaw-only frame and put raw
        # distance in channel 17.
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        error_vehicle = torch.stack(
            (
                cos_yaw * position_error_nwu[0] + sin_yaw * position_error_nwu[1],
                -sin_yaw * position_error_nwu[0] + cos_yaw * position_error_nwu[1],
                position_error_nwu[2],
            )
        )
        distance = torch.linalg.vector_norm(error_vehicle)
        goal_direction = error_vehicle / (distance + 1.0e-6)

        live = torch.cat(
            (
                goal_direction,
                orientation_xyzw,
                linear_velocity_flu,
                angular_velocity_flu,
                previous_action.to(device=self.device, dtype=torch.float32).flatten(),
                distance.reshape(1),
            )
        )
        self._history = torch.roll(self._history, shifts=1, dims=0)
        self._history[0] = live[7:17]
        history = self._history[self._history_taps].flatten()
        return torch.cat((live, history)).unsqueeze(0), current_rpy_flu
