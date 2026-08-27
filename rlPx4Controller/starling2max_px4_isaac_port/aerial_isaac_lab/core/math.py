"""Quaternion helpers with an explicit ``wxyz`` convention.

Aerial Gym used Isaac Gym's ``xyzw`` tensors.  Isaac Lab uses ``wxyz``.  This
module keeps all internal control calculations in Isaac Lab order and makes
conversion to the old policy convention explicit at the boundary.
"""

from __future__ import annotations

import torch


def quat_wxyz_to_xyzw(quat: torch.Tensor) -> torch.Tensor:
    return quat[..., (1, 2, 3, 0)]


def quat_xyzw_to_wxyz(quat: torch.Tensor) -> torch.Tensor:
    return quat[..., (3, 0, 1, 2)]


def quat_conjugate(quat: torch.Tensor) -> torch.Tensor:
    return torch.cat((quat[..., :1], -quat[..., 1:]), dim=-1)


def quat_inv(quat: torch.Tensor, eps: float = 1.0e-9) -> torch.Tensor:
    return quat_conjugate(quat) / quat.square().sum(dim=-1, keepdim=True).clamp_min(eps)


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def quat_apply(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate ``vector`` from the quaternion's local frame into its parent frame."""
    q_vec = quat[..., 1:]
    return vector + 2.0 * (
        quat[..., :1] * torch.cross(q_vec, vector, dim=-1)
        + torch.cross(q_vec, torch.cross(q_vec, vector, dim=-1), dim=-1)
    )


def quat_apply_inverse(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    return quat_apply(quat_conjugate(quat), vector)


def quat_from_euler_xyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    return torch.stack(
        (
            cy * cr * cp + sy * sr * sp,
            cy * sr * cp - sy * cr * sp,
            cy * cr * sp + sy * sr * cp,
            sy * cr * cp - cy * sr * sp,
        ),
        dim=-1,
    )


def euler_xyz_from_quat(quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    w, x, y, z = quat.unbind(dim=-1)
    roll = torch.atan2(2.0 * (w * x + y * z), w * w - x * x - y * y + z * z)
    pitch = torch.asin((2.0 * (w * y - z * x)).clamp(-1.0, 1.0))
    yaw = torch.atan2(2.0 * (w * z + x * y), w * w + x * x - y * y - z * z)
    return roll, pitch, yaw


def matrix_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """Return local-to-world rotation matrices for normalized ``wxyz`` quaternions."""
    w, x, y, z = quat.unbind(dim=-1)
    two_s = 2.0 / quat.square().sum(dim=-1).clamp_min(1.0e-9)
    return torch.stack(
        (
            1.0 - two_s * (y * y + z * z), two_s * (x * y - z * w), two_s * (x * z + y * w),
            two_s * (x * y + z * w), 1.0 - two_s * (x * x + z * z), two_s * (y * z - x * w),
            two_s * (x * z - y * w), two_s * (y * z + x * w), 1.0 - two_s * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(quat.shape[:-1] + (3, 3))


def quat_from_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to normalized ``wxyz`` quaternions."""
    m00, m01, m02 = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
    m10, m11, m12 = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
    m20, m21, m22 = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]
    q_abs = torch.sqrt(
        torch.stack((1.0 + m00 + m11 + m22, 1.0 + m00 - m11 - m22,
                     1.0 - m00 + m11 - m22, 1.0 - m00 - m11 + m22), dim=-1).clamp_min(1.0e-8)
    )
    candidates = torch.stack(
        (
            torch.stack((q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01), dim=-1),
            torch.stack((m21 - m12, q_abs[..., 1] ** 2, m01 + m10, m02 + m20), dim=-1),
            torch.stack((m02 - m20, m01 + m10, q_abs[..., 2] ** 2, m12 + m21), dim=-1),
            torch.stack((m10 - m01, m02 + m20, m12 + m21, q_abs[..., 3] ** 2), dim=-1),
        ),
        dim=-2,
    )
    candidates = candidates / (2.0 * q_abs[..., :, None]).clamp_min(1.0e-8)
    quat = candidates.gather(-2, q_abs.argmax(dim=-1)[..., None, None].expand(*q_abs.shape[:-1], 1, 4)).squeeze(-2)
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
    return torch.where(quat[..., :1] < 0.0, -quat, quat)


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))
