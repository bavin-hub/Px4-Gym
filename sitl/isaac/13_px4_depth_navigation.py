#!/usr/bin/env python
"""Run PX4 SITL inside the depth-navigation obstacle course and publish depth.

This standalone Pegasus example reproduces the evaluation half of the
``starling_aerial_isaac`` Deep Collision Encoder navigation task
(``aerial_isaac_lab/tasks/starling2max_navigation_env.py``) with PX4 in the
control loop instead of the Isaac Lab Lee velocity controller.

It exists so the ROS 2 policy node ``depth_policy/depth_velocity_sp.py`` can be
evaluated against renders produced by the same engine family the policy was
trained on, rather than against Gazebo:

* the **obstacle course** is the training ``ObstacleFieldCfg`` box -- 3 panels,
  47 objects drawn from the same four box variants, and the six enclosing walls
  -- rebuilt from procedural cuboids with the training sampling ranges;
* the **depth camera** matches ``Starling2MaxNavigationEnvCfg.depth_camera``:
  240x135 at an 87 deg horizontal FOV, mounted at (0.10, 0.0, 0.03) in the body
  FLU frame, running at the 10 Hz policy rate;
* depth is published as ``distance_to_image_plane`` on ``/starling/raw_depth``,
  which is the exact quantity the training environment feeds to the frozen
  Deep Collision Encoder.

Depth is **ray-cast, not rendered**.  ``MultiMeshRayCasterCamera`` casts one ray
per pixel against the obstacle meshes; going through Isaac Sim's RTX pipeline
instead would reproduce the intrinsics but not the per-pixel behaviour at edges,
thin geometry and grazing surfaces.  Because every solid in this course is a
box of known pose and size, :class:`BoxField` reproduces the training sensor
analytically: the same pinhole ray pattern, exact ray-box intersection in place
of Warp's ray-triangle intersection, the same projection onto the camera's
forward axis, and the same far-plane clipping.  There is no near plane, exactly
as in training.

What is deliberately *not* reproduced is the controller: PX4's multicopter
velocity controller replaces ``LeeVelocityController``, which is the point of
the exercise -- everything else is held as close to training as possible so any
behaviour gap can be attributed to the controller rather than to perception.

:data:`SCENE_MODE` swaps the world between that training corridor and a replica
of the PX4 Gazebo ``default.sdf`` tree world, so the *scene* can be A/B tested
while the simulator, PX4 and the policy are all held fixed.

Frames
------
The training world is +X forward / +Y left / +Z up with the corridor running
along +X.  Isaac Sim (and Pegasus) is ENU, and Pegasus converts ENU -> NED for
PX4 as ``N = ENU_y``, ``E = ENU_x``, ``D = -ENU_z``.  The corridor is therefore
laid out along **ENU +Y** so that PX4 -- and the policy node's NED +X -- point
down the corridor.  The mapping from training-corridor coordinates to ENU is a
+90 deg yaw about Z plus a vertical shift that puts the box floor on the ground
plane, since PX4 has to take off from the ground where the training environment
started its episodes in mid-air.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import NamedTuple

import carb
import numpy as np
from isaacsim import SimulationApp


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vehicle-usd",
        type=Path,
        required=True,
        help="Starling 2 Max USD exported with body and rotor0..rotor3 rigid bodies.",
    )
    parser.add_argument(
        "--px4-dir",
        type=Path,
        default=None,
        help="PX4-Autopilot checkout; defaults to the Pegasus configuration.",
    )
    parser.add_argument(
        "--px4-airframe",
        default=None,
        help="PX4_SIM_MODEL value; defaults to the Pegasus configuration.",
    )
    parser.add_argument("--scene", choices=("corridor", "trees"), default="corridor")
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


ARGS = _parse_args()
ARGS.vehicle_usd = ARGS.vehicle_usd.expanduser().resolve()
if not ARGS.vehicle_usd.is_file():
    raise FileNotFoundError(f"Starling vehicle USD not found: {ARGS.vehicle_usd}")

# SimulationApp must be created before importing the remaining Isaac modules.
simulation_app = SimulationApp({"headless": ARGS.headless})

import omni.timeline  # noqa: E402
import torch  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.api.objects import (  # noqa: E402
    FixedCuboid,
    FixedCylinder,
    FixedSphere,
    VisualSphere,
)
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
from isaacsim.core.utils.prims import create_prim  # noqa: E402
from isaacsim.core.utils.viewports import set_camera_view  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

enable_extension("isaacsim.ros2.bridge")

import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402
from sensor_msgs.msg import Image  # noqa: E402

from pegasus.simulator.logic.backends.px4_mavlink_backend import (  # noqa: E402
    PX4MavlinkBackend,
    PX4MavlinkBackendConfig,
)
from pegasus.simulator.logic.backends.ros2_backend import ROS2Backend  # noqa: E402
from pegasus.simulator.logic.interface.pegasus_interface import (  # noqa: E402
    PegasusInterface,
)
from pegasus.simulator.logic.vehicles.multirotor import (  # noqa: E402
    Multirotor,
    MultirotorConfig,
)
from pegasus.simulator.params import SIMULATION_ENVIRONMENTS  # noqa: E402


SCENE_MODE = ARGS.scene
"""Which world to build.

``corridor``
    The training ``ObstacleFieldCfg`` box: panels and objects inside six walls.
    This is the distribution the policy was trained on.
``trees``
    A replica of the PX4 Gazebo ``default.sdf`` world -- 14 cylinder-and-sphere
    trees over an open ground plane, no enclosing box -- with the drone spawned
    where and how PX4's gz bridge spawns it. Use this to A/B the *scene* against
    the corridor while holding the simulator, PX4 and the policy fixed: if the
    policy fails here too, the Gazebo failure is the scene, not the plumbing.
"""


# --------------------------------------------------------------------------- #
# Gazebo default.sdf tree world, transcribed for SCENE_MODE = "trees".
# --------------------------------------------------------------------------- #

# (x_east, y_north, trunk_radius, trunk_length, canopy_radius, canopy_z), in
# metres. Gazebo's world frame is already ENU, so these need no remapping.
TREES_ENU = (
    (3.0, 2.0, 0.18, 4.0, 1.25, 4.20),
    (3.0, -2.0, 0.20, 4.0, 1.35, 4.30),
    (6.0, 0.8, 0.16, 4.0, 1.15, 4.10),
    (-3.0, 2.0, 0.19, 4.0, 1.30, 4.25),
    (-4.0, -1.5, 0.17, 4.0, 1.20, 4.15),
    (0.0, -3.5, 0.20, 4.0, 1.40, 4.35),
    (7.0, -4.5, 0.18, 4.0, 1.25, 4.20),
    (9.0, 4.0, 0.20, 4.0, 1.35, 4.30),
    (-7.0, 4.5, 0.17, 4.0, 1.20, 4.15),
    (-9.0, -4.0, 0.19, 4.0, 1.30, 4.25),
    (2.0, 7.0, 0.16, 4.0, 1.15, 4.10),
    (-2.0, -7.0, 0.20, 4.0, 1.40, 4.35),
    (12.0, 0.5, 0.18, 4.0, 1.25, 4.20),
    (-12.0, -0.5, 0.18, 4.0, 1.25, 4.20),
)

TRUNK_COLOR = np.array([0.32, 0.14, 0.05])
CANOPY_COLOR = np.array([0.05, 0.34, 0.07])

TREE_LAYOUT = "corridor"
"""How the trees are arranged.

``gazebo``
    The verbatim ``default.sdf`` positions, with the drone spawned at the origin
    facing east exactly as PX4's gz bridge spawns it. The goal is north, so it
    starts 90 deg off the nose -- faithful to the Gazebo run, but the policy has
    to yaw a quarter turn before it can make any progress.
``corridor``
    The same trees re-scattered into a band along the flight path, with the
    drone spawned facing the goal. This isolates obstacle *geometry* -- trunks,
    canopies and open sky rather than boxes in a closed room -- from the
    initial-heading problem, matching how the training corridor starts.
"""

TREE_GOAL_NED_M = np.array([5.0, 0.0, -3.0])
"""Goal in PX4 local NED, matching the policy node's waypoint."""

# Band the corridor layout scatters trees into, relative to the spawn: metres
# along the flight path (north) and across it (east). The near edge keeps the
# takeoff column clear; the far edge runs well past the goal so the view ahead
# has something in it. With nothing beyond the goal the upper image sits
# entirely on the far plane, which is nothing like the closed room of training.
TREE_ALONG_TRACK_M = (2.5, 14.0)
TREE_CROSS_TRACK_M = (-4.5, 4.5)
TREE_CORRIDOR_COUNT = 16
TREE_BACKGROUND_COUNT = 36
TREE_BACKGROUND_NORTH_M = (-15.0, 30.0)
TREE_BACKGROUND_EAST_M = (-20.0, 20.0)
TREE_MIN_SEPARATION_M = 2.2
"""Minimum trunk-to-trunk spacing.

At the 3 m flight altitude a canopy presents only a ~0.5 m radius cross-section,
so this keeps the forward region populated without closing every route through
the trees. The separate goal-clear radius below keeps the waypoint reachable.
"""
GOAL_CLEAR_RADIUS_M = 2.5
"""No trunk is placed within this horizontal radius of the goal.

Wide enough that the nearest canopy surface still clears the policy node's
1.0 m waypoint threshold, so the goal region itself is always reachable.
"""
SPAWN_CLEAR_RADIUS_M = 2.5
"""No trunk is placed within this horizontal radius of the takeoff point."""
TREE_SEED = 1


# --------------------------------------------------------------------------- #
# Obstacle course -- ported verbatim from ObstacleFieldCfg.
# --------------------------------------------------------------------------- #

# Corridor-frame bounds, in metres. The box is 11.0 x 6.5 x 5.0 m.
BOUNDS_MIN = np.array([-1.5, -3.25, -2.5])
BOUNDS_MAX = np.array([9.5, 3.25, 2.5])

PANEL_SIZE = (0.1, 1.2, 3.0)
"""``environment_assets/panels/panel.urdf``."""

OBJECT_SIZES = (
    (0.1, 0.5, 0.5),   # wall_0_5
    (0.1, 1.0, 1.0),   # wall_1
    (0.1, 0.1, 2.0),   # rod
    (0.4, 0.4, 0.4),   # cube
)
"""The four box variants in ``environment_assets/objects/``."""

NUM_PANELS = 3
NUM_OBJECTS = 47

PANEL_MIN_RATIO = np.array([0.30, 0.05, 0.05])
PANEL_MAX_RATIO = np.array([0.85, 0.95, 0.95])
PANEL_MIN_EULER = np.array([0.0, 0.0, -math.pi / 3.0])
PANEL_MAX_EULER = np.array([0.0, 0.0, math.pi / 3.0])

OBJECT_MIN_RATIO = np.array([0.30, 0.05, 0.05])
OBJECT_MAX_RATIO = np.array([0.85, 0.90, 0.90])
OBJECT_MIN_EULER = np.array([-math.pi, -math.pi, -math.pi])
OBJECT_MAX_EULER = np.array([math.pi, math.pi, math.pi])

WALL_THICKNESS = 0.2

PANEL_COLOR = np.array([0.67, 0.26, 0.26])
OBJECT_COLOR = np.array([0.35, 0.45, 0.55])
WALL_COLOR = np.array([0.55, 0.55, 0.58])

CURRICULUM_LEVEL = 18
"""Active obstacle count, i.e. the training curriculum level.

The original DCE evaluation used 36; this is easier while staying inside the
training curriculum's 15-50 range, so the course is still in distribution.
Raise it back to 36 for the paper's setting.

The pool is ordered panels-first, so lowering this drops the *last* slots. The
tightest gaps along the flight line come from the first few obstacles and barely
move; what changes is the overall clutter the camera sees.
"""

OBSTACLE_SEED = 53
"""Seed for the obstacle pose draw, so a run can be reproduced exactly.

Worth tuning rather than leaving at 1: the level only trims the *end* of the
pool, so the tightest gaps -- which come from the first few slots -- do not move
when it is lowered. Re-drawing repositions everything. This seed keeps the
direct line blocked while leaving a ~0.9 m through-route.
"""


# --------------------------------------------------------------------------- #
# Vehicle, camera and task geometry.
# --------------------------------------------------------------------------- #

# Starling2MaxNavigationEnvCfg.depth_camera / models/starling2max_depth/model.sdf.
# The real Starling 2 Max C29 forward ToF (PMD M0178), per CHANGE10 in the
# training env: 240x180, HFOV 86 deg, VFOV 106 deg, 0.2-6 m.
#
# The pixels are NOT square. Isaac Lab derives vertical_aperture from the aspect
# ratio only when it is left None; the training env now sets it explicitly, so
# fy must be computed from the vertical FOV rather than copied from fx.
DEPTH_IMAGE_WIDTH = 240
DEPTH_IMAGE_HEIGHT = 180
DEPTH_HORIZONTAL_FOV_DEG = 86.0
DEPTH_VERTICAL_FOV_DEG = 106.0
DEPTH_CAMERA_OFFSET_FLU_M = np.array([0.10, 0.0, 0.03])
DEPTH_TOPIC = "/starling/raw_depth"
DEPTH_RATE_HZ = 10.0

# MultiMeshRayCasterCameraCfg.max_distance, with depth_clipping_behavior="max":
# ray misses and out-of-range hits both land on the far plane, which the policy
# node's normalize_depth_image() then maps to +1.0.
DEPTH_MAX_DISTANCE_M = 6.0

# There is deliberately no near plane. The training ray-caster had none, so
# sub-0.2 m returns arrived as real distances and became the encoder's -1 "too
# close" sentinel. A near plane would swallow them and the sentinel would never
# fire -- which is exactly what happens with the Gazebo model's <near>0.2</near>.

# Spawn at the low-x end of the corridor, on the floor. The training episodes
# started in mid-air at an x ratio of 0.1-0.2, i.e. x in [-0.4, 0.7]; PX4 has to
# start landed, so the drone sits at x = 0 and climbs under its own takeoff.
SPAWN_CORRIDOR_XY_M = np.array([0.0, 0.0])
SPAWN_HEIGHT_M = 0.07
SPAWN_YAW_OFFSET_DEG = 0.0
"""Yaw offset from straight-down-corridor. Training reset sampled +/- 30 deg."""

NODE_TAKEOFF_ALTITUDE_M = 3.0
"""Mirrors ``depth_velocity_sp.py``'s ``takeoff_height = -3.0`` in NED."""

# Training sampled the goal at an x ratio of 0.90-0.94, i.e. x in [8.4, 8.84],
# and never resampled it within an episode. A fixed goal keeps the evaluation
# deterministic; its height is pinned to the policy node's takeoff altitude so
# the printed waypoint needs no vertical correction. The resulting height above
# the floor sits inside the training goal band of 0.5-4.5 m.
GOAL_CORRIDOR_M = np.array(
    [8.6, 0.0, SPAWN_HEIGHT_M + NODE_TAKEOFF_ALTITUDE_M + BOUNDS_MIN[2]]
)
GOAL_MARKER_RADIUS_M = 0.30
"""Radius of the goal gizmo.

Visual only: it has no collider, and it is not in the ray-cast set, so it never
appears in the depth image the policy sees.
"""
SUCCESS_DISTANCE_M = 1.0
"""Starling2MaxNavigationEnvCfg.success_distance."""

# Arm half-span (hypot(0.095, 0.130) = 0.161 m) plus the 0.090 m propeller
# radius, rounded up. Used by the analytic crash check below.
COLLISION_RADIUS_M = 0.20
CRASH_ARM_HEIGHT_M = 1.0
"""Crash reporting starts only once the drone has climbed this high."""


# --------------------------------------------------------------------------- #
# Viewport camera.
# --------------------------------------------------------------------------- #

VIEW_MODE = "chase"
"""Which camera follows the drone.

``chase``
    Trails behind and above the drone, rotating with its yaw, so the drone stays
    centred with the corridor ahead of it. This is the one to watch it fly.
``follow``
    Reproduces the training ``ViewerCfg``: the same offsets, but as a pure world
    translation that does *not* rotate with yaw, so the camera always trails
    down the corridor.
``fpv``
    Rides the depth camera itself. You see what the policy sees, not the drone.
``static``
    Leaves the viewport alone so you can fly it manually.
"""

# Training ViewerCfg eye/lookat, in the corridor frame and relative to the root.
VIEW_EYE_OFFSET_M = np.array([-3.0, 0.0, 1.2])
VIEW_TARGET_OFFSET_M = np.array([3.0, 0.0, 0.0])

VIEW_SMOOTHING = 0.15
"""Per-update blend toward the desired pose. 1.0 is rigid, lower is smoother."""

VIEW_RATE_HZ = 30.0


def corridor_to_enu(points: np.ndarray) -> np.ndarray:
    """Map training-corridor coordinates to Isaac Sim ENU.

    The corridor frame is +X forward / +Y left / +Z up with the origin at the
    box centre in x and y.  ENU is +X east / +Y north / +Z up.  Sending the
    corridor's forward axis to north makes it PX4's (and the policy node's)
    NED +X, and the vertical shift puts the box floor on the ground plane.

    Args:
        points: Corridor-frame points, shape ``(..., 3)``, in metres.

    Returns:
        ENU points of the same shape, in metres.
    """
    points = np.asarray(points, dtype=float)
    enu = np.empty_like(points)
    enu[..., 0] = -points[..., 1]
    enu[..., 1] = points[..., 0]
    enu[..., 2] = points[..., 2] - BOUNDS_MIN[2]
    return enu


# The corridor -> ENU map above is a pure +90 deg yaw plus a translation, so
# obstacle and vehicle orientations only need the same yaw applied on the left.
CORRIDOR_TO_ENU_ROTATION = Rotation.from_euler("z", 90.0, degrees=True)


class OrientedBox:
    """An oriented box used for the analytic crash check.

    PhysX contact reporting is not wired up in this standalone app, so crashes
    are detected geometrically instead: the drone is treated as a sphere of
    :data:`COLLISION_RADIUS_M` and tested against every solid in the scene.
    This is a proxy for the training environment's ``ContactSensor`` with
    ``collision_force_threshold``, not a reproduction of it -- it reports the
    moment of geometric overlap rather than a measured contact force.
    """

    def __init__(
        self,
        center_enu: np.ndarray,
        half_extents: np.ndarray,
        rotation_enu: Rotation,
    ) -> None:
        """Store the box in the form the point-distance test needs.

        Args:
            center_enu: Box centre in ENU metres, shape ``(3,)``.
            half_extents: Half sizes along the box's own axes, shape ``(3,)``.
            rotation_enu: Box-to-ENU rotation.
        """
        self._center = np.asarray(center_enu, dtype=float)
        self._half_extents = np.asarray(half_extents, dtype=float)
        self._rotation_matrix = rotation_enu.as_matrix()

    @property
    def center(self) -> np.ndarray:
        """(np.ndarray) Box centre in ENU metres, shape ``(3,)``."""
        return self._center

    @property
    def half_extents(self) -> np.ndarray:
        """(np.ndarray) Half sizes along the box's own axes, shape ``(3,)``."""
        return self._half_extents

    @property
    def rotation_matrix(self) -> np.ndarray:
        """(np.ndarray) Box-to-ENU rotation matrix, shape ``(3, 3)``."""
        return self._rotation_matrix


class BoxField:
    """Every solid in the course, stacked for clearance and ray-cast queries.

    The crash check runs on every physics step at 250 Hz and the depth ray-cast
    fires 32 400 rays at 10 Hz, so both are evaluated as batched array
    operations rather than in a Python loop over ~40 boxes.
    """

    def __init__(self, boxes: list[OrientedBox], device: str = "cpu") -> None:
        """Stack the boxes into contiguous arrays.

        Args:
            boxes: The solids to test against.
            device: Torch device for the ray-cast. The clearance check stays on
                the CPU because it runs inline with the physics step.
        """
        self._centers = np.stack([box.center for box in boxes])
        self._half_extents = np.stack([box.half_extents for box in boxes])
        # Transposed once here so the query is a plain box-frame projection.
        self._rotations_t = np.stack([box.rotation_matrix.T for box in boxes])

        self._device = device
        self._centers_t = torch.as_tensor(
            self._centers, dtype=torch.float32, device=device
        )
        self._half_extents_t = torch.as_tensor(
            self._half_extents, dtype=torch.float32, device=device
        )
        self._rotations_tt = torch.as_tensor(
            self._rotations_t, dtype=torch.float32, device=device
        )

    def __len__(self) -> int:
        """Return the number of solids in the field."""
        return self._centers.shape[0]

    @property
    def device(self) -> str:
        """(str) Torch device the ray-cast runs on."""
        return self._device

    def raycast(
        self, origin_enu: np.ndarray, directions_enu: torch.Tensor
    ) -> torch.Tensor:
        """Return the hit distance along each ray, or ``inf`` where it misses.

        Exact ray-box intersection by the slab method, which is what Warp's
        ray-triangle intersection resolves to for these axis-aligned box meshes.

        Args:
            origin_enu: Shared ray origin in ENU metres, shape ``(3,)``.
            directions_enu: Unit ray directions in ENU, shape ``(R, 3)``.

        Returns:
            Hit distances, shape ``(R,)``, with ``inf`` for rays that miss.
        """
        origin = torch.as_tensor(
            origin_enu, dtype=torch.float32, device=self._device
        )
        nearest = torch.full(
            (directions_enu.shape[0],), float("inf"), device=self._device
        )

        for index in range(len(self)):
            rotation_t = self._rotations_tt[index]
            half_extents = self._half_extents_t[index]
            origin_box = rotation_t @ (origin - self._centers_t[index])
            # rotation_t @ d for every ray, i.e. d @ rotation_t.T.
            directions_box = directions_enu @ rotation_t.T

            # Guard the division rather than the result: a ray exactly parallel
            # to a slab yields +/-inf bounds, which the min/max below handle.
            safe = torch.where(
                directions_box.abs() < 1.0e-9,
                torch.full_like(directions_box, 1.0e-9),
                directions_box,
            )
            bound_low = (-half_extents - origin_box) / safe
            bound_high = (half_extents - origin_box) / safe
            entry = torch.minimum(bound_low, bound_high).amax(dim=-1)
            exit_ = torch.maximum(bound_low, bound_high).amin(dim=-1)

            # Rays starting inside a box hit at the exit face.
            hit = (exit_ >= torch.clamp(entry, min=0.0)) & (exit_ >= 0.0)
            distance = torch.where(entry > 0.0, entry, exit_)
            nearest = torch.where(hit & (distance < nearest), distance, nearest)

        return nearest

    def min_distance_to(self, point_enu: np.ndarray) -> float:
        """Return the distance from ``point_enu`` to the nearest solid.

        Args:
            point_enu: Query point in ENU metres, shape ``(3,)``.

        Returns:
            Euclidean distance in metres to the closest box surface, or ``0.0``
            when the point lies inside any box.
        """
        offsets = np.asarray(point_enu, dtype=float) - self._centers
        offsets_box = np.einsum("nij,nj->ni", self._rotations_t, offsets)
        overshoot = np.maximum(np.abs(offsets_box) - self._half_extents, 0.0)
        return float(np.sqrt(np.einsum("ni,ni->n", overshoot, overshoot)).min())


class SphereField:
    """Stacked spheres, for the tree canopies."""

    def __init__(self, spheres: list[tuple[np.ndarray, float]], device: str) -> None:
        """Stack sphere centres and radii.

        Args:
            spheres: ``(centre_enu, radius)`` pairs, in metres.
            device: Torch device for the ray-cast.
        """
        self._device = device
        self._centers = np.stack([np.asarray(c, dtype=float) for c, _ in spheres])
        self._radii = np.array([r for _, r in spheres], dtype=float)
        self._centers_t = torch.as_tensor(
            self._centers, dtype=torch.float32, device=device
        )
        self._radii_t = torch.as_tensor(
            self._radii, dtype=torch.float32, device=device
        )

    def min_distance_to(self, point_enu: np.ndarray) -> float:
        """Return the distance to the nearest sphere surface, 0.0 if inside.

        Args:
            point_enu: Query point in ENU metres, shape ``(3,)``.

        Returns:
            Distance in metres.
        """
        offsets = np.asarray(point_enu, dtype=float) - self._centers
        return float(
            np.maximum(np.linalg.norm(offsets, axis=-1) - self._radii, 0.0).min()
        )

    def raycast(
        self, origin_enu: np.ndarray, directions_enu: torch.Tensor
    ) -> torch.Tensor:
        """Return the nearest hit distance per ray, ``inf`` where it misses.

        Args:
            origin_enu: Shared ray origin in ENU metres, shape ``(3,)``.
            directions_enu: Unit ray directions in ENU, shape ``(R, 3)``.

        Returns:
            Hit distances, shape ``(R,)``.
        """
        origin = torch.as_tensor(
            origin_enu, dtype=torch.float32, device=self._device
        )
        nearest = torch.full(
            (directions_enu.shape[0],), float("inf"), device=self._device
        )
        for index in range(self._centers_t.shape[0]):
            to_center = origin - self._centers_t[index]
            # Directions are unit length, so the quadratic's leading term is 1.
            half_b = directions_enu @ to_center
            constant = to_center @ to_center - self._radii_t[index] ** 2
            discriminant = half_b * half_b - constant
            root = torch.sqrt(discriminant.clamp_min(0.0))
            entry = -half_b - root
            exit_ = -half_b + root
            distance = torch.where(entry > 0.0, entry, exit_)
            hit = (discriminant > 0.0) & (exit_ > 0.0)
            nearest = torch.where(hit & (distance < nearest), distance, nearest)
        return nearest


class CylinderField:
    """Stacked vertical capped cylinders, for the tree trunks.

    A capped cylinder is the intersection of an infinite cylinder with a
    horizontal slab, so entry/exit bounds from both are simply merged.
    """

    def __init__(
        self, cylinders: list[tuple[np.ndarray, float, float, float]], device: str
    ) -> None:
        """Stack cylinder axes, radii and vertical extents.

        Args:
            cylinders: ``(centre_xy, radius, z_min, z_max)`` tuples, in metres.
            device: Torch device for the ray-cast.
        """
        self._device = device
        self._centers_xy = np.stack(
            [np.asarray(c, dtype=float) for c, _, _, _ in cylinders]
        )
        self._radii = np.array([r for _, r, _, _ in cylinders], dtype=float)
        self._z_min = np.array([z0 for _, _, z0, _ in cylinders], dtype=float)
        self._z_max = np.array([z1 for _, _, _, z1 in cylinders], dtype=float)

        as_tensor = lambda a: torch.as_tensor(  # noqa: E731
            a, dtype=torch.float32, device=device
        )
        self._centers_xy_t = as_tensor(self._centers_xy)
        self._radii_t = as_tensor(self._radii)
        self._z_min_t = as_tensor(self._z_min)
        self._z_max_t = as_tensor(self._z_max)

    def min_distance_to(self, point_enu: np.ndarray) -> float:
        """Return the distance to the nearest cylinder surface, 0.0 if inside.

        Args:
            point_enu: Query point in ENU metres, shape ``(3,)``.

        Returns:
            Distance in metres.
        """
        point = np.asarray(point_enu, dtype=float)
        radial = np.maximum(
            np.linalg.norm(point[:2] - self._centers_xy, axis=-1) - self._radii, 0.0
        )
        vertical = np.maximum(
            np.maximum(self._z_min - point[2], point[2] - self._z_max), 0.0
        )
        return float(np.sqrt(radial**2 + vertical**2).min())

    def raycast(
        self, origin_enu: np.ndarray, directions_enu: torch.Tensor
    ) -> torch.Tensor:
        """Return the nearest hit distance per ray, ``inf`` where it misses.

        Args:
            origin_enu: Shared ray origin in ENU metres, shape ``(3,)``.
            directions_enu: Unit ray directions in ENU, shape ``(R, 3)``.

        Returns:
            Hit distances, shape ``(R,)``.
        """
        origin = torch.as_tensor(
            origin_enu, dtype=torch.float32, device=self._device
        )
        nearest = torch.full(
            (directions_enu.shape[0],), float("inf"), device=self._device
        )
        direction_z = directions_enu[:, 2]
        safe_z = torch.where(
            direction_z.abs() < 1.0e-9, torch.full_like(direction_z, 1.0e-9), direction_z
        )

        for index in range(self._centers_xy_t.shape[0]):
            offset_xy = origin[:2] - self._centers_xy_t[index]
            direction_xy = directions_enu[:, :2]
            quad_a = (direction_xy * direction_xy).sum(dim=-1)
            half_b = direction_xy @ offset_xy
            constant = offset_xy @ offset_xy - self._radii_t[index] ** 2
            discriminant = half_b * half_b - quad_a * constant
            root = torch.sqrt(discriminant.clamp_min(0.0))
            safe_a = quad_a.clamp_min(1.0e-12)
            radial_entry = (-half_b - root) / safe_a
            radial_exit = (-half_b + root) / safe_a

            # A ray running straight up the axis never leaves the infinite
            # cylinder, so its radial bounds are unbounded.
            inside_axis = (quad_a < 1.0e-12) & (constant < 0.0)
            radial_entry = torch.where(
                inside_axis, torch.full_like(radial_entry, -float("inf")), radial_entry
            )
            radial_exit = torch.where(
                inside_axis, torch.full_like(radial_exit, float("inf")), radial_exit
            )

            slab_low = (self._z_min_t[index] - origin[2]) / safe_z
            slab_high = (self._z_max_t[index] - origin[2]) / safe_z
            slab_entry = torch.minimum(slab_low, slab_high)
            slab_exit = torch.maximum(slab_low, slab_high)

            entry = torch.maximum(radial_entry, slab_entry)
            exit_ = torch.minimum(radial_exit, slab_exit)
            distance = torch.where(entry > 0.0, entry, exit_)
            hit = (
                ((discriminant > 0.0) | inside_axis)
                & (exit_ >= torch.clamp(entry, min=0.0))
                & (exit_ > 0.0)
            )
            nearest = torch.where(hit & (distance < nearest), distance, nearest)
        return nearest


class GroundPlane:
    """The infinite ``z = 0`` ground plane the Gazebo world sits on."""

    def __init__(self, device: str) -> None:
        """Store the device the ray-cast runs on.

        Args:
            device: Torch device for the ray-cast.
        """
        self._device = device

    def min_distance_to(self, point_enu: np.ndarray) -> float:
        """Return the height above the plane.

        Args:
            point_enu: Query point in ENU metres, shape ``(3,)``.

        Returns:
            Distance in metres, 0.0 at or below the ground.
        """
        return float(max(point_enu[2], 0.0))

    def raycast(
        self, origin_enu: np.ndarray, directions_enu: torch.Tensor
    ) -> torch.Tensor:
        """Return the hit distance per ray, ``inf`` for rays not aimed down.

        Args:
            origin_enu: Shared ray origin in ENU metres, shape ``(3,)``.
            directions_enu: Unit ray directions in ENU, shape ``(R, 3)``.

        Returns:
            Hit distances, shape ``(R,)``.
        """
        height = float(origin_enu[2])
        direction_z = directions_enu[:, 2]
        distance = -height / direction_z.clamp(max=-1.0e-9)
        return torch.where(
            direction_z < -1.0e-9,
            distance,
            torch.full_like(distance, float("inf")),
        )


class Scene:
    """Every ray-cast target in the world, whatever its primitive type."""

    def __init__(self, fields: list, device: str) -> None:
        """Collect the primitive fields making up the scene.

        Args:
            fields: Objects exposing ``raycast`` and ``min_distance_to``.
            device: Torch device the ray-cast runs on.
        """
        self._fields = [field for field in fields if field is not None]
        self._device = device

    @property
    def device(self) -> str:
        """(str) Torch device the ray-cast runs on."""
        return self._device

    def min_distance_to(self, point_enu: np.ndarray) -> float:
        """Return the distance to the nearest solid of any type.

        Args:
            point_enu: Query point in ENU metres, shape ``(3,)``.

        Returns:
            Distance in metres.
        """
        return min(field.min_distance_to(point_enu) for field in self._fields)

    def raycast(
        self, origin_enu: np.ndarray, directions_enu: torch.Tensor
    ) -> torch.Tensor:
        """Return the nearest hit distance per ray across all primitive types.

        Args:
            origin_enu: Shared ray origin in ENU metres, shape ``(3,)``.
            directions_enu: Unit ray directions in ENU, shape ``(R, 3)``.

        Returns:
            Hit distances, shape ``(R,)``, with ``inf`` where every field misses.
        """
        nearest = None
        for field in self._fields:
            distances = field.raycast(origin_enu, directions_enu)
            nearest = distances if nearest is None else torch.minimum(nearest, distances)
        return nearest


class SceneLayout(NamedTuple):
    """Everything the app needs that depends on which scene was built."""

    solids: Scene
    spawn_enu: np.ndarray
    spawn_rotation: Rotation
    goal_enu: np.ndarray
    description: str


def build_scene(world: World) -> SceneLayout:
    """Build the world selected by :data:`SCENE_MODE`.

    Args:
        world: Active Isaac Sim world.

    Returns:
        The scene geometry plus its spawn pose and goal.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if SCENE_MODE == "trees":
        return _build_tree_world(world, device)
    return _build_corridor(world, device)


def _build_tree_world(world: World, device: str) -> SceneLayout:
    """Build the open tree world.

    The Gazebo layout keeps its original fourteen trees. The corridor layout
    combines route obstacles with a wider background forest while leaving the
    spawn and goal regions clear.

    Args:
        world: Active Isaac Sim world.
        device: Torch device for the ray-cast.

    Returns:
        The tree scene with its spawn pose and goal.
    """
    create_prim("/World/Trees", "Xform")
    placements = _tree_placements()
    trunks: list[tuple[np.ndarray, float, float, float]] = []
    canopies: list[tuple[np.ndarray, float]] = []

    for index, (east, north, trunk_radius, trunk_length, canopy_radius, canopy_z) in enumerate(
        placements
    ):
        trunk_center = np.array([east, north, 0.5 * trunk_length])
        canopy_center = np.array([east, north, canopy_z])
        world.scene.add(
            FixedCylinder(
                prim_path=f"/World/Trees/trunk_{index:02d}",
                name=f"trunk_{index:02d}",
                position=trunk_center,
                radius=trunk_radius,
                height=trunk_length,
                color=TRUNK_COLOR,
            )
        )
        world.scene.add(
            FixedSphere(
                prim_path=f"/World/Trees/canopy_{index:02d}",
                name=f"canopy_{index:02d}",
                position=canopy_center,
                radius=canopy_radius,
                color=CANOPY_COLOR,
            )
        )
        trunks.append((trunk_center[:2], trunk_radius, 0.0, trunk_length))
        canopies.append((canopy_center, canopy_radius))

    scene = Scene(
        [
            CylinderField(trunks, device),
            SphereField(canopies, device),
            GroundPlane(device),
        ],
        device,
    )

    spawn_enu = np.array([0.0, 0.0, SPAWN_HEIGHT_M])
    # gazebo: face east, as PX4's gz bridge spawns it, leaving the northward
    # goal 90 deg off the nose. corridor: face the goal, as training did.
    spawn_yaw_deg = 0.0 if TREE_LAYOUT == "gazebo" else 90.0
    spawn_rotation = Rotation.from_euler("z", spawn_yaw_deg, degrees=True)
    # NED -> ENU: east = ned_y, north = ned_x, up = -ned_down.
    goal_enu = spawn_enu + np.array(
        [TREE_GOAL_NED_M[1], TREE_GOAL_NED_M[0], -TREE_GOAL_NED_M[2]]
    )
    print(f"[scene] {TREE_LAYOUT} tree world: {len(placements)} trees, "
          "open sky, no enclosing box")
    return SceneLayout(
        scene,
        spawn_enu,
        spawn_rotation,
        goal_enu,
        f"{TREE_LAYOUT} tree world ({len(placements)} trees, open sky)",
    )


def _tree_placements() -> list[tuple[float, float, float, float, float, float]]:
    """Return the trees to spawn, as ``TREES_ENU``-shaped tuples.

    For ``TREE_LAYOUT = "gazebo"`` this is the transcribed world verbatim. For
    ``"corridor"`` the same trunk and canopy dimensions are reused for a dense
    flight-path band plus a wider background forest. Placement is rejection-
    sampled so the spawn, goal and trunk separation clearances are respected.

    Returns:
        ``(east, north, trunk_radius, trunk_length, canopy_radius, canopy_z)``.
    """
    if TREE_LAYOUT == "gazebo":
        return list(TREES_ENU)

    rng = np.random.default_rng(TREE_SEED)
    goal_east, goal_north = TREE_GOAL_NED_M[1], TREE_GOAL_NED_M[0]
    placements: list[tuple[float, float, float, float, float, float]] = []

    def add_trees(
        count: int,
        north_bounds: tuple[float, float],
        east_bounds: tuple[float, float],
    ) -> None:
        added = 0
        attempts = 0
        while added < count and attempts < 20000:
            attempts += 1
            north = rng.uniform(*north_bounds)
            east = rng.uniform(*east_bounds)
            if math.hypot(east - goal_east, north - goal_north) < GOAL_CLEAR_RADIUS_M:
                continue
            if math.hypot(east, north) < SPAWN_CLEAR_RADIUS_M:
                continue
            if any(
                math.hypot(east - other[0], north - other[1]) < TREE_MIN_SEPARATION_M
                for other in placements
            ):
                continue
            # Reuse the Gazebo trees' trunk and canopy dimensions in order.
            _, _, trunk_radius, trunk_length, canopy_radius, canopy_z = TREES_ENU[
                len(placements) % len(TREES_ENU)
            ]
            placements.append(
                (east, north, trunk_radius, trunk_length, canopy_radius, canopy_z)
            )
            added += 1

    add_trees(TREE_CORRIDOR_COUNT, TREE_ALONG_TRACK_M, TREE_CROSS_TRACK_M)
    add_trees(
        TREE_BACKGROUND_COUNT,
        TREE_BACKGROUND_NORTH_M,
        TREE_BACKGROUND_EAST_M,
    )
    return placements


def _build_corridor(world: World, device: str) -> SceneLayout:
    """Build the training obstacle box and its spawn pose and goal.

    Args:
        world: Active Isaac Sim world.
        device: Torch device for the ray-cast.

    Returns:
        The corridor scene with its spawn pose and goal.
    """
    boxes = build_obstacle_course(world)
    scene = Scene([BoxField(boxes, device=device)], device)

    spawn_enu = corridor_to_enu(
        np.array([SPAWN_CORRIDOR_XY_M[0], SPAWN_CORRIDOR_XY_M[1], BOUNDS_MIN[2]])
    )
    spawn_enu[2] = SPAWN_HEIGHT_M
    spawn_rotation = CORRIDOR_TO_ENU_ROTATION * Rotation.from_euler(
        "z", SPAWN_YAW_OFFSET_DEG, degrees=True
    )
    return SceneLayout(
        scene,
        spawn_enu,
        spawn_rotation,
        corridor_to_enu(GOAL_CORRIDOR_M),
        f"training corridor ({CURRICULUM_LEVEL} obstacles)",
    )


def build_obstacle_course(world: World) -> list[OrientedBox]:
    """Spawn the training obstacle box and return its solids for crash testing.

    Obstacle poses are drawn once at startup from the training ranges, with
    ``CURRICULUM_LEVEL`` slots active.  The training environment kept a
    fixed-size pool and parked inactive slots far below the box; here the
    inactive slots are simply not created, which is equivalent for a run that
    never resets.

    Args:
        world: Active Isaac Sim world.

    Returns:
        Every solid in the course, as oriented boxes in ENU.
    """
    rng = np.random.default_rng(OBSTACLE_SEED)
    span = BOUNDS_MAX - BOUNDS_MIN
    solids: list[OrientedBox] = []

    # Grouping Xforms must exist before anything is spawned beneath them, so the
    # course does not leave untyped ancestor prims on the stage.
    for group in ("/World/Obstacles", "/World/Walls"):
        create_prim(group, "Xform")

    # Pool ordering is panels first, then objects cycling through the variants,
    # so low curriculum levels still contain the large occluding panels.
    num_active = int(np.clip(CURRICULUM_LEVEL, 0, NUM_PANELS + NUM_OBJECTS))
    for slot in range(num_active):
        is_panel = slot < NUM_PANELS
        if is_panel:
            size = PANEL_SIZE
            min_ratio, max_ratio = PANEL_MIN_RATIO, PANEL_MAX_RATIO
            min_euler, max_euler = PANEL_MIN_EULER, PANEL_MAX_EULER
            color = PANEL_COLOR
        else:
            size = OBJECT_SIZES[(slot - NUM_PANELS) % len(OBJECT_SIZES)]
            min_ratio, max_ratio = OBJECT_MIN_RATIO, OBJECT_MAX_RATIO
            min_euler, max_euler = OBJECT_MIN_EULER, OBJECT_MAX_EULER
            color = OBJECT_COLOR

        ratio = min_ratio + rng.random(3) * (max_ratio - min_ratio)
        position_corridor = BOUNDS_MIN + span * ratio
        euler = min_euler + rng.random(3) * (max_euler - min_euler)

        # quat_from_euler_xyz in the training repo is the extrinsic xyz
        # convention, which is scipy's lowercase "xyz".
        rotation_enu = CORRIDOR_TO_ENU_ROTATION * Rotation.from_euler("xyz", euler)
        position_enu = corridor_to_enu(position_corridor)
        quaternion_xyzw = rotation_enu.as_quat()

        world.scene.add(
            FixedCuboid(
                prim_path=f"/World/Obstacles/obstacle_{slot:02d}",
                name=f"obstacle_{slot:02d}",
                position=position_enu,
                orientation=np.roll(quaternion_xyzw, 1),
                scale=np.array(size),
                size=1.0,
                color=color,
            )
        )
        solids.append(
            OrientedBox(position_enu, 0.5 * np.array(size), rotation_enu)
        )

    solids.extend(_build_boundary_walls(world))
    print(f"[scene] training corridor: {num_active} obstacles "
          f"+ {len(solids) - num_active} walls")
    return solids


def _build_boundary_walls(world: World) -> list[OrientedBox]:
    """Spawn the five enclosing walls and return them as oriented boxes.

    The training environment spawned six walls and no ground plane.  Here the
    bottom wall is dropped in favour of the environment's ground plane, because
    PX4 has to take off from a surface that its EKF and land detector agree is
    the ground.

    Args:
        world: Active Isaac Sim world.

    Returns:
        The wall solids, in ENU.
    """
    thickness = WALL_THICKNESS
    size_x, size_y, size_z = BOUNDS_MAX - BOUNDS_MIN
    center_x, center_y, center_z = 0.5 * (BOUNDS_MAX + BOUNDS_MIN)

    walls = {
        "back": (
            (thickness, size_y + 2 * thickness, size_z + 2 * thickness),
            (BOUNDS_MIN[0] - 0.5 * thickness, center_y, center_z),
        ),
        "front": (
            (thickness, size_y + 2 * thickness, size_z + 2 * thickness),
            (BOUNDS_MAX[0] + 0.5 * thickness, center_y, center_z),
        ),
        "right": (
            (size_x + 2 * thickness, thickness, size_z + 2 * thickness),
            (center_x, BOUNDS_MIN[1] - 0.5 * thickness, center_z),
        ),
        "left": (
            (size_x + 2 * thickness, thickness, size_z + 2 * thickness),
            (center_x, BOUNDS_MAX[1] + 0.5 * thickness, center_z),
        ),
        "top": (
            (size_x + 2 * thickness, size_y + 2 * thickness, thickness),
            (center_x, center_y, BOUNDS_MAX[2] + 0.5 * thickness),
        ),
        "bottom": (
            (size_x + 2 * thickness, size_y + 2 * thickness, thickness),
            (center_x, center_y, BOUNDS_MIN[2] - 0.5 * thickness),
        ),
    }

    solids: list[OrientedBox] = []
    for name, (size, position_corridor) in walls.items():
        position_enu = corridor_to_enu(np.array(position_corridor))
        quaternion_xyzw = CORRIDOR_TO_ENU_ROTATION.as_quat()

        # The bottom wall's top face lands exactly on the ground plane, so
        # spawning it too would leave PhysX with coplanar colliders under the
        # drone. It is still ray-cast against, so the depth image sees the floor
        # the training environment had.
        if name != "bottom":
            world.scene.add(
                FixedCuboid(
                    prim_path=f"/World/Walls/wall_{name}",
                    name=f"wall_{name}",
                    position=position_enu,
                    orientation=np.roll(quaternion_xyzw, 1),
                    scale=np.array(size),
                    size=1.0,
                    color=WALL_COLOR,
                )
            )
        solids.append(
            OrientedBox(
                position_enu, 0.5 * np.array(size), CORRIDOR_TO_ENU_ROTATION
            )
        )
    return solids


def create_goal_marker(world: World, goal_enu: np.ndarray) -> VisualSphere:
    """Create a collision-free sphere showing the navigation goal.

    Args:
        world: Active Isaac Sim world.
        goal_enu: Goal position in ENU metres.

    Returns:
        Visual sphere marking the goal in the GUI.
    """
    marker = VisualSphere(
        prim_path="/World/navigation_goal",
        name="navigation_goal",
        position=goal_enu,
        radius=GOAL_MARKER_RADIUS_M,
        color=np.array([1.0, 0.9, 0.1]),
    )
    world.scene.add(marker)
    return marker


class RayCastDepthCamera:
    """Ray-cast depth sensor reproducing ``MultiMeshRayCasterCamera``.

    The ray pattern is Isaac Lab's ``pinhole_camera_pattern`` reproduced exactly:
    pixel centres through the inverse intrinsic matrix, reordered into the
    robotics camera frame (+X forward, +Y left, +Z up) and normalised.

    The intrinsics match the training sensor bit for bit. Isaac Lab computes
    ``fx = width * focal_length / horizontal_aperture`` and
    ``fy = height * focal_length / vertical_aperture``; substituting the env's
    aperture definitions collapses those to ``fx = width / (2 tan(HFOV/2))`` and
    ``fy = height / (2 tan(VFOV/2))``. Because the C29 ToF has non-square pixels,
    ``fy != fx`` -- deriving one from the other would give a 70 deg vertical FOV
    instead of 106.

    Depth is projected onto the camera's forward axis and clipped to the far
    plane, which is what ``depth_clipping_behavior="max"`` did in training.
    """

    def __init__(self, solids: Scene, ros_node: Node) -> None:
        """Build the ray pattern and the ROS 2 publisher.

        Args:
            solids: Geometry to cast against.
            ros_node: Existing Pegasus ROS 2 node to publish from.
        """
        self._solids = solids
        self._node = ros_node
        self._publish_period_s = 1.0 / DEPTH_RATE_HZ
        self._time_since_publish_s = 0.0
        self._vehicle = None
        self._frames_published = 0

        self._publisher = ros_node.create_publisher(
            Image, DEPTH_TOPIC, rclpy.qos.qos_profile_sensor_data
        )

        focal_x_px = 0.5 * DEPTH_IMAGE_WIDTH / math.tan(
            0.5 * math.radians(DEPTH_HORIZONTAL_FOV_DEG)
        )
        focal_y_px = 0.5 * DEPTH_IMAGE_HEIGHT / math.tan(
            0.5 * math.radians(DEPTH_VERTICAL_FOV_DEG)
        )
        self._directions_camera = self._build_ray_pattern(
            focal_x_px, focal_y_px, solids.device
        )

        # The forward component of each unit ray direction turns a hit distance
        # into a distance to the image plane. Constant in the camera frame, so
        # it is folded in once here.
        self._forward_component = self._directions_camera[:, 0].contiguous()

        message = Image()
        message.height = DEPTH_IMAGE_HEIGHT
        message.width = DEPTH_IMAGE_WIDTH
        message.encoding = "32FC1"
        message.is_bigendian = 0
        message.step = DEPTH_IMAGE_WIDTH * 4
        message.header.frame_id = "depth_camera"
        self._message = message

    @staticmethod
    def _build_ray_pattern(
        focal_x_px: float, focal_y_px: float, device: str
    ) -> torch.Tensor:
        """Return unit ray directions in the camera frame, shape ``(H * W, 3)``.

        Args:
            focal_x_px: Horizontal focal length in pixels.
            focal_y_px: Vertical focal length in pixels. Not equal to
                ``focal_x_px``: the C29 ToF has non-square pixels.
            device: Torch device to build the pattern on.

        Returns:
            Ray directions ordered row-major from the top-left pixel, matching
            the training sensor's image layout.
        """
        intrinsics = torch.tensor(
            [
                [focal_x_px, 0.0, 0.5 * DEPTH_IMAGE_WIDTH],
                [0.0, focal_y_px, 0.5 * DEPTH_IMAGE_HEIGHT],
                [0.0, 0.0, 1.0],
            ],
            device=device,
        )
        grid = torch.meshgrid(
            torch.arange(DEPTH_IMAGE_WIDTH, dtype=torch.int32, device=device),
            torch.arange(DEPTH_IMAGE_HEIGHT, dtype=torch.int32, device=device),
            indexing="xy",
        )
        pixels = torch.vstack(list(map(torch.ravel, grid))).T.float()
        pixels = torch.hstack([pixels, torch.ones((len(pixels), 1), device=device)])
        # Sample the centre of each pixel, not its corner.
        pixels += torch.tensor([[0.5, 0.5, 0.0]], device=device)

        in_camera = torch.inverse(intrinsics) @ pixels.T
        # (x right, y down, z forward) -> (x forward, y left, z up).
        in_camera = in_camera[[2, 0, 1], :] * torch.tensor(
            [1.0, -1.0, -1.0], device=device
        ).unsqueeze(1)
        return (in_camera / in_camera.norm(dim=0, keepdim=True)).T.contiguous()

    def attach(self, vehicle) -> None:
        """Bind the sensor to the vehicle it rides on.

        Args:
            vehicle: Pegasus vehicle providing the camera pose.
        """
        self._vehicle = vehicle

    def update(self, dt: float) -> None:
        """Cast and publish one depth frame when the period elapses.

        Args:
            dt: Physics-step duration in seconds.
        """
        self._time_since_publish_s += dt
        if self._time_since_publish_s < self._publish_period_s:
            return
        self._time_since_publish_s %= self._publish_period_s
        if self._vehicle is None:
            return

        state = self._vehicle.state
        body_to_enu = Rotation.from_quat(state.attitude)
        origin_enu = np.asarray(state.position, dtype=float) + body_to_enu.apply(
            DEPTH_CAMERA_OFFSET_FLU_M
        )

        # The camera's local rotation is identity in the body FLU frame, so the
        # body rotation carries the rays straight from camera frame to ENU.
        rotation = torch.as_tensor(
            body_to_enu.as_matrix(), dtype=torch.float32, device=self._solids.device
        )
        directions_enu = self._directions_camera @ rotation.T

        distances = self._solids.raycast(origin_enu, directions_enu)
        depth = distances * self._forward_component
        depth = torch.nan_to_num(
            depth, nan=DEPTH_MAX_DISTANCE_M, posinf=DEPTH_MAX_DISTANCE_M
        ).clamp(max=DEPTH_MAX_DISTANCE_M)

        self._message.header.stamp = self._node.get_clock().now().to_msg()
        self._message.data = (
            depth.to("cpu", torch.float32).numpy().tobytes()
        )
        self._publisher.publish(self._message)

        if self._frames_published == 0:
            carb.log_warn(
                f"[depth] publishing {DEPTH_TOPIC} "
                f"({DEPTH_IMAGE_WIDTH}x{DEPTH_IMAGE_HEIGHT} 32FC1) "
                f"on {self._solids.device}"
            )
        self._frames_published += 1



class ViewportCamera:
    """Drive the viewport camera so the drone stays in frame.

    Isaac Lab's ``ViewerCfg`` with ``origin_type="asset_root"`` does this
    natively; a standalone Isaac Sim app has to move the viewport camera itself,
    which is what this does once per :data:`VIEW_RATE_HZ` tick.
    """

    def __init__(self, vehicle) -> None:
        """Bind to the vehicle the camera should track.

        Args:
            vehicle: Pegasus vehicle providing the pose to follow.
        """
        self._vehicle = vehicle
        self._period_s = 1.0 / VIEW_RATE_HZ
        self._time_since_update_s = 0.0
        self._eye = None
        self._target = None

    def update(self, dt: float) -> None:
        """Move the viewport camera toward its desired pose.

        Args:
            dt: Physics-step duration in seconds.
        """
        if VIEW_MODE == "static":
            return

        self._time_since_update_s += dt
        if self._time_since_update_s < self._period_s:
            return
        self._time_since_update_s %= self._period_s

        state = self._vehicle.state
        position = np.asarray(state.position, dtype=float)
        body_to_enu = Rotation.from_quat(state.attitude)

        if VIEW_MODE == "fpv":
            # Sit on the depth camera and look straight down its axis.
            eye = position + body_to_enu.apply(DEPTH_CAMERA_OFFSET_FLU_M)
            target = eye + body_to_enu.apply(np.array([10.0, 0.0, 0.0]))
        elif VIEW_MODE == "follow":
            # Training ViewerCfg: a pure world translation, no yaw tracking.
            eye = position + CORRIDOR_TO_ENU_ROTATION.apply(VIEW_EYE_OFFSET_M)
            target = position + CORRIDOR_TO_ENU_ROTATION.apply(VIEW_TARGET_OFFSET_M)
        else:
            # Chase: the same offsets, but carried around by the drone's yaw so
            # the camera stays behind it through turns. Yaw only -- rolling the
            # camera with the airframe makes the view unreadable.
            forward = body_to_enu.apply(np.array([1.0, 0.0, 0.0]))
            yaw = Rotation.from_euler("z", math.atan2(forward[1], forward[0]))
            eye = position + yaw.apply(VIEW_EYE_OFFSET_M)
            target = position + yaw.apply(VIEW_TARGET_OFFSET_M)

        if self._eye is None:
            self._eye, self._target = eye, target
        else:
            self._eye += VIEW_SMOOTHING * (eye - self._eye)
            self._target += VIEW_SMOOTHING * (target - self._target)

        set_camera_view(eye=self._eye, target=self._target)


class NavigationMonitor:
    """Report crashes and goal arrival to the console during a run."""

    def __init__(self, vehicle, solids: Scene, goal_enu: np.ndarray) -> None:
        """Store what the per-step geometric checks need.

        Args:
            vehicle: Pegasus vehicle to track.
            solids: Every collidable solid in the scene, in ENU.
            goal_enu: Goal position in ENU metres.
        """
        self._vehicle = vehicle
        self._solids = solids
        self._goal_enu = np.asarray(goal_enu, dtype=float)
        self._armed = False
        self._crashed = False
        self._arrived = False
        self._time_since_report_s = 0.0

    def update(self, dt: float) -> None:
        """Run the crash and arrival checks for one physics step.

        Args:
            dt: Physics-step duration in seconds.
        """
        position_enu = np.asarray(self._vehicle.state.position, dtype=float)

        # The drone starts on the ground, so the checks only make sense once it
        # has actually taken off.
        if not self._armed:
            if position_enu[2] > CRASH_ARM_HEIGHT_M:
                self._armed = True
                carb.log_warn(
                    f"[nav] airborne, tracking. goal at ENU "
                    f"{np.round(self._goal_enu, 2).tolist()} m"
                )
            return

        distance_to_goal = float(np.linalg.norm(self._goal_enu - position_enu))

        if not self._arrived and distance_to_goal <= SUCCESS_DISTANCE_M:
            self._arrived = True
            carb.log_warn(
                f"[nav] GOAL reached: {distance_to_goal:.2f} m from target "
                f"(success radius {SUCCESS_DISTANCE_M:.1f} m)"
            )

        if not self._crashed:
            clearance = self._solids.min_distance_to(position_enu)
            if clearance <= COLLISION_RADIUS_M or position_enu[2] <= COLLISION_RADIUS_M:
                self._crashed = True
                carb.log_warn(
                    f"[nav] CRASH at ENU {np.round(position_enu, 2).tolist()} m, "
                    f"{distance_to_goal:.2f} m from the goal"
                )

        self._time_since_report_s += dt
        if self._time_since_report_s >= 1.0:
            self._time_since_report_s = 0.0
            status = "crashed" if self._crashed else ("arrived" if self._arrived else "flying")
            print(
                f"[nav] {status} | goal distance {distance_to_goal:5.2f} m | "
                f"ENU {np.round(position_enu, 2).tolist()}"
            )


class PegasusDepthNavigationApp:
    """Standalone PX4, obstacle-course and depth-publishing Pegasus app."""

    def __init__(self) -> None:
        """Build the world, obstacle course, vehicle, camera and monitor."""
        self.timeline = omni.timeline.get_timeline_interface()
        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world
        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Default Environment"])

        layout = build_scene(self.world)
        create_goal_marker(self.world, layout.goal_enu)

        px4_config = PX4MavlinkBackendConfig(
            {
                "vehicle_id": 0,
                "px4_autolaunch": True,
                "px4_dir": str(ARGS.px4_dir.expanduser().resolve())
                if ARGS.px4_dir is not None
                else self.pg.px4_path,
                "px4_vehicle_model": ARGS.px4_airframe
                or self.pg.px4_default_airframe,
                "input_scaling": [1048.0, 1048.0, 1048.0, 1048.0],
                "zero_position_armed": [209.0, 209.0, 209.0, 209.0],
            }
        )
        ros_backend = ROS2Backend(
            vehicle_id=0,
            config={
                "namespace": "drone",
                "pub_sensors": False,
                # Depth does not go through Pegasus' graphical-sensor path: it
                # is ray-cast rather than rendered, and published directly.
                "pub_graphical_sensors": False,
                "pub_state": True,
                "pub_tf": True,
                "sub_control": False,
            },
        )

        multirotor_config = MultirotorConfig()
        # PX4 must remain first because Pegasus takes rotor input from backend 0.
        multirotor_config.backends = [PX4MavlinkBackend(px4_config), ros_backend]

        drone = Multirotor(
            "/World/quadrotor",
            str(ARGS.vehicle_usd),
            0,
            layout.spawn_enu.tolist(),
            layout.spawn_rotation.as_quat(),
            config=multirotor_config,
        )

        self.depth_camera = RayCastDepthCamera(layout.solids, ros_backend.node)
        self.depth_camera.attach(drone)
        self.monitor = NavigationMonitor(drone, layout.solids, layout.goal_enu)
        self.viewport_camera = ViewportCamera(drone)

        # These run off the physics clock rather than the render clock, so the
        # depth frame lands on a fixed 10 Hz grid the way the training
        # environment's decimation=10 did.
        self.world.add_physics_callback("/World/depth_camera", self._depth_step)
        self.world.add_physics_callback("/World/nav_monitor", self._monitor_step)
        self.world.add_physics_callback("/World/viewport", self._viewport_step)

        self.world.reset()
        self.stop_sim = False
        self._print_banner(layout)

    def _depth_step(self, dt: float) -> None:
        """Physics callback wrapper for the depth ray-cast.

        Args:
            dt: Physics-step duration in seconds.
        """
        self.depth_camera.update(dt)

    def _monitor_step(self, dt: float) -> None:
        """Physics callback wrapper for the navigation monitor.

        Args:
            dt: Physics-step duration in seconds.
        """
        self.monitor.update(dt)

    def _viewport_step(self, dt: float) -> None:
        """Physics callback wrapper for the viewport camera.

        Args:
            dt: Physics-step duration in seconds.
        """
        self.viewport_camera.update(dt)

    def _print_banner(self, layout: SceneLayout) -> None:
        """Print the goal in the NED frame the policy node expects.

        PX4 sets its local NED origin at the spawn point, so the scene goal
        converts to a waypoint the policy node can use directly.

        Args:
            layout: The scene that was built.
        """
        offset_enu = layout.goal_enu - layout.spawn_enu
        waypoint_ned = [
            round(float(offset_enu[1]), 2),
            round(float(offset_enu[0]), 2),
            round(float(-offset_enu[2]), 2),
        ]
        heading_enu = layout.spawn_rotation.apply([1.0, 0.0, 0.0])
        heading_deg = math.degrees(math.atan2(heading_enu[0], heading_enu[1]))
        goal_bearing_deg = math.degrees(math.atan2(offset_enu[0], offset_enu[1]))
        print("=" * 72)
        print(f"Scene: {layout.description}")
        print(f"Depth published on {DEPTH_TOPIC} at {DEPTH_RATE_HZ:.0f} Hz "
              f"({DEPTH_IMAGE_WIDTH}x{DEPTH_IMAGE_HEIGHT}, 32FC1 metres)")
        print(f"Set the policy node's waypoint to {waypoint_ned}  (PX4 local NED)")
        print(f"Spawn heading {heading_deg:+.0f} deg from north; goal bearing "
              f"{goal_bearing_deg:+.0f} deg -> {abs(goal_bearing_deg - heading_deg):.0f} "
              f"deg off the nose at takeoff")
        print("=" * 72)

    def run(self) -> None:
        """Run the simulation until the app closes or a stop is requested."""
        self.timeline.play()
        while simulation_app.is_running() and not self.stop_sim:
            self.world.step(render=True)

        carb.log_warn("Pegasus depth-navigation simulation is closing.")
        self.timeline.stop()
        simulation_app.close()


def main() -> None:
    """Create and run the standalone Pegasus depth-navigation application."""
    PegasusDepthNavigationApp().run()


if __name__ == "__main__":
    main()
