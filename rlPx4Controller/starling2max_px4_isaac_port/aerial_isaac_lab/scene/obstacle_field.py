"""Curriculum-driven obstacle field for the depth navigation task.

Port of Aerial Gym's ``EnvWithObstaclesCfg`` (``aerial_gym/config/env_config/env_with_obstacles.py``)
and the panel/object/wall asset parameters from
``aerial_gym/config/asset_config/env_object_config.py``.

Obstacle geometry is built entirely from Isaac Lab procedural primitives --
boxes, cylinders and spheres -- all of which the ray-caster parses directly
(``Cube``, ``Cylinder``, ``Sphere`` are in ``PRIMITIVE_MESH_TYPES``).  URDF-backed
obstacles do not work here: a URDF import applies ``RigidBodyAPI`` to a child link
rather than the slot root, so ``RigidObjectCollection`` never moves them and they
remain invisible to depth.  See :data:`SLOT_KINDS`.

Two groups are created:

* **Obstacles** -- a fixed pool of ``sum(kind_counts)`` kinematic rigid
  bodies in a :class:`~isaaclab.assets.RigidObjectCollection`.  Poses are
  resampled per environment on reset; slots beyond the current curriculum level
  are parked far below the environment so they neither occlude the depth image
  nor generate contacts.  This is how Aerial Gym varies obstacle count
  (``obs_dict["num_obstacles_in_env"] = curriculum_level``) while keeping a
  fixed-size scene.
* **Walls** -- six static box colliders closing the environment box.  Aerial Gym
  uses ``create_ground_plane = False`` and a ``bottom_wall`` instead of a ground
  plane, matching the other tasks in this repository which also spawn no ground.

The obstacle pool has ``max_level`` slots so the curriculum can reach its
maximum.  Which slots activate is chosen per environment by a scene *archetype*
(forest / mixed, see :data:`ARCHETYPES`), so low curriculum levels give a
coherent sparse scene rather than a random subset -- with a minority of box
obstacles present in every archetype.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, RigidObjectCollection, RigidObjectCollectionCfg
from isaaclab.sensors import MultiMeshRayCasterCfg
from isaaclab.utils import configclass

from aerial_isaac_lab.core.math import quat_from_euler_xyz

OBSTACLE_PRIM_ROOT = "/World/envs/env_.*/Obstacles"
WALL_PRIM_ROOT = "/World/envs/env_.*/Walls"


@dataclass(frozen=True)
class SlotKind:
    """One obstacle archetype's geometry and how it may be posed.  CHANGE6.

    Replaces the box-only ``BoxShape``.  ``geometry`` selects the spawner:

    * ``"box"``      -- ``size`` is (x, y, z) extents
    * ``"cylinder"`` -- ``size`` is (radius, height, unused)
    * ``"sphere"``   -- ``size`` is (radius, unused, unused)

    URDF-backed kinds were tried and removed; see :data:`SLOT_KINDS`.

    ``z_ratio`` places the slot vertically within the box (0 = floor, 1 = ceiling)
    so desks sit low and canopies sit high, and ``upright`` restricts rotation to
    yaw for anything that would look wrong lying on its side.
    """

    name: str
    geometry: str
    size: tuple[float, float, float] = (0.0, 0.0, 0.0)
    z_ratio: tuple[float, float] = (0.05, 0.95)
    upright: bool = False
    group: str = "box"
    """One of ``"forest"`` or ``"box"`` -- used for archetype weighting."""


SLOT_KINDS: dict[str, SlotKind] = {
    # --- forest -----------------------------------------------------------
    # All procedural primitives. Aerial Gym's trees/*.urdf and thin/*.urdf were
    # tried first but do NOT work here: a URDF import applies RigidBodyAPI to a
    # child link rather than the slot root, so RigidObjectCollection never moves
    # them and they stay invisible to the depth camera (measured: far-plane
    # fraction 1.000 with the slot teleported 2.5 m in front of the lens).
    # Primitives are verified visible, so a tree is a trunk plus a canopy sphere
    # occupying two nearby slots.
    "trunk": SlotKind("trunk", "cylinder", size=(0.15, 7.0, 0.0),
                      z_ratio=(0.35, 0.65), upright=True, group="forest"),
    "canopy": SlotKind("canopy", "sphere", size=(1.1, 0.0, 0.0),
                       z_ratio=(0.60, 0.90), group="forest"),
    "sapling": SlotKind("sapling", "cylinder", size=(0.07, 4.0, 0.0),
                        z_ratio=(0.20, 0.50), upright=True, group="forest"),
    "bush": SlotKind("bush", "sphere", size=(0.6, 0.0, 0.0),
                     z_ratio=(0.10, 0.35), group="forest"),
    # --- boxes (the original Aerial Gym obstacles, kept as a minority) -----
    "panel": SlotKind("panel", "box", size=(0.1, 1.2, 3.0), group="box"),
    "cube": SlotKind("cube", "box", size=(0.4, 0.4, 0.4), group="box"),
    "rod": SlotKind("rod", "box", size=(0.1, 0.1, 2.0), group="box"),
    "wall_1": SlotKind("wall_1", "box", size=(0.1, 1.0, 1.0), group="box"),
}

ARCHETYPES: tuple[str, ...] = ("forest", "mixed")
"""Scene styles sampled per environment per reset.  CHANGE6."""


@configclass
class ObstacleFieldCfg:
    """Geometry and randomisation ranges for the navigation obstacle field."""

    # CHANGE2: an 11 m cube centred on the environment origin, replacing Aerial
    # Gym's 11 x 6.5 x 5 corridor.  Centring it means the box centre *is* the
    # origin, which is where the goal now sits.
    #
    # Aerial Gym randomises the corridor per environment between
    # ``lower_bound_min`` / ``lower_bound_max`` and ``upper_bound_min`` /
    # ``upper_bound_max``; the box is kept identical across environments here so
    # ``replicate_physics`` stays enabled.
    #
    # To revert CHANGE2: bounds_min = (-1.5, -3.25, -2.5),
    #                    bounds_max = ( 9.5,  3.25,  2.5).
    bounds_min: tuple[float, float, float] = (-5.5, -5.5, -5.5)
    bounds_max: tuple[float, float, float] = (5.5, 5.5, 5.5)

    # CHANGE5: pool sized so the cube has the same obstacle *density* as Aerial
    # Gym's original corridor -- 1331 m^3 vs 357 m^3 is 3.73x, so 50 slots becomes
    # 186. ``curriculum.min_level`` / ``max_level`` in both depth tasks are scaled
    # by the same factor (15/50 -> 56/186).
    #
    # CHANGE6 replaces CHANGE5's panel/object split with per-kind counts, keeping
    # the same 186 total. Boxes are deliberately a minority (37 of 186, ~20%) so
    # the course reads as trees and structure with boxes mixed in rather than a
    # field of random boxes.
    # To revert CHANGE6: use only the box kinds, e.g.
    #   {"panel": 11, "cube": 60, "rod": 60, "wall_1": 55}
    kind_counts: dict[str, int] = {
        # forest: 120 slots
        "trunk": 42,
        "canopy": 33,
        "sapling": 27,
        "bush": 18,
        # boxes: 66 slots (~35%)
        "panel": 18,
        "cube": 21,
        "rod": 16,
        "wall_1": 11,
    }
    """Slot count per :data:`SLOT_KINDS` entry. Sums to the 186-slot pool."""

    # CHANGE6: relative chance each scene style is drawn per environment per
    # reset, in :data:`ARCHETYPES` order (forest, mixed).
    archetype_weights: tuple[float, float] = (0.5, 0.5)

    # CHANGE6: how strongly an archetype prefers each group when choosing which
    # slots to activate. Boxes keep a non-zero weight in *every* archetype, so
    # there are always some box obstacles present, just not many.
    group_weights: dict[str, tuple[float, float]] = {
        # (forest, mixed)
        "forest": (1.0, 0.5),
        "box": (0.15, 0.5),
    }

    num_archetype_orderings: int = 8
    """Precomputed activation orders per archetype, so composition still varies."""

    # CHANGE2: obstacles now fill the whole cube. The original x range started at
    # ratio 0.30 purely to keep a fixed obstacle-free slab around the spawn; that
    # job is now done by the dynamic spawn/goal clearances below. The z range is
    # per-kind (see SlotKind.z_ratio) since CHANGE6.
    # To revert CHANGE2: min ratio x back to 0.30, max x to 0.85.
    plan_min_ratio: tuple[float, float] = (0.05, 0.05)
    """Lower (x, y) placement ratio for every slot."""

    plan_max_ratio: tuple[float, float] = (0.95, 0.95)
    """Upper (x, y) placement ratio for every slot."""

    upright_yaw_limit: float = math.pi
    """Yaw range for ``upright`` kinds; roll and pitch stay zero."""

    # CHANGE2: obstacle-free spheres around the goal and around the drone's spawn
    # point. The goal now sits in the middle of the obstacle field rather than
    # beyond it, so without the goal clearance some episodes would be unwinnable.
    goal_clearance_radius: float = 1.5
    """Obstacle-free radius about the goal. Should exceed ``success_distance``."""

    spawn_clearance_radius: float = 1.5
    """Obstacle-free radius about the drone's spawn point."""

    spawn_forward_only: bool = True
    """Place obstacles only in the initial forward half-space of the drone.

    This is used when ``spawn_forward_local`` is supplied by a task. It matches
    the forward-only depth sensor and prevents initially invisible side/rear
    obstacles. Previous behavior: ``False`` (or omit ``spawn_forward_local``).
    """

    obstacle_core_radius: float | None = None
    """Horizontal radius about the box centre outside which obstacles are rejected."""

    clearance_attempts: int = 8
    """Resampling passes used to satisfy both clearances before parking a slot."""

    # CHANGE4: the six boundary walls are kinematic and can be parked per
    # environment at reset, so the drone sees both sealed-box depth (every ray
    # hits something inside 10 m) and open-space depth (most rays miss and return
    # the far sentinel).  The virtual box -- obstacle, goal and spawn sampling --
    # is unchanged either way.
    # To revert CHANGE4: set box_probability = 1.0 to always keep the walls.
    box_probability: float = 0.5
    """Chance the boundary walls are present, sampled per environment per reset."""

    wall_thickness: float = 0.2
    """``environment_assets/walls/*.urdf`` box thickness."""

    parked_depth: float = 1000.0
    """How far below the box inactive obstacle slots are parked, in metres."""

    parked_spacing: float = 5.0
    """Lateral spacing between parked slots, so they never overlap each other."""

    wall_color: tuple[float, float, float] = (0.55, 0.55, 0.58)
    """Obstacle colours come from ``_GROUP_COLORS`` per slot group since CHANGE6."""

    @property
    def box_size(self) -> tuple[float, float, float]:
        return tuple(hi - lo for hi, lo in zip(self.bounds_max, self.bounds_min))

    @property
    def box_center(self) -> tuple[float, float, float]:
        return tuple(0.5 * (hi + lo) for hi, lo in zip(self.bounds_max, self.bounds_min))

    @property
    def num_obstacles(self) -> int:
        return sum(self.kind_counts.values())

    def slot_kinds(self) -> list[SlotKind]:
        """Pool ordering: all slots of one kind, then the next.  CHANGE6.

        Order within the pool is irrelevant to difficulty -- which slots activate
        is decided by the per-archetype orderings in
        :class:`ObstacleFieldRandomizer`, not by slot index.
        """
        kinds: list[SlotKind] = []
        for name, count in self.kind_counts.items():
            if name not in SLOT_KINDS:
                raise KeyError(f"Unknown slot kind '{name}'. Known: {sorted(SLOT_KINDS)}")
            kinds.extend([SLOT_KINDS[name]] * count)
        return kinds


_GROUP_COLORS: dict[str, tuple[float, float, float]] = {
    "forest": (0.30, 0.42, 0.26),
    "office": (0.55, 0.54, 0.50),
    "box": (0.67, 0.26, 0.26),
}

# Kinematic rigid props shared by every obstacle spawn. Aerial Gym sets
# ``fix_base_link = True`` on all obstacle assets, so they are static; kinematic
# bodies hold a teleported pose, ignore gravity, and still report contacts
# against the (dynamic) drone.
def _rigid_props(kinematic: bool) -> sim_utils.RigidBodyPropertiesCfg:
    return sim_utils.RigidBodyPropertiesCfg(
        kinematic_enabled=kinematic, disable_gravity=True
    )


def _cuboid_spawn(
    size: tuple[float, float, float], color: tuple[float, float, float], kinematic: bool
) -> sim_utils.CuboidCfg:
    return sim_utils.CuboidCfg(
        size=size,
        rigid_props=_rigid_props(kinematic),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
    )


def _slot_spawn(kind: SlotKind, kinematic: bool = True):
    """Spawn config for one slot, dispatched on :attr:`SlotKind.geometry`.  CHANGE6.

    All geometries end up as primitives the ray-caster understands: ``Cube``,
    ``Cylinder`` and ``Sphere`` are in ``PRIMITIVE_MESH_TYPES``, and Aerial Gym's
    tree/thin URDFs are themselves built from ``<cylinder>`` / ``<box>`` links,
    merged into one body by ``merge_fixed_joints``.
    """
    color = _GROUP_COLORS[kind.group]
    if kind.geometry == "box":
        return _cuboid_spawn(kind.size, color, kinematic)
    if kind.geometry == "cylinder":
        return sim_utils.CylinderCfg(
            radius=kind.size[0],
            height=kind.size[1],
            rigid_props=_rigid_props(kinematic),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        )
    if kind.geometry == "sphere":
        return sim_utils.SphereCfg(
            radius=kind.size[0],
            rigid_props=_rigid_props(kinematic),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        )
    if kind.geometry == "urdf":
        files = sorted(_AERIAL_GYM_ASSETS.glob(kind.urdf_glob))
        if not files:
            raise FileNotFoundError(
                f"No files match '{kind.urdf_glob}' under {_AERIAL_GYM_ASSETS}. "
                "CHANGE6 uses Aerial Gym's tree/thin URDFs; adjust "
                "_AERIAL_GYM_ASSETS or drop the 'tree'/'thin' kinds."
            )
        # Deterministic pick so the same slot always gets the same asset, but
        # different slots spread over the available files.
        asset = files[slot % len(files)]
        return sim_utils.UrdfFileCfg(
            asset_path=str(asset),
            fix_base=False,
            # Trees are 13 links joined by fixed joints; merge them into one body.
            merge_fixed_joints=True,
            joint_drive=None,
            # CHANGE6: must stay False. make_instanceable=True turns the imported
            # geometry into USD instance proxies, which MultiMeshRayCaster's mesh
            # search does not traverse -- the slot then registers but is invisible
            # to depth (verified: far-plane fraction 1.000 even with the obstacle
            # teleported 2.5 m in front of the camera).
            make_instanceable=False,
            rigid_props=_rigid_props(kinematic),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        )
    raise ValueError(f"Unknown geometry '{kind.geometry}' for slot kind '{kind.name}'")


def create_group_prims() -> None:
    """Create the ``Obstacles`` and ``Walls`` grouping Xforms inside ``env_0``.

    Isaac Lab's ``@clone`` spawn decorator resolves a spawn path by splitting off
    the leaf and looking up the parent, so the parent must already exist before
    anything is spawned underneath it.  Call this at the very top of
    ``_setup_scene``, before the obstacle collection or the walls.
    """
    for root in (OBSTACLE_PRIM_ROOT, WALL_PRIM_ROOT):
        sim_utils.create_prim(root.replace("env_.*", "env_0"), "Xform")


def build_obstacle_collection_cfg(cfg: ObstacleFieldCfg) -> RigidObjectCollectionCfg:
    """Build the fixed-size pool of randomisable obstacles."""
    rigid_objects: dict[str, RigidObjectCfg] = {}
    for slot, kind in enumerate(cfg.slot_kinds()):
        rigid_objects[f"obstacle_{slot:03d}"] = RigidObjectCfg(
            prim_path=f"{OBSTACLE_PRIM_ROOT}/obstacle_{slot:03d}",
            spawn=_slot_spawn(kind, kinematic=True),
            # Spawned parked; every slot gets a real pose on the first reset.
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(slot * cfg.parked_spacing, 0.0, -cfg.parked_depth)
            ),
        )
    return RigidObjectCollectionCfg(rigid_objects=rigid_objects)


WALL_NAMES: tuple[str, ...] = ("back", "front", "right", "left", "bottom", "top")
"""Wall slot order. Fixes the object index used by :class:`BoxToggler`."""


def wall_specs(cfg: ObstacleFieldCfg) -> dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]]:
    """``{name: (size, env-local position)}`` for the six boundary walls.

    Each wall is oversized by the thickness on its in-plane axes so the corners
    close.  Shared by the collection builder and :class:`BoxToggler` so the
    nominal poses cannot drift apart.
    """
    size_x, size_y, size_z = cfg.box_size
    cx, cy, cz = cfg.box_center
    t = cfg.wall_thickness
    return {
        "back": ((t, size_y + 2 * t, size_z + 2 * t), (cfg.bounds_min[0] - 0.5 * t, cy, cz)),
        "front": ((t, size_y + 2 * t, size_z + 2 * t), (cfg.bounds_max[0] + 0.5 * t, cy, cz)),
        "right": ((size_x + 2 * t, t, size_z + 2 * t), (cx, cfg.bounds_min[1] - 0.5 * t, cz)),
        "left": ((size_x + 2 * t, t, size_z + 2 * t), (cx, cfg.bounds_max[1] + 0.5 * t, cz)),
        "bottom": ((size_x + 2 * t, size_y + 2 * t, t), (cx, cy, cfg.bounds_min[2] - 0.5 * t)),
        "top": ((size_x + 2 * t, size_y + 2 * t, t), (cx, cy, cfg.bounds_max[2] + 0.5 * t)),
    }


def build_wall_collection_cfg(cfg: ObstacleFieldCfg) -> RigidObjectCollectionCfg:
    """Build the six boundary walls as a toggleable kinematic collection.  CHANGE4.

    Previously these were plain static colliders spawned by ``spawn_boundary_walls``.
    They are kinematic rigid bodies now so :class:`BoxToggler` can park them per
    environment at reset, which is the only way to remove them from the depth
    image: ``MultiMeshRayCaster`` parses its warp meshes once and thereafter tracks
    only transforms, so hiding a wall or scaling it to zero would leave it in the
    BVH and depth would not change.

    To revert CHANGE4: restore ``spawn_boundary_walls`` (static prims), set the
    wall ray-cast target back to ``track_mesh_transforms=False``, and drop the
    toggler.
    """
    specs = wall_specs(cfg)
    rigid_objects: dict[str, RigidObjectCfg] = {}
    for name in WALL_NAMES:
        size, pos = specs[name]
        rigid_objects[name] = RigidObjectCfg(
            prim_path=f"{WALL_PRIM_ROOT}/{name}",
            spawn=_cuboid_spawn(size, cfg.wall_color, kinematic=True),
            init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
        )
    return RigidObjectCollectionCfg(rigid_objects=rigid_objects)


def raycast_targets() -> list[MultiMeshRayCasterCfg.RaycastTargetCfg]:
    """Ray-cast targets for the depth camera.

    ``MultiMeshRayCaster`` keeps a per-environment mesh list, so each drone only
    ever sees the geometry in its own environment -- there is no cross-env bleed
    and ``env_spacing`` only has to be large enough to keep the *physics* separate.

    Both targets are transform-tracked: obstacles are teleported every reset, and
    since CHANGE4 the walls are too.  Walls stay ``is_shared`` because that flag
    only controls mesh *parsing* reuse across environments (matched per prim name),
    which is still valid -- it is independent of transform tracking.
    """
    return [
        MultiMeshRayCasterCfg.RaycastTargetCfg(
            prim_expr=f"{OBSTACLE_PRIM_ROOT}/.*",
            is_shared=False,
            track_mesh_transforms=True,
            merge_prim_meshes=True,
        ),
        MultiMeshRayCasterCfg.RaycastTargetCfg(
            prim_expr=f"{WALL_PRIM_ROOT}/.*",
            is_shared=True,
            # CHANGE4: was False when the walls were static.
            track_mesh_transforms=True,
            merge_prim_meshes=True,
        ),
    ]


class BoxToggler:
    """Places or parks the six boundary walls per environment.  CHANGE4.

    With the box present the drone flies in a sealed 11 m cube and every ray hits
    something within the 10 m sensor range.  With it parked, only the obstacles
    remain, so most rays miss and return the far sentinel -- which is the point:
    the encoder sees near-obstacle structure against open space instead of always
    having a wall 5-11 m out.

    The sampling volume is unaffected either way; ``bounds_min/max`` stay the
    virtual box used for obstacle, goal and spawn placement.

    Args:
        cfg: Field configuration.
        collection: The spawned wall collection.
        env_origins: ``(num_envs, 3)`` world positions of the environment origins.
        device: Torch device.
    """

    def __init__(
        self,
        cfg: ObstacleFieldCfg,
        collection: RigidObjectCollection,
        env_origins: torch.Tensor,
        device: str | torch.device,
    ):
        self.cfg = cfg
        self.collection = collection
        self.env_origins = env_origins
        self.device = device
        self.num_walls = len(WALL_NAMES)

        specs = wall_specs(cfg)
        self._nominal_local = torch.tensor(
            [specs[name][1] for name in WALL_NAMES], device=device, dtype=torch.float32
        )
        # Parked far in +y as well as -z, so parked walls never coincide with
        # parked obstacle slots (kinematic bodies do not collide with each other,
        # but keeping them apart keeps the scene readable in the viewport).
        parked = torch.zeros((self.num_walls, 3), device=device)
        parked[:, 0] = torch.arange(self.num_walls, device=device) * (2.0 * cfg.parked_spacing)
        parked[:, 1] = cfg.parked_depth
        parked[:, 2] = -cfg.parked_depth
        self._parked_local = parked

        self._present = torch.ones(env_origins.shape[0], dtype=torch.bool, device=device)

    @property
    def present(self) -> torch.Tensor:
        """``(num_envs,)`` mask of environments that currently have the box."""
        return self._present

    def randomize(self, env_ids: torch.Tensor, probability: float) -> torch.Tensor:
        """Sample the box on/off per environment and write the wall poses.

        Args:
            env_ids: Environments to resample.
            probability: Chance the box is present. 1.0 always keeps it, 0.0 never.

        Returns:
            ``(len(env_ids),)`` bool mask of which of them got the box.
        """
        count = env_ids.numel()
        if count == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.device)

        present = torch.rand(count, device=self.device) < probability
        self._present[env_ids] = present

        pos_local = torch.where(
            present.view(count, 1, 1),
            self._nominal_local.unsqueeze(0),
            self._parked_local.unsqueeze(0),
        )
        pose = torch.zeros((count, self.num_walls, 7), device=self.device)
        pose[..., :3] = pos_local + self.env_origins[env_ids].unsqueeze(1)
        pose[..., 3] = 1.0  # identity quaternion; walls are axis-aligned
        self.collection.write_object_link_pose_to_sim(pose, env_ids=env_ids)
        return present


class ObstacleFieldRandomizer:
    """Samples obstacle poses per environment, gated by the curriculum level.

    Args:
        cfg: Field configuration.
        collection: The spawned obstacle collection.
        env_origins: ``(num_envs, 3)`` world positions of the environment origins.
        device: Torch device.
    """

    def __init__(
        self,
        cfg: ObstacleFieldCfg,
        collection: RigidObjectCollection,
        env_origins: torch.Tensor,
        device: str | torch.device,
    ):
        self.cfg = cfg
        self.collection = collection
        self.env_origins = env_origins
        self.device = device
        self.num_slots = cfg.num_obstacles

        # CHANGE6: per-slot sampling ranges derived from each slot's SlotKind, so
        # desks sit low, canopies sit high, trees and pillars stay upright, and
        # boxes tumble freely -- all sampled in one shot.
        kinds = cfg.slot_kinds()
        self.slot_kind_names = [k.name for k in kinds]
        self.slot_groups = [k.group for k in kinds]

        min_ratio, max_ratio, min_euler, max_euler = [], [], [], []
        for kind in kinds:
            min_ratio.append((cfg.plan_min_ratio[0], cfg.plan_min_ratio[1], kind.z_ratio[0]))
            max_ratio.append((cfg.plan_max_ratio[0], cfg.plan_max_ratio[1], kind.z_ratio[1]))
            if kind.upright:
                min_euler.append((0.0, 0.0, -cfg.upright_yaw_limit))
                max_euler.append((0.0, 0.0, cfg.upright_yaw_limit))
            else:
                min_euler.append((-math.pi, -math.pi, -math.pi))
                max_euler.append((math.pi, math.pi, math.pi))

        self._min_ratio = torch.tensor(min_ratio, device=device)
        self._max_ratio = torch.tensor(max_ratio, device=device)
        self._min_euler = torch.tensor(min_euler, device=device)
        self._max_euler = torch.tensor(max_euler, device=device)
        self._bounds_min = torch.tensor(cfg.bounds_min, device=device)
        self._bounds_max = torch.tensor(cfg.bounds_max, device=device)

        # Parked pose: far below the box, spread laterally so parked kinematic
        # bodies never coincide.
        parked = torch.zeros((self.num_slots, 3), device=device)
        parked[:, 0] = torch.arange(self.num_slots, device=device) * cfg.parked_spacing
        parked[:, 2] = -cfg.parked_depth
        self._parked_local = parked

        self._slot_index = torch.arange(self.num_slots, device=device)
        self._archetype_rank = self._build_archetype_ranks(cfg, kinds)
        self._archetype_probs = torch.tensor(cfg.archetype_weights, device=device).clamp_min(0.0)
        self._archetype_probs /= self._archetype_probs.sum()
        self._num_orderings = cfg.num_archetype_orderings
        self._archetype = torch.zeros(env_origins.shape[0], dtype=torch.long, device=device)

    def _build_archetype_ranks(
        self, cfg: ObstacleFieldCfg, kinds: list[SlotKind]
    ) -> torch.Tensor:
        """Precompute activation orders per (archetype, variant).  CHANGE6.

        The curriculum level says *how many* slots are active; this says *which*.
        For each archetype the slots are ranked by a weighted random draw
        (Gumbel-top-k), so a forest scene activates trunks and canopies first and
        a mixed scene draws trees and boxes evenly -- while box slots keep a
        non-zero weight everywhere, so a few boxes always appear.

        Several variants per archetype are precomputed and one is drawn per reset,
        so the *composition* varies too, not just the poses.

        Returns:
            ``(num_archetypes * num_orderings, num_slots)`` rank tensor; a slot is
            active when its rank is below the curriculum level.
        """
        weights = torch.tensor(
            [[cfg.group_weights[k.group][a] for k in kinds] for a in range(len(ARCHETYPES))],
            device=self.device,
        ).clamp_min(1.0e-6)
        # Repeat each archetype's weights once per ordering variant.
        weights = weights.repeat_interleave(cfg.num_archetype_orderings, dim=0)
        # Gumbel-top-k: sorting log(w) + Gumbel noise draws without replacement
        # with probability proportional to w.
        gumbel = -torch.log(-torch.log(torch.rand_like(weights).clamp_min(1.0e-12)))
        order = torch.argsort(weights.log() + gumbel, dim=1, descending=True)
        rank = torch.empty_like(order)
        rank.scatter_(1, order, self._slot_index.expand_as(order))
        return rank

    def randomize(
        self,
        env_ids: torch.Tensor,
        num_active: int,
        spawn_pos_local: torch.Tensor | None = None,
        goal_pos_local: torch.Tensor | None = None,
        spawn_forward_local: torch.Tensor | None = None,
    ) -> None:
        """Resample obstacle poses for ``env_ids``, activating ``num_active`` slots.

        Args:
            env_ids: Environments to resample.
            num_active: Curriculum level, i.e. the number of in-box obstacles.
            spawn_pos_local: ``(len(env_ids), 3)`` drone spawn positions in
                environment-local metres.  Obstacles are kept
                ``spawn_clearance_radius`` away from these.  CHANGE2.
            goal_pos_local: ``(len(env_ids), 3)`` goal positions in
                environment-local metres.  Obstacles are kept
                ``goal_clearance_radius`` away from these.  CHANGE2.
            spawn_forward_local: ``(len(env_ids), 3)`` unit vectors along the
                drone's initial forward heading. When supplied and
                ``spawn_forward_only`` is enabled, obstacle origins are kept in
                the corresponding forward half-space.
        """
        count = env_ids.numel()
        if count == 0:
            return
        num_active = int(max(0, min(num_active, self.num_slots)))

        def sample_positions() -> torch.Tensor:
            ratio = self._min_ratio + torch.rand(
                (count, self.num_slots, 3), device=self.device
            ) * (self._max_ratio - self._min_ratio)
            return self._bounds_min + (self._bounds_max - self._bounds_min) * ratio

        pos_local = sample_positions()

        euler = self._min_euler + torch.rand(
            (count, self.num_slots, 3), device=self.device
        ) * (self._max_euler - self._min_euler)

        # CHANGE2: keep the goal and the drone's spawn point obstacle-free. Both
        # sit inside the obstacle field now, so overlapping slots would make some
        # episodes unwinnable or end them in an instant collision.
        #
        # Rejection sampling: the two clearance spheres together occupy ~2% of the
        # cube, so a handful of passes clears essentially every conflict. Anything
        # still conflicting after that is parked, which keeps the guarantee exact
        # at the cost of an occasional missing obstacle.
        # To revert CHANGE2: pass no spawn/goal positions from the task.
        if (
            spawn_pos_local is not None
            or goal_pos_local is not None
            or self.cfg.obstacle_core_radius is not None
            or (self.cfg.spawn_forward_only and spawn_forward_local is not None)
        ):
            conflict = self._placement_conflict(
                pos_local, spawn_pos_local, goal_pos_local, spawn_forward_local
            )
            for _ in range(self.cfg.clearance_attempts):
                if not bool(conflict.any()):
                    break
                pos_local = torch.where(conflict.unsqueeze(-1), sample_positions(), pos_local)
                conflict = self._placement_conflict(
                    pos_local, spawn_pos_local, goal_pos_local, spawn_forward_local
                )
            if bool(conflict.any()):
                pos_local = torch.where(
                    conflict.unsqueeze(-1), self._parked_local.unsqueeze(0), pos_local
                )

        # CHANGE6: which slots are active is set by the archetype's ordering, not
        # by raw slot index -- so a forest env activates trunks/canopies first and
        # an office env activates partitions/desks first, both with a few boxes.
        # A scene style plus an ordering variant is drawn per environment.
        archetype = torch.multinomial(
            self._archetype_probs, count, replacement=True
        )
        variant = torch.randint(self._num_orderings, (count,), device=self.device)
        self._archetype[env_ids] = archetype
        rows = archetype * self._num_orderings + variant
        active = (self._archetype_rank[rows] < num_active).unsqueeze(-1)

        pos_local = torch.where(active, pos_local, self._parked_local.unsqueeze(0))
        euler = torch.where(active, euler, torch.zeros_like(euler))

        pose = torch.zeros((count, self.num_slots, 7), device=self.device)
        pose[..., :3] = pos_local + self.env_origins[env_ids].unsqueeze(1)
        # quat_from_euler_xyz is elementwise, so it broadcasts over (count, num_slots).
        pose[..., 3:] = quat_from_euler_xyz(euler[..., 0], euler[..., 1], euler[..., 2])

        self.collection.write_object_link_pose_to_sim(pose, env_ids=env_ids)

    def _placement_conflict(
        self,
        pos_local: torch.Tensor,
        spawn_pos_local: torch.Tensor | None,
        goal_pos_local: torch.Tensor | None,
        spawn_forward_local: torch.Tensor | None,
    ) -> torch.Tensor:
        """Mask slots violating a clearance sphere, core radius, or half-space.

        Distances use the slot *origin*, not its bounding box, so a large panel can
        still clip the sphere at the margin. The radii are sized with that in mind.
        """
        conflict = torch.zeros(pos_local.shape[:2], dtype=torch.bool, device=self.device)
        if self.cfg.obstacle_core_radius is not None:
            centre = pos_local.new_tensor(self.cfg.box_center)
            radius_xy = (pos_local[..., :2] - centre[:2]).norm(dim=-1)
            conflict |= radius_xy > self.cfg.obstacle_core_radius
        if goal_pos_local is not None:
            distance = (pos_local - goal_pos_local.unsqueeze(1)).norm(dim=-1)
            conflict |= distance < self.cfg.goal_clearance_radius
        if spawn_pos_local is not None:
            distance = (pos_local - spawn_pos_local.unsqueeze(1)).norm(dim=-1)
            conflict |= distance < self.cfg.spawn_clearance_radius
        if (
            self.cfg.spawn_forward_only
            and spawn_pos_local is not None
            and spawn_forward_local is not None
        ):
            relative_xy = pos_local[..., :2] - spawn_pos_local[:, None, :2]
            forward_xy = spawn_forward_local[:, None, :2]
            forward_projection = (relative_xy * forward_xy).sum(dim=-1)
            conflict |= forward_projection <= 0.0
        return conflict

    @property
    def archetype(self) -> torch.Tensor:
        """``(num_envs,)`` index into :data:`ARCHETYPES` per environment.  CHANGE6."""
        return self._archetype

    def obstacle_positions_w(self, env_ids: torch.Tensor) -> torch.Tensor:
        """World positions of every obstacle slot, ``(len(env_ids), num_slots, 3)``."""
        return self.collection.data.object_link_pose_w[env_ids, :, :3]


def sample_perimeter_spawn_positions(
    cfg: ObstacleFieldCfg,
    count: int,
    min_radius: float,
    max_radius: float,
    z_half_range: float,
    device: str | torch.device,
) -> torch.Tensor:
    """Sample spawns in the obstacle-free shell around the box centre."""

    centre = torch.tensor(cfg.box_center, device=device)
    bounds_min = torch.tensor(cfg.bounds_min, device=device)
    bounds_max = torch.tensor(cfg.bounds_max, device=device)

    bearing = torch.rand(count, device=device) * (2.0 * math.pi)
    unit = torch.rand(count, device=device)
    radius = torch.sqrt(min_radius**2 + unit * (max_radius**2 - min_radius**2))

    pos_local = torch.empty((count, 3), device=device)
    pos_local[:, 0] = centre[0] + radius * torch.cos(bearing)
    pos_local[:, 1] = centre[1] + radius * torch.sin(bearing)
    pos_local[:, 2] = centre[2] + (torch.rand(count, device=device) * 2.0 - 1.0) * z_half_range

    margin = 0.5
    return torch.max(torch.min(pos_local, bounds_max - margin), bounds_min + margin)


def sample_spawn_positions(
    cfg: ObstacleFieldCfg,
    count: int,
    goal_pos_local: torch.Tensor,
    min_ratio: tuple[float, float, float],
    max_ratio: tuple[float, float, float],
    min_goal_distance: float,
    device: str | torch.device,
    attempts: int = 8,
) -> torch.Tensor:
    """Sample drone spawn positions in environment-local metres.  CHANGE2.

    Uniform over the ratio box, but rejected within ``min_goal_distance`` of the
    goal so the drone never starts on top of it.  Any sample still too close after
    ``attempts`` passes is pushed radially outward to exactly the minimum
    distance, which makes the guarantee exact rather than probabilistic.

    Args:
        cfg: Field configuration, for the box bounds.
        count: Number of positions to sample.
        goal_pos_local: ``(count, 3)`` goal positions in environment-local metres.
        min_ratio: Lower ratio corner of the spawn region.
        max_ratio: Upper ratio corner of the spawn region.
        min_goal_distance: Minimum spawn-to-goal distance in metres.
        device: Torch device.
        attempts: Rejection-sampling passes before falling back to the push.

    Returns:
        ``(count, 3)`` spawn positions in environment-local metres.
    """
    bounds_min = torch.tensor(cfg.bounds_min, device=device)
    bounds_max = torch.tensor(cfg.bounds_max, device=device)
    span = bounds_max - bounds_min
    ratio_min = torch.tensor(min_ratio, device=device)
    ratio_max = torch.tensor(max_ratio, device=device)

    def draw() -> torch.Tensor:
        ratio = ratio_min + torch.rand((count, 3), device=device) * (ratio_max - ratio_min)
        return bounds_min + span * ratio

    pos_local = draw()
    for _ in range(attempts):
        too_close = (pos_local - goal_pos_local).norm(dim=-1) < min_goal_distance
        if not bool(too_close.any()):
            break
        pos_local = torch.where(too_close.unsqueeze(-1), draw(), pos_local)

    # Guaranteed fallback: push the stragglers radially out to the minimum radius.
    offset = pos_local - goal_pos_local
    distance = offset.norm(dim=-1, keepdim=True)
    # A sample exactly at the goal has no direction to push along; use +x.
    direction = torch.where(
        distance > 1.0e-6, offset / distance.clamp_min(1.0e-6), offset.new_tensor([1.0, 0.0, 0.0])
    )
    pushed = goal_pos_local + direction * (min_goal_distance * 1.01)
    pushed = pushed.clamp(bounds_min + span * ratio_min, bounds_min + span * ratio_max)
    return torch.where(distance < min_goal_distance, pushed, pos_local)
