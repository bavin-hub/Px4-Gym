"""Scene construction helpers shared by perception-driven tasks.

Import this module only after :class:`isaaclab.app.AppLauncher` has started the
simulation application.
"""

from .obstacle_field import (
    ARCHETYPES,
    OBSTACLE_PRIM_ROOT,
    SLOT_KINDS,
    WALL_NAMES,
    WALL_PRIM_ROOT,
    BoxToggler,
    ObstacleFieldCfg,
    ObstacleFieldRandomizer,
    build_obstacle_collection_cfg,
    build_wall_collection_cfg,
    create_group_prims,
    raycast_targets,
    sample_perimeter_spawn_positions,
    sample_spawn_positions,
    SlotKind,
    wall_specs,
)

__all__ = [
    "ARCHETYPES",
    "OBSTACLE_PRIM_ROOT",
    "SLOT_KINDS",
    "WALL_NAMES",
    "WALL_PRIM_ROOT",
    "BoxToggler",
    "SlotKind",
    "ObstacleFieldCfg",
    "ObstacleFieldRandomizer",
    "build_obstacle_collection_cfg",
    "build_wall_collection_cfg",
    "create_group_prims",
    "raycast_targets",
    "sample_perimeter_spawn_positions",
    "sample_spawn_positions",
    "wall_specs",
]
