"""Starling 2 Max values inferred from the checked-in URDF and PX4 params."""

from __future__ import annotations

from pathlib import Path

from .model_cfg import MultirotorModelCfg


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
STARLING2MAX_URDF_PATH = str(_PROJECT_ROOT / "assets/robots/starling2max/starling2max.urdf")

STARLING2MAX_MODEL = MultirotorModelCfg(
    urdf_path=STARLING2MAX_URDF_PATH,
    root_body_name="vehicle",
    motor_body_names=("vehicle_rotor0", "vehicle_rotor1", "vehicle_rotor2", "vehicle_rotor3"),
    motor_directions=(1.0, 1.0, -1.0, -1.0),
    # Rows [Fx, Fy, Fz, Mx, My, Mz]. Mx = rotor y, My = -rotor x, using the URDF
    # joint origins (see starling2max.urdf) so the moment arms match where the
    # forces are physically applied. The PX4 CA_ROTOR params flip the y sign and
    # use a smaller frame, which inverts the roll axis; do not use them here.
    allocation_matrix=(
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0, 1.0),
        (-0.13, 0.13, 0.13, -0.13),
        (-0.095, 0.095, -0.095, 0.095),
        (-0.05, -0.05, 0.05, 0.05),
    ),
    thrust_constant=5.925386e-06,
    motor_tau_increasing=0.055,
    motor_tau_decreasing=0.085,
    min_thrust=0.0,
    max_thrust=9.356928461538462,
    max_thrust_rate=100000.0,
    thrust_to_torque_ratio=0.05,
    rigid_linear_damping=0.02,
    rigid_angular_damping=0.02,
    aerodynamic_linear_damping=(0.0, 0.0, 0.0),
    aerodynamic_quadratic_damping=(0.0, 0.0, 0.0),
    aerodynamic_angular_linear_damping=(0.0, 0.0, 0.0),
    aerodynamic_angular_quadratic_damping=(0.0, 0.0, 0.0),
)
