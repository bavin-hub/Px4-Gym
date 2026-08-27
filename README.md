# Px4-Gym

[![PX4](https://img.shields.io/badge/PX4-v1.14.3-silver.svg)](https://github.com/PX4/PX4-Autopilot/releases/tag/v1.14.3)
[![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-5.1.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/)
[![Pegasus](https://img.shields.io/badge/Pegasus-v5.1.0-silver.svg)](https://pegasussimulator.github.io/PegasusSimulator/)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://docs.python.org/3.11/)
[![Ubuntu](https://img.shields.io/badge/Ubuntu-22.04-orange.svg)](https://releases.ubuntu.com/22.04/)

PX4-Gym provides reinforcement-learning training, in-simulator evaluation, and
PX4 SITL deployment for the Starling 2 Max quadrotor. The `s2m_rlpx4` branch
wires a Torch-native port of the RL-PX4 attitude, rate, allocation, motor, and
velocity cascade into Isaac Lab, then preserves the same policy contracts in
ROS 2 deployment nodes.

Training and same-environment evaluation run in Isaac Lab. SITL policy
evaluation can run against Gazebo or against Isaac Sim through Pegasus
Simulator. PX4 state and offboard setpoints pass through Micro XRCE-DDS and the
checked-in ROS 2 packages.

## Tasks

| Task | Gymnasium ID | Obs. | Action | Deployment interface |
|---|---|---:|---|---|
| Velocity waypoint | `AerialIsaac-Starling2Max-Velocity-v0` | 18 | Body-FLU velocity and yaw rate | PX4 trajectory setpoint |
| Position waypoint | `AerialIsaac-Starling2Max-Position-v0` | 18 | Relative position and yaw | PX4 trajectory setpoint |
| Attitude delta | `AerialIsaac-Starling2Max-AttitudeDelta-v0` | 98 | Collective and attitude deltas | PX4 attitude setpoint |
| Direct body rate | `AerialIsaac-Starling2Max-Rates-v0` | 98 | CTBR | PX4 body-rate setpoint |
| Depth velocity navigation | `AerialIsaac-Starling2Max-Navigation-v0` | 81 | Speed, inclination, and yaw delta | PX4 trajectory setpoint |
| Depth attitude navigation | `AerialIsaac-Starling2Max-DepthAttitudeDelta-v0` | 81 | Collective, pitch delta, and yaw delta | PX4 attitude setpoint |

The 98-D policies include strided command/response history. The 81-D depth
policies include a frozen 64-D Deep Collision Encoder latent from a 240x180,
0.2-to-6 m forward depth stream.

## Repository Layout

- `rlPx4Controller/`: upstream C++/pybind controller plus the isolated
  `starling2max_px4_isaac_port/` Isaac Lab package.
- `ros_ws/src/policy_test/`: position, velocity, attitude-delta, and direct-rate
  PX4 offboard deployment nodes.
- `ros_ws/src/depth_policy/`: the two DCE depth-policy deployment nodes and a
  depth viewer.
- `sitl/gz/`: Starling 2 Max Gazebo models, depth model, airframe, and tree world.
- `sitl/isaac/`: standalone Pegasus/PX4 depth-navigation evaluator.
- `dds_topics.yaml`: PX4 v1.14 uXRCE-DDS topic map used by the ROS nodes.
- `docs/`: setup, training, architecture, and SITL procedures.

The Isaac training package uses the Torch-native controller port. Installing
the root C++ `rlPx4Controller` Python bindings is optional and is not required
for vectorized training or policy deployment.

## Documentation

1. [Workspace setup](docs/workspace_setup.md)
2. [Isaac Lab training and evaluation](docs/isaac_lab_training_eval.md)
3. [PX4 SITL policy evaluation](docs/px4_sitl_policy_test.md)
4. [Starling RL-PX4 architecture](docs/starling2max_rlpx4.md)

## Tested Stack

- Ubuntu 22.04
- Python 3.11
- Isaac Sim 5.1.0
- Isaac Lab `main`
- Pegasus Simulator v5.1.0
- PX4-Autopilot v1.14.3
- `px4_msgs` `release/1.14`
- `px4_ros_com` `release/v1.14`

Record the exact Isaac Lab commit for reproducible experiments because its
`main` branch changes independently of this repository.

## Remaining Work

- Run hardware-in-the-loop and real-vehicle validation.
- Add a sensor/EKF noise evaluation path.
- Add fixed-wing tasks and deployment nodes.

## Acknowledgements

The controller stack derives from
[rlPx4Controller](https://github.com/emNavi/rlPx4Controller) and the geometric
controller/task lineage in
[Aerial Gym Simulator](https://github.com/ntnu-arl/aerial_gym_simulator).
[Pegasus Simulator](https://pegasussimulator.github.io/PegasusSimulator/)
provides the Isaac Sim/PX4 SITL integration used by the standalone evaluator.
