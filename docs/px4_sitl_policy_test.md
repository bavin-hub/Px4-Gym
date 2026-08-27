# PX4 SITL Policy Evaluation

Complete [workspace setup](workspace_setup.md), train or obtain a checkpoint,
and keep the task's YAML and checkpoint together. The six tasks can use the
same Starling/PX4 state bridge; the two depth tasks additionally require the
matched depth stream and frozen DCE checkpoint.

## Choose a Simulator

### Gazebo

Use the base model for the four non-depth policies:

```bash
cd "${PX4_AUTOPILOT_ROOT}"
PX4_GZ_WORLD=default_trees make px4_sitl gz_starling2max
```

Use the depth model for either navigation policy:

```bash
cd "${PX4_AUTOPILOT_ROOT}"
PX4_GZ_WORLD=default_trees make px4_sitl gz_starling2max_depth
```

Bridge the Gazebo depth image to the policy contract with the ROS-Gazebo image
bridge. Confirm the source topic with `gz topic -l`; the final ROS topic must be
`/starling/raw_depth`, type `sensor_msgs/msg/Image`, encoding `32FC1`, in metres.

### Isaac Sim through Pegasus

Pegasus provides the PX4 MAVLink/HIL simulator backend and auto-launches PX4.
The checked-in standalone app provides the training-style corridor or tree
scene and publishes metric depth directly through Isaac Sim's ROS 2 bridge.

```bash
cd "${PX4_GYM_ROOT}"
isaac_run sitl/isaac/13_px4_depth_navigation.py \
  --vehicle-usd /absolute/path/to/starling2max.usd \
  --px4-dir "${PX4_AUTOPILOT_ROOT}" \
  --px4-airframe pegasus_starling2max \
  --scene corridor
```

Add `--headless` for a non-GUI run. The USD must expose `body` and
`rotor0`...`rotor3` rigid bodies, which Pegasus uses to apply forces. The
checked-in `10050_pegasus_starling2max` airframe carries the Starling controller,
allocation, hover-thrust, and rate-limit parameters while Pegasus owns the
simulated dynamics.

The app prints the goal in PX4 local NED. Pass the matching waypoint parameters
to the selected ROS node when they are supported, or keep the node's default
goal aligned with the printed value.

## Common Terminals

Terminal 1 starts DDS:

```bash
MicroXRCEAgent udp4 -p 8888
```

Terminal 2 runs either Gazebo/PX4 or the Pegasus command above.

Terminal 3 sources ROS and checks connectivity:

```bash
cd "${PX4_GYM_ROOT}/ros_ws"
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 topic list | grep /fmu
ros2 topic echo /fmu/out/vehicle_odometry
ros2 topic echo /fmu/out/vehicle_status
```

Run one policy node at a time from a fourth terminal sourced the same way.

## Non-Depth Policies

Set a convenience path:

```bash
export STARLING_PORT="${PX4_GYM_ROOT}/rlPx4Controller/starling2max_px4_isaac_port"
```

Position:

```bash
ros2 run policy_test position_sim_controller --ros-args \
  -p policy_config:="${STARLING_PORT}/config/rl_games_starling2max_position.yaml" \
  -p checkpoint:=/absolute/path/to/starling2max_position.pth \
  -p device:=cuda:0
```

Velocity:

```bash
ros2 run policy_test velocity_sim_controller --ros-args \
  -p policy_config:="${STARLING_PORT}/config/rl_games_starling2max_velocity.yaml" \
  -p checkpoint:=/absolute/path/to/starling2max_velocity.pth \
  -p device:=cuda:0
```

Attitude delta:

```bash
ros2 run policy_test delta_attitude_sp --ros-args \
  -p policy_config:="${STARLING_PORT}/config/rl_games_starling2max_attitude_delta.yaml" \
  -p checkpoint:=/absolute/path/to/starling2max_attitude_delta.pth \
  -p device:=cuda:0 \
  -p takeoff_height:=-3.0
```

Direct CTBR/body rate:

```bash
ros2 run policy_test body_rates_sim_controller --ros-args \
  -p policy_config:="${STARLING_PORT}/config/rl_games_starling2max_rates.yaml" \
  -p checkpoint:=/absolute/path/to/starling2max_rates.pth \
  -p device:=cuda:0 \
  -p takeoff_height:=-3.0
```

The attitude and CTBR executables build the current 98-D observation including
the eight strided history frames. The CTBR node publishes rates in PX4 FRD and
collective thrust through `/fmu/in/vehicle_rates_setpoint`.

## Depth Policies

The depth nodes require explicit policy checkpoints. Their training root,
config, encoder checkpoint, device, and depth topic are ROS parameters; defaults
for everything except the policy checkpoint resolve from this checkout.

Depth velocity navigation:

```bash
ros2 run depth_policy depth_velocity_sp --ros-args \
  -p training_root:="${STARLING_PORT}" \
  -p policy_config:="${STARLING_PORT}/config/rl_games_starling2max_navigation.yaml" \
  -p checkpoint:=/absolute/path/to/starling2max_navigation.pth \
  -p encoder_checkpoint:="${STARLING_PORT}/aerial_isaac_lab/perception/weights/ICRA_test_set_more_sim_data_kld_beta_3_LD_64_epoch_49.pth" \
  -p depth_topic:=/starling/raw_depth \
  -p device:=cuda:0
```

Depth attitude-delta navigation:

```bash
ros2 run depth_policy depth_attitude_sp --ros-args \
  -p training_root:="${STARLING_PORT}" \
  -p policy_config:="${STARLING_PORT}/config/rl_games_starling2max_depth_attitude_delta.yaml" \
  -p checkpoint:=/absolute/path/to/starling2max_depth_attitude_delta.pth \
  -p encoder_checkpoint:="${STARLING_PORT}/aerial_isaac_lab/perception/weights/ICRA_test_set_more_sim_data_kld_beta_3_LD_64_epoch_49.pth" \
  -p depth_topic:=/starling/raw_depth \
  -p device:=cuda:0
```

Verify depth independently:

```bash
ros2 topic hz /starling/raw_depth
ros2 topic echo /starling/raw_depth --once
ros2 run depth_policy test_depth --ros-args \
  -p topic:=/starling/raw_depth -p max_depth:=6.0
```

Expected depth is 240x180 `32FC1` metres at 10 Hz. A different FOV, clipping
range, encoding, normalization, or DCE checkpoint changes the observation and
invalidates the comparison.

## PX4 Topic Mapping

The checked-in `dds_topics.yaml` includes all required inputs:

- `/fmu/in/trajectory_setpoint`
- `/fmu/in/vehicle_attitude_setpoint`
- `/fmu/in/vehicle_rates_setpoint`
- `/fmu/in/offboard_control_mode`
- `/fmu/in/vehicle_command`

It also publishes odometry, status, global position, and hover-thrust estimates
used by the deployment nodes. Rebuild PX4 after changing the map.

## Troubleshooting

- No `/fmu` topics: verify PX4, Micro XRCE-DDS Agent, DDS domain, and the rebuilt
  `dds_topics.yaml`.
- Policy shape/load error: the YAML, checkpoint, observation size, and ROS
  executable do not match.
- No depth: check the ROS 2 bridge and confirm `/starling/raw_depth` type,
  encoding, dimensions, and rate.
- Pegasus import error: install `pegasus.simulator` with Isaac Sim's Python, not
  the system Python.
- Pegasus vehicle lookup error: pass a valid `--vehicle-usd` with the required
  body/rotor prim names.
