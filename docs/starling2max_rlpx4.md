# Starling 2 Max RL-PX4 Architecture

The Isaac Lab package is isolated under
`rlPx4Controller/starling2max_px4_isaac_port/`. It ports the relevant
RL-PX4/PX4 cascade into batched Torch so thousands of environments remain on
the GPU. It does not call the root pybind modules during training.

## Controller Paths

- Position and velocity waypoint tasks retain the Lee controller interfaces.
- Attitude-delta uses the Torch PX4 attitude loop followed by the PX4 rate loop.
- Direct CTBR bypasses attitude control and enters the PX4 rate loop.
- Depth velocity navigation uses the PX4 velocity, attitude, and rate cascade.
- Depth attitude navigation enters the PX4 attitude and rate loops.
- `motor_px4.py` applies allocation, output transforms, motor lag, rotor forces,
  and drag using `config/allocation_starling2max.yaml`.

The navigation multi-rate schedule is:

| Component | Rate |
|---|---:|
| Physics, rate controller, and motor model | 1000 Hz |
| PX4 attitude controller | 250 Hz |
| PX4 velocity controller | 50 Hz |
| Depth velocity policy and DCE | 10 Hz |
| Depth attitude policy | 50 Hz |

## Policy Contracts

The 18-D waypoint observation is:

```text
[position_error_nwu(3), orientation_xyzw(4),
 body_linear_velocity_flu(3), body_angular_velocity_flu(3),
 previous_action(4), yaw_error_over_pi(1)]
```

The attitude-delta and CTBR tasks reparameterize the first three values as a
unit goal direction in the vehicle yaw frame, place raw goal distance in channel
17, and append eight 10-D history samples from channels 7:17 at a stride of five
policy steps. Their resulting observation is 98-D.

The two navigation tasks replace the history with a frozen 64-D Deep Collision
Encoder latent, producing an 81-D observation. Training and SITL must agree on:

- 240x180 depth images;
- 86 degree horizontal and 106 degree vertical FOV;
- 0.2-to-6 m normalization range;
- 10 Hz depth updates;
- the checked-in DCE checkpoint;
- `32FC1` metric depth on `/starling/raw_depth` during deployment.

## Frame Boundary

Isaac policies use NWU world axes and FLU body axes. PX4 uses NED world axes and
FRD body axes. ROS deployment nodes perform this conversion exactly once at
the policy boundary. Quaternion observations are `xyzw`; PX4 messages carry
`wxyz` where required by the v1.14 message definition.

## Deployment Mapping

| Policy | ROS executable | PX4 input |
|---|---|---|
| Position | `position_sim_controller` | `/fmu/in/trajectory_setpoint` |
| Velocity | `velocity_sim_controller` | `/fmu/in/trajectory_setpoint` |
| Attitude delta | `delta_attitude_sp` | `/fmu/in/vehicle_attitude_setpoint` |
| Direct CTBR | `body_rates_sim_controller` | `/fmu/in/vehicle_rates_setpoint` |
| Depth velocity | `depth_velocity_sp` | `/fmu/in/trajectory_setpoint` |
| Depth attitude | `depth_attitude_sp` | `/fmu/in/vehicle_attitude_setpoint` |

The attitude and CTBR nodes share the same 98-D observation/history adapter so
their live command channels match the training task. The depth nodes load the
perception implementation from the selected training tree and receive config,
policy checkpoint, encoder checkpoint, device, and depth topic as ROS
parameters.
