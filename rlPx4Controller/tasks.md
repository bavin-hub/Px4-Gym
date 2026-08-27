# Policy training tasks

The Isaac Lab port under `starling2max_px4_isaac_port` registers six Starling 2 Max
policy-training tasks.

| Vehicle | Task | Gymnasium ID | Observation | Policy action | Controller path |
|---|---|---|---:|---|---|
| Starling 2 Max | Velocity waypoint tracking | `AerialIsaac-Starling2Max-Velocity-v0` | 18 | Body-FLU `[vx, vy, vz, yaw_rate]`, scaled to ±1 m/s and ±π/5 rad/s | Lee velocity |
| Starling 2 Max | Position-setpoint waypoint tracking | `AerialIsaac-Starling2Max-Position-v0` | 18 | `[dx, dy, dz, yaw]`; relative position offset scaled by `5/sqrt(3)` m and absolute world yaw in ±π | Lee position |
| Starling 2 Max | Attitude-delta goal tracking | `AerialIsaac-Starling2Max-AttitudeDelta-v0` | 98 | `[delta_thrust, delta_roll, delta_pitch, delta_yaw]`; measured-attitude-relative commands | PX4 attitude and rate loops |
| Starling 2 Max | Direct body-rate goal tracking | `AerialIsaac-Starling2Max-Rates-v0` | 98 | CTBR `[roll_rate, pitch_rate, yaw_rate, collective_thrust]` after a final `tanh` | Direct PX4 rate loop |
| Starling 2 Max | Depth velocity navigation | `AerialIsaac-Starling2Max-Navigation-v0` | 81 | `[forward_speed, inclination, delta_yaw]`; body-FLU `[vx, 0, vz]` plus a measured-yaw-relative ±20° heading setpoint | PX4 velocity, attitude, and rate cascade |
| Starling 2 Max | Depth attitude-delta navigation | `AerialIsaac-Starling2Max-DepthAttitudeDelta-v0` | 81 | `[collective, delta_pitch, delta_yaw]`; roll fixed at zero | PX4 attitude and rate loops |
Both depth tasks share the 3.5 m obstacle core, 4.2–4.8 m perimeter spawn,
goal-facing ±30° spawn yaw, 45–140 curriculum, and yaw-relative chase viewer. They use a
frozen 64-D Deep Collision Encoder latent. The Starling attitude
and direct-rate tasks use a 98-D observation with command/response history; the remaining
state-only tasks use the common 18-D waypoint observation.
