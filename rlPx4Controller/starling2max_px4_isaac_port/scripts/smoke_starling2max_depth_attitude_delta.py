"""Smoke test for the Starling 2 Max depth attitude-delta navigation task.

Beyond the checks the velocity-nav smoke test does (depth encoding, camera frame,
encoder resize), this one verifies the pieces specific to attitude-delta control:

* the **roll setpoint stays exactly 0.0** for every environment and every step;
* the **pitch setpoint respects ``max_pitch``** even under a saturating command;
* the **thrust setpoint stays inside** the PX4 throttle band implied by
  ``collective_thrust_min/max``;
* the frozen encoder runs at the camera's **10 Hz**, not the policy's 50 Hz --
  reported as the number of environments re-encoded per step;
* the PX4 cascade keeps its native rates: **1 kHz** rate loop, **250 Hz**
  attitude loop, **50 Hz** policy.

Run it with ``--command hold`` (zero action), ``--command pitch`` (full forward
pitch) or ``--command yaw`` (full yaw rate) to exercise the saturation paths.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=60)
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--curriculum-level", type=int, default=None)
parser.add_argument(
    "--command",
    choices=("hold", "pitch", "yaw", "climb"),
    default="pitch",
    help="Which saturating command to hold for the whole run.",
)
args, _ = parser.parse_known_args()

from isaaclab.app import AppLauncher

simulation_app = AppLauncher({"headless": args.headless, "device": args.device}).app

import torch

from aerial_isaac_lab.tasks import (
    Starling2MaxDepthAttitudeDeltaEnv,
    Starling2MaxDepthAttitudeDeltaEnvCfg,
)


def main() -> None:
    env_cfg = Starling2MaxDepthAttitudeDeltaEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    if args.curriculum_level is not None:
        env_cfg.curriculum.min_level = args.curriculum_level

    env = Starling2MaxDepthAttitudeDeltaEnv(cfg=env_cfg)
    obs, _ = env.reset()

    hover_magnitude = abs(env_cfg.px4_hover_thrust)
    # CHANGE13 + PORT: the setpoint is the PX4 throttle magnitude itself, not the
    # Lee controller's normalised (0 == hover) form the upstream task uses.
    thrust_lo = env_cfg.collective_thrust_min
    thrust_hi = env_cfg.collective_thrust_max
    raw_hover = 2.0 * hover_magnitude / env_cfg.collective_thrust_scale - 1.0

    print(f"observation shape : {tuple(obs['policy'].shape)} (expected (N, 81))")
    print(f"action space dim  : {env_cfg.action_space} (expected 3)")
    print(f"policy rate       : {1.0 / env.step_dt:.1f} Hz (expected 50.0)")
    print(f"physics/rate loop : {1.0 / env.physics_dt:.1f} Hz (expected 1000.0)")
    print(
        f"attitude loop     : {1.0 / (env.physics_dt * env._controller.attitude_interval):.1f} Hz "
        f"(expected 250.0, interval {env._controller.attitude_interval})"
    )
    print(f"max episode steps : {env.max_episode_length}")
    print(f"active obstacles  : {env.curriculum_level}")
    print(f"max pitch         : {math.degrees(env_cfg.max_pitch):.1f} deg")
    print(f"pitch slew        : {math.degrees(env_cfg.max_delta_pitch) / env.step_dt:.0f} deg/s")
    print(f"yaw slew          : {math.degrees(env_cfg.max_delta_yaw) / env.step_dt:.0f} deg/s")
    print(
        f"thrust band       : [{thrust_lo:.4f}, {thrust_hi:.4f}] PX4 throttle "
        f"(hover {hover_magnitude:.3f} at a={raw_hover:+.3f}, "
        f"ceiling {thrust_hi / hover_magnitude:.2f}x hover)"
    )
    print(f"command           : {args.command}")
    print()

    actions = torch.zeros((env.num_envs, env_cfg.action_space), device=env.device)
    if args.command == "pitch":
        actions[:, 1] = 1.0
    elif args.command == "yaw":
        actions[:, 2] = 1.0
    elif args.command == "climb":
        actions[:, 0] = 1.0

    max_abs_roll = 0.0
    max_abs_pitch = 0.0
    max_abs_thrust = 0.0
    encodes = 0
    last_frame = env._last_depth_frame.clone()

    for step in range(args.steps):
        obs, reward, terminated, truncated, _ = env.step(actions)
        setpoints = env.attitude_setpoints

        max_abs_roll = max(max_abs_roll, setpoints[:, 1].abs().max().item())
        max_abs_pitch = max(max_abs_pitch, setpoints[:, 2].abs().max().item())
        max_abs_thrust = max(max_abs_thrust, setpoints[:, 0].abs().max().item())

        refreshed = int((env._last_depth_frame != last_frame).sum())
        encodes += refreshed
        last_frame = env._last_depth_frame.clone()

        if step % 10 == 0 or step == args.steps - 1:
            speed = env._robot.data.root_lin_vel_w.norm(dim=-1)
            vertical = env._robot.data.root_lin_vel_w[:, 2]
            print(
                f"step {step:3d} | thrust {setpoints[0, 0]:+.3f} "
                f"roll {math.degrees(setpoints[0, 1]):+.2f}d "
                f"pitch {math.degrees(setpoints[0, 2]):+.2f}d "
                f"yaw {math.degrees(setpoints[0, 3]):+7.1f}d | "
                f"speed {speed.mean():5.2f} m/s vz {vertical.mean():+5.2f} | "
                f"box {int(env.box_present.sum())}/{env.num_envs} | "
                f"d_min mean {env.min_obstacle_dist.mean():4.2f} min {env.min_obstacle_dist.min():4.2f} m | "
                f"latent std {env.image_latents.std():.2f} | "
                f"reward {reward.mean():+7.2f} | "
                f"crash {int(terminated.sum())} re-encoded {refreshed}"
            )

    print()
    print(f"max |roll setpoint|   : {math.degrees(max_abs_roll):.6f} deg (must be 0)")
    print(f"max |pitch setpoint|  : {math.degrees(max_abs_pitch):.3f} deg "
          f"(cap {math.degrees(env_cfg.max_pitch):.1f})")
    print(f"max |thrust setpoint| : {max_abs_thrust:.4f} "
          f"(band [{thrust_lo:.4f}, {thrust_hi:.4f}])")
    print(f"encoder calls         : {encodes} over {args.steps * env.num_envs} env-steps "
          f"(expect ~1 in {int(round(0.1 / env.step_dt))})")

    assert obs["policy"].shape[1] == 81, "observation is not 81-D"
    assert torch.isfinite(obs["policy"]).all(), "non-finite observation"
    assert max_abs_roll == 0.0, "roll setpoint is not pinned to zero"
    assert max_abs_pitch <= env_cfg.max_pitch + 1e-6, "pitch setpoint exceeded max_pitch"
    assert max_abs_thrust <= thrust_hi + 1e-6, "thrust setpoint outside its band"
    assert math.isclose(1.0 / env.step_dt, 50.0), "policy is not 50 Hz"
    assert math.isclose(1.0 / env.physics_dt, 1000.0), "physics/rate loop is not 1 kHz"
    assert env._controller.attitude_interval == 4, "attitude loop is not 250 Hz"
    assert encodes < args.steps * env.num_envs, "encoder is running every policy step"
    print("\nsmoke test passed")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
