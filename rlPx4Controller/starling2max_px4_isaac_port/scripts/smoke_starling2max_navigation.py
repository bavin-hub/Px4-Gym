"""Smoke test / depth-pipeline validation for the Starling 2 Max navigation task.

This is the cheapest way to de-risk the three things that fail *silently* in the
port: the ``[-1] U [0, 1]`` depth encoding, the camera frame convention, and the
180x240 -> 270x480 resize the frozen encoder expects.

It steps the environment with zero actions and reports, per step:

* depth statistics -- min/max, and the fraction of pixels sitting on the near
  (``-1.0``) and far (``+1.0``) sentinels;
* latent statistics -- the encoder was trained to produce roughly unit-Gaussian
  latents, so a mean far from 0 or a std far from 1 means the depth encoding is
  wrong even if nothing crashes;
* collision and reward bookkeeping.

With ``--dump-images`` it also writes ``depth_XXX.png`` (the normalised depth
seen by the encoder) next to ``recon_XXX.png`` (that depth round-tripped through
``encode`` -> ``decode``).  If the reconstruction does not resemble the input,
the encoding or the camera frame is wrong -- fix that before training anything.
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
parser.add_argument("--steps", type=int, default=20)
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--curriculum-level", type=int, default=None)
parser.add_argument(
    "--dump-env",
    type=int,
    default=None,
    help="Which env to dump images for. Default: the first env that has the box, "
         "since a box-less env in open space renders as an all-white far plane.",
)
parser.add_argument(
    "--dump-images",
    type=Path,
    default=None,
    help="Directory to write depth/reconstruction PNG pairs for env 0.",
)
args, _ = parser.parse_known_args()

from isaaclab.app import AppLauncher

simulation_app = AppLauncher({"headless": args.headless, "device": args.device}).app

import torch

from aerial_isaac_lab.tasks import Starling2MaxNavigationEnv, Starling2MaxNavigationEnvCfg


def save_pair(depth: torch.Tensor, recon: torch.Tensor, out_dir: Path, step: int) -> None:
    """Write the normalised depth and its reconstruction as 8-bit PNGs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
    except ImportError:
        # No PIL in this interpreter: fall back to raw arrays.
        import numpy as np

        np.save(out_dir / f"depth_{step:03d}.npy", depth.cpu().numpy())
        np.save(out_dir / f"recon_{step:03d}.npy", recon.cpu().numpy())
        return

    # Depth is [-1] U [0, 1]; map to [0, 255] for display.
    for name, image in (("depth", depth), ("recon", recon)):
        array = image.clamp(-1.0, 1.0).cpu().numpy()
        array = ((array + 1.0) * 0.5 * 255.0).astype("uint8")
        Image.fromarray(array).save(out_dir / f"{name}_{step:03d}.png")


def main() -> None:
    env_cfg = Starling2MaxNavigationEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    if args.curriculum_level is not None:
        env_cfg.curriculum.min_level = args.curriculum_level

    env = Starling2MaxNavigationEnv(cfg=env_cfg)
    obs, _ = env.reset()
    print(f"observation shape : {tuple(obs['policy'].shape)} (expected (N, 81))")
    print(f"action space dim  : {env_cfg.action_space} (expected 3)")
    print(f"policy rate       : {1.0 / env.step_dt:.1f} Hz")
    print(f"physics rate      : {1.0 / env.physics_dt:.1f} Hz")
    print(f"velocity control  : {env_cfg.velocity_control_hz:.1f} Hz")
    print(f"attitude control  : {env_cfg.attitude_control_hz:.1f} Hz")
    print(f"rate control      : {env_cfg.rate_control_hz:.1f} Hz")
    print(f"max episode steps : {env.max_episode_length}")
    print(f"obstacle slots    : {env_cfg.obstacles.num_obstacles}")
    print(f"active obstacles  : {env.curriculum_level}")
    print(f"depth image shape : {tuple(env.depth_image.shape)} (expected (N, 180, 240))")
    print()

    dump_env = args.dump_env
    if dump_env is None:
        boxed = env.box_present.nonzero(as_tuple=False).flatten()
        dump_env = int(boxed[0]) if boxed.numel() else 0
    if args.dump_images is not None:
        print(f"dumping images for env {dump_env} (box={bool(env.box_present[dump_env])})\n")

    actions = torch.zeros((env.num_envs, env_cfg.action_space), device=env.device)
    for step in range(args.steps):
        obs, reward, terminated, truncated, _ = env.step(actions)

        depth = env.depth_image
        latents = env.image_latents
        near = (depth <= -1.0 + 1e-6).float().mean()
        far = (depth >= 1.0 - 1e-6).float().mean()
        valid = depth[(depth > -1.0 + 1e-6) & (depth < 1.0 - 1e-6)]
        valid_min = valid.min().item() if valid.numel() else float("nan")
        valid_max = valid.max().item() if valid.numel() else float("nan")

        print(
            f"step {step:3d} | depth valid [{valid_min:+.3f}, {valid_max:+.3f}] "
            f"near {near:.3f} far {far:.3f} | box {int(env.box_present.sum())}/{env.num_envs} | "
            f"latent mean {latents.mean():+.3f} std {latents.std():.3f} | "
            f"reward {reward.mean():+8.2f} | "
            f"crash {int(terminated.sum())} timeout {int(truncated.sum())}"
        )

        if args.dump_images is not None:
            recon = env._dce_encoder.decode(latents[dump_env : dump_env + 1])
            save_pair(depth[dump_env], recon[0, 0], args.dump_images, step)

    if args.dump_images is not None:
        print(f"\nwrote depth/reconstruction pairs to {args.dump_images}")

    # Sanity assertions -- these catch the silent failure modes.
    assert obs["policy"].shape[1] == 81, "observation is not 81-D"
    assert math.isclose(1.0 / env.step_dt, 10.0), "navigation policy is not 10 Hz"
    assert math.isclose(1.0 / env.physics_dt, 1000.0), "physics/rate loop is not 1 kHz"
    assert env._velocity_interval == 20, "velocity loop is not 50 Hz"
    assert env._controller.attitude_interval == 4, "attitude loop is not 250 Hz"
    assert torch.isfinite(obs["policy"]).all(), "non-finite observation"
    assert torch.isfinite(env.depth_image).all(), "non-finite depth (nan clipping is broken)"
    print("\nsmoke test passed")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
