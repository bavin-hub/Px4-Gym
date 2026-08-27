"""Frozen Deep Collision Encoder for single-channel depth images.

Port of Aerial Gym's ``VAEImageEncoder`` (``aerial_gym/utils/vae/vae_image_encoder.py``)
plus the depth range/normalisation semantics that Aerial Gym applies inside its
Warp sensor before the image ever reaches the encoder
(``aerial_gym/sensors/warp/warp_sensor.py::apply_range_limits`` / ``normalize_observation``).

The encoder is used **frozen**: weights are loaded once, the module is put in
``eval()`` mode, every parameter has ``requires_grad=False``, and ``encode()``
runs under ``torch.no_grad()``.  Because the task writes the resulting latent
into the observation tensor rather than exposing this module to the RL
framework, there is no gradient path into it at all -- rl_games never sees these
parameters and cannot register them with its optimiser.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from .vae import VAE

_PERCEPTION_ROOT = Path(__file__).resolve().parent
DCE_WEIGHTS_PATH = str(
    _PERCEPTION_ROOT / "weights" / "ICRA_test_set_more_sim_data_kld_beta_3_LD_64_epoch_49.pth"
)
"""Deep Collision Encoder weights from Kulkarni & Alexis, arXiv:2402.03947."""


def clean_state_dict(state_dict: dict) -> dict:
    """Strip ``DataParallel``/legacy prefixes so the checkpoint matches :class:`VAE`."""
    clean_dict = {}
    for key, value in state_dict.items():
        if "module." in key:
            key = key.replace("module.", "")
        if "dronet." in key:
            key = key.replace("dronet.", "encoder.")
        clean_dict[key] = value
    return clean_dict


@dataclass
class DceEncoderCfg:
    """Configuration mirroring Aerial Gym's ``navigation_task_config.vae_config``."""

    latent_dims: int = 64
    """Latent dimensionality. Fixed by the checkpoint; do not change."""

    model_file: str = DCE_WEIGHTS_PATH
    """Absolute path to the frozen encoder weights."""

    image_res: tuple[int, int] = (270, 480)
    """Resolution the encoder expects. Depth is resized to this before encoding."""

    interpolation_mode: str = "nearest"
    """Resize mode. ``nearest`` is what the encoder was trained against."""

    return_sampled_latent: bool = True
    """Sample from the posterior (training) instead of returning the mean.

    Aerial Gym trains with ``True`` -- the reparameterisation noise acts as a
    regulariser -- and deploys with the mean (``sim2real/config.py``
    ``SAMPLE_FROM_LATENT = False``).  Keep ``True`` for training runs.
    """


@dataclass
class DepthRangeCfg:
    """Aerial Gym depth range semantics (``BaseDepthCameraConfig``).

    Reproduces the ``[-1] U [0, 1]`` encoding the frozen encoder was trained on:
    valid returns are divided by :attr:`max_range`; anything beyond the far plane
    (including ray misses) becomes ``+1.0``; anything nearer than
    :attr:`min_range` becomes ``-1.0``.
    """

    max_range: float = 10.0
    min_range: float = 0.2


def normalize_depth_image(depth: torch.Tensor, cfg: DepthRangeCfg) -> torch.Tensor:
    """Convert metric depth into Aerial Gym's ``[-1] U [0, 1]`` encoding, in place.

    ``depth`` is expected to already have far-plane clipping applied by the
    sensor (Isaac Lab's ``depth_clipping_behavior="max"`` maps both ``nan``
    misses and out-of-range hits to ``max_distance``).  This function applies the
    remaining two steps from Aerial Gym, in the same order:

    1. ``pixels[pixels < min_range] = -max_range``  (near out-of-range sentinel)
    2. ``pixels /= max_range``                      (normalise)

    Isaac Lab has no ``depth_clipping_behavior`` that produces the negative near
    sentinel, which is why this step is explicit.
    """
    # Defensive: if the sensor was configured without far clipping, misses arrive
    # as nan/inf. Fold them onto the far plane before the ordered clamps below.
    depth = torch.nan_to_num(
        depth, nan=cfg.max_range, posinf=cfg.max_range, neginf=cfg.max_range
    )
    depth = depth.clamp(max=cfg.max_range)
    depth[depth < cfg.min_range] = -cfg.max_range
    return depth / cfg.max_range


class DceDepthEncoder(nn.Module):
    """Wraps the frozen VAE for batched depth-image inference.

    Args:
        cfg: Encoder configuration.
        device: Device to place the frozen weights on.
    """

    def __init__(self, cfg: DceEncoderCfg, device: str | torch.device = "cuda:0"):
        super().__init__()
        self.cfg = cfg
        self.vae_model = VAE(input_dim=1, latent_dim=cfg.latent_dims).to(device)

        state_dict = clean_state_dict(torch.load(cfg.model_file, map_location=device))
        self.vae_model.load_state_dict(state_dict)

        # Freeze: eval mode disables nothing stochastic here (the sampling is
        # explicit), but requires_grad_(False) makes the freeze structural.
        self.vae_model.eval()
        self.vae_model.requires_grad_(False)

    def encode(self, depth_images: torch.Tensor) -> torch.Tensor:
        """Encode a batch of normalised depth images into latent vectors.

        Args:
            depth_images: ``(N, H, W)`` or ``(N, 1, H, W)`` normalised depth.

        Returns:
            ``(N, latent_dims)`` latent vectors.
        """
        with torch.no_grad():
            if depth_images.dim() == 3:
                depth_images = depth_images.unsqueeze(1)
            elif depth_images.dim() != 4:
                raise ValueError(
                    f"Expected depth of shape (N, H, W) or (N, 1, H, W), got {tuple(depth_images.shape)}"
                )

            x_res, y_res = depth_images.shape[-2], depth_images.shape[-1]
            if self.cfg.image_res != (x_res, y_res):
                depth_images = torch.nn.functional.interpolate(
                    depth_images,
                    self.cfg.image_res,
                    mode=self.cfg.interpolation_mode,
                )
            z_sampled, means, *_ = self.vae_model.encode(depth_images)

        return z_sampled if self.cfg.return_sampled_latent else means

    def forward(self, depth_images: torch.Tensor) -> torch.Tensor:
        return self.encode(depth_images)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Reconstruct depth images from latents -- for debugging/validation only."""
        with torch.no_grad():
            if latents.shape[-1] != self.cfg.latent_dims:
                raise ValueError(
                    f"Latent size {latents.shape[-1]} does not match encoder size {self.cfg.latent_dims}"
                )
            return self.vae_model.decode(latents)

    def get_latent_dims_size(self) -> int:
        return self.cfg.latent_dims
