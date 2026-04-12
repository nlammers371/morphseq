"""arch_spec.py — Lightweight model descriptor for checkpoint-free inference.

``ArchiveSpec`` captures the minimum information needed to reconstruct the
model architecture (encoder + decoder) from a checkpoint without the full
Hydra / loss / data-pipeline stack.

Typical workflow
----------------
**During training** (called automatically by ``train_vae``):

    from src.core.models.arch_spec import save_arch_spec
    save_arch_spec(model_config, run_dir)   # writes arch_spec.json

**At inference time** (no Hydra, no loss modules required):

    from src.core.models.arch_spec import load_encoder

    model, spec = load_encoder("path/to/run")
    # or explicitly choose a checkpoint:
    model, spec = load_encoder("path/to/run", ckpt_name="epoch09.ckpt")

    model.eval()
    with torch.no_grad():
        out = model(batch)      # ModelOutput with .mu, .logvar, .recon_x
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, List

import torch


# ---------------------------------------------------------------------------
# ArchiveSpec dataclass
# ---------------------------------------------------------------------------

@dataclass
class ArchiveSpec:
    """Minimal, JSON-serialisable descriptor for a trained VAE checkpoint.

    Parameters
    ----------
    model_class : str
        ``"VAE"`` or ``"metricVAE"``.
    backbone : str
        Architecture name understood by ``ARCH_REGISTRY``/``resolve_arch``.
        E.g. ``"Swin-Tiny"``, ``"convVAE"``, ``"ldmVAE"``.
    latent_dim : int
        Total latent dimension (biological + nuisance).
    input_dim : list[int]
        ``[C, H, W]`` of the input images (e.g. ``[1, 288, 128]``).
    nuisance_dim : int
        Number of nuisance latent dimensions.  0 for plain ``VAE``.
    dec_use_local_attn : bool
        Whether the decoder uses local attention (Swin / ViT decoders only).
    n_out_channels : int
        Base channel count for legacy ConvVAE encoder/decoder.
    n_conv_layers : int
        Number of conv layers in legacy ConvVAE.
    kernel_size : int
        Kernel size for legacy ConvVAE.
    stride : int
        Stride for legacy ConvVAE.
    ldm_params : dict
        Extra keyword arguments passed to ``ArchitectureAELDM`` (only used
        when ``backbone == "ldmVAE"``).
    """

    model_class: str = "VAE"
    backbone: str = "Swin-Tiny"
    latent_dim: int = 64
    input_dim: List[int] = field(default_factory=lambda: [1, 288, 128])
    nuisance_dim: int = 0

    # Decoder flag (Timm arches only)
    dec_use_local_attn: bool = False

    # Legacy ConvVAE extras (ignored for Timm / LDM arches)
    n_out_channels: int = 16
    n_conv_layers: int = 5
    kernel_size: int = 4
    stride: int = 2

    # LDM extras (ignored for Timm / Conv arches)
    ldm_params: dict = field(default_factory=dict)

    # -----------------------------------------------------------------------
    # Convenience properties
    # -----------------------------------------------------------------------

    @property
    def is_timm_arch(self) -> bool:
        return self.backbone not in ("convVAE", "ldmVAE")

    @property
    def biological_dim(self) -> int:
        return self.latent_dim - self.nuisance_dim

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ArchiveSpec":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------

_SPEC_FILENAME = "arch_spec.json"


def save_arch_spec(model_config: Any, run_dir: str | Path) -> Path:
    """Derive an ``ArchiveSpec`` from a ``VAEConfig`` / ``metricVAEConfig``
    and write it as ``arch_spec.json`` inside *run_dir*.

    Parameters
    ----------
    model_config :
        A ``VAEConfig`` or ``metricVAEConfig`` instance (the object returned
        by ``initialize_model``).
    run_dir : str or Path
        Output directory (the same directory that Hydra writes ``.hydra/``
        and ``checkpoints/`` into).

    Returns
    -------
    Path
        Path to the written JSON file.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    dd = model_config.ddconfig
    lc = model_config.lossconfig

    # --- nuisance_dim ---
    nuisance_dim = 0
    if model_config.name == "metricVAE" and hasattr(lc, "latent_dim_nuisance"):
        nuisance_dim = int(lc.latent_dim_nuisance)

    # --- LDM extras ---
    ldm_params: dict = {}
    if getattr(dd, "name", "") == "ldmVAE":
        ldm_fields = ("double_z", "z_channels", "resolution", "in_channels",
                       "out_ch", "ch", "ch_mult", "num_res_blocks",
                       "attn_resolutions", "dropout", "freeze_encoder_trunk")
        ldm_params = {f: getattr(dd, f) for f in ldm_fields if hasattr(dd, f)}

    spec = ArchiveSpec(
        model_class=model_config.name,
        backbone=str(dd.name),
        latent_dim=int(dd.latent_dim),
        input_dim=list(dd.input_dim),
        nuisance_dim=nuisance_dim,
        dec_use_local_attn=bool(getattr(dd, "dec_use_local_attn", False)),
        # Legacy Conv fields (harmless for Timm / LDM)
        n_out_channels=int(getattr(dd, "n_out_channels", 16)),
        n_conv_layers=int(getattr(dd, "n_conv_layers", 5)),
        kernel_size=int(getattr(dd, "kernel_size", 4)),
        stride=int(getattr(dd, "stride", 2)),
        ldm_params=ldm_params,
    )

    out_path = run_dir / _SPEC_FILENAME
    with open(out_path, "w") as fh:
        json.dump(spec.to_dict(), fh, indent=2)

    return out_path


def load_arch_spec(run_dir: str | Path) -> ArchiveSpec:
    """Load an ``ArchiveSpec`` from *run_dir*.

    Tries ``arch_spec.json`` first.  If not found, raises ``FileNotFoundError``
    with a helpful message pointing to the fallback (``load_trained_model``).

    Parameters
    ----------
    run_dir : str or Path
        The run directory.
    """
    run_dir = Path(run_dir)
    spec_path = run_dir / _SPEC_FILENAME
    if not spec_path.exists():
        raise FileNotFoundError(
            f"No arch_spec.json found at {spec_path}.\n"
            "This run predates ArchiveSpec support.  Either:\n"
            "  • Use load_trained_model() (requires Hydra config) instead, or\n"
            "  • Re-run training to generate arch_spec.json automatically."
        )
    with open(spec_path) as fh:
        data = json.load(fh)
    return ArchiveSpec.from_dict(data)


# ---------------------------------------------------------------------------
# Model construction from spec
# ---------------------------------------------------------------------------

def _build_model_from_spec(spec: ArchiveSpec):
    """Instantiate a ``VAE`` or ``metricVAE`` from an ``ArchiveSpec``.

    Does **not** load any weights.  Does **not** import loss modules,
    discriminators, or data configs.
    """
    from src.core.models.model_components.arch_configs import (
        TimmArchitecture, LegacyArchitecture, ArchitectureAELDM,
    )
    from src.core.models.factories import build_from_config
    from src.core.models.legacy_models import VAE, metricVAE

    # --- Build arch config ---
    if spec.is_timm_arch:
        dd = TimmArchitecture(
            name=spec.backbone,
            latent_dim=spec.latent_dim,
            input_dim=tuple(spec.input_dim),
            dec_use_local_attn=spec.dec_use_local_attn,
        )
    elif spec.backbone == "ldmVAE":
        dd = ArchitectureAELDM(
            latent_dim=spec.latent_dim,
            **spec.ldm_params,
        )
    else:  # convVAE
        dd = LegacyArchitecture(
            latent_dim=spec.latent_dim,
            input_dim=tuple(spec.input_dim),
            n_out_channels=spec.n_out_channels,
            n_conv_layers=spec.n_conv_layers,
            kernel_size=spec.kernel_size,
            stride=spec.stride,
        )

    # Minimal namespace understood by build_from_config
    class _FakeLoss:
        target = "NT-Xent"  # metricVAE.forward() reads this

    class _FakeCfg:
        pass

    cfg = _FakeCfg()
    cfg.name = spec.model_class
    cfg.ddconfig = dd
    cfg.lossconfig = _FakeLoss()

    return build_from_config(cfg)


# ---------------------------------------------------------------------------
# Public inference API
# ---------------------------------------------------------------------------

def load_encoder(
    run_path: str | Path,
    ckpt_name: str = "last.ckpt",
    map_location: str = "cpu",
):
    """Load a trained encoder (and decoder) from a checkpoint directory.

    This is the **lightweight inference path**: it requires only the
    ``arch_spec.json`` written at training time — no Hydra config, no loss
    modules, no discriminators, no data pipeline.

    Parameters
    ----------
    run_path : str or Path
        Path to the run directory (must contain ``arch_spec.json`` and
        ``checkpoints/``).
    ckpt_name : str, optional
        Checkpoint filename inside ``run_path/checkpoints/``.
        Defaults to ``"last.ckpt"``.  Pass ``None`` to auto-select the newest.
    map_location : str, optional
        Device for ``torch.load``.  Defaults to ``"cpu"``.

    Returns
    -------
    model : VAE or metricVAE (nn.Module)
        Frozen model in eval mode.  Call ``model(batch_tensor)`` to get a
        ``ModelOutput`` with ``.mu``, ``.logvar``, ``.recon_x``.
    spec : ArchiveSpec
        The spec that was used to build the model.

    Examples
    --------
    >>> model, spec = load_encoder("path/to/run")
    >>> model.eval()
    >>> with torch.no_grad():
    ...     out = model(images)          # images: (B, C, H, W)
    ...     mu  = out.mu                 # (B, latent_dim)
    ...     recon = out.recon_x          # (B, C, H, W)
    """
    run_path = Path(run_path)

    # 1. Load arch spec
    spec = load_arch_spec(run_path)

    # 2. Resolve checkpoint
    ckpt_dir = run_path / "checkpoints"
    if ckpt_name is not None:
        ckpt_path = ckpt_dir / ckpt_name
    else:
        ckpts = sorted(ckpt_dir.glob("*.ckpt"))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
        ckpt_path = ckpts[-1]
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # 3. Build model architecture (no loss / data imports)
    model = _build_model_from_spec(spec)

    # 4. Load state dict (LitModel stores model weights under "model.*")
    ckpt = torch.load(str(ckpt_path), map_location=map_location, weights_only=False)
    raw_sd = ckpt.get("state_dict", ckpt)

    model_sd = model.state_dict()
    filtered: dict = {}
    skipped: list = []
    for raw_key, val in raw_sd.items():
        # Strip the "model." prefix added by LitModel
        key = raw_key[len("model."):] if raw_key.startswith("model.") else raw_key
        if key in model_sd and val.shape == model_sd[key].shape:
            filtered[key] = val
        else:
            skipped.append(raw_key)

    model.load_state_dict(filtered, strict=False)

    if skipped:
        warnings.warn(
            f"load_encoder: skipped {len(skipped)} state_dict key(s) due to shape "
            f"mismatches or missing keys (e.g. {skipped[:3]}).  "
            "Loss / discriminator weights are expected to be absent.",
            stacklevel=2,
        )

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return model, spec
