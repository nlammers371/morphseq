# Training Guide

This document explains how to launch both a **singleton training run** and a
**parameter sweep** using the morphseq VAE framework.

---

## Prerequisites

All commands assume:
- You are in the repository root (`/path/to/morphseq`).
- The `MORPHSEQ_OUTPUT_ROOT` environment variable is set to a directory where
  training outputs (checkpoints, logs, `arch_spec.json`) should be written.
- A W&B account is configured (run `wandb login` once).

```bash
export MORPHSEQ_OUTPUT_ROOT=/data/morphseq/training_outputs
```

The framework uses [Hydra](https://hydra.cc) for config management and
[PyTorch Lightning](https://lightning.ai) for the training loop.  The entry
point is `src/core/run/training.py`.

---

## 1. Singleton training run

A singleton run trains a single model with fixed hyperparameters.

### Command

```bash
python -m src.core.run.training \
    model=vae_timm \
    model.ddconfig.name="Swin-Tiny" \
    model.ddconfig.latent_dim=64 \
    model.lossconfig.kld_weight=1.0 \
    model.lossconfig.pips_weight=7.5 \
    model.lossconfig.use_gan=True \
    model.lossconfig.gan_net="style2" \
    model.dataconfig.batch_size=64 \
    model.trainconfig.max_epochs=50 \
    hydra.job.name=my_run
```

### What each key controls

| Key | Default | Description |
|-----|---------|-------------|
| `model` | `vae_timm` | Which model YAML to load from `hydra_configs/model/` |
| `model.ddconfig.name` | `"Swin-Tiny"` | Encoder backbone (see `ARCH_REGISTRY` in `model_configs.py`) |
| `model.ddconfig.latent_dim` | `64` | Total latent dimension |
| `model.lossconfig.kld_weight` | `1.0` | KLD regularisation weight |
| `model.lossconfig.pips_weight` | `7.5` | Perceptual (LPIPS) loss weight |
| `model.lossconfig.use_gan` | `True` | Enable GAN discriminator |
| `model.lossconfig.gan_net` | `"patch4scale"` | Discriminator architecture |
| `model.trainconfig.max_epochs` | `50` | Total training epochs |
| `model.trainconfig.lr_base` | `1e-4` | Base learning rate |
| `model.trainconfig.encoder_lr_scale` | `0.1` | Backbone LR = lr_base × this |
| `model.trainconfig.grad_clip_norm` | `0.0` | Gradient clip norm (0 = off) |
| `model.dataconfig.batch_size` | `64` | Batch size per GPU |
| `model.dataconfig.root` | *(from ancestor resolver)* | Dataset root |
| `hydra.job.name` | *(script name)* | Human-readable run label |

### Output layout

```
$MORPHSEQ_OUTPUT_ROOT/
  runs/
    my_run_20250317_120000/
      arch_spec.json          ← lightweight model descriptor (for load_encoder)
      .hydra/
        config.yaml           ← full resolved config snapshot
      checkpoints/
        last.ckpt
        epoch09.ckpt          ← if save_epochs=[9,19,29]
        epoch19.ckpt
      tensorboard/
      wandb/
```

### MetricVAE (contrastive) singleton

```bash
python -m src.core.run.training \
    model=metric_vae_timm \
    model.ddconfig.latent_dim=128 \
    model.lossconfig.metric_weight=1.0 \
    model.lossconfig.temperature=0.1 \
    model.trainconfig.max_epochs=50 \
    hydra.job.name=metric_run
```

---

## 2. Parameter sweep (multirun)

A sweep launches multiple runs in parallel (or sequentially, depending on your
cluster), each with a different combination of hyperparameters.  Hydra's
`--multirun` flag + its grid-search launcher handles this automatically.

### Command — grid search

Separate multiple values for a single parameter with commas.  Hydra produces
the Cartesian product across all comma-separated parameters.

```bash
python -m src.core.run.training --multirun \
    hydra.job.name=sweep_kld_pips \
    model=vae_timm \
    model.lossconfig.kld_weight=0.5,1.0,2.0 \
    model.lossconfig.pips_weight=5.0,7.5,10.0 \
    model.ddconfig.name="Swin-Tiny" \
    model.trainconfig.max_epochs=50
```

This launches 3 × 3 = **9 runs**, each in its own subdirectory:

```
$MORPHSEQ_OUTPUT_ROOT/
  sweeps/
    sweep_kld_pips_20250317_120000/
      0/    ← kld=0.5, pips=5.0
      1/    ← kld=0.5, pips=7.5
      2/    ← kld=0.5, pips=10.0
      3/    ← kld=1.0, pips=5.0
      ...
      multirun.yaml
```

Each subdirectory has the same layout as a singleton run (including
`arch_spec.json`).

### Command — backbone + GAN sweep (shell script style)

The `src/core/run/sweep*/` shell scripts show real sweep patterns.
A minimal template:

```bash
#!/usr/bin/env bash
set -euo pipefail

python -m src.core.run.training --multirun \
    hydra.job.name=sweep_arch_gan \
    model=vae_timm \
    model.ddconfig.name="Swin-Tiny","Swin-Large" \
    model.lossconfig.gan_net="style2","patch4scale" \
    model.lossconfig.pips_weight=7.5 \
    model.lossconfig.kld_weight=1.0 \
    model.trainconfig.max_epochs=50 \
    model.dataconfig.batch_size=64
```

Save this as `src/core/run/sweep_next/my_sweep.sh`, make it executable
(`chmod +x`), and run it from the repo root.

### Aggregating sweep results

After a multirun completes, `collect_results_recursive` in `run_utils.py`
walks the sweep directory and produces a `job_summary_df.csv`:

```python
from src.core.run.run_utils import collect_results_recursive

df = collect_results_recursive("$MORPHSEQ_OUTPUT_ROOT/sweeps/sweep_arch_gan_...")
print(df.sort_values("val/lpips_val").head())
```

---

## 3. Key tunable hyperparameters

### Loss weights and schedules

All loss weights can be scheduled (ramped up gradually during training) or
held constant.  The schedules are cosine ramps controlled by `ramp_scale` and
`hold_scale` in `BasicLoss`:

```
epoch 0                                    max_epochs
  ├── [metric ramp]                           (NTXentLoss only)
  │       └── [KLD hold] → [KLD ramp]
  │                              └── [PIPS hold] → [PIPS ramp]
  │                                                     └── [GAN hold] → [GAN ramp]
```

| Flag | Effect |
|------|--------|
| `schedule_kld=True` | KLD starts at 0, ramps to `kld_weight` |
| `schedule_pips=True` | PIPS starts after KLD reaches target |
| `schedule_gan=True` | GAN starts after PIPS reaches target |
| `schedule_metric=True` | (metricVAE only) metric loss ramps first |

To skip scheduling for a term, set `schedule_<term>=False`.

### Gradient clipping

Enable gradient clipping to stabilise GAN training:

```bash
model.trainconfig.grad_clip_norm=1.0
```

### Encoder learning rate

The encoder backbone uses a reduced LR by default (10× lower than the decoder)
to preserve pretrained features.  To sweep over this:

```bash
model.trainconfig.encoder_lr_scale=0.05,0.1,0.2
```

### Available discriminator architectures

| `gan_net` value | Architecture | Notes |
|-----------------|--------------|-------|
| `"patch"` | 70×70 PatchGAN | Fast, lightweight |
| `"ms_patch"` | Multi-scale PatchGAN (3 scales) | Richer signal |
| `"patch4scale"` | Multi-scale PatchGAN (4 scales) | Default |
| `"style2"` | StyleGAN-2 (default depth) | Strong |
| `"style2_small"` | StyleGAN-2 (4 blocks) | Faster |
| `"style2_big"` | StyleGAN-2 (7 blocks) | Strongest |
| `"resnet_sn"` | ResNet-50 with spectral norm | Global context |

### Available encoder backbones

| `ddconfig.name` | Architecture | Param count |
|-----------------|--------------|-------------|
| `"Swin-Tiny"` | Swin Transformer Tiny | ~28M |
| `"Swin-Large"` | Swin Transformer Large | ~197M |
| `"ViT-Tiny"` | ViT-Tiny (augreg) | ~5M |
| `"ViT-Large"` | ViT-Large (DINOv2) | ~307M |
| `"Efficient-B0-RA"` | EfficientNet-B0 | ~5M |
| `"Efficient-B4"` | EfficientNet-B4 | ~19M |
| `"ConvNeXt-Tiny"` | ConvNeXt-Tiny | ~28M |
| `"MaxViT-Tiny"` | MaxViT-Tiny | ~31M |
| `"MaxViT-Small"` | MaxViT-Small | ~69M |
| `"DeiT-Tiny"` | DeiT-Tiny | ~5M |
| `"RegNet-Y"` | RegNetY-064 | ~30M |

---

## 4. Cluster / SLURM submission

Wrap the singleton or multirun command in a SLURM batch script:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=morphseq_sweep
#SBATCH --gpus-per-node=4
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --output=slurm_%j.log

export MORPHSEQ_OUTPUT_ROOT=/data/morphseq/training_outputs

python -m src.core.run.training --multirun \
    hydra.job.name=sweep_kld_pips \
    model=vae_timm \
    model.lossconfig.kld_weight=0.5,1.0,2.0 \
    model.lossconfig.pips_weight=5.0,7.5,10.0 \
    model.trainconfig.max_epochs=50
```

The Lightning `DDPStrategy` is pre-configured; Lightning auto-detects all
available GPUs (`devices="auto"`).
