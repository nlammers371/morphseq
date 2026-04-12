# Inference Guide: Loading a Trained Model with ArchiveSpec

This document covers the two ways to load a trained VAE checkpoint:

1. **`load_encoder`** — lightweight path, no Hydra required (recommended for
   embeddings and reconstructions)
2. **`load_trained_model`** — full path, loads the complete `LitModel`
   including loss functions and data config (required for training resumption
   or if the run pre-dates `arch_spec.json`)

---

## How ArchiveSpec works

Every training run now writes a small JSON file called `arch_spec.json` into
the run output directory alongside the checkpoints:

```
$MORPHSEQ_OUTPUT_ROOT/runs/my_run_20250317_120000/
  arch_spec.json          ← this file
  checkpoints/
    last.ckpt
    epoch09.ckpt
  .hydra/
    config.yaml
```

`arch_spec.json` contains only the information needed to reconstruct the model
architecture — backbone name, latent dimension, input size, etc.:

```json
{
  "model_class": "VAE",
  "backbone": "Swin-Tiny",
  "latent_dim": 64,
  "input_dim": [1, 288, 128],
  "nuisance_dim": 0,
  "dec_use_local_attn": false,
  "n_out_channels": 16,
  "n_conv_layers": 5,
  "kernel_size": 4,
  "stride": 2,
  "ldm_params": {}
}
```

`load_encoder` reads this file and loads the checkpoint weights directly into
the reconstructed model, **without** importing Hydra, loss functions,
discriminators, or data configs.

---

## 1. Quick start — `load_encoder`

```python
from src.core.models.arch_spec import load_encoder
import torch

RUN_PATH = "/data/morphseq/training_outputs/runs/my_run_20250317_120000"

# Load the model (last checkpoint by default)
model, spec = load_encoder(RUN_PATH)

# Optional: move to GPU
model = model.cuda()

# --- Embed a batch ---
images = torch.rand(8, 1, 288, 128).cuda()   # (B, C, H, W) in [0, 1]

model.eval()
with torch.no_grad():
    out = model(images)

mu      = out.mu        # (B, latent_dim)  posterior mean
logvar  = out.logvar    # (B, latent_dim)  posterior log-variance
recon   = out.recon_x   # (B, C, H, W)    reconstruction
z       = out.z         # (B, latent_dim)  reparameterised sample
```

### Choosing a specific checkpoint

```python
# By filename:
model, spec = load_encoder(RUN_PATH, ckpt_name="epoch09.ckpt")

# Auto-select the newest:
model, spec = load_encoder(RUN_PATH, ckpt_name=None)
```

### Inspecting the spec

```python
print(spec.backbone)        # "Swin-Tiny"
print(spec.latent_dim)      # 64
print(spec.model_class)     # "VAE"
print(spec.biological_dim)  # latent_dim - nuisance_dim
print(spec.input_dim)       # [1, 288, 128]
```

---

## 2. Generating embeddings for a dataset

```python
import torch
from torch.utils.data import DataLoader
from src.core.models.arch_spec import load_encoder

model, spec = load_encoder(RUN_PATH)
model = model.cuda().eval()

# Assume your dataset returns {"data": tensor, "label": [...]}
loader = DataLoader(your_dataset, batch_size=64, num_workers=4)

all_mu, all_labels = [], []

with torch.no_grad():
    for batch in loader:
        images = batch["data"].cuda()
        out = model(images)
        all_mu.append(out.mu.cpu())
        all_labels.extend(batch["label"])

embeddings = torch.cat(all_mu, dim=0)   # (N, latent_dim)
```

---

## 3. Encoder-only extraction (skip the decoder)

For embedding generation you don't need the decoder forward pass.  Access the
encoder sub-module directly:

```python
encoder = model.encoder

with torch.no_grad():
    enc_out = encoder(images)

mu     = enc_out.embedding          # (B, latent_dim)
logvar = enc_out.log_covariance     # (B, latent_dim)
```

This is ~2× faster than the full forward pass because it skips the decoder.

---

## 4. MetricVAE — accessing biological vs. nuisance latents

If the model was trained as a `metricVAE` (contrastive metric learning),
`spec.nuisance_dim > 0` and the latent space is partitioned:

```
z[:, :nuisance_dim]           ← nuisance dimensions (batch effects, technical noise)
z[:, nuisance_dim:]           ← biological dimensions (morphological signal)
```

```python
model, spec = load_encoder(RUN_PATH)

with torch.no_grad():
    out = model(images)

nuis_dim = spec.nuisance_dim
mu_bio   = out.mu[:, nuis_dim:]    # biological latents
mu_nuis  = out.mu[:, :nuis_dim]   # nuisance latents
```

---

## 5. Fallback: `load_trained_model` (legacy / Hydra path)

Use this if the run directory does **not** contain `arch_spec.json` (i.e., the
run was produced before the `arch_spec` feature was added):

```python
from src.core.run.run_utils import load_trained_model
import pytorch_lightning as pl

lit_model, eval_data_cfg, model_cfg = load_trained_model(
    run_path=RUN_PATH,
    ckpt_name="last.ckpt",
)

# Run prediction via Lightning Trainer
trainer = pl.Trainer(accelerator="gpu", devices=1)
predictions = trainer.predict(lit_model, dataloaders=your_dataloader)

# Each prediction dict contains:
#   predictions[i]["mu"]       → (B, latent_dim)
#   predictions[i]["log_var"]  → (B, latent_dim)
#   predictions[i]["recon"]    → (B, C, H, W)
#   predictions[i]["snip_ids"] → list[str]
```

`load_trained_model` requires:
- The `.hydra/config.yaml` snapshot in the run directory.
- A resolvable dataset root (`metadata/age_key.csv` must be findable above the
  run directory).
- All loss / discriminator modules to be importable (they are loaded even if
  you only want embeddings).

---

## 6. Saving arch_spec.json retroactively for old runs

If you have pre-existing runs without `arch_spec.json`, you can generate one
from the Hydra config:

```python
from pathlib import Path
from src.core.run.run_utils import load_trained_model
from src.core.models.arch_spec import save_arch_spec

RUN_PATH = "/path/to/old/run"

# load_trained_model also initialises model_config from .hydra/config.yaml
_, _, model_config = load_trained_model(RUN_PATH)

spec_path = save_arch_spec(model_config, run_dir=RUN_PATH)
print(f"Written: {spec_path}")
```

After this, `load_encoder(RUN_PATH)` will work without Hydra.

---

## 7. Summary of `ArchiveSpec` fields

| Field | Type | Description |
|-------|------|-------------|
| `model_class` | str | `"VAE"` or `"metricVAE"` |
| `backbone` | str | Encoder architecture name (e.g. `"Swin-Tiny"`) |
| `latent_dim` | int | Total latent dimension |
| `input_dim` | list[int] | `[C, H, W]` of expected input images |
| `nuisance_dim` | int | Number of nuisance latent dims (0 for plain VAE) |
| `dec_use_local_attn` | bool | Whether decoder uses local attention |
| `n_out_channels` | int | Base channels for legacy ConvVAE (ignored for Timm) |
| `n_conv_layers` | int | Conv depth for legacy ConvVAE (ignored for Timm) |
| `kernel_size` | int | Kernel size for legacy ConvVAE (ignored for Timm) |
| `stride` | int | Stride for legacy ConvVAE (ignored for Timm) |
| `ldm_params` | dict | Extra kwargs for LDM encoder (ignored for Timm / Conv) |
| `is_timm_arch` | property | `True` when backbone is not `"convVAE"` or `"ldmVAE"` |
| `biological_dim` | property | `latent_dim - nuisance_dim` |
