# MorphSeq VAE — Architecture Reference (for an external architect)

> **Audience & purpose.** This document is written for an LLM collaborator who
> *cannot see the repo*. It describes the **model machinery** that learns a
> morphological latent space from zebrafish embryo imaging, so that refactors and
> three planned extensions (predictive/forward-contrastive loss; slice-aware
> encoding with a focal-depth token; latent-distance↔developmental-time
> calibration) can be designed from this document alone.
>
> **Verbatim policy.** Anything likely to be modified is reproduced *verbatim*
> with `file:line` citations. Prose is orientation only. Where the code
> contradicts the project framing, or where something is unresolved, it is
> flagged with **⚠️ FLAG**.
>
> **Single most important orientation fact.** There are *four* parallel copies of
> the VAE stack in `src/` (`src/core/`, `src/vae/`, `src/legacy/`, `src/_Archive/`).
> **Only `src/core/` is live.** The Hydra+Lightning entry point is
> `src/core/run/training.py`. Everything in §3–§7 below is from `src/core/` unless
> explicitly stated. `src/vae/`, `src/legacy/`, `src/_Archive/` are
> dead/superseded and should be treated as read-only history.
>
> **Second most important fact (a landmine).** The *training-time data config* in
> `src/core` (`src/data/dataset_configs.py`) is currently a **minimal analysis
> shim** that does **not** implement the paired-view + metadata API that the
> MetricVAE training step and NT-Xent loss require. See §3.4 and §8. The real
> paired sampler (`SeqPairDatasetCached`) exists in
> `src/core/functions/dataset_utils.py` but is **not wired** into the config.
> Any change to the input pipeline must reconcile this.

---

## 1. Repo map

### 1.1 High-level (non-`src`) tree — annotated, shallow

```
morphseq/
├── docs/                     # human docs (training_guide.md, inference_guide.md, this file)
├── src/                      # ← ALL core model machinery lives here (focus of this doc)
├── dev/particle_prediction/  # separate sub-project; has its own conda rules in CLAUDE.md
├── metadata/                 # age_key.csv etc. (developmental-age lookup; consumed at data-config build time)
├── results/, jupyter/, morphseq_playground/, morphseq_vault/   # notebooks, outputs, scratch
├── tests/                    # repo-level tests (mostly analysis/build, not the VAE core)
├── _Archive/, segmentation_sandbox/, visualization/, image_building/, mathematica/, seq/   # out of scope
└── *.md, *.yml, check_*.py   # status notes, conda envs, staleness checks
```

Non-`src` folders are **out of scope** except `metadata/` (the `metadata/age_key.csv`
file is how inference utilities locate the data root — see
`_find_data_root`, `run_utils.py:353`).

### 1.2 `src/` tree — what is live vs. dead

```
src/
├── core/                  ★ LIVE training/model stack (Hydra + PyTorch Lightning)
│   ├── run/
│   │   ├── training.py            ★ ENTRY POINT (@hydra.main)
│   │   ├── run_utils.py           ★ train_vae(), initialize_model(), load_trained_model(), result aggregation
│   │   ├── compat.py              legacy sys.modules aliasing for old checkpoints
│   │   ├── registry.py, training_cluster.py
│   │   └── sweep01_files … sweep11/   shell scripts for Hydra multirun sweeps
│   ├── lightning/
│   │   ├── pl_wrappers.py         ★ LitModel (the training/val/predict loop)
│   │   ├── train_config.py        ★ LitTrainConfig (LRs, epochs, grad clip)
│   │   ├── callbacks.py           SaveRunMetadata, EpochListCheckpoint
│   │   └── pl_utils.py            ramp_weight / cosine_ramp_weight (loss schedules)
│   ├── losses/
│   │   ├── loss_functions.py      ★ VAELossBasic, NTXentLoss, _VAELossBase, EVALPIPSLOSS
│   │   ├── loss_configs.py        ★ BasicLoss, MetricLoss (pydantic configs + schedules + bio/nuisance indices)
│   │   ├── loss_helpers.py        ★ lpips_score, ssim_score, LatentCovarianceLoss (eval metrics)
│   │   ├── discriminators.py      7 GAN discriminator variants
│   │   ├── perceptual_custom.py   (6-line stub)
│   │   └── ldm_loss_functions.py  for the LDM autoencoder branch (secondary)
│   ├── models/
│   │   ├── factories.py           ★ build_from_config() — dispatches name → encoder/decoder/model
│   │   ├── model_configs.py       ★ VAEConfig, metricVAEConfig (+ ARCH_REGISTRY)
│   │   ├── legacy_models.py       ★ VAE, metricVAE nn.Modules (forward + latent split)  ⚠️ misleading filename: this is LIVE
│   │   ├── arch_spec.py           ★ ArchiveSpec — checkpoint-free inference (load_encoder)
│   │   ├── model_utils.py         ModelOutput, deep_merge, prune_empty
│   │   ├── ldm_models.py          AutoencoderKLModel (LDM branch, secondary)
│   │   └── model_components/
│   │       ├── timm_components.py     ★ TimmEncoder, UniDecLite (the live encoder/decoder)
│   │       ├── arch_configs.py        ★ TimmArchitecture, LegacyArchitecture, ArchitectureAELDM
│   │       ├── legacy_components.py    EncoderConvVAE, DecoderConvVAEUpsamp (old conv path)
│   │       ├── window_attention.py     optional decoder local attention (flagged buggy in YAML)
│   │       ├── ldm_components_ae.py     LDM encoder/decoder wrappers (secondary)
│   │       └── timm_helpers.py          vit_resize / swin_resize
│   ├── functions/
│   │   └── dataset_utils.py       ⚠️ SeqPairDatasetCached/TripletDatasetCached — the REAL paired sampler, NOT wired in (see §3.4/§8)
│   ├── hydra_configs/
│   │   ├── base.yaml              root config (output dirs, wandb)
│   │   └── model/{vae_timm,metric_vae_timm,vae_timm_no_pips}.yaml
│   └── diffusion/, lightning_logs/, …   LDM experiments (secondary)
│
├── data/                  ⚠️ SHIM ONLY
│   ├── dataset_configs.py     BaseDataConfig/EvalDataConfig/NTXentDataConfig — analysis shim (single images, no pairs/metadata)
│   └── data_transforms.py     basic_transform()
│
├── analyze/               downstream analysis (embeddings, trajectories, classification, OT, viz) — consumes trained models
│   ├── analysis_utils.py      uses load_encoder/load_trained_model
│   └── assess_hydra_results.py  UMAP / result assessment  ⚠️ has stale imports (`from run.run_utils …`, `from data.dataset_configs …`)
│
├── build/                 image build pipeline → produces 2D "full-focus" snips the VAE consumes (see §3.1). Out of model scope.
│   ├── export_utils.py         LoG_focus_stacker  ← the full-focus / EDF projection (BUILD-TIME, not in the model)
│   ├── build01A/01B_compile_*   z-slice compilation + focus stacking
│   └── build03A/03B/05_*        embryo segmentation, snip export, training-snip assembly
├── data_pipeline/         newer Snakemake-based rebuild of `build/` (parallel effort; out of model scope)
├── run_morphseq_pipeline/ CLI orchestration of build + embedding generation (services/gen_embeddings.py uses load_encoder)
│
├── vae/        ✗ SUPERSEDED pythae-style stack (seq_vae, morph_iaf_vae, metric_vae, normalizing_flows). NOT live.
├── legacy/     ✗ DEAD copy of the vae stack
├── _Archive/   ✗ DEAD older copies (build_orig, vae)
├── models/, diffusion/, ml_preprocessing/, crossmodal/, app/, misc/, core/diffusion/…   ✗ peripheral / experimental
```

**Outmoded/unused folders inside `src/` (do not modify, do not design against):**
`src/vae/`, `src/legacy/`, `src/_Archive/`, `src/models/`, `src/ml_preprocessing/`,
`src/diffusion/` (top-level), and the `_Archive/` subdirs under `src/build` and
`src/vae`. The `src/core/diffusion/` + `src/core/models/ldm_*` LDM branch is real
code but a *secondary* experiment (a KL-regularized latent-diffusion autoencoder),
not the metric-VAE that §3–§7 describe.

⚠️ **FLAG — naming traps:**
- `src/core/models/legacy_models.py` is the **live** model file despite "legacy".
- `src/core/functions/` duplicates `src/functions/` and `src/core/functions/dataset_utils.py` is the real paired sampler. Confusing; see §8.

### 1.3 Entry points & exact commands

**Train (singleton)** — from repo root, with `MORPHSEQ_OUTPUT_ROOT` set:
```bash
export MORPHSEQ_OUTPUT_ROOT=/data/morphseq/training_outputs
# plain VAE
python -m src.core.run.training \
    model=vae_timm \
    model.ddconfig.name="Swin-Tiny" model.ddconfig.latent_dim=64 \
    model.lossconfig.kld_weight=1.0 model.lossconfig.pips_weight=7.5 \
    model.lossconfig.use_gan=True model.lossconfig.gan_net="style2" \
    model.dataconfig.batch_size=64 model.trainconfig.max_epochs=50 \
    hydra.job.name=my_run
# MetricVAE (contrastive)
python -m src.core.run.training \
    model=metric_vae_timm model.ddconfig.latent_dim=128 \
    model.lossconfig.metric_weight=1.0 model.lossconfig.temperature=0.1 \
    model.trainconfig.max_epochs=50 hydra.job.name=metric_run
```

**Sweep (Hydra multirun):**
```bash
python -m src.core.run.training --multirun \
    hydra.job.name=sweep_kld_pips model=vae_timm \
    model.lossconfig.kld_weight=0.5,1.0,2.0 model.lossconfig.pips_weight=5.0,7.5,10.0
```
Real sweep shell scripts live in `src/core/run/sweep*/`. After a multirun,
`collect_results_recursive(results_dir)` (`run_utils.py:56`) writes
`job_summary_df.csv`.

**Evaluate / inference** — two paths (`docs/inference_guide.md`):
- Lightweight: `from src.core.models.arch_spec import load_encoder; model, spec = load_encoder(run_path)` — needs only `arch_spec.json` + checkpoint.
- Full: `from src.core.run.run_utils import load_trained_model; lit, eval_cfg, mcfg = load_trained_model(run_path)` — rebuilds the full `LitModel` from `.hydra/config.yaml`.

⚠️ **FLAG:** there is no single canonical "evaluate" CLI in `src/core`.
`src/analyze/assess_hydra_results.py` exists but has broken top-level imports
(`from run.run_utils import …`, `from data.dataset_configs import …`) and looks
stale. Embedding generation for downstream analysis goes through
`src/run_morphseq_pipeline/services/gen_embeddings.py` and
`src/analyze/analysis_utils.py`. See §7.

---

## 2. Stack & config

- **Framework:** PyTorch + **PyTorch Lightning** (`LitModel(pl.LightningModule)`),
  with **manual optimization** (`self.automatic_optimization = False`,
  `pl_wrappers.py:44`) because of the VAE+GAN two-optimizer setup.
- **Config management:** **Hydra** (`@hydra.main`, `training.py:23`) composes YAML
  in `src/core/hydra_configs/`. The composed `DictConfig` is immediately turned
  into a plain dict (`OmegaConf.to_container(cfg, resolve=True)`, `training.py:29`)
  and passed to `train_vae`. Typed validation is then done by **pydantic
  dataclasses** (`VAEConfig`/`metricVAEConfig` in `model_configs.py`;
  `BasicLoss`/`MetricLoss` in `loss_configs.py`; `LitTrainConfig` in
  `train_config.py`; `TimmArchitecture`/`LegacyArchitecture` in `arch_configs.py`).
  A custom OmegaConf resolver `ancestor` (`training.py:16-21`) resolves the data
  root from the Hydra run dir.
- **Experiment tracking:** **Weights & Biases** + **TensorBoard** (both attached
  to the Trainer, `run_utils.py:171-185, 216`).
- **Backbones:** **timm** pretrained models (Swin/ViT/ConvNeXt/MaxViT/EfficientNet/
  RegNet/DeiT) — see `timm_dict`, `arch_configs.py:21-32`. **LPIPS** (`lpips` pkg)
  for perceptual loss; **piq** for MS-SSIM eval.
- **Hardware assumptions (hard-coded):** `accelerator="gpu"`, `precision=16`,
  `strategy=DDPStrategy(find_unused_parameters=True)`, `devices="auto"`
  (`run_utils.py:216-224`). `torch.set_float32_matmul_precision("medium")`.
  Multi-GPU DDP is the default; there is **no CPU fallback** in `train_vae` (the
  `pick_devices` helper exists, `run_utils.py:261`, but is unused). The
  `dev/particle_prediction/` subtree must run via `conda run -n morphseq-env`
  (per repo `CLAUDE.md`); the core VAE has no such constraint documented.
- **Pixel-scale assumption:** the loss hard-codes a **128×288** image
  (`_pixel_scale`, `loss_functions.py:183-187`); input is `(1, 288, 128)`
  grayscale throughout (`arch_configs.py:18,47`).

---

## 3. Data pipeline

### 3.1 How z-stacks become "full-focus" images (UPSTREAM, build-time)

The VAE **never sees z-stacks**. Confocal z-slices are projected to a single 2D
"full-focus" (extended-depth-of-field) grayscale image **at build time**, by a
Laplacian-of-Gaussian focus stacker:

- `LoG_focus_stacker` in `src/build/export_utils.py` (used by
  `src/build/build01A_compile_keyence_torch.py` / `build01B_compile_yx1_images_torch.py`;
  benchmarked in `src/build/benchmark_focus_stacker.py`).
- Downstream, `build03A/03B/05` segment embryos and export per-embryo **snips**
  into `…/training_data/bf_embryo_snips/<experiment>/*.jpg`.

⚠️ **FLAG (critical for Extension #2):** focus-stacking collapses δ (focal depth)
**before** the model. By the time data reaches the VAE, depth information is gone
and each sample is one 2D grayscale snip. I read only the *locations* of the
build-time projection, not its internals (out of `src/` model scope). Confirm
`LoG_focus_stacker`'s signature before designing slice-aware encoding.

### 3.2 The transform actually used by the wired config (VERBATIM)

`src/data/data_transforms.py` (entire file):
```python
from __future__ import annotations

from typing import Callable, Optional, Tuple

from torchvision import transforms


def basic_transform(target_size: Optional[Tuple[int, int]] = None) -> Callable:
    """Minimal transform used for legacy embedding generation.

    The legacy VAE expects grayscale tensors. Many call sites pass
    `target_size=(288, 128)`.
    """
    ops = [transforms.Grayscale(num_output_channels=1)]
    if target_size is not None:
        ops.append(transforms.Resize(target_size))
    ops.append(transforms.ToTensor())
    return transforms.Compose(ops)
```

### 3.3 The dataset wired into `LitModel` today (VERBATIM — the shim)

`LitModel.train_dataloader` (`pl_wrappers.py:319`) calls
`self.data_cfg.create_dataset()`. With the current `src/core` import graph,
`data_cfg` is a `BaseDataConfig`/`NTXentDataConfig` from
`src/data/dataset_configs.py`, whose `create_dataset` returns `_SnipDataset`:

`src/data/dataset_configs.py:12-33, 69-115` (verbatim, abridged to the relevant classes):
```python
class _SnipDataset(Dataset):
    def __init__(self, image_paths: Sequence[Path], transform: Optional[Callable] = None):
        self._paths = list(image_paths)
        self._transform = transform

    def __len__(self) -> int:  # noqa: D401
        return len(self._paths)

    def __getitem__(self, idx: int):
        p = self._paths[idx]
        img = Image.open(p).convert("L")
        if self._transform is not None:
            x = self._transform(img)
        else:
            # Default: 1×H×W float tensor in [0, 1]
            x = torch.from_numpy(torch.tensor(img, dtype=torch.uint8).numpy()).float() / 255.0
            x = x.unsqueeze(0)
        return {"data": x, "label": (str(p), 0)}
```
```python
@dataclass
class BaseDataConfig:
    """Minimal shim for analysis scripts. ... only supports the subset used by
    downstream analysis utilities."""
    root: Union[str, Path]
    return_sample_names: bool = True
    transforms: Optional[Callable] = None
    transform_name: str = "basic"
    num_workers: int = 0
    train_indices: Optional[Sequence[int]] = None
    test_indices: Optional[Sequence[int]] = None
    eval_indices: Optional[Sequence[int]] = None

    def make_metadata(self) -> None:
        # Legacy API hook; no-op for the snip-folder dataset.
        return None

    def create_dataset(self) -> Dataset:
        root_p = Path(self.root)
        paths = _collect_snip_paths(root_p, experiments=None)
        return _SnipDataset(paths, transform=self.transforms)


class NTXentDataConfig(BaseDataConfig):
    """Placeholder to satisfy imports; not used by the pipeline embedding path."""
    pass
```

This shim yields **single images** `{"data": (1,H,W), "label": (path,0)}` with
**no** `self_stats`/`other_stats`, **no** `metric_array`, **no** paired views,
and `make_metadata()` is a no-op (so `data_config.metric_array` does not exist).

### 3.4 The dataset the training step + NT-Xent loss actually REQUIRE (VERBATIM)

`metricVAE.forward` expects `x` of shape **(B, 2, C, H, W)** and `NTXentLoss`
reads `model_input["self_stats"]` / `["other_stats"]`. The only code that
produces these is `SeqPairDatasetCached.__getitem__` in
`src/core/functions/dataset_utils.py:214-309` (verbatim):
```python
    def __getitem__(self, index):
        if index in self.cache:
            X = self.cache[index]
        else:
            X = Image.open(self.samples[index][0])
            if self.transform:
                X = self.transform(X)

        # determine if we're in train or eval partition
        train_flag = index in self.model_config.train_indices
        if train_flag:
            group_bool_vec = self.model_config.train_bool
        else:
            group_bool_vec = self.model_config.eval_bool

        key_dict = self.model_config.seq_key_dict  # [self.mode]

        pert_id_vec = key_dict["pert_id_vec"]
        e_id_vec = key_dict["e_id_vec"]
        age_hpf_vec = key_dict["age_hpf_vec"]

        time_window = self.model_config.time_window
        self_target = self.model_config.self_target_prob
        other_age_penalty = self.model_config.other_age_penalty

        #############3
        # Select sequential pair
        pert_id_input = pert_id_vec[index]
        e_id_input = e_id_vec[index]
        age_hpf_input = age_hpf_vec[index]

        # load metric array
        metric_array = self.model_config.metric_array
        pos_pert_ids = np.where(metric_array[pert_id_input, :]==1)[0]

        pert_match_array = np.isin(pert_id_vec, pos_pert_ids)
        if self.time_only_flag: # if true, disregard class match info
            pert_match_array = np.ones_like(pert_match_array, dtype=np.bool_)

        e_match_array = e_id_vec == e_id_input
        age_delta_array = np.abs(age_hpf_vec - age_hpf_input)
        age_match_array = age_delta_array <= time_window

        # positive options
        self_option_array = e_match_array & age_match_array & group_bool_vec
        other_option_array = (~e_match_array) & age_match_array & pert_match_array & group_bool_vec

        if (np.random.rand() <= self_target) or (np.sum(other_option_array) == 0):
            options = np.nonzero(self_option_array)[0]
            seq_pair_index = np.random.choice(options, 1, replace=False)[0]
            weight_hpf = age_delta_array[seq_pair_index] + 1
        else:
            options = np.nonzero(other_option_array)[0]
            seq_pair_index = np.random.choice(options, 1, replace=False)[0]
            weight_hpf = age_delta_array[seq_pair_index] + 1 + other_age_penalty

        if (seq_pair_index in self.cache) and (seq_pair_index != index):
            Y = self.cache[seq_pair_index]
        else:
            Y = Image.open(self.samples[seq_pair_index][0])
            if self.transform:
                Y = self.transform(Y)
            self.cache[seq_pair_index] = Y

        X = torch.reshape(X, (1, X.shape[0], X.shape[1], X.shape[2]))
        Y = torch.reshape(Y, (1, Y.shape[0], Y.shape[1], Y.shape[2]))
        XY = torch.cat([X, Y], axis=0)

        weight_hpf = torch.ones(weight_hpf.shape)  # ignore age-based weighting for now
        return DatasetOutput(data=XY, label=[self.samples[index][0], seq_pair_index], index=[index, seq_pair_index],
                             weight_hpf=weight_hpf,
                             self_stats=[e_id_input, age_hpf_input, pert_id_input],
                             other_stats=[e_id_vec[seq_pair_index], age_hpf_vec[seq_pair_index], pert_id_vec[seq_pair_index]])
```

**Augmentation transform** used by the contrastive path (verbatim,
`dataset_utils.py:617-629`):
```python
    @staticmethod
    def get_simclr_pipeline_transform():#(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(brightness=0.3)
        data_transforms = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                              transforms.RandomAffine(degrees=15, scale=tuple([0.7, 1.3])),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomVerticalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.ToTensor()])
        return data_transforms
```

### 3.5 Labels / metadata / developmental-time attachment

The contrastive supervision signal is built from three per-sample fields carried
in `seq_key_dict` (constructed by a data-config `make_metadata()` — present in the
old `src/vae` stack via `make_training_key.py`, **absent** from the `src/core`
shim):
- `pert_id_vec` — perturbation/genotype class id (→ `metric_array` for
  class-relatedness; `+1` related, `-1` cross/negative).
- `e_id_vec` — embryo id (a "self" pair = same embryo, nearby time).
- `age_hpf_vec` — **developmental age in hours-post-fertilization** (the
  developmental-time axis; sourced from `metadata/age_key.csv` upstream).

These flow to the loss as `self_stats = [e_id, age_hpf, pert_id]` and
`other_stats` (the paired sample's triple). See §5 for how they build the
positive/negative target matrix.

### 3.6 Batching & splits

`SubsetRandomSampler(train_indices)` / `eval_indices` (`pl_wrappers.py:319-345`);
`drop_last=True`; `batch_size`/`num_workers` from the data config (256/8 for
metric, 64/8 for plain VAE per YAML). Splits are persisted to
`split_indices.pkl` by the `SaveRunMetadata` callback (`callbacks.py:7-31`).

---

## 4. Model

### 4.1 Construction dispatch

`build_from_config` (`factories.py:12-44`, verbatim) selects encoder/decoder by
`cfg.name` (`"VAE"` or `"metricVAE"`) and `cfg.ddconfig.name`:
```python
def build_from_config(cfg):
    if cfg.name == "VAE":
        if "convVAE" in cfg.ddconfig.name:
            encoder = EncoderConvVAE(cfg.ddconfig)
            decoder = DecoderConvVAEUpsamp(cfg.ddconfig)
        elif "ldmVAE" in cfg.ddconfig.name:
            encoder = WrappedLDMEncoderPool(asdict(cfg.ddconfig))
            decoder = WrappedLDMDecoder(asdict(cfg.ddconfig))
        elif cfg.ddconfig.is_timm_arch:
            encoder = TimmEncoder(cfg.ddconfig)
            decoder = UniDecLite(cfg=cfg.ddconfig, enc_ch_last=encoder.embed_dim)
        else:
            raise NotImplementedError
        model = VAE(cfg, encoder=encoder, decoder=decoder)
    elif cfg.name == "metricVAE":
        if "convVAE" in cfg.ddconfig.name:
            encoder = EncoderConvVAE(cfg.ddconfig)
            decoder = DecoderConvVAEUpsamp(cfg.ddconfig)
        elif "ldmVAE" in cfg.ddconfig.name:
            encoder = WrappedLDMEncoderPool(asdict(cfg.ddconfig))
            decoder = WrappedLDMDecoder(asdict(cfg.ddconfig))
        elif cfg.ddconfig.is_timm_arch:
            encoder = TimmEncoder(cfg.ddconfig)
            decoder = UniDecLite(cfg=cfg.ddconfig, enc_ch_last=encoder.embed_dim)
        else:
            raise NotImplementedError
        model = metricVAE(cfg, encoder=encoder, decoder=decoder)
    else:
        raise NotImplementedError
    return model
```
The live YAMLs use `ddconfig.name="Swin-Tiny"` → the **`is_timm_arch`** branch →
`TimmEncoder` + `UniDecLite`.

### 4.2 Encoder — `TimmEncoder` (VERBATIM forward, `timm_components.py:84-107`)

A timm backbone → mean-pool → **two linear heads** `embedding` (μ) and `log_var`
(logσ²). Both heads have width `latent_dim`.
```python
    def forward(self, x):
        if self._is_patch_family():                         # ViT / DeiT
            # forward to patch tokens
            tokens = self.backbone.forward_features(x)
            if self.model_name[:4] == "swin":
                cls_or_mean = tokens.mean(dim=[1, 2])
            else:
                cls_or_mean = tokens.mean(dim=1)
                # mean-pool tokens
            mu, logvar = self.embedding(cls_or_mean), self.log_var(cls_or_mean)

            return ModelOutput(embedding=mu, log_covariance=logvar)


        else:                                               # Conv / Swin / MaxViT
            feats = self.backbone(x)                        # list of stage maps
            penult = feats[-1]                              # deepest
            vec = self.pool(penult).flatten(1)              # [B,C]
            mu, logvar = self.embedding(vec), self.log_var(vec)
            return ModelOutput(embedding=mu, log_covariance=logvar)
```
Heads are created at `timm_components.py:58-59`:
```python
        self.embedding     = nn.Linear(self.embed_dim, self.latent_dim)
        self.log_var = nn.Linear(self.embed_dim, self.latent_dim)
```
There is an optional (default-off) orthonormal projection buffer
(`_init_orthonormal`, `timm_components.py:72-77`) gated by `cfg.orth_flag`.

### 4.3 Decoder — `UniDecLite`

`z → fc → reshape to (enc_ch_last, H/32, W/32) → 5× UpBlock (pixel-shuffle ↑2)
→ 1×1 conv → sigmoid`. Output channels = input channels (1). Optional
`WindowAttention` block after UpBlock idx 1, gated by `dec_use_local_attn`
(YAML comment flags it as buggy; default False). Constructor at
`timm_components.py:135-176`; forward at `timm_components.py:179-222` (returns
`ModelOutput(reconstruction=sigmoid(...))`).

### 4.4 The latent and the z_b / z_n split — **WHERE & HOW** (VERBATIM)

**There is no architectural split.** The encoder produces a single μ/logσ² of
width `latent_dim`. The "biological vs nuisance" partition is **purely an index
convention applied inside the loss**, not separate heads or separate networks.

The split is defined in `MetricLoss` (`loss_configs.py:183-202`, verbatim):
```python
    @property
    def latent_dim_bio(self) -> int:
        # at least 1, rounding up the nuisance count
        bio = self.latent_dim - math.ceil(self.frac_nuisance_latents * self.latent_dim)
        return max(bio, 1)

    @property
    def metric_cfg(self):
        return dict(n_warmup=self.metric_warmup, n_rampup=self.metric_rampup, w_min=0, w_max=self.metric_weight)

    @property
    def latent_dim_nuisance(self) -> int:
        return self.latent_dim - self.latent_dim_bio

    @property
    def biological_indices(self):
        return torch.arange(self.latent_dim_nuisance, self.latent_dim, dtype=torch.int64)

    @property
    def nuisance_indices(self):
        return torch.arange(0, self.latent_dim_nuisance, dtype=torch.int64)
```
So with `frac_nuisance_latents=0.2` (default, `loss_configs.py:124`) and
`latent_dim=128`: `latent_dim_nuisance=26`, `latent_dim_bio=102`.
**Convention:** dims `[0 : n_nuisance)` = **z_n** (nuisance), dims
`[n_nuisance : latent_dim)` = **z_b** (biological). The NT-Xent metric acts on
`mu[:, biological_indices]` only (§5); both subspaces still receive the standard
KLD (there is a `bio_only_kld` flag, `loss_configs.py:139`, but it is **not**
consulted anywhere in `loss_functions.py` — ⚠️ FLAG, dead flag).

### 4.5 Model forward — `metricVAE.forward` (VERBATIM, `legacy_models.py:73-127`)

Note the paired-view handling: in training, `x` is 5-D `(B, 2, C, H, W)`; both
views are encoded (→ μ,logσ² of shape `(2B, D)`), but the reconstruction is built
from **view 0 only**.
```python
    def forward(self, x: torch.Tensor) -> ModelOutput:
        self.vanilla = False
        if (len(x.shape) != 5):
            self.vanilla = True

        if self.vanilla: # do normal VAE pass if not training
            encoder_output = self.encoder(x)
            mu, logvar = encoder_output.embedding, encoder_output.log_covariance
            z = self.reparametrize(mu, logvar)
            recon_x = self.decoder(z)["reconstruction"]

        elif self.config.lossconfig.target == "NT-Xent":
            # 1) split out the two views
            x0, x1 = x.unbind(dim=1)  # each is (B, C, H, W)
            # 2) stack them into a single 2B batch, *block-wise*
            x_all = torch.cat([x0, x1], dim=0)  # (2B, C, H, W)
            # 3) run everything in one shot
            enc = self.encoder(x_all)
            mu = enc.embedding  # (2B, D)
            logvar = enc.log_covariance  # (2B, D)
            B = x0.shape[0]
            z = self.reparametrize(mu[:B], logvar[:B]) # we only need the actual samples (not positive pairs)
            recon_x = self.decoder(z)["reconstruction"]
        else:
            raise NotImplementedError

        output = ModelOutput(
                        mu=mu,
                        logvar=logvar,
                        recon_x=recon_x,
                        z=z
                    )
        return output

    @staticmethod
    def reparametrize(mu: Tensor, logvar: Tensor) -> Tensor:
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std
```
Plain `VAE.forward` (`legacy_models.py:26-49`) is the un-paired version and
additionally **clamps** logvar to `[-10, 5]` (`legacy_models.py:38`) — note
`metricVAE` does **not** clamp. ⚠️ FLAG: inconsistent logvar clamping between the
two models.

### 4.6 Existing conditioning

**None.** No δ/focal-depth token, no class-conditioning, no time-conditioning is
fed to encoder or decoder. The decoder takes only `z`. (This is the clean hook
point for Extension #2 — see §9.)

---

## 5. Objective

All loss terms live in `src/core/losses/loss_functions.py`. Two top-level loss
modules share a base `_VAELossBase`:
- `VAELossBasic` — pixel + KLD + optional LPIPS + optional GAN.
- `NTXentLoss` — the above **plus** the NT-Xent contrastive metric term on z_b.

### 5.1 The four shared VAE terms (VERBATIM, `loss_functions.py:192-229`)

```python
    def _compute_vae_terms(self, x, recon_x, mu, logvar):
        # Standard Gaussian KL: -0.5 * E[1 + logvar - mu² - exp(logvar)]
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)

        pixel_loss = recon_module(self, x, recon_x)

        pips_loss = 0
        if self.pips_flag and self.pips_weight > 0:
            pips_loss = calc_pips_loss(self, x, recon_x)

        gan_loss = 0
        if self.use_gan and self.gan_weight > 0:
            pred_fake = self.D(recon_x)
            if isinstance(pred_fake, (list, tuple)):   # multi-scale discriminator
                gan_loss = sum([-p.mean() for p in pred_fake]) / len(pred_fake)
            else:
                gan_loss = -pred_fake.mean()

        pixel_loss_w = pixel_loss.mean() * self._pixel_scale()
        kld_loss_w   = KLD.mean()        # kld_scale_factor = 1 (weights are in kld_weight)

        return pixel_loss_w, kld_loss_w, pips_loss, gan_loss
```
- **Reconstruction (pixel):** L1/L2/bce via `recon_module` (`loss_functions.py:34-55`),
  then **rescaled** by `_pixel_scale = (128*288)/100` (L2) or `/10/100` (L1)
  (`loss_functions.py:183-187`) to be O(1) and comparable to the KLD scale.
- **KLD:** standard diagonal-Gaussian KL, mean over latent dims then over batch.
- **Perceptual (LPIPS):** `calc_pips_loss` (`loss_functions.py:20-31`) — frozen
  `lpips.LPIPS(net=cfg.pips_net)` (default `"vgg"`), grayscale repeated to 3ch,
  under `autocast("cuda")`.
- **GAN (generator term):** `-D(recon_x).mean()` (hinge GAN; the **discriminator**
  update is in the training step, §6).

### 5.2 Plain VAE combination (VERBATIM, `loss_functions.py:239-263`)

```python
    def forward(self, model_input, model_output, batch_key="data"):
        x       = model_input[batch_key]
        recon_x = model_output.recon_x
        mu      = model_output.mu
        logvar  = model_output.logvar

        pixel_loss_w, kld_loss_w, pips_loss, gan_loss = self._compute_vae_terms(
            x, recon_x, mu, logvar
        )

        recon_loss = (
            pixel_loss_w
            + self.kld_weight  * kld_loss_w
            + self.pips_weight * pips_loss
            + self.gan_weight  * gan_loss
        )

        return ModelOutput(
            loss=recon_loss, recon_loss=recon_loss, gan_loss=gan_loss,
            pips_loss=pips_loss, pixel_loss=pixel_loss_w, kld_loss=kld_loss_w,
        )
```

### 5.3 NT-Xent loss — combination + metric construction (VERBATIM)

`NTXentLoss.forward` (`loss_functions.py:289-319`):
```python
    def forward(self, model_input, model_output, batch_key="data"):
        x = model_input[batch_key]
        x0, _ = x.unbind(dim=1)   # split paired views; reconstruct from view 0 only

        pixel_loss_w, kld_loss_w, pips_loss, gan_loss = self._compute_vae_terms(
            x0, model_output.recon_x, model_output.mu, model_output.logvar
        )

        metric_loss = self._nt_xent_loss_euclidean(
            features=model_output.mu,
            self_stats=model_input["self_stats"],
            other_stats=model_input["other_stats"],
        )

        total_loss = (
            pixel_loss_w
            + self.kld_weight   * kld_loss_w
            + self.pips_weight  * pips_loss
            + self.gan_weight   * gan_loss
            + self.metric_weight * metric_loss
        )

        return ModelOutput(
            loss=total_loss, recon_loss=total_loss, metric_loss=metric_loss,
            gan_loss=gan_loss, pips_loss=pips_loss, pixel_loss=pixel_loss_w, kld_loss=kld_loss_w,
        )
```

The metric itself (`loss_functions.py:325-379`, verbatim):
```python
    def _nt_xent_loss_euclidean(self, features, self_stats=None, other_stats=None, n_views=2):

        temperature = self.cfg.temperature
        features    = features[:, self.cfg.biological_indices]
        device      = features.device
        batch_size  = int(features.shape[0] / n_views)

        # Positive-pair indicator matrix (1 = positive, -1 = exclude self)
        pair_matrix = torch.cat([torch.arange(batch_size) for _ in range(n_views)], dim=0)
        pair_matrix = (pair_matrix.unsqueeze(0) == pair_matrix.unsqueeze(1)).float()
        mask = torch.eye(pair_matrix.shape[0], dtype=torch.bool, device=device)
        pair_matrix[mask] = -1

        # Normalised squared Euclidean distance
        dist_matrix = torch.cdist(features, features, p=2).pow(2)
        N     = self.cfg.latent_dim_bio / 2
        sigma = N
        dist_normed = (-(dist_matrix / sigma).pow(0.5) + self.cfg.margin) / temperature

        # Build target matrix: 1 = positive, 0 = negative, -1 = exclude
        target_matrix = torch.zeros(pair_matrix.shape, dtype=torch.float32)
        if self_stats is not None:
            age_vec    = torch.cat([self_stats[1], other_stats[1]], axis=0)
            age_deltas = torch.abs(age_vec.unsqueeze(-1) - age_vec.unsqueeze(0))
            age_bool   = age_deltas <= (self.cfg.time_window + 1.5)

            pert_cross = torch.zeros_like(age_bool, dtype=torch.bool)
            pert_bool  = torch.ones_like(age_bool,  dtype=torch.bool)

            if self.cfg.self_target_prob < 1.0:
                pert_vec      = torch.cat([self_stats[2], other_stats[2]], axis=0)
                metric_array  = torch.tensor(self.cfg.metric_array).type(torch.int8)
                metric_matrix = metric_array.clone().to(pert_vec.device)
                metric_matrix = metric_matrix[pert_vec, :][:, pert_vec]
                pert_bool  = metric_matrix == 1
                pert_cross = metric_matrix == -1

            target_matrix[age_bool & pert_bool] = 1
            target_matrix[pert_cross]            = -1

        target_matrix[pair_matrix == 1]  = 1
        target_matrix[pair_matrix == -1] = -1
        target_matrix = target_matrix.to(device)

        return self._nt_xent_loss_multiclass(dist_normed, target_matrix)

    def _nt_xent_loss_multiclass(self, logits_tempered, target):
        logits_tempered[target == -1] = -torch.inf   # exclude flagged pairs

        logits_num = logits_tempered.clone()
        logits_num[target == 0] = -torch.inf          # exclude negatives from numerator

        numerator   = torch.logsumexp(logits_num,      axis=1)
        denominator = torch.logsumexp(logits_tempered, axis=1)
        return torch.mean(-(numerator - denominator))
```

**Key facts for the architect:**
- **Operates on `mu` (not `z`)**, sliced to `biological_indices` (z_b only).
- **Kernel:** `logit_ij = (margin − ‖z_b,i − z_b,j‖₂ / √σ) / temperature`, with
  `σ = latent_dim_bio/2`. `dist_matrix` is squared Euclidean, `.pow(0.5)` makes it
  Euclidean. The loss is a softmax/`logsumexp` cross-entropy: numerator over
  positives, denominator over positives+negatives, excluded pairs set to `−inf`.
- ⚠️ **FLAG vs. project framing:** the framing calls this a *"generalized-Gaussian
  NT-Xent"*. The **live** code (`loss_functions.py:342`) uses a plain
  **Euclidean-distance** kernel with a margin offset (exponent fixed at 0.5),
  **not** a tunable generalized-Gaussian exponent. If a `β`-parameterized
  generalized-Gaussian kernel was intended, it is **not present here** (it may
  exist in the dead `src/vae` stack). Confirm before relying on it.
- **Positive construction:** two images are positives if (a) they are the
  designated augmented/sequential pair (`pair_matrix==1`), OR (b) their
  **developmental age** differs by `≤ time_window + 1.5` hpf **and** their classes
  are "related" in `metric_array` (`==1`). Cross/incompatible classes
  (`metric_array==-1`) are **excluded** (`target=-1`, set to `−inf`). Everything
  else is a negative (`target=0`).
- **`self_target_prob` semantics differ** between the dataset sampler (probability
  of drawing a self-pair vs. a cross-embryo pair, `dataset_utils.py:270`) and the
  loss (`<1.0` simply switches on the `metric_array` class logic,
  `loss_functions.py:354`). ⚠️ FLAG: same name, two roles.

### 5.4 Where the weights are applied / combined

The weighted sum is inside each `forward` (above). The **per-epoch weight values**
are set on the loss object by `LitModel` *before* calling it
(`pl_wrappers.py:120-129`): `kld_weight`, `pips_weight`, `gan_weight`,
`metric_weight` are recomputed every step from the ramp schedules (§6.3).

---

## 6. Training loop

### 6.1 The core training step (VERBATIM, `pl_wrappers.py:108-204`)

Manual optimization, generator-then-discriminator, hinge GAN.
```python
    def training_step(self, batch, batch_idx) -> torch.Tensor:

        # 1) get optimizers
        opts = self.optimizers()
        if isinstance(opts, (list, tuple)):
            opt_G = opts[0]
            opt_D = opts[1] if len(opts) > 1 else None
        else:
            opt_G = opts
            opt_D = None

        # 2) update weights
        kld_w = self._kld_weight()
        self.loss_fn.kld_weight = kld_w
        pips_w = self._pips_weight()
        self.loss_fn.pips_weight = pips_w
        gan_w = self._gan_weight()
        self.loss_fn.gan_weight = gan_w

        if hasattr(self.loss_fn, "metric_weight"):
            metric_w = self._metric_weight()
            self.loss_fn.metric_weight = metric_w

        # ------------------------------------------------
        # a) GENERATOR / VAE update
        # ------------------------------------------------
        x = batch[self.batch_key]
        out = self(x)  # forward VAE

        # compute loss
        loss_output = self.loss_fn(model_input=batch,
                                   model_output=out,
                                   batch_key=self.batch_key)

        # get batch size
        bsz = x.size(0)
        # log
        self._log_metrics(loss_output=loss_output, stage="train", bsz=bsz)

        self.manual_backward(loss_output.loss)

        if self.train_cfg.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.train_cfg.grad_clip_norm
            )

        gn_G = self._grad_norm(module=self.model)
        self.log(f"train/grad_G", gn_G, on_step=True, on_epoch=self.log_epoch, rank_zero_only=True)

        opt_G.step()
        opt_G.zero_grad()

        # ------------------------------------------------
        # b) DISCRIMINATOR update
        # ------------------------------------------------
        if (self.loss_fn.gan_weight > 0) and (self.loss_fn.use_gan):

            with torch.no_grad():
                x_hat = out.recon_x.detach()

            if hasattr(self.loss_fn, "metric_weight"):
                x, _ = x.unbind(dim=1)  # (B, C, H, W)

            pred_real = self.loss_fn.D(x)
            pred_fake = self.loss_fn.D(x_hat)

            # --- GAN loss ---
            if self.loss_fn.gan_net in ["ms_patch", "patch4scale"]:
                loss_D_list = [(F.relu(1 - pred_real[i]).mean() +
                                F.relu(1 + pred_fake[i]).mean()) for i in range(len(pred_real))]
                loss_D = torch.stack(loss_D_list).mean()
            else:
                loss_D = (F.relu(1 - pred_real).mean() +
                          F.relu(1 + pred_fake).mean())

            self.log("train/loss_D", loss_D, prog_bar=True, on_step=True, on_epoch=self.log_epoch)

            # --- Combine and backward ---
            self.manual_backward(loss_D)

            if self.train_cfg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.loss_fn.D.parameters(), self.train_cfg.grad_clip_norm
                )

            gn_D = self._grad_norm(module=self.loss_fn.D)

            opt_D.step()
            opt_D.zero_grad()

            self.log("train/grad_D", gn_D, prog_bar=True, on_step=True, on_epoch=self.log_epoch)
        else:
            self.log("train/loss_D", 0, prog_bar=True, on_step=True, on_epoch=self.log_epoch)
            self.log("train/grad_D",0, prog_bar=True, on_step=True, on_epoch=self.log_epoch)
```
⚠️ Note: `accumulate_grad_batches` is set in `LitTrainConfig` but with manual
optimization Lightning does **not** auto-accumulate; the step always `opt.step()`s
every batch. Likely ineffective — FLAG.

### 6.2 Optimizer & LR groups (VERBATIM, `pl_wrappers.py:292-316`)

Three-group generator optimizer (encoder trunk / decoder / VAE heads) + optional
discriminator optimizer.
```python
    def configure_optimizers(self):
        head_names = {"embedding", "log_var"}  # whatever you called them

        enc_base_params, head_params = [], []
        for name, p in self.model.encoder.named_parameters():
            if any(name.startswith(h) for h in head_names):
                head_params.append(p)  # LR ×2
            else:
                enc_base_params.append(p)

        dec_params = self.model.decoder.parameters()  # UniDec-Lite
        opt_G = torch.optim.Adam(
            [
                {"params": enc_base_params, "lr": self.train_cfg.lr_encoder},  # 0.25×
                {"params": dec_params, "lr": self.train_cfg.lr_decoder},  # 1×
                {"params": head_params, "lr": self.train_cfg.lr_head},  # 2×
            ]
        )

        if self.loss_fn.use_gan:  # discriminator present?
            opt_D = torch.optim.Adam(self.loss_fn.D.parameters(), lr=self.train_cfg.lr_gan, betas=(0.5, 0.999))
            return [opt_G, opt_D]
        else:
            return [opt_G]
```
⚠️ FLAG: `head_names` matches encoder params named `embedding*`/`log_var*`; the
named loop uses `name.startswith(...)` against `named_parameters()` names like
`embedding.weight` — works, but brittle if heads are renamed. No LR scheduler
(constant LR; only the *loss-weight* ramps vary over epochs).

### 6.3 Loss-weight schedules

Per-epoch weights come from `cosine_ramp_weight(step_curr=current_epoch, **cfg)`
(`pl_utils.py:28-47`), gated by `schedule_<term>` flags. The warmup/rampup
windows are **derived properties** in the loss config — terms ramp in sequence:
(metric →) KLD → PIPS → GAN. See `BasicLoss`/`MetricLoss` properties
(`loss_configs.py:49-86, 142-190`); `ramp_scale`/`hold_scale` default 5.0 each,
scaled by `train_scale = min(max_epochs/30, 1)`.

### 6.4 ALL hyperparameters with defaults

**`LitTrainConfig`** (`train_config.py`): `benchmark=True`,
`accumulate_grad_batches=2`, `max_epochs=100`, `lr_base=1e-4`, `save_every_n=50`,
`eval_gpu_flag=True`, `save_epochs=[]`, `encoder_lr_scale=0.1`,
`grad_clip_norm=0.0`, `gan_net="patch"` (overwritten by lossconfig). Derived:
`lr_encoder=lr_base*0.1`, `lr_decoder=lr_base`, `lr_head=lr_base*2`,
`lr_gan=lr_base/4` if `"style"` in gan_net else `lr_base/2`.

**`BasicLoss`** (`loss_configs.py`): `kld_weight=1.0`, `schedule_kld=True`,
`kld_warmup=0`, `reconstruction_loss="L2"`, `pips_net="vgg"`, `pips_flag=True`,
`pips_weight=7.5`, `schedule_pips=True`, `use_gan=False`, `gan_weight=1.0`,
`gan_net="patch4scale"`, `schedule_gan=True`, `use_pips_eval=True`,
`eval_pips_net="alex"`, `ramp_scale=5.0`, `hold_scale=5.0`, `tv_weight=0`,
`max_epochs=25` (overwritten by trainconfig).

**`MetricLoss`** (extends `BasicLoss`, `loss_configs.py:110-`): `target="NT-Xent"`,
`schedule_metric=True`, `metric_warmup=0`, `frac_nuisance_latents=0.2`,
`latent_dim=None` (set from ddconfig), `metric_array=[]` (set from dataconfig),
`temperature=0.1`, `metric_weight=1.0`, `margin=1.0`,
`distance_metric="euclidean"`, `time_window=1.5`, `self_target_prob=0.5`,
`bio_only_kld=False` (⚠️ unused).

**`TimmArchitecture`** (`arch_configs.py`): `name="Efficient-B0-RA"`,
`is_timm_arch=True`, `latent_dim=64`, `orth_flag=False`,
`use_pretrained_weights=True`, `dec_use_local_attn=False`,
`input_dim=(1,288,128)`. **`LegacyArchitecture`**: `name="convVAE"`,
`latent_dim=64`, `n_out_channels=16`, `n_conv_layers=5`, `kernel_size=4`,
`stride=2`, `input_dim=(1,288,128)`.

**YAML overrides actually used** (`metric_vae_timm.yaml`):
`name="Swin-Tiny"`, `latent_dim=128`, `batch_size=256`, `num_workers=8`,
`max_epochs=50`, `save_epochs=[9,19,29]`, `use_gan=True`, `gan_net="patch4scale"`.
(`vae_timm.yaml`: `latent_dim=64`, `batch_size=64`, `gan_net="style2"`.)
⚠️ FLAG: `dataconfig.target="BasicDataset"`, `wrap`, and `lambda_feat_match=10.0`
appear in YAML but have **no consumer** in `src/core` (the shim ignores `target`;
`lambda_feat_match` is commented out in `BasicLoss`). Dead config keys.

---

## 7. Evaluation

### 7.1 In-training validation metrics (VERBATIM)

`validation_step` (`pl_wrappers.py:60-105`) logs LPIPS, MS-SSIM, and **per-subspace
latent anisotropy / condition number** (a key diagnostic for the z_b/z_n split):
```python
            if hasattr(self.loss_fn, "metric_weight"):
                cov_loss_fn_b = LatentCovarianceLoss(latent_indices=self.loss_fn.biological_indices)
                cov_loss_b, cond_b = cov_loss_fn_b(model_output=out)
                self.log("val/z_b_anisotropy", cov_loss_b, sync_dist=False, prog_bar=True)
                self.log("val/z_b_cond", cond_b, sync_dist=False, prog_bar=True)
                cov_loss_fn_n = LatentCovarianceLoss(latent_indices=self.loss_fn.nuisance_indices)
                cov_loss_n, cond_n = cov_loss_fn_n(model_output=out)
                self.log("val/z_n_anisotropy", cov_loss_n, sync_dist=False, prog_bar=True)
                self.log("val/z_n_cond", cond_n, sync_dist=False, prog_bar=True)
            else:
                cov_loss_fn = LatentCovarianceLoss(latent_indices=None)
                cov_loss, cond = cov_loss_fn(model_output=out)
```

The metric implementations (`loss_helpers.py`):
- `lpips_score` (`loss_helpers.py:22-39`) — cached frozen LPIPS(alex), CPU/GPU.
- `ssim_score` (`loss_helpers.py:42-66`) — `piq.MultiScaleSSIMLoss`.
- `LatentCovarianceLoss` (`loss_helpers.py:70-131`, verbatim core):
```python
    def forward(self, model_output) -> Tensor:
        # extract latent vector
        z = model_output.mu
        if self.latent_indices is not None:
            z = z[:, self.latent_indices]

        # get covariance
        B, D = z.shape
        zc = z - z.mean(dim=0, keepdim=True)                   # centre
        cov = (zc.t() @ zc) / (B - 1 + self.eps)               # unbiased Σ̂
        eye = torch.eye(D, device=z.device, dtype=z.dtype)

        eigvals = torch.linalg.eigvalsh(cov.float())
        eps = 1e-8
        eigvals_clipped = eigvals.clamp(min=eps)
        cond = eigvals_clipped.max() / eigvals_clipped.min()
        if self.mode == "fro":
            diff = cov - eye
            loss = (diff ** 2).sum()
            if self.normalize:
                loss = loss / (D * D)
        else:  # 'off'
            off_diag_mask = torch.ones_like(cov) - eye
            loss = ((cov * off_diag_mask) ** 2).sum()
            if self.normalize:
                loss = loss / (D * (D - 1))
            loss = loss.sqrt()
        return loss, cond
```
Computed on **`mu`**, optionally sliced to a subspace; returns
`(off-diagonal-covariance-norm, condition-number)`.

### 7.2 Inference / embedding extraction

- `LitModel.predict_step` (`pl_wrappers.py:350-376`) returns `snip_ids, recon,
  orig, recon_loss(mse), mu, log_var` per batch.
- `load_encoder` (`arch_spec.py:261-349`) → frozen model → `out.mu` directly.
- Downstream embedding generation: `src/run_morphseq_pipeline/services/gen_embeddings.py`
  and `src/analyze/analysis_utils.py` (both call the load helpers).

⚠️ FLAG: `src/analyze/assess_hydra_results.py` (UMAP + result assessment) has
**broken module imports** (`from run.run_utils …`, `from data.dataset_configs …`)
and references `data_config.make_metadata()` / `metric_array`; it appears stale and
is not a reliable evaluation entry point. There is **no automated
quantitative-eval harness** for the latent space in `src/core` beyond the
in-training logs above.

---

## 8. Refactor targets

1. **Data-config schism (highest priority, blocks Extensions #1 & #3).** The
   `src/core` training stack imports its data config from `src/data/dataset_configs.py`,
   which is a **single-image analysis shim** lacking `metric_array`, `seq_key_dict`,
   paired-view `create_dataset`, real `make_metadata`, and `train_bool`/`eval_bool`.
   But `initialize_model` (`run_utils.py:339-340`) does
   `model_config.lossconfig.metric_array = data_config.metric_array`, and
   `NTXentLoss`/`metricVAE.forward` require paired 5-D tensors + `self_stats`. With
   the shim, **MetricVAE training cannot run as imported** (AttributeError on
   `metric_array`; wrong tensor rank). The real sampler `SeqPairDatasetCached`
   lives in `src/core/functions/dataset_utils.py` but is **orphaned** (no config
   builds `seq_key_dict`/splits/`metric_array` in `src/core`; that logic only
   exists in the dead `src/vae` stack via `make_training_key.py`). **Action:** port
   a real `NTXentDataConfig` into `src/core` that (a) builds metadata
   (`age_hpf_vec`, `pert_id_vec`, `e_id_vec`, `metric_array`, splits), and (b)
   returns `SeqPairDatasetCached`. Reconcile `import` of
   `from data.dataset_configs …` (model_configs.py:8) vs.
   `from src.core.data.dataset_configs …` (run_utils.py:429) vs. actual file at
   `src/data/dataset_configs.py` — three inconsistent module paths for one concept.

2. **Quadruple-duplicated VAE stacks.** `src/core` (live), `src/vae`, `src/legacy`,
   `src/_Archive` each contain a full `models/`, `losses/`, `trainers/`. Plus
   `src/core/functions/` duplicates `src/functions/`. Delete or quarantine the dead
   trees to prevent edits landing in the wrong copy. The "legacy" name on the live
   `src/core/models/legacy_models.py` is actively misleading.

3. **`build_from_config` duplication.** The `VAE` and `metricVAE` branches
   (`factories.py:12-44`) are byte-identical except the final model class. Collapse
   to one encoder/decoder builder + a model-class lookup.

4. **`_VAELossBase` subclassing vs. composition.** `VAELossBasic` and `NTXentLoss`
   both inherit `_VAELossBase` and re-implement the weighted sum. Adding a new loss
   term (Extension #1) currently means editing both `forward`s and the
   `LitModel._log_metrics`/`training_step` weight-setting block
   (`pl_wrappers.py:120-129, 206-234`). The weight-set/ramp plumbing
   (`_kld_weight`/`_pips_weight`/`_metric_weight`/`_gan_weight`) is copy-paste; a
   registry of `{name: (schedule_flag, cfg)}` would let new terms be added in one
   place.

5. **Logvar clamping inconsistency.** `VAE` clamps logvar to `[-10,5]`
   (`legacy_models.py:38`); `metricVAE` does not. Unify.

6. **Dead/ignored config knobs.** `bio_only_kld` (never read), `tv_weight`
   (computed by `calc_tv_loss` but never added to any total), `lambda_feat_match`
   (commented out), YAML `dataconfig.target`/`wrap`, `accumulate_grad_batches`
   (ineffective under manual optimization), `save_every_n`. Each is a latent
   correctness trap.

7. **Hard-coded image geometry.** `_pixel_scale` hard-codes `128*288`
   (`loss_functions.py:185`); `UniDecLite` assumes stride-32 / `H/32` latent grid
   and exactly 5 upsampling stages (`timm_components.py:141,153-159`). Changing
   input resolution silently miscalibrates the loss and may break the decoder.

8. **No LR scheduler / single global step granularity.** Schedules key off
   `current_epoch` only; fine-grained warmup is impossible.

---

## 9. Extension surface (locations & impact only — do NOT implement)

### Extension #1 — Predictive / forward-contrastive loss
- **Loss term:** add a new method on `NTXentLoss` (or a new sibling of
  `_VAELossBase`) in `src/core/losses/loss_functions.py`, summed into `total_loss`
  in `NTXentLoss.forward` (`loss_functions.py:303-309`). Add its weight +
  `schedule_*` + `*_cfg` ramp to `MetricLoss` (`loss_configs.py:110-218`) mirroring
  `metric_weight`/`metric_cfg`.
- **Training plumbing:** set the new per-epoch weight in
  `LitModel.training_step` (`pl_wrappers.py:120-129`), add a `_<name>_weight()`
  helper (mirror `_metric_weight`, `pl_wrappers.py:266-274`), and a log line in
  `_log_metrics` (`pl_wrappers.py:229-234`).
- **Data:** a "forward" prediction needs **temporally-ordered pairs** (sample at
  time t and t+Δ). The pairing logic already exists in `SeqPairDatasetCached`
  (`age_delta_array`, `time_window`, `weight_hpf`, `dataset_utils.py:263-277`) but
  see §8.1 — the sampler must first be wired into a real `src/core` data config.
  The two encoded views (`mu[:B]`, `mu[B:]` in `metricVAE.forward`,
  `legacy_models.py:104-108`) are the natural anchor/target representations; a
  predictor head mapping `z_b(t) + Δt → z_b(t+Δ)` would attach to the model output.
- **Touches:** `loss_functions.py`, `loss_configs.py`, `pl_wrappers.py`,
  `dataset_utils.py` (+ the new data config), optionally `legacy_models.py` (if a
  predictor sub-module is added) and `arch_spec.py` (if the predictor must be
  reloadable at inference).

### Extension #2 — Slice-aware encoding with a focal-depth (δ) conditioning token
- **Replaces full-focus input.** Today the projection happens at **build time**
  (`LoG_focus_stacker`, `src/build/export_utils.py`) and the model only sees one 2D
  snip (§3.1). To go slice-aware you must: (a) stop collapsing z in the build (or
  export per-slice snips + their δ), (b) change the dataset to emit a stack +
  δ-vector, and (c) feed δ as a token.
- **Encoder hook:** `TimmEncoder.forward` (`timm_components.py:84-107`) currently
  mean-pools tokens then applies two linear heads. A δ token would be concatenated
  to the token sequence (ViT/Swin patch families) **before** pooling, or
  FiLM-modulate `cls_or_mean`/`vec` before the `embedding`/`log_var` heads
  (`timm_components.py:58-59,93,102`). The backbone is built with
  `in_chans=input_dim[0]` (`timm_components.py:38-39,50`); a multi-slice input
  changes channel/seq assumptions.
- **Decoder:** `UniDecLite` (`timm_components.py:129-222`) is δ-agnostic; if
  reconstruction should be slice-conditioned, δ must be injected into `self.fc`
  input or the up-blocks.
- **Config:** add δ fields to `TimmArchitecture` (`arch_configs.py:35-51`) and
  `ArchiveSpec` (`arch_spec.py:42-90`, plus `_build_model_from_spec`
  `arch_spec.py:207-254`) so slice-aware models are reloadable.
- **Touches:** `src/build/export_utils.py` + `build01A/01B` (export per-slice + δ),
  the new data config / `dataset_utils.py`, `timm_components.py`, `arch_configs.py`,
  `arch_spec.py`, and `legacy_models.py` (forward signature now carries δ).
- ⚠️ This is the **most invasive** extension: it crosses the build/model boundary
  that everything else respects.

### Extension #3 — Calibration: unit latent distance ≡ fixed developmental time
- **Where time lives:** `age_hpf` enters via `self_stats[1]`/`other_stats[1]`
  (`dataset_utils.py:308-309`) and is consumed by the NT-Xent `age_bool`
  (`loss_functions.py:347-349`). A calibration loss would compare
  `‖z_b,i − z_b,j‖₂` (already computed as `dist_matrix`,
  `loss_functions.py:339`) against `|age_hpf_i − age_hpf_j|` on **WT
  trajectories** (filter by `pert_id`/`metric_array`).
- **Loss term:** new method on `NTXentLoss` reusing `dist_matrix` + `age_vec`
  (`loss_functions.py:339,347`), restricted to WT pairs; summed into `total_loss`
  with its own weight/ramp (same plumbing as Extension #1 §9).
- **Identifying WT:** needs the WT class id from `metric_array`/`pert_id_vec`
  metadata — again depends on the §8.1 data-config port (the WT label is only
  meaningful once `seq_key_dict` is built in `src/core`).
- **Touches:** `loss_functions.py`, `loss_configs.py`, `pl_wrappers.py`, and the
  data config (to expose a WT mask + calibrated time units). Optionally add a
  `val/` calibration metric in `validation_step` (`pl_wrappers.py:60-105`)
  alongside the existing anisotropy logs.

---

## 10. Open questions (unresolved from code alone)

1. **Is MetricVAE training currently runnable in `src/core`?** As imported, the
   data config is the shim (§3.4/§8.1) and `initialize_model` would fail on
   `data_config.metric_array`. Either (a) there is an un-committed/alternate
   `dataset_configs.py` on the training machine, (b) `src/` sys.path ordering
   resolves `from data.dataset_configs` to a different file than I found, or (c)
   the `src/core` metric path is mid-refactor and last trained from `src/vae`.
   I could not determine which. **This is the single most important thing to
   confirm.**
2. **"Generalized-Gaussian" kernel.** The framing says generalized-Gaussian
   NT-Xent; the live kernel is Euclidean-with-margin (exponent 0.5 fixed,
   `loss_functions.py:342`). Is a tunable exponent expected (and only present in
   the dead `src/vae` stack), or has the design moved to the simpler kernel?
3. **`metric_array` provenance & encoding.** Values `{1, 0, -1}` (related /
   negative / cross-excluded) indexed by `pert_id`. Where is it constructed in the
   `src/core` world? (Only found being *consumed*; the producer is in the dead
   `src/vae` `make_training_key.py`.)
4. **`weight_hpf` / `other_age_penalty`.** The sampler computes age-based weights
   then overwrites them with ones (`dataset_utils.py:300`) and the loss ignores
   `weight_hpf` entirely. Is age-weighting intended to be revived (relevant to
   Extension #3)?
5. **`orth_flag` orthonormal projection.** `TimmEncoder` can register an
   orthonormal basis buffer (`timm_components.py:66-77`) but `forward` never applies
   `self.project`. Dead, or intended to be applied to μ?
6. **`dec_use_local_attn`.** YAML comments call it buggy and default it off
   (`metric_vae_timm.yaml`). Is the decoder window-attention path expected to work?
7. **Exact full-focus projection.** I located `LoG_focus_stacker`
   (`src/build/export_utils.py`) but did not read its internals (out of `src/`
   model scope). Its input/output contract must be confirmed before Extension #2.
8. **`predict_step` reads `out.logvar`** (`pl_wrappers.py:357`) while `mu`/`logvar`
   are the `ModelOutput` keys — consistent, but the plain-VAE path clamps logvar and
   metric path does not, so embeddings' uncertainty estimates differ by model type
   (§4.5). Intended?
```
