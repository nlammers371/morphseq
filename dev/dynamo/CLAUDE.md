# CLAUDE.md

## Project Overview

This project studies morphogenetic dynamics in embryos via live imaging. Embryo images are embedded into a learned latent space (pretrained VAE); this submodule builds a stochastic dynamical model that predicts forward trajectories in that latent space, conditioned on partial observations.

The full model specification is in `docs/model_spec.md`. **Read it before writing any code.**

## Repository Structure

```
dev/
├── dynamo/                          # This submodule
│   ├── CLAUDE.md
│   ├── docs/
│   │   └── model_spec.md           # Full model specs — read before coding
│   ├── data/                        # Data loading, fragment sampling
│   │   ├── loading.py              #   load_trajectories() → TrajectoryDataset
│   │   └── dataset.py              #   FragmentDataset (PyTorch), collation
│   ├── models/                      # Model definitions (potentials, baselines)
│   ├── inference/                   # Closed-form solvers for c, v, R
│   ├── training/                    # Training loops (staged)
│   ├── eval/                        # Metrics, W&B logging
│   ├── viz/                         # Three-panel visualization
│   └── tests/                       # Unit and integration tests
│       └── test_data.py            #   Tests for data loading + fragment sampling
├── scripts/                         # Entry points: train, evaluate, visualize
└── configs/                         # Hyperparameter configs (YAML or similar)
```

## Key Constraints

- **Framework**: plain PyTorch. No Lightning. No unnecessary abstractions.
- **Python**: 3.10+
- **Logging**: Weights & Biases (wandb)
- **Differentiation**: all per-embryo inference (c, v, R) must be batched and differentiable. Use `torch.linalg.solve` for ridge systems. Gradients must flow through the linear solve into network parameters.
- **Smooth activations only**: use softplus or ELU in all potential networks. Never ReLU.

## Build Sequence

Implement in this order. Each step must be runnable and testable before moving to the next. See `docs/model_spec.md` §15.2 for rationale.

1. Data loading and fragment sampling
2. Evaluation and visualization stack (against dummy predictions)
3. Kernel baseline
4. phi0-only model (Model v1)
5. Implement closed-form solvers for c, v, R (inference module)
6. Add orthogonal baseline flux modes (Model v2; see 3.4 in doc/model_spec.md)
7. Add full mode learning machinery with options to (i) learn only potential    modes (Model v3), (ii) only flux modes (Model v4), or (iii) both (Model v5)

## Data Format

**Source**: CSV files at `morphseq_playground/metadata/build06_output/df03_final_output_with_latents_{experiment_id}.csv` (one per experiment). See `results/nlammers/20260209/pbx_notebook.ipynb` for usage examples.

**Key columns**: `embryo_id`, `frame_index`, `relative_time_s`, `genotype`, `temperature`, `experiment_id`, `use_embryo_flag`, 80 × `z_mu_b_*` (VAE latents, indices 20–99).

**Pipeline** (implemented in `data/loading.py`):
1. Load CSVs → filter `use_embryo_flag == True` → drop NaN embeddings
2. Fit PCA on all `z_mu_b_*` columns → project to PC space (default 10 components)
3. Group by `embryo_id`, sort by `frame_index` → `EmbryoTrajectory` per embryo
4. Each trajectory carries: PC vectors (T×D), time array, experiment-level median Δt, temperature, genotype

**Fragment sampling** (implemented in `data/dataset.py`, per spec §7.2):
- Random embryo → random contiguous fragment → random horizon k ∈ {1,2,3,4}
- Yields batches (B×L×D) with padding mask, time deltas, and metadata
- Heteroscedasticity correction (§4.1) uses real inter-frame Δt values, not assumed-uniform spacing

## Testing Expectations

- Unit tests for the closed-form solvers: verify c* minimizes the quadratic, verify R* is the correct projection, verify gradients flow through `torch.linalg.solve`.
- Integration test: overfit a small synthetic dataset (known potential, known modes) and verify recovery of ground truth parameters.
- Never commit code that breaks existing tests.

## Style

- Type hints on all function signatures.
- Docstrings on all public functions (one-line summary + args/returns).
- Keep modules focused: one file should not do data loading AND model definition.
- Prefer explicit over clever. This model has enough inherent complexity; the code should be as boring as possible.

## Common Commands

```bash
PYTHON=/net/trapnell/vol1/home/nlammers/micromamba/envs/vae-env-cluster/bin/python

# Run data loading / fragment sampling tests
"$PYTHON" -m pytest dev/dynamo/tests/test_data.py -v

# Run all dynamo tests
"$PYTHON" -m pytest dev/dynamo/tests/ -v

# Train stage 1 (phi0 only) — not yet implemented
# "$PYTHON" dev/scripts/train.py --config dev/configs/stage1.yaml

# Train stage 2 (modes) — not yet implemented
# "$PYTHON" dev/scripts/train.py --config dev/configs/stage2.yaml --checkpoint <stage1_ckpt>

# Evaluate all three models — not yet implemented
# "$PYTHON" dev/scripts/evaluate.py --models kernel,phi0,full --test-tier all

# Generate visualization panels — not yet implemented
# "$PYTHON" dev/scripts/visualize.py --checkpoint <ckpt> --output figures/
```

## Things to Watch For

- **Identifiability**: beta, D, and R_e interact. D is global. R_e is per-embryo. Mean of lambda_e is constrained to 1. See spec §3.4 and §6.6.
- **Gradient flow**: backprop through `torch.linalg.solve` can produce NaNs if the system is ill-conditioned. Add a small diagonal jitter (1e-6) to H^T H + Lambda before solving.
- **Mode initialization**: all phi_m networks must initialize near zero so training starts on the baseline landscape. Use small weight init (e.g., scale default init by 0.01).
- **S_m initialization**: always zero. Modes start as pure potentials.
