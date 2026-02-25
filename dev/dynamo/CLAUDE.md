# CLAUDE.md

## Project Overview

This project studies morphogenetic dynamics in embryos via live imaging. Embryo images are embedded into a learned latent space (pretrained VAE); this submodule builds a stochastic dynamical model that predicts forward trajectories in that latent space, conditioned on partial observations.

The full model specification is in `docs/model_spec.md`. **Read it before writing any code.**

## Repository Structure

<!-- TODO: fill in actual top-level structure -->
```
project_root/
├── CLAUDE.md
├── docs/
│   └── model_spec.md          # Full technical specification (the source of truth)
├── dynamo/                     # <-- this submodule (name TBD)
│   ├── data/                   # Data loading, fragment sampling
│   ├── models/                 # Model definitions (potentials, baselines)
│   ├── inference/              # Closed-form solvers for c, v, R
│   ├── training/               # Training loops (staged)
│   ├── eval/                   # Metrics, W&B logging
│   ├── viz/                    # Visualization panels (trajectory view, prediction fan, phenotype space, mode deflection)
│   └── tests/                  # Unit and integration tests
├── scripts/                    # Entry points: train, evaluate, visualize
└── configs/                    # Hyperparameter configs (YAML or similar)
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
3. Simple kernel regression baseline (nearest-point lookup — trivial, implement first)
4. Branching particle filter baseline (non-trivial — see spec §9.2 carefully; implement in layers: selection first, then single-shot prediction, then recruitment)
5. phi0-only model (Stage 1 training)
6. Orthogonal modes (S_m applied to nabla phi0, no phi_m networks)
7. Full Helmholtz modes with S_m = 0 (independent phi_m networks, potential-only)
8. Unfreeze S_m on full model
9. Add regularization terms incrementally

## Data Format

<!-- TODO: fill in specifics about your data -->
Each embryo provides:
- `trajectory`: tensor of shape `(T, d)` — latent embeddings over time
- `dt`: scalar — time step for this experiment (uniform within experiment)
- `temperature`: scalar — experimental temperature
- `perturbation_class`: string or int — class label (may be absent for novel perturbations)
- `embryo_id`: unique identifier

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

<!-- TODO: fill in your actual commands -->
```bash
# Run tests
pytest dynamo/tests/ -v

# Train stage 1 (phi0 only)
python scripts/train.py --config configs/stage1.yaml

# Train stage 2 (modes)
python scripts/train.py --config configs/stage2.yaml --checkpoint <stage1_ckpt>

# Evaluate all five models
python scripts/evaluate.py --models simple_kernel,particle_filter,phi0,orthogonal,full --test-tier all

# Generate visualization panels
python scripts/visualize.py --checkpoint <ckpt> --output figures/
```

## Things to Watch For

- **Particle filter baseline complexity**: the branching particle filter (§9.2) is NOT a simple nearest-neighbor lookup. It involves direction-aware matching via linear fits, per-reference speed ratios matched on developmental progress, local recruitment with weight inheritance, and particle caps. Read spec §9.2 in full before implementing. Key: recruitment is local per-reference, never against a global centroid. This preserves multimodal predictions at bifurcations. Implement in layers: selection → single-shot prediction → recruitment.
- **Identifiability**: beta, D, and R_e interact. D is global. R_e is per-embryo. Mean of lambda_e is constrained to 1. See spec §3.5 and §6.6.
- **Gradient flow**: backprop through `torch.linalg.solve` can produce NaNs if the system is ill-conditioned. Add a small diagonal jitter (1e-6) to H^T H + Lambda before solving.
- **Mode initialization**: all phi_m networks must initialize near zero so training starts on the baseline landscape. Use small weight init (e.g., scale default init by 0.01).
- **S_m initialization**: always zero. Modes start as pure potentials.
