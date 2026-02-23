# Task 01 — `compute_minibatch_uot` (MVP, `bias_mode='none'`)

**File:** `src/flux/transport/minibatch_uot.py`

## Interface (proposed, stable)
```python
def compute_minibatch_uot(
    x_t,            # (B, d) features at time t
    x_tp1,          # (B, d) features at time t+Δ
    t, tp1,         # scalar or (B,) times
    cost_fn,        # callable(x_t, x_tp1) -> (B, B) pairwise costs
    eps: float,     # entropic regularization (Sinkhorn)
    reg: float,     # unbalanced KL/TV strength
    max_iter: int = 200,
    tol: float = 1e-6,
    bias_mode: str = "none",
    bias_kwargs: dict | None = None,
    rng=None,
) -> tuple:
    """Return transport plan P (B×B) and teacher velocities v_teacher (B, d)."""
```

## MVP Logic
1. Compute pairwise cost matrix `C = cost_fn(x_t, x_tp1)`.
2. Run **unbalanced entropic OT** (e.g., Sinkhorn‑Knopp with KL mass penalty) to obtain `P`.
3. **No biasing**: ignore `bias_mode` and `bias_kwargs` in MVP.
4. Compute teacher velocities  
   `v_teacher[i] = Σ_j P[i,j] * (x_tp1[j] - x_t[i]) / max(Σ_j P[i,j], 1e-8)`.
5. Return `(P, v_teacher)`.

## Numerical Hygiene
- Work in log‑space where possible.
- Clip exponents to avoid overflow.
- Assert finite values and expected shapes.
- Add tiny epsilon to row/col normalizations.

## Unit & Smoke Tests
- **Shapes:** B=16, d=8 random Gaussians; assert finite `P`, row sums > 0.
- **Determinism:** fix RNG; rerun; hash equality.
- **Degenerate cases:** identical `x_t == x_tp1` → `v_teacher` near zero.
- **Time monotonicity (sanity):** when `x_tp1 = x_t + Δv`, `v_teacher` ≈ Δv.

## Reference / Current Code to Preserve
```python
# File: src/flux/transport/minibatch_uot.py

import numpy as np
import ot
from typing import Optional, Tuple, Dict, Literal

def compute_minibatch_uot(
    X: np.ndarray,  # (m, d) source embeddings
    Y: np.ndarray,  # (n, d) target embeddings  
    dt: float = 1.0,  # time gap
    a: Optional[np.ndarray] = None,  # (m,) source masses
    b: Optional[np.ndarray] = None,  # (n,) target masses
    eps: float = 0.05,  # entropic regularization
    reg_m: float = 1.0,  # mass penalty (tau)
    metric: str = "euclidean",
    **kwargs
) -> Dict:
    """
    Compute unbalanced optimal transport between minibatches.

    Returns:
        dict with keys:
        - 'plan': (m, n) transport plan
        - 'velocities': (m, d) teacher velocities
        - 'row_masses': (m,) transported mass per source
        - 'cost': scalar, total transport cost
    """
    m, n = len(X), len(Y)

    # Default uniform masses
    if a is None:
        a = np.ones(m) / m
    if b is None:
        b = np.ones(n) / n

    # Compute cost matrix
    if metric == "euclidean":
        C = ot.dist(X, Y, metric='euclidean') ** 2
    else:
        C = ot.dist(X, Y, metric=metric)

    # Solve UOT
    Pi = ot.unbalanced.sinkhorn_unbalanced(a, b, C, eps, reg_m)

    # Extract velocities
    row_masses = Pi.sum(axis=1)
    velocities = np.zeros_like(X)

    for i in range(m):
        if row_masses[i] > 1e-10:  # threshold for numerical stability
            weighted_displacement = Pi[i, :] @ (Y - X[i])
            velocities[i] = weighted_displacement / (dt * row_masses[i])

    return {
        'plan': Pi,
        'velocities': velocities,
        'row_masses': row_masses,
        'cost': np.sum(Pi * C)
    }
```

```python
def compute_biased_uot(
    X: np.ndarray,
    Y: np.ndarray,
    dt: float = 1.0,
    self_indices: Optional[np.ndarray] = None,  # (m,) with -1 for no track
    bias_mode: Literal["cost_shaping", "prior", "lower_bound", "none"] = "none",
    bias_strength: float = 5.0,  # γ for cost_shaping, λ for prior
    lower_bounds: Optional[np.ndarray] = None,  # for lower_bound mode
    eps: float = 0.05,
    reg_m: float = 1.0,
    **kwargs
) -> Dict:
    """
    UOT with trajectory biasing for known next-frame matches.

    Args:
        self_indices: Array where self_indices[i] = j means x_i tracks to y_j
                     Use -1 for no known track
        bias_mode: Method to incorporate tracking
            - "cost_shaping": Reduce cost on self edges
            - "prior": KL penalty toward reference plan
            - "lower_bound": Reserve minimum mass on self edges
            - "none": Standard UOT
    """
    m, n = len(X), len(Y)
    a = np.ones(m) / m
    b = np.ones(n) / n

    # Compute base cost
    C = ot.dist(X, Y) ** 2

    if bias_mode == "none" or self_indices is None:
        return compute_minibatch_uot(X, Y, dt, a, b, eps, reg_m)

    elif bias_mode == "cost_shaping":
        # Method 1: Make self-edges cheaper
        C_modified = C.copy()
        for i, j in enumerate(self_indices):
            if j >= 0 and j < n:
                C_modified[i, j] -= bias_strength
                C_modified[i, j] = max(0, C_modified[i, j])  # keep non-negative

        Pi = ot.unbalanced.sinkhorn_unbalanced(a, b, C_modified, eps, reg_m)

    elif bias_mode == "prior":
        # Method 2: Prior plan regularization
        # Build reference plan concentrated on self edges
        Pi0 = np.full((m, n), 1e-12)  # small floor
        for i, j in enumerate(self_indices):
            if j >= 0 and j < n:
                Pi0[i, j] = a[i]  # concentrate mass

        # Equivalent to reweighted kernel
        K = np.exp(-C / eps)
        ab = np.outer(a, b)
        R = (Pi0 / np.maximum(ab, 1e-30)) ** (bias_strength / eps)
        K_modified = K * R
        C_effective = -eps * np.log(np.maximum(K_modified, 1e-300))

        Pi = ot.unbalanced.sinkhorn_unbalanced(a, b, C_effective, eps, reg_m)

    elif bias_mode == "lower_bound":
        # Method 3: Reserve mass on self edges
        if lower_bounds is None:
            # Default: reserve 50% of mass on tracked edges
            lower_bounds = np.zeros(m)
            for i, j in enumerate(self_indices):
                if j >= 0 and j < n:
                    lower_bounds[i] = 0.5 * a[i]

        # Reserve mass
        Pi_fixed = np.zeros((m, n))
        a_residual = a.copy()
        b_residual = b.copy()

        for i, j in enumerate(self_indices):
            if j >= 0 and j < n and lower_bounds[i] > 0:
                mass_to_fix = min(lower_bounds[i], a_residual[i], b_residual[j])
                Pi_fixed[i, j] = mass_to_fix
                a_residual[i] -= mass_to_fix
                b_residual[j] -= mass_to_fix

        # Solve for residual
        Pi_residual = ot.unbalanced.sinkhorn_unbalanced(
            a_residual, b_residual, C, eps, reg_m
        )

        Pi = Pi_fixed + Pi_residual

    # Extract velocities (same as before)
    row_masses = Pi.sum(axis=1)
    velocities = np.zeros_like(X)

    for i in range(m):
        if row_masses[i] > 1e-10:
            weighted_displacement = Pi[i, :] @ (Y - X[i])
            velocities[i] = weighted_displacement / (dt * row_masses[i])

    return {
        'plan': Pi,
        'velocities': velocities, 
        'row_masses': row_masses,
        'cost': np.sum(Pi * C),
        'bias_mode': bias_mode
    }
```

```python
def sample_minibatch_pair(
    embeddings: np.ndarray,  # (N, d) all embeddings
    times: np.ndarray,  # (N,) time values
    cluster_labels: np.ndarray,  # (N,) from spectral clustering
    embryo_ids: np.ndarray,  # (N,) for tracking
    source_time: float,
    target_time: float,
    batch_size: int = 256,
    time_tolerance: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample source and target minibatches from adjacent time clusters.

    Returns:
        X: (m, d) source embeddings
        Y: (n, d) target embeddings  
        self_indices: (m,) tracking indices (-1 if no track)
    """
    # Find points near source and target times
    source_mask = np.abs(times - source_time) < time_tolerance
    target_mask = np.abs(times - target_time) < time_tolerance

    source_indices = np.where(source_mask)[0]
    target_indices = np.where(target_mask)[0]

    # Sample up to batch_size points
    if len(source_indices) > batch_size:
        source_indices = np.random.choice(source_indices, batch_size, replace=False)
    if len(target_indices) > batch_size:
        target_indices = np.random.choice(target_indices, batch_size, replace=False)

    X = embeddings[source_indices]
    Y = embeddings[target_indices]

    # Build tracking indices
    self_indices = np.full(len(source_indices), -1, dtype=int)
    source_embryos = embryo_ids[source_indices]
    target_embryos = embryo_ids[target_indices]

    for i, emb_id in enumerate(source_embryos):
        # Find if this embryo exists in target set
        matches = np.where(target_embryos == emb_id)[0]
        if len(matches) > 0:
            self_indices[i] = matches[0]  # take first match

    return X, Y, self_indices
```

```python
def train_ode_with_uot(
    model,  # Neural ODE model
    data_loader,  # Provides (embeddings, times, embryo_ids)
    optimizer,
    num_epochs: int = 100,
    uot_config: Dict = None
):
    """
    Training loop using UOT-derived teacher velocities.
    """
    if uot_config is None:
        uot_config = {
            'eps': 0.05,
            'reg_m': 1.0, 
            'bias_mode': 'lower_bound',
            'batch_size': 256,
            'dt_range': (0.5, 2.0),  # sample time gaps
            'min_row_mass': 0.1  # threshold for velocity trust
        }

    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0

        for embeddings, times, embryo_ids in data_loader:
            # Sample time gap
            dt = np.random.uniform(*uot_config['dt_range'])

            # Get unique time points and sample a source time
            unique_times = np.unique(times)
            valid_source_times = unique_times[unique_times + dt < unique_times.max()]

            if len(valid_source_times) == 0:
                continue

            source_time = np.random.choice(valid_source_times)
            target_time = source_time + dt

            # Sample minibatches
            X, Y, self_indices = sample_minibatch_pair(
                embeddings, times, None, embryo_ids,
                source_time, target_time,
                uot_config['batch_size']
            )

            if len(X) == 0 or len(Y) == 0:
                continue

            # Compute UOT with biasing
            uot_result = compute_biased_uot(
                X, Y, dt,
                self_indices=self_indices,
                bias_mode=uot_config['bias_mode'],
                eps=uot_config['eps'],
                reg_m=uot_config['reg_m']
            )

            # Get ODE predictions
            X_torch = torch.from_numpy(X).float()
            t_torch = torch.full((len(X),), source_time)
            predicted_velocities = model(X_torch, t_torch)

            # Compute loss (mass-weighted MSE)
            row_masses = uot_result['row_masses']
            mask = row_masses > uot_config['min_row_mass']

            if mask.sum() > 0:
                target_velocities = torch.from_numpy(
                    uot_result['velocities'][mask]
                ).float()
                pred_velocities_masked = predicted_velocities[mask]
                weights = torch.from_numpy(row_masses[mask]).float()

                loss = (weights * (pred_velocities_masked - target_velocities).pow(2).sum(1)).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

        if num_batches > 0:
            print(f"Epoch {epoch}: Loss = {epoch_loss / num_batches:.4f}")
```

```python
def grid_search_uot_params(data_subset, param_grid):
    """
    Simple grid search for UOT hyperparameters.
    """
    results = []

    for params in param_grid:
        # Run UOT with params
        uot_result = compute_biased_uot(**params)

        # Evaluate quality metrics
        metrics = {
            'params': params,
            'plan_entropy': compute_entropy(uot_result['plan']),
            'self_edge_fraction': compute_self_edge_mass(uot_result['plan'], self_indices),
            'velocity_smoothness': compute_smoothness(uot_result['velocities'])
        }
        results.append(metrics)

    return results
```

```python
# File: tests/test_minibatch_uot.py

def test_perfect_tracking():
    """Test that perfect tracks are preserved."""
    # Create identical point clouds at t and t+1
    X = Y = np.random.randn(100, 10)
    self_indices = np.arange(100)

    result = compute_biased_uot(
        X, Y, dt=1.0,
        self_indices=self_indices,
        bias_mode="lower_bound"
    )

    # Check diagonal dominance
    diag_mass = np.diag(result['plan']).sum()
    total_mass = result['plan'].sum()
    assert diag_mass / total_mass > 0.8, "Perfect tracks not preserved"

def test_no_targets():
    """Test pure deletion when no targets exist."""
    X = np.random.randn(50, 10)
    Y = np.empty((0, 10))  # no targets

    result = compute_minibatch_uot(X, Y)
    assert result['row_masses'].sum() < 0.1, "Mass not deleted when no targets"

def test_bifurcation():
    """Test one source splitting to two targets."""
    X = np.array([[0, 0]])  # one source
    Y = np.array([[-1, 0], [1, 0]])  # two targets

    result = compute_minibatch_uot(X, Y, eps=0.1, reg_m=10.0)
    # Should split roughly equally
    assert abs(result['plan'][0, 0] - result['plan'][0, 1]) < 0.1
```

```python
def create_minibatch_generator(embeddings, times, cluster_labels, embryo_ids):
    """
    Generator that yields minibatch pairs using cluster structure.
    """
    unique_clusters = np.unique(cluster_labels)

    while True:
        # Sample source cluster
        source_cluster = np.random.choice(unique_clusters)

        # Find temporally adjacent clusters
        source_times = times[cluster_labels == source_cluster]
        mean_time = source_times.mean()

        # Find clusters with slightly later mean time
        target_clusters = []
        for c in unique_clusters:
            target_mean = times[cluster_labels == c].mean()
            if 0.5 < target_mean - mean_time < 3.0:
                target_clusters.append(c)

        if len(target_clusters) == 0:
            continue

        target_cluster = np.random.choice(target_clusters)

        # Extract points from chosen clusters
        source_mask = cluster_labels == source_cluster
        target_mask = cluster_labels == target_cluster

        yield (
            embeddings[source_mask],
            embeddings[target_mask],
            embryo_ids[source_mask],
            embryo_ids[target_mask]
        )
```