Marazzano Colon

# Experimental Design: Variance Decomposition of Zebrafish Embryo Phenotypes

## 1. Problem Statement
**Objective:** Quantify the relative contributions of genetics (Pair ID) versus individual variability (Embryo ID) to the developmental trajectory of zebrafish embryos.

**The Core Question:**
We observe time-series data of phenotype changes (e.g., curvature) in developing embryos. While visual inspection suggests that the mating pair (genetics) is the main determinant of the trajectory shape, we need a rigorous statistical framework to answer:
> *"What percentage of the variation in the data is explained by the mating Pair (Genetics), the specific Embryo (Individual), and Measurement Noise?"*
s
## 2. Data Structure & Challenges

### The Data
* **Structure:** Hierarchical / Nested.
    * **Level 1 (Pair):** Mating pairs of zebrafish mutants.
    * **Level 2 (Embryo):** Multiple embryos spawned from each pair.
    * **Observations:** Time-series measurements of curvature ($y$) at time $t$.
* **Temporal Coverage:** Overlapping but incomplete time windows (e.g., some sampled $t=11 \to 80$, others $t=44 \to 120$).

### Key Statistical Challenges
1.  **Anti-Correlated Trends:** Distinct pairs may exhibit opposite behaviors (e.g., Pair A increases over time, Pair B decreases). A standard "global mean" model would average these to zero, failing to capture the signal.
2.  **Autocorrelation:** Data points within a trajectory are not independent; they form a smooth curve.
3.  **Extrapolation Risk:** Due to incomplete time windows, models might generate "unsupported trends" in gaps where data is missing.

---

## 3. Modeling Approach

We will use a **Hierarchical Bayesian Model** using **Hilbert Space Gaussian Processes (HSGP)**.

### Why this approach?
* **Bayesian Framework:** Allows us to estimate uncertainty and handle smaller sample sizes robustly via partial pooling.
* **HSGP (Splines):** Models the *shape* of the trajectory non-parametrically (letting the data define the curve) rather than forcing a linear or quadratic fit.
* **Factor-Smooth Interactions:** We will fit a specific curve for **each Pair**, allowing them to have completely different shapes (crossing lines), while still sharing a global "smoothness" prior.

### The Equation (Conceptually)
$$y_{ijk} = \underbrace{f_{pair}(t_k)}_{\text{Genetic Trend}} + \underbrace{u_{ij}}_{\text{Individual Offset}} + \underbrace{\epsilon_{ijk}}_{\text{Noise}}$$

* $f_{pair}(t)$: A smooth curve unique to Pair $j$.
* $u_{ij}$: A random intercept (offset) for Embryo $i$ nested in Pair $j$.
* $\epsilon_{ijk}$: Residual variation not explained by the pair trend or the linear embryo shift (includes measurement noise + non-linear shape deviations).

---

## 4. Variance Decomposition Strategy

To calculate the percentage of variance explained by each source, we will use a **Finite Population Variance** approach via model predictions (counterfactuals).

### Component 1: Genetic Variance ($\sigma^2_{pair}$)
**Definition:** The variance of the "Idealized/Synthetic" trajectories.
**Method:** We predict the curvature for all observations using **only** the Pair-specific smooth curve, ignoring the embryo offset.
* *Concept:* "What would the curvature be if this embryo were a generic representative of its parents?"

### Component 2: Individual Variance ($\sigma^2_{embryo}$)
**Definition:** The variance of the *deviation* between the specific embryo and its genetic average.
**Method:** We predict the full value (Curve + Offset) and subtract the Genetic Prediction.
* *Concept:* "How much does this specific embryo rebel against its genetic programming?"

### Component 3: Noise Variance ($\sigma^2_{error}$)
**Definition:** The variance of the residuals.
**Method:** The difference between the Full Prediction and the Raw Observed Data.

---

## 5. Implementation Plan

**Software Stack:**
* **Language:** Python
* **Modeling Engine:** `Bambi` (High-level interface for PyMC)
* **Backend:** `PyMC` (Probabilistic Programming) / `ArviZ` (Exploration)

### Code Example: Model Definition & Fitting

```python
import bambi as bmb
import pandas as pd
import numpy as np

# 1. Data Prep: Ensure strictly unique IDs for hierarchy
df['unique_embryo_id'] = df['pair_id'].astype(str) + "_" + df['embryo_id'].astype(str)

# 2. Model Definition
# 'hsgp(time, by=pair_id)': Separate smooth curve for every pair (Genetics)
# '(1|unique_embryo_id)':   Random intercept for every embryo (Individual)
# '0 +':                    Remove global intercept to let curves float freely
model_definition = "curvature ~ 0 + hsgp(time, by=pair_id) + (1|unique_embryo_id)"

model = bmb.Model(model_definition, data=df, dropna=True)

# 3. Fit Model (MCMC Sampling)
# Draws=1000 means we get 4000 total posterior samples (4 chains * 1000 draws)
idata = model.fit(draws=1000, tune=1000, chains=4)
```

### Code Example: Variance Decomposition Calculation
```python
# A. Get "Genetic" (Synthetic) Predictions
# We predict on the original data, effectively isolating the Pair trend
# Note: This accesses the 'mu' (mean) of the posterior predictive
preds = model.predict(idata, inplace=False)
full_prediction = preds.posterior["mu"].mean(dim=["chain", "draw"]).values

# B. Isolate the Embryo Effect
# Extract the learned random intercepts for each embryo
random_intercepts = idata.posterior["1|unique_embryo_id"].mean(dim=["chain", "draw"])

# Map these intercepts back to the original dataframe rows to align them
embryo_offsets = np.array([
    random_intercepts.sel(unique_embryo_id_dim=uid).values 
    for uid in df['unique_embryo_id']
])

# C. Calculate Components
# 1. Genetic Signal = Full Prediction - Embryo Offset
genetic_signal = full_prediction - embryo_offsets

# 2. Residuals = Raw Data - Full Prediction
residuals = df['curvature'].values - full_prediction

# D. Calculate Variances
var_genetics = np.var(genetic_signal)
var_individual = np.var(embryo_offsets)
var_noise = np.var(residuals)

# E. Final Percentages
total_var = var_genetics + var_individual + var_noise

print(f"Variance Explained by Genetics (Pair): {var_genetics / total_var:.1%}")
print(f"Variance Explained by Individual:      {var_individual / total_var:.1%}")
print(f"Variance Unexplained (Noise):          {var_noise / total_var:.1%}")


```

### Code Example: Visualization (Handling Extrapolation)

To prevent the model from displaying "unsupported trends" (extrapolating curves into time windows where a specific pair was not observed), we apply post-hoc clipping.

```python 
import matplotlib.pyplot as plt

# Generate high-resolution predictions for plotting
t_grid = np.linspace(df['time'].min(), df['time'].max(), 200)
# (Assuming code to generate predictions on t_grid exists here...)

for pair in df['pair_id'].unique():
    # 1. Identify valid data range for this pair
    valid_t_min = df[df['pair_id'] == pair]['time'].min()
    valid_t_max = df[df['pair_id'] == pair]['time'].max()
    
    # 2. Clip the plotting line to this range
    pair_curve = get_curve_for_pair(pair) # Pseudocode accessor
    
    # Mask out extrapolation
    mask = (t_grid >= valid_t_min) & (t_grid <= valid_t_max)
    
    plt.plot(t_grid[mask], pair_curve[mask], label=f"Pair {pair}")

plt.legend()
plt.show()
```