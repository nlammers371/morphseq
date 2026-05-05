# Metric Loss Equations

This note documents the actual metric-loss equations implemented in the current training stack, with emphasis on the `NTXentLoss` path in [src/core/losses/loss_functions.py](/net/trapnell/vol1/home/nlammers/projects/repositories/morphseq/src/core/losses/loss_functions.py:270). The configuration for this loss lives in [src/core/losses/loss_configs.py](/net/trapnell/vol1/home/nlammers/projects/repositories/morphseq/src/core/losses/loss_configs.py:110), and the training pipeline injects the dataset-derived `metric_array` before the loss module is instantiated in [src/core/run/run_utils.py](/net/trapnell/vol1/home/nlammers/projects/repositories/morphseq/src/core/run/run_utils.py:339).

## Combined training objective

For a batch of paired views, the implemented objective is

$$
\mathcal{L}_{\text{total}}
=
\mathcal{L}_{\text{pixel}}
+ \lambda_{\text{kld}} \mathcal{L}_{\text{kld}}
+ \lambda_{\text{pips}} \mathcal{L}_{\text{pips}}
+ \lambda_{\text{gan}} \mathcal{L}_{\text{gan}}
+ \lambda_{\text{metric}} \mathcal{L}_{\text{metric}},
$$

where

$$
\lambda_{\text{kld}} = \texttt{kld\_weight}, \quad
\lambda_{\text{pips}} = \texttt{pips\_weight}, \quad
\lambda_{\text{gan}} = \texttt{gan\_weight}, \quad
\lambda_{\text{metric}} = \texttt{metric\_weight}.
$$

In code, this sum is assembled in `NTXentLoss.forward()` at [src/core/losses/loss_functions.py](/net/trapnell/vol1/home/nlammers/projects/repositories/morphseq/src/core/losses/loss_functions.py:289).

## VAE terms

Only the first view is reconstructed. If the input batch is

$$
x = \{(x_i^{(0)}, x_i^{(1)})\}_{i=1}^B,
$$

then reconstruction is computed against $x_i^{(0)}$ only.

### Pixel term

Let $\hat{x}_i$ denote the reconstruction of $x_i^{(0)}$. The raw pixel loss is either

$$
\ell_i^{\text{L1}} = \frac{1}{P}\| \hat{x}_i - x_i^{(0)} \|_1
$$

or

$$
\ell_i^{\text{L2}} = \frac{1}{2P}\| \hat{x}_i - x_i^{(0)} \|_2^2,
$$

where $P$ is the number of pixels per sample after flattening.

The implemented reconstruction contribution is then

$$
\mathcal{L}_{\text{pixel}} = s_{\text{pixel}} \cdot \frac{1}{B}\sum_{i=1}^B \ell_i,
$$

with hard-coded scale

$$
s_{\text{pixel}} =
\begin{cases}
\frac{128 \cdot 288}{100} & \text{for L2} \\
\frac{128 \cdot 288}{1000} & \text{for L1}.
\end{cases}
$$

This scaling is implemented in `_pixel_scale()` and applied in `_compute_vae_terms()`.

### KLD term

If $\mu_i \in \mathbb{R}^D$ and $\log \sigma_i^2 \in \mathbb{R}^D$ are the encoder outputs for the latent Gaussian, the code uses

$$
\mathcal{L}_{\text{kld}}
=
\frac{1}{B}\sum_{i=1}^B
\left[
-\frac{1}{2}
\sum_{d=1}^{D}
\left(
1 + \log \sigma_{id}^2 - \mu_{id}^2 - \sigma_{id}^2
\right)
\right]
$$

written in vectorized form as

$$
-\frac{1}{2}\,\mathrm{mean}\!\left(1 + \log \sigma^2 - \mu^2 - e^{\log \sigma^2}\right).
$$

### Optional LPIPS and GAN terms

If enabled, the objective also includes:

$$
\mathcal{L}_{\text{pips}} = \mathrm{LPIPS}(\hat{x}, x^{(0)})
$$

and a generator-side adversarial term

$$
\mathcal{L}_{\text{gan}} = -\mathbb{E}[D(\hat{x})]
$$

or the average of that quantity across multi-scale discriminator outputs.

## Metric term: implemented NT-Xent variant

The implemented metric term is not the standard cosine SimCLR objective. It is a masked, multi-positive cross-entropy built from Euclidean-distance logits over the biological subset of latent coordinates.

### 1. Biological latent subspace

The latent vector is split into nuisance and biological coordinates using `frac_nuisance_latents`; see [src/core/losses/loss_configs.py](/net/trapnell/vol1/home/nlammers/projects/repositories/morphseq/src/core/losses/loss_configs.py:182).

If the full latent dimension is $D$, then

$$
D_{\text{bio}} = D - \left\lceil \rho D \right\rceil,
$$

where $\rho = \texttt{frac\_nuisance\_latents}$.

Only the biological coordinates are used in the metric loss:

$$
z_i = \mu_i[\text{biological\_indices}] \in \mathbb{R}^{D_{\text{bio}}}.
$$

The metric term is computed on `model_output.mu`, not sampled latents.

### 2. Two-view batch construction

After encoding both views, the implementation forms a single stack of $2B$ latent vectors:

$$
Z = \{z_1^{(0)}, \dots, z_B^{(0)}, z_1^{(1)}, \dots, z_B^{(1)}\}.
$$

The nominal paired-view positives are defined by sample identity across the two views.

### 3. Euclidean-distance logits

The code first computes the squared pairwise Euclidean distance matrix

$$
\Delta_{ij} = \|z_i - z_j\|_2^2.
$$

It then sets

$$
N = \frac{D_{\text{bio}}}{2},
\qquad
\sigma = N,
$$

and converts distances to logits via

$$
\ell_{ij}
=
\frac{
-\sqrt{\Delta_{ij}/\sigma} + m
}{\tau},
$$

where

$$
m = \texttt{margin}, \qquad \tau = \texttt{temperature}.
$$

Since $\sqrt{\Delta_{ij}} = \|z_i-z_j\|_2$, this is equivalent to

$$
\ell_{ij}
=
\frac{
m - \|z_i-z_j\|_2/\sqrt{\sigma}
}{\tau}.
$$

So the logit is an affine transform of negative Euclidean distance. Nearby points receive larger logits, and the `margin` acts as an additive offset before temperature scaling.

### 4. Target matrix

The target matrix $T \in \{-1,0,1\}^{2B \times 2B}$ is assembled in two stages.

#### Stage A: paired-view structure

The code first creates a matrix encoding same-sample relationships across the two views:

$$
P_{ij} =
\begin{cases}
1 & \text{if } i,j \text{ correspond to the same base sample and } i \neq j \\
-1 & \text{if } i=j \\
0 & \text{otherwise.}
\end{cases}
$$

Here:

- `1` means forced positive
- `-1` means excluded
- `0` means unresolved for now

The diagonal is always excluded.

#### Stage B: metadata-defined positives and exclusions

If metadata is present, additional structure is added using `self_stats`, `other_stats`, and `metric_array`.

Let $a_i$ denote age and $c_i$ denote perturbation class for each of the $2B$ latent vectors. Then the age gate is

$$
A_{ij} = \mathbf{1}\left(|a_i-a_j| \le w + 1.5\right),
$$

where $w = \texttt{time\_window}$.

If `self_target_prob < 1.0`, a perturbation compatibility matrix is pulled from `metric_array`:

$$
M_{ij} \in \{-1,0,1\}.
$$

The code interprets this as

$$
C^{+}_{ij} = \mathbf{1}(M_{ij}=1), \qquad
C^{-}_{ij} = \mathbf{1}(M_{ij}=-1).
$$

Then the metadata rules are

$$
T_{ij} = 1 \quad \text{if} \quad A_{ij} \land C^{+}_{ij},
$$

$$
T_{ij} = -1 \quad \text{if} \quad C^{-}_{ij}.
$$

If `self_target_prob = 1.0`, perturbation compatibility defaults to all-true and no cross-class exclusions are introduced.

Finally, the paired-view matrix overrides metadata defaults:

$$
T_{ij} = 1 \quad \text{where } P_{ij}=1,
$$

$$
T_{ij} = -1 \quad \text{where } P_{ij}=-1.
$$

So the final semantics are:

$$
T_{ij} =
\begin{cases}
1 & \text{positive pair} \\
0 & \text{negative competitor} \\
-1 & \text{masked / excluded.}
\end{cases}
$$

### 5. Multi-positive masked cross-entropy

Given logits $\ell_{ij}$ and target matrix $T_{ij}$, the implementation computes for each anchor $i$:

$$
\mathcal{P}_i = \{j : T_{ij}=1\},
\qquad
\mathcal{V}_i = \{j : T_{ij}\neq -1\}.
$$

The per-anchor loss is

$$
\mathcal{L}_i
=
-\log
\frac{
\sum_{j \in \mathcal{P}_i} \exp(\ell_{ij})
}{
\sum_{j \in \mathcal{V}_i} \exp(\ell_{ij})
}.
$$

The batch metric loss is

$$
\mathcal{L}_{\text{metric}}
=
\frac{1}{2B}\sum_{i=1}^{2B} \mathcal{L}_i.
$$

This is exactly what `_nt_xent_loss_multiclass()` computes via masked `logsumexp`.

## Interpretation

The implemented metric term can be read as follows:

- It is contrastive, but not one-positive-per-anchor.
- It uses Euclidean geometry on the encoder mean, restricted to biological latent axes.
- It treats same-sample paired views as mandatory positives.
- It can also treat nearby ages and compatible perturbations as positives.
- It can explicitly exclude biologically invalid comparisons from the denominator.

So the objective is closer to a metadata-structured neighborhood classification loss than to vanilla NT-Xent.

## Hyperparameters that directly shape the equation

From [src/core/losses/loss_configs.py](/net/trapnell/vol1/home/nlammers/projects/repositories/morphseq/src/core/losses/loss_configs.py:128), the main equation-level parameters are:

- `temperature` $\tau$: softmax sharpness
- `margin` $m$: additive offset on the distance-derived logits
- `metric_weight`: outer multiplier on the metric term
- `frac_nuisance_latents`: determines $D_{\text{bio}}$
- `time_window`: age tolerance before the hard-coded extra `+1.5`
- `self_target_prob`: toggles whether perturbation-compatibility structure is used

## One implementation detail worth noting

The method is named `NTXentLoss`, but mathematically the current active version is not the usual normalized temperature-scaled cross-entropy from SimCLR:

- it does not use cosine similarity
- it does not require exactly one positive per anchor
- it masks some comparisons out entirely
- it uses metadata-conditioned positive sets

That naming mismatch is worth keeping in mind when comparing runs against published NT-Xent baselines.
