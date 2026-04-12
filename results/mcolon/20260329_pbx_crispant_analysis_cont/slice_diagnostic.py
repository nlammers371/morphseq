"""
slice_diagnostic.py
-------------------
2D single-slice force diagnostic for tuning attraction topology and force balance.
"""
from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

ANALYSIS_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ANALYSIS_ROOT))

MULTI_PATH = (
    ANALYSIS_ROOT
    / 'results/phenotypic_positioning_multiclass_bridge_bin4_perm500'
    / 'multiclass_probability_vectors.csv'
)

GENOTYPE_COLORS = {
    'inj_ctrl': '#2166AC',
    'pbx1b_crispant': '#F7B267',
    'pbx4_crispant': '#9467bd',
    'pbx1b_pbx4_crispant': '#B2182B',
}

TARGET_TIMES = [26.0, 78.0]


def gaussian_kernel(pos: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    diff = pos[:, None, :] - pos[None, :, :]
    sq = (diff ** 2).sum(axis=-1)
    K = np.exp(-sq / (2.0 * sigma ** 2))
    return K, diff, sq


def knn_mask_from_sq_dist(sq_dist: np.ndarray, k_attract: int | None) -> np.ndarray | float:
    n = sq_dist.shape[0]
    if k_attract is None or n <= 1:
        return 1.0
    k_eff = min(k_attract, n - 1)
    if k_eff <= 0 or k_eff >= n - 1:
        return 1.0
    sq_for_knn = sq_dist.copy()
    np.fill_diagonal(sq_for_knn, np.inf)
    knn_idx = np.argpartition(sq_for_knn, kth=k_eff - 1, axis=1)[:, :k_eff]
    knn_mask = np.zeros((n, n), dtype=float)
    knn_mask[np.arange(n)[:, None], knn_idx] = 1.0
    knn_mask = np.maximum(knn_mask, knn_mask.T)
    np.fill_diagonal(knn_mask, 0.0)
    return knn_mask


def attraction_grad(
    pos: np.ndarray,
    sigma: float,
    C: np.ndarray | None = None,
    k_attract: int | None = None,
    subtract_mean: bool = False,
) -> tuple[float, np.ndarray]:
    K, diff, sq = gaussian_kernel(pos, sigma)
    if C is None:
        C = np.ones_like(K)
    W = K * C * knn_mask_from_sq_dist(sq, k_attract)
    np.fill_diagonal(W, 0.0)
    energy = -W.sum()
    W_sym = W + W.T
    grad = (W_sym[:, :, None] * diff).sum(axis=1) / (sigma ** 2)
    if subtract_mean:
        grad = grad - grad.mean(axis=0, keepdims=True)
    return energy, grad


def same_label_coherence(labels: np.ndarray) -> np.ndarray:
    C = (labels[:, None] == labels[None, :]).astype(float)
    np.fill_diagonal(C, 0.0)
    return C


def repulsion_grad(pos: np.ndarray, epsilon_r: float, eta: float = 1e-4) -> tuple[float, np.ndarray]:
    diff = pos[:, None, :] - pos[None, :, :]
    sq = (diff ** 2).sum(axis=-1)
    denom = sq + eta
    valid = np.ones_like(denom)
    np.fill_diagonal(valid, 0.0)
    energy = (epsilon_r / denom * valid).sum()
    coeff = -2.0 * epsilon_r / (denom ** 2) * valid
    coeff_sym = coeff + coeff.T
    grad = (coeff_sym[:, :, None] * diff).sum(axis=1)
    return energy, grad


def radial_spread(pos: np.ndarray) -> float:
    center = pos.mean(axis=0)
    return float(np.sqrt(((pos - center) ** 2).sum(axis=1).mean()))


def run_slice(
    pos0: np.ndarray,
    sigma: float,
    epsilon_r: float,
    n_iter: int = 300,
    lr: float = 5e-4,
    C: np.ndarray | None = None,
    k_attract: int | None = None,
    subtract_mean: bool = False,
) -> tuple[np.ndarray, list[dict]]:
    pos = pos0.copy()
    history = []
    for i in range(n_iter):
        e_att, g_att = attraction_grad(pos, sigma, C=C, k_attract=k_attract, subtract_mean=subtract_mean)
        e_rep, g_rep = repulsion_grad(pos, epsilon_r)
        grad = g_att + g_rep
        pos = pos - lr * grad
        if i % 50 == 0 or i == n_iter - 1:
            history.append({
                'iter': i,
                'e_att': e_att,
                'e_rep': e_rep,
                'spread': radial_spread(pos),
            })
    return pos, history


def separation_score(pos: np.ndarray, labels: np.ndarray) -> float:
    groups = sorted(set(labels))
    if len(groups) < 2:
        return float('nan')
    centroids = np.array([pos[labels == g].mean(axis=0) for g in groups])
    cdiff = centroids[:, None, :] - centroids[None, :, :]
    cdist = np.sqrt((cdiff ** 2).sum(axis=-1))
    between = cdist[np.triu_indices(len(groups), k=1)].mean()
    within = np.mean([
        np.sqrt(((pos[labels == g] - c) ** 2).sum(axis=-1)).mean()
        for g, c in zip(groups, centroids)
    ])
    return between / (within + 1e-8)


def plot_before_after(ax_before, ax_after, pos0, pos_final, labels, meta):
    for ax, pos, title in [
        (ax_before, pos0, meta['before_title']),
        (ax_after, pos_final, meta['after_title']),
    ]:
        for geno, color in GENOTYPE_COLORS.items():
            sel = labels == geno
            if sel.sum() == 0:
                continue
            ax.scatter(pos[sel, 0], pos[sel, 1], color=color, s=18, alpha=0.7, linewidths=0)
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])


def load_slice(t: float, init: str = 'umap') -> tuple[np.ndarray, np.ndarray, str]:
    df = pd.read_csv(MULTI_PATH)
    prob_cols = [c for c in df.columns if c.startswith('pred_proba_')]
    sub = df[df['time_bin_center'] == t].reset_index(drop=True)
    X = sub[prob_cols].values.astype(float)
    labels = sub['true_class'].values

    if init == 'umap':
        try:
            import umap as umap_lib
            reducer = umap_lib.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
            pos = reducer.fit_transform(X)
            return pos, labels, 'umap'
        except Exception as exc:
            print(f'  UMAP failed ({exc}), falling back to PCA')

    pca = PCA(n_components=2, random_state=0)
    pos = pca.fit_transform(X)
    return pos, labels, 'pca'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--output-dir', default='results/mcolon/20260329_pbx_crispant_analysis_cont/results/slice_diagnostic/')
    p.add_argument('--n-iter', type=int, default=300)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--init', choices=['pca', 'umap', 'both'], default='both')
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sigma_fracs = [0.3, 0.5, 0.7, 1.0]
    eps_mults = [0.3, 0.6, 1.2]
    k_values = [5, 10, 20, None]
    subtract_values = [False, True]
    inits = ['pca', 'umap'] if args.init == 'both' else [args.init]

    summary_rows = []

    for t in TARGET_TIMES:
        for init_method in inits:
            pos0, labels, init_name = load_slice(t, init=init_method)
            n = len(pos0)
            print(f"\n=== t={t:.0f}hpf init={init_name} n={n} ===")
            diff = pos0[:, None, :] - pos0[None, :, :]
            dists = np.sqrt(((diff ** 2).sum(axis=-1))[np.triu_indices(n, k=1)])
            sr = radial_spread(pos0)
            print(f"  scale_ref={sr:.4f} median_pair_dist={np.median(dists):.4f}")
            score_before = separation_score(pos0, labels)
            spread_before = radial_spread(pos0)
            C_oracle = same_label_coherence(labels)

            for coherence_mode, C_mat in [('all2all', None), ('oracle', C_oracle)]:
                n_params = len(sigma_fracs) * len(eps_mults) * len(k_values) * len(subtract_values)
                fig, axes = plt.subplots(n_params, 2, figsize=(9, 2.8 * n_params))
                axes = np.array(axes).reshape(n_params, 2)
                row_idx = 0
                for sf, em, k_attract, subtract_mean in product(sigma_fracs, eps_mults, k_values, subtract_values):
                    sigma = sf * sr
                    epsilon_r = em * 0.6 * sigma ** 2
                    pos_final, history = run_slice(
                        pos0,
                        sigma,
                        epsilon_r,
                        n_iter=args.n_iter,
                        lr=args.lr,
                        C=C_mat,
                        k_attract=k_attract,
                        subtract_mean=subtract_mean,
                    )
                    score_after = separation_score(pos_final, labels)
                    spread_after = radial_spread(pos_final)
                    summary_rows.append({
                        't': t,
                        'init': init_name,
                        'coherence': coherence_mode,
                        'sigma_frac': sf,
                        'eps_mult': em,
                        'sigma': sigma,
                        'epsilon_r': epsilon_r,
                        'k_attract': k_attract,
                        'subtract_mean': subtract_mean,
                        'sep_before': score_before,
                        'sep_after': score_after,
                        'sep_gain': score_after - score_before,
                        'spread_before': spread_before,
                        'spread_after': spread_after,
                        'spread_ratio': spread_after / (spread_before + 1e-8),
                        'history_final_spread': history[-1]['spread'],
                    })
                    before_title = (
                        f"t={t:.0f} {coherence_mode} before\n"
                        f"sigma={sigma:.3f} eps_r={epsilon_r:.4f} "
                        f"k={k_attract} mean={subtract_mean} sep={score_before:.2f}"
                    )
                    after_title = f"after sep={score_after:.2f} spread={spread_after:.2f}"
                    plot_before_after(
                        axes[row_idx, 0],
                        axes[row_idx, 1],
                        pos0,
                        pos_final,
                        labels,
                        {'before_title': before_title, 'after_title': after_title},
                    )
                    row_idx += 1
                fig.suptitle(
                    f"Slice diagnostic — t={t:.0f}hpf init={init_name} coherence={coherence_mode}",
                    fontsize=10,
                    y=0.999,
                )
                fig.tight_layout()
                path = out / f"slice_t{t:.0f}_{init_name}_{coherence_mode}.png"
                fig.savefig(path, dpi=140, bbox_inches='tight')
                plt.close(fig)
                print(f"  -> {path}")

    summary = pd.DataFrame(summary_rows)
    summary_path = out / 'summary.csv'
    summary.to_csv(summary_path, index=False)
    print(f"\nSummary: {summary_path}")
    print(summary.sort_values(['sep_gain', 'spread_ratio'], ascending=[False, False]).head(20).to_string(index=False))


if __name__ == '__main__':
    main()
