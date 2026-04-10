# Within-Bin Wik_ab vs Inj_ctrl Null Audit

The target quantity is time-bin-internal `inj_ctrl` vs `wik_ab` AUROC. These controls should be near-null within matched biological stage.

## Requested anchor 26.0 hpf
| stage                             |   time_bin_center |   auroc_obs |   auroc_null_mean |      pval |      qval |   n_inj_ctrl |   n_wik_ab |
|:----------------------------------|------------------:|------------:|------------------:|----------:|----------:|-------------:|-----------:|
| binned_embeddings                 |                26 |    0.726852 |          0.462963 | 0.0909091 | 0.490909  |           18 |         12 |
| build06_rows                      |                26 |    0.859599 |          0.501509 | 0.0909091 | 0.0981818 |          192 |        116 |
| m_raw_injctrl_vs_wikab            |                26 |    0.333333 |          0.472222 | 0.909091  | 0.944056  |           18 |         12 |
| pairwise_raw_aligned_umap_init    |                26 |    0.921296 |          0.527315 | 0.0909091 | 0.0944056 |           18 |         12 |
| pairwise_raw_condensed_final      |                26 |    0.944444 |          0.535648 | 0.0909091 | 0.0944056 |           18 |         12 |
| pairwise_shrunk_aligned_umap_init |                26 |    0.87963  |          0.508796 | 0.0909091 | 0.102273  |           18 |         12 |
| pairwise_shrunk_condensed_final   |                26 |    0.888889 |          0.497222 | 0.0909091 | 0.102273  |           18 |         12 |
| raw_coordinates                   |                26 |    0.671296 |          0.525463 | 0.272727  | 0.294545  |           18 |         12 |
| shrunk_coordinates                |                26 |    0.685185 |          0.519444 | 0.181818  | 0.196364  |           18 |         12 |

## Requested anchor 54.0 hpf
| stage                             |   time_bin_center |   auroc_obs |   auroc_null_mean |      pval |      qval |   n_inj_ctrl |   n_wik_ab |
|:----------------------------------|------------------:|------------:|------------------:|----------:|----------:|-------------:|-----------:|
| binned_embeddings                 |                54 |    0.666667 |          0.553333 | 0.272727  | 0.818182  |            5 |         12 |
| build06_rows                      |                54 |    0.951923 |          0.499439 | 0.0909091 | 0.0981818 |           39 |         96 |
| m_raw_injctrl_vs_wikab            |                54 |    0.733333 |          0.371667 | 0.0909091 | 0.245455  |            5 |         12 |
| pairwise_raw_aligned_umap_init    |                54 |    0.95     |          0.501667 | 0.0909091 | 0.0944056 |            5 |         12 |
| pairwise_raw_condensed_final      |                54 |    0.983333 |          0.511667 | 0.0909091 | 0.0944056 |            5 |         12 |
| pairwise_shrunk_aligned_umap_init |                54 |    0.733333 |          0.53     | 0.181818  | 0.181818  |            5 |         12 |
| pairwise_shrunk_condensed_final   |                54 |    0.833333 |          0.523333 | 0.181818  | 0.188811  |            5 |         12 |
| raw_coordinates                   |                54 |    0.783333 |          0.466667 | 0.181818  | 0.204545  |            5 |         12 |
| shrunk_coordinates                |                54 |    0.783333 |          0.466667 | 0.181818  | 0.196364  |            5 |         12 |

## Requested anchor 78.0 hpf
| stage                             |   time_bin_center |   auroc_obs |   auroc_null_mean |      pval |      qval |   n_inj_ctrl |   n_wik_ab |
|:----------------------------------|------------------:|------------:|------------------:|----------:|----------:|-------------:|-----------:|
| binned_embeddings                 |                78 |    0.55     |          0.529167 | 0.636364  | 1         |           12 |         10 |
| build06_rows                      |                78 |    0.952042 |          0.483504 | 0.0909091 | 0.0981818 |          103 |        116 |
| m_raw_injctrl_vs_wikab            |                78 |    0.441667 |          0.415    | 0.454545  | 0.721925  |           12 |         10 |
| pairwise_raw_aligned_umap_init    |                78 |    1        |          0.515    | 0.0909091 | 0.0944056 |           12 |         10 |
| pairwise_raw_condensed_final      |                78 |    1        |          0.523333 | 0.0909091 | 0.0944056 |           12 |         10 |
| pairwise_shrunk_aligned_umap_init |                78 |    1        |          0.516667 | 0.0909091 | 0.102273  |           12 |         10 |
| pairwise_shrunk_condensed_final   |                78 |    1        |          0.516667 | 0.0909091 | 0.102273  |           12 |         10 |
| raw_coordinates                   |                78 |    0.983333 |          0.471667 | 0.0909091 | 0.116883  |           12 |         10 |
| shrunk_coordinates                |                78 |    0.983333 |          0.471667 | 0.0909091 | 0.116883  |           12 |         10 |
