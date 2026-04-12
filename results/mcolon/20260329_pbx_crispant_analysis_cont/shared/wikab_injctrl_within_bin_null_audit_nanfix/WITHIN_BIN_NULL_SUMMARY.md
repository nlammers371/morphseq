# Within-Bin Wik_ab vs Inj_ctrl Null Audit

The target quantity is time-bin-internal `inj_ctrl` vs `wik_ab` AUROC. These controls should be near-null within matched biological stage.

## Requested anchor 26.0 hpf
| stage                             |   time_bin_center |   auroc_obs |   auroc_null_mean |      pval |      qval |   n_inj_ctrl |   n_wik_ab |
|:----------------------------------|------------------:|------------:|------------------:|----------:|----------:|-------------:|-----------:|
| binned_embeddings                 |                26 |    0.726852 |          0.462963 | 0.0909091 | 0.490909  |           18 |         12 |
| build06_rows                      |                26 |    0.859599 |          0.501504 | 0.0909091 | 0.0981818 |          192 |        116 |
| m_raw_injctrl_vs_wikab            |                26 |    0.333333 |          0.472222 | 0.909091  | 0.944056  |           18 |         12 |
| pairwise_raw_aligned_umap_init    |                26 |    0.231481 |          0.494444 | 1         | 1         |           18 |         12 |
| pairwise_raw_condensed_final      |                26 |    0.217593 |          0.510648 | 1         | 1         |           18 |         12 |
| pairwise_shrunk_aligned_umap_init |                26 |    0.365741 |          0.406481 | 0.636364  | 1         |           18 |         12 |
| pairwise_shrunk_condensed_final   |                26 |    0.333333 |          0.40787  | 0.636364  | 1         |           18 |         12 |
| raw_coordinates                   |                26 |    0.333333 |          0.472222 | 0.909091  | 0.944056  |           18 |         12 |
| shrunk_coordinates                |                26 |    0.5      |          0.5      | 1         | 1         |           18 |         12 |

## Requested anchor 54.0 hpf
| stage                             |   time_bin_center |   auroc_obs |   auroc_null_mean |      pval |      qval |   n_inj_ctrl |   n_wik_ab |
|:----------------------------------|------------------:|------------:|------------------:|----------:|----------:|-------------:|-----------:|
| binned_embeddings                 |                54 |    0.666667 |          0.553333 | 0.272727  | 0.818182  |            5 |         12 |
| build06_rows                      |                54 |    0.951923 |          0.499439 | 0.0909091 | 0.0981818 |           39 |         96 |
| m_raw_injctrl_vs_wikab            |                54 |    0.733333 |          0.371667 | 0.0909091 | 0.245455  |            5 |         12 |
| pairwise_raw_aligned_umap_init    |                54 |    0.333333 |          0.441667 | 0.727273  | 1         |            5 |         12 |
| pairwise_raw_condensed_final      |                54 |    0.333333 |          0.438333 | 0.727273  | 0.981818  |            5 |         12 |
| pairwise_shrunk_aligned_umap_init |                54 |    0.266667 |          0.53     | 1         | 1         |            5 |         12 |
| pairwise_shrunk_condensed_final   |                54 |    0.283333 |          0.523333 | 0.909091  | 1         |            5 |         12 |
| raw_coordinates                   |                54 |    0.733333 |          0.371667 | 0.0909091 | 0.245455  |            5 |         12 |
| shrunk_coordinates                |                54 |    0.733333 |          0.371667 | 0.0909091 | 0.613636  |            5 |         12 |

## Requested anchor 78.0 hpf
| stage                             |   time_bin_center |   auroc_obs |   auroc_null_mean |      pval |      qval |   n_inj_ctrl |   n_wik_ab |
|:----------------------------------|------------------:|------------:|------------------:|----------:|----------:|-------------:|-----------:|
| binned_embeddings                 |                78 |    0.55     |          0.529167 | 0.636364  | 1         |           12 |         10 |
| build06_rows                      |                78 |    0.952042 |          0.483504 | 0.0909091 | 0.0981818 |          103 |        116 |
| m_raw_injctrl_vs_wikab            |                78 |    0.441667 |          0.415    | 0.454545  | 0.721925  |           12 |         10 |
| pairwise_raw_aligned_umap_init    |                78 |    0.416667 |          0.524167 | 0.818182  | 1         |           12 |         10 |
| pairwise_raw_condensed_final      |                78 |    0.441667 |          0.528333 | 0.818182  | 0.981818  |           12 |         10 |
| pairwise_shrunk_aligned_umap_init |                78 |    0.308333 |          0.468333 | 0.818182  | 1         |           12 |         10 |
| pairwise_shrunk_condensed_final   |                78 |    0.275    |          0.455833 | 0.818182  | 1         |           12 |         10 |
| raw_coordinates                   |                78 |    0.441667 |          0.415    | 0.454545  | 0.721925  |           12 |         10 |
| shrunk_coordinates                |                78 |    0.441667 |          0.415    | 0.454545  | 1         |           12 |         10 |
