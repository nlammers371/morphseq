# Within-Bin Wik_ab vs Inj_ctrl Null Audit

The target quantity is time-bin-internal `inj_ctrl` vs `wik_ab` AUROC. These controls should be near-null within matched biological stage.

## Requested anchor 26.0 hpf
| stage                             |   time_bin_center |   auroc_obs |   auroc_null_mean |     pval |     qval |   n_inj_ctrl |   n_wik_ab |
|:----------------------------------|------------------:|------------:|------------------:|---------:|---------:|-------------:|-----------:|
| binned_embeddings                 |                26 |    0.726852 |          0.45     | 0.166667 | 0.75     |           18 |         12 |
| build06_rows                      |                26 |    0.859599 |          0.513721 | 0.166667 | 0.173077 |          192 |        116 |
| m_raw_injctrl_vs_wikab            |                26 |    0.333333 |          0.515741 | 1        | 1        |           18 |         12 |
| pairwise_raw_aligned_umap_init    |                26 |    0.921296 |          0.634259 | 0.166667 | 0.173077 |           18 |         12 |
| pairwise_raw_condensed_final      |                26 |    0.944444 |          0.639815 | 0.166667 | 0.173077 |           18 |         12 |
| pairwise_shrunk_aligned_umap_init |                26 |    0.87963  |          0.608333 | 0.166667 | 0.173077 |           18 |         12 |
| pairwise_shrunk_condensed_final   |                26 |    0.888889 |          0.608333 | 0.166667 | 0.173077 |           18 |         12 |
| raw_coordinates                   |                26 |    0.671296 |          0.544444 | 0.333333 | 0.333333 |           18 |         12 |
| shrunk_coordinates                |                26 |    0.685185 |          0.533333 | 0.333333 | 0.333333 |           18 |         12 |

## Requested anchor 54.0 hpf
| stage                             |   time_bin_center |   auroc_obs |   auroc_null_mean |     pval |     qval |   n_inj_ctrl |   n_wik_ab |
|:----------------------------------|------------------:|------------:|------------------:|---------:|---------:|-------------:|-----------:|
| binned_embeddings                 |                54 |    0.666667 |          0.513333 | 0.166667 | 0.75     |            5 |         12 |
| build06_rows                      |                54 |    0.951923 |          0.499466 | 0.166667 | 0.173077 |           39 |         96 |
| m_raw_injctrl_vs_wikab            |                54 |    0.733333 |          0.356667 | 0.166667 | 0.375    |            5 |         12 |
| pairwise_raw_aligned_umap_init    |                54 |    0.95     |          0.396667 | 0.166667 | 0.173077 |            5 |         12 |
| pairwise_raw_condensed_final      |                54 |    0.983333 |          0.42     | 0.166667 | 0.173077 |            5 |         12 |
| pairwise_shrunk_aligned_umap_init |                54 |    0.733333 |          0.396667 | 0.166667 | 0.173077 |            5 |         12 |
| pairwise_shrunk_condensed_final   |                54 |    0.833333 |          0.45     | 0.166667 | 0.173077 |            5 |         12 |
| raw_coordinates                   |                54 |    0.783333 |          0.36     | 0.166667 | 0.1875   |            5 |         12 |
| shrunk_coordinates                |                54 |    0.783333 |          0.36     | 0.166667 | 0.1875   |            5 |         12 |

## Requested anchor 78.0 hpf
| stage                             |   time_bin_center |   auroc_obs |   auroc_null_mean |     pval |     qval |   n_inj_ctrl |   n_wik_ab |
|:----------------------------------|------------------:|------------:|------------------:|---------:|---------:|-------------:|-----------:|
| binned_embeddings                 |                78 |    0.55     |          0.663333 | 0.833333 | 1        |           12 |         10 |
| build06_rows                      |                78 |    0.952042 |          0.501607 | 0.166667 | 0.173077 |          103 |        116 |
| m_raw_injctrl_vs_wikab            |                78 |    0.441667 |          0.496667 | 0.666667 | 0.818182 |           12 |         10 |
| pairwise_raw_aligned_umap_init    |                78 |    1        |          0.57     | 0.166667 | 0.173077 |           12 |         10 |
| pairwise_raw_condensed_final      |                78 |    1        |          0.565    | 0.166667 | 0.173077 |           12 |         10 |
| pairwise_shrunk_aligned_umap_init |                78 |    1        |          0.581667 | 0.166667 | 0.173077 |           12 |         10 |
| pairwise_shrunk_condensed_final   |                78 |    1        |          0.576667 | 0.166667 | 0.173077 |           12 |         10 |
| raw_coordinates                   |                78 |    0.983333 |          0.596667 | 0.166667 | 0.1875   |           12 |         10 |
| shrunk_coordinates                |                78 |    0.983333 |          0.596667 | 0.166667 | 0.1875   |           12 |         10 |
