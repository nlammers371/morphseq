# Wik_ab vs Inj_ctrl Representation Ablation Audit

Variants:
- direct: ['inj_ctrl__vs__wik_ab']
- focal_vs_pbx1b_pbx4: ['inj_ctrl__vs__pbx1b_pbx4_crispant', 'pbx1b_crispant__vs__pbx1b_pbx4_crispant', 'pbx1b_pbx4_crispant__vs__pbx4_crispant', 'pbx1b_pbx4_crispant__vs__wik_ab']
- all_pairs: ['inj_ctrl__vs__pbx1b_crispant', 'inj_ctrl__vs__pbx1b_pbx4_crispant', 'inj_ctrl__vs__pbx4_crispant', 'inj_ctrl__vs__wik_ab', 'pbx1b_crispant__vs__pbx1b_pbx4_crispant', 'pbx1b_crispant__vs__pbx4_crispant', 'pbx1b_crispant__vs__wik_ab', 'pbx1b_pbx4_crispant__vs__pbx4_crispant', 'pbx1b_pbx4_crispant__vs__wik_ab', 'pbx4_crispant__vs__wik_ab']

## vector_space
| variant                    |   requested_anchor |   time_bin_center |   auroc_obs |   auroc_null_mean |     pval |     qval |   n_inj_ctrl |   n_wik_ab |
|:---------------------------|-------------------:|------------------:|------------:|------------------:|---------:|---------:|-------------:|-----------:|
| all_pairs_raw              |                 26 |                26 |    0.671296 |          0.544444 | 0.333333 | 0.333333 |           18 |         12 |
| all_pairs_shrunk           |                 26 |                26 |    0.685185 |          0.533333 | 0.333333 | 0.333333 |           18 |         12 |
| direct_raw                 |                 26 |                26 |    0.333333 |          0.515741 | 1        | 1        |           18 |         12 |
| direct_shrunk              |                 26 |                26 |    0.5      |          0.5      | 1        | 1        |           18 |         12 |
| focal_vs_pbx1b_pbx4_raw    |                 26 |                26 |    0.652778 |          0.566667 | 0.333333 | 0.333333 |           18 |         12 |
| focal_vs_pbx1b_pbx4_shrunk |                 26 |                26 |    0.652778 |          0.566667 | 0.333333 | 0.333333 |           18 |         12 |
| all_pairs_raw              |                 54 |                54 |    0.783333 |          0.36     | 0.166667 | 0.25     |            5 |         12 |
| all_pairs_shrunk           |                 54 |                54 |    0.783333 |          0.36     | 0.166667 | 0.25     |            5 |         12 |
| direct_raw                 |                 54 |                54 |    0.733333 |          0.356667 | 0.166667 | 0.5      |            5 |         12 |
| direct_shrunk              |                 54 |                54 |    0.733333 |          0.356667 | 0.166667 | 0.5      |            5 |         12 |
| focal_vs_pbx1b_pbx4_raw    |                 54 |                54 |    0.9      |          0.456667 | 0.166667 | 0.25     |            5 |         12 |
| focal_vs_pbx1b_pbx4_shrunk |                 54 |                54 |    0.9      |          0.456667 | 0.166667 | 0.25     |            5 |         12 |
| all_pairs_raw              |                 78 |                78 |    0.983333 |          0.596667 | 0.166667 | 0.25     |           12 |         10 |
| all_pairs_shrunk           |                 78 |                78 |    0.983333 |          0.596667 | 0.166667 | 0.25     |           12 |         10 |
| direct_raw                 |                 78 |                78 |    0.441667 |          0.496667 | 0.666667 | 1        |           12 |         10 |
| direct_shrunk              |                 78 |                78 |    0.441667 |          0.496667 | 0.666667 | 1        |           12 |         10 |
| focal_vs_pbx1b_pbx4_raw    |                 78 |                78 |    1        |          0.573333 | 0.166667 | 0.25     |           12 |         10 |
| focal_vs_pbx1b_pbx4_shrunk |                 78 |                78 |    1        |          0.573333 | 0.166667 | 0.25     |           12 |         10 |

## aligned_umap_init
| variant                    |   requested_anchor |   time_bin_center |   auroc_obs |   auroc_null_mean |     pval |     qval |   n_inj_ctrl |   n_wik_ab |
|:---------------------------|-------------------:|------------------:|------------:|------------------:|---------:|---------:|-------------:|-----------:|
| all_pairs_raw              |                 26 |                26 |    0.944444 |          0.62963  | 0.166667 | 0.166667 |           18 |         12 |
| all_pairs_shrunk           |                 26 |                26 |    0.939815 |          0.62963  | 0.166667 | 0.166667 |           18 |         12 |
| direct_raw                 |                 26 |                26 |    0.37963  |          0.510185 | 1        | 1        |           18 |         12 |
| direct_shrunk              |                 26 |                26 |    0.657407 |          0.383333 | 0.166667 | 0.25     |           18 |         12 |
| focal_vs_pbx1b_pbx4_raw    |                 26 |                26 |    0.847222 |          0.634259 | 0.166667 | 0.25     |           18 |         12 |
| focal_vs_pbx1b_pbx4_shrunk |                 26 |                26 |    0.962963 |          0.551852 | 0.166667 | 0.166667 |           18 |         12 |
| all_pairs_raw              |                 54 |                54 |    0.9      |          0.376667 | 0.166667 | 0.166667 |            5 |         12 |
| all_pairs_shrunk           |                 54 |                54 |    0.883333 |          0.36     | 0.166667 | 0.166667 |            5 |         12 |
| direct_raw                 |                 54 |                54 |    0.766667 |          0.253333 | 0.166667 | 0.5      |            5 |         12 |
| direct_shrunk              |                 54 |                54 |    0.7      |          0.39     | 0.166667 | 0.25     |            5 |         12 |
| focal_vs_pbx1b_pbx4_raw    |                 54 |                54 |    0.733333 |          0.573333 | 0.5      | 0.5      |            5 |         12 |
| focal_vs_pbx1b_pbx4_shrunk |                 54 |                54 |    0.95     |          0.416667 | 0.166667 | 0.166667 |            5 |         12 |
| all_pairs_raw              |                 78 |                78 |    1        |          0.571667 | 0.166667 | 0.166667 |           12 |         10 |
| all_pairs_shrunk           |                 78 |                78 |    1        |          0.566667 | 0.166667 | 0.166667 |           12 |         10 |
| direct_raw                 |                 78 |                78 |    0.25     |          0.511667 | 1        | 1        |           12 |         10 |
| direct_shrunk              |                 78 |                78 |    0.266667 |          0.546667 | 0.833333 | 0.833333 |           12 |         10 |
| focal_vs_pbx1b_pbx4_raw    |                 78 |                78 |    1        |          0.561667 | 0.166667 | 0.25     |           12 |         10 |
| focal_vs_pbx1b_pbx4_shrunk |                 78 |                78 |    1        |          0.548333 | 0.166667 | 0.166667 |           12 |         10 |
