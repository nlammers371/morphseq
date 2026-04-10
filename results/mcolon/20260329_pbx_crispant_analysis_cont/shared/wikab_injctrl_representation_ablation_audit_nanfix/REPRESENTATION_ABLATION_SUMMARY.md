# Wik_ab vs Inj_ctrl Representation Ablation Audit

Variants:
- direct: ['inj_ctrl__vs__wik_ab']
- focal_vs_pbx1b_pbx4: ['inj_ctrl__vs__pbx1b_pbx4_crispant', 'pbx1b_crispant__vs__pbx1b_pbx4_crispant', 'pbx1b_pbx4_crispant__vs__pbx4_crispant', 'pbx1b_pbx4_crispant__vs__wik_ab']
- all_pairs: ['inj_ctrl__vs__pbx1b_crispant', 'inj_ctrl__vs__pbx1b_pbx4_crispant', 'inj_ctrl__vs__pbx4_crispant', 'inj_ctrl__vs__wik_ab', 'pbx1b_crispant__vs__pbx1b_pbx4_crispant', 'pbx1b_crispant__vs__pbx4_crispant', 'pbx1b_crispant__vs__wik_ab', 'pbx1b_pbx4_crispant__vs__pbx4_crispant', 'pbx1b_pbx4_crispant__vs__wik_ab', 'pbx4_crispant__vs__wik_ab']

Vector-space AUROC is now computed only on probes with support in both `inj_ctrl` and `wik_ab` within the anchor bin, with train-fold mean imputation rather than zero-imputation.

## vector_space
| variant                    |   requested_anchor |   time_bin_center |   auroc_obs |   auroc_null_mean |   n_supported_features |        pval |       qval |   n_inj_ctrl |   n_wik_ab |
|:---------------------------|-------------------:|------------------:|------------:|------------------:|-----------------------:|------------:|-----------:|-------------:|-----------:|
| all_pairs_raw              |                 26 |                26 |    0.333333 |          0.472222 |                      1 |   0.909091  |   0.909091 |           18 |         12 |
| all_pairs_shrunk           |                 26 |                26 |    0.5      |          0.5      |                      1 |   1         |   1        |           18 |         12 |
| direct_raw                 |                 26 |                26 |    0.333333 |          0.472222 |                      1 |   0.909091  |   0.909091 |           18 |         12 |
| direct_shrunk              |                 26 |                26 |    0.5      |          0.5      |                      1 |   1         |   1        |           18 |         12 |
| focal_vs_pbx1b_pbx4_raw    |                 26 |                26 |  nan        |        nan        |                      0 | nan         | nan        |           18 |         12 |
| focal_vs_pbx1b_pbx4_shrunk |                 26 |                26 |  nan        |        nan        |                      0 | nan         | nan        |           18 |         12 |
| all_pairs_raw              |                 54 |                54 |    0.733333 |          0.371667 |                      1 |   0.0909091 |   0.272727 |            5 |         12 |
| all_pairs_shrunk           |                 54 |                54 |    0.733333 |          0.371667 |                      1 |   0.0909091 |   0.272727 |            5 |         12 |
| direct_raw                 |                 54 |                54 |    0.733333 |          0.371667 |                      1 |   0.0909091 |   0.272727 |            5 |         12 |
| direct_shrunk              |                 54 |                54 |    0.733333 |          0.371667 |                      1 |   0.0909091 |   0.272727 |            5 |         12 |
| focal_vs_pbx1b_pbx4_raw    |                 54 |                54 |  nan        |        nan        |                      0 | nan         | nan        |            5 |         12 |
| focal_vs_pbx1b_pbx4_shrunk |                 54 |                54 |  nan        |        nan        |                      0 | nan         | nan        |            5 |         12 |
| all_pairs_raw              |                 78 |                78 |    0.441667 |          0.415    |                      1 |   0.454545  |   0.681818 |           12 |         10 |
| all_pairs_shrunk           |                 78 |                78 |    0.441667 |          0.415    |                      1 |   0.454545  |   0.681818 |           12 |         10 |
| direct_raw                 |                 78 |                78 |    0.441667 |          0.415    |                      1 |   0.454545  |   0.681818 |           12 |         10 |
| direct_shrunk              |                 78 |                78 |    0.441667 |          0.415    |                      1 |   0.454545  |   0.681818 |           12 |         10 |
| focal_vs_pbx1b_pbx4_raw    |                 78 |                78 |  nan        |        nan        |                      0 | nan         | nan        |           12 |         10 |
| focal_vs_pbx1b_pbx4_shrunk |                 78 |                78 |  nan        |        nan        |                      0 | nan         | nan        |           12 |         10 |

## aligned_umap_init
| variant                    |   requested_anchor |   time_bin_center |   auroc_obs |   auroc_null_mean |   n_supported_features |        pval |       qval |   n_inj_ctrl |   n_wik_ab |
|:---------------------------|-------------------:|------------------:|------------:|------------------:|-----------------------:|------------:|-----------:|-------------:|-----------:|
| all_pairs_raw              |                 26 |                26 |    0.333333 |          0.469444 |                      1 |   0.727273  |   0.909091 |           18 |         12 |
| all_pairs_shrunk           |                 26 |                26 |    0.231481 |          0.548148 |                      1 |   1         |   1        |           18 |         12 |
| direct_raw                 |                 26 |                26 |    0.375    |          0.462963 |                      1 |   0.727273  |   0.909091 |           18 |         12 |
| direct_shrunk              |                 26 |                26 |    0.324074 |          0.525463 |                      1 |   0.909091  |   0.909091 |           18 |         12 |
| focal_vs_pbx1b_pbx4_raw    |                 26 |                26 |  nan        |        nan        |                      0 | nan         | nan        |           18 |         12 |
| focal_vs_pbx1b_pbx4_shrunk |                 26 |                26 |  nan        |        nan        |                      0 | nan         | nan        |           18 |         12 |
| all_pairs_raw              |                 54 |                54 |    0.683333 |          0.48     |                      1 |   0.272727  |   0.818182 |            5 |         12 |
| all_pairs_shrunk           |                 54 |                54 |    0.7      |          0.44     |                      1 |   0.272727  |   0.818182 |            5 |         12 |
| direct_raw                 |                 54 |                54 |    0.816667 |          0.415    |                      1 |   0.0909091 |   0.272727 |            5 |         12 |
| direct_shrunk              |                 54 |                54 |    0.766667 |          0.428333 |                      1 |   0.272727  |   0.818182 |            5 |         12 |
| focal_vs_pbx1b_pbx4_raw    |                 54 |                54 |  nan        |        nan        |                      0 | nan         | nan        |            5 |         12 |
| focal_vs_pbx1b_pbx4_shrunk |                 54 |                54 |  nan        |        nan        |                      0 | nan         | nan        |            5 |         12 |
| all_pairs_raw              |                 78 |                78 |    0.275    |          0.4725   |                      1 |   0.909091  |   0.909091 |           12 |         10 |
| all_pairs_shrunk           |                 78 |                78 |    0.408333 |          0.471667 |                      1 |   0.727273  |   1        |           12 |         10 |
| direct_raw                 |                 78 |                78 |    0.316667 |          0.455833 |                      1 |   0.909091  |   0.909091 |           12 |         10 |
| direct_shrunk              |                 78 |                78 |    0.25     |          0.514167 |                      1 |   0.909091  |   0.909091 |           12 |         10 |
| focal_vs_pbx1b_pbx4_raw    |                 78 |                78 |  nan        |        nan        |                      0 | nan         | nan        |           12 |         10 |
| focal_vs_pbx1b_pbx4_shrunk |                 78 |                78 |  nan        |        nan        |                      0 | nan         | nan        |           12 |         10 |
