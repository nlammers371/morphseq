# Wik_ab vs Inj_ctrl Focal Probe Zoom

Descriptive-only audit: no permutations, no p-values.

## Raw focal-reference probes
| table   |   requested_anchor |   time_bin_center | probe                                   |   n_inj_ctrl |   n_wik_ab |     auroc |   inj_mean |   wik_mean |   inj_median |   wik_median |   mean_diff_inj_minus_wik |
|:--------|-------------------:|------------------:|:----------------------------------------|-------------:|-----------:|----------:|-----------:|-----------:|-------------:|-------------:|--------------------------:|
| raw     |                 26 |                26 | inj_ctrl__vs__pbx1b_pbx4_crispant       |           18 |         12 | 0.111111  |   0.235011 |   0        |     0.347044 |     0        |                  0.235011 |
| raw     |                 26 |                26 | pbx1b_crispant__vs__pbx1b_pbx4_crispant |           18 |         12 | 0.5       |   0        |   0        |     0        |     0        |                  0        |
| raw     |                 26 |                26 | pbx1b_pbx4_crispant__vs__pbx4_crispant  |           18 |         12 | 0.5       |   0        |   0        |     0        |     0        |                  0        |
| raw     |                 26 |                26 | pbx1b_pbx4_crispant__vs__wik_ab         |           18 |         12 | 0.25      |   0        |  -0.192902 |     0        |    -0.300716 |                  0.192902 |
| raw     |                 54 |                54 | inj_ctrl__vs__pbx1b_pbx4_crispant       |            5 |         12 | 0         |   0.284829 |   0        |     0.227051 |     0        |                  0.284829 |
| raw     |                 54 |                54 | pbx1b_crispant__vs__pbx1b_pbx4_crispant |            5 |         12 | 0.5       |   0        |   0        |     0        |     0        |                  0        |
| raw     |                 54 |                54 | pbx1b_pbx4_crispant__vs__pbx4_crispant  |            5 |         12 | 0.5       |   0        |   0        |     0        |     0        |                  0        |
| raw     |                 54 |                54 | pbx1b_pbx4_crispant__vs__wik_ab         |            5 |         12 | 0.333333  |   0        |  -0.149859 |     0        |    -0.211087 |                  0.149859 |
| raw     |                 78 |                78 | inj_ctrl__vs__pbx1b_pbx4_crispant       |           12 |         10 | 0.0833333 |   0.401478 |   0        |     0.456864 |     0        |                  0.401478 |
| raw     |                 78 |                78 | pbx1b_crispant__vs__pbx1b_pbx4_crispant |           12 |         10 | 0.5       |   0        |   0        |     0        |     0        |                  0        |
| raw     |                 78 |                78 | pbx1b_pbx4_crispant__vs__pbx4_crispant  |           12 |         10 | 0.5       |   0        |   0        |     0        |     0        |                  0        |
| raw     |                 78 |                78 | pbx1b_pbx4_crispant__vs__wik_ab         |           12 |         10 | 0         |   0        |  -0.486989 |     0        |    -0.510404 |                  0.486989 |

## Shrunk focal-reference probes
| table   |   requested_anchor |   time_bin_center | probe                                   |   n_inj_ctrl |   n_wik_ab |     auroc |   inj_mean |   wik_mean |   inj_median |   wik_median |   mean_diff_inj_minus_wik |
|:--------|-------------------:|------------------:|:----------------------------------------|-------------:|-----------:|----------:|-----------:|-----------:|-------------:|-------------:|--------------------------:|
| shrunk  |                 26 |                26 | inj_ctrl__vs__pbx1b_pbx4_crispant       |           18 |         12 | 0.111111  |   0.166397 |  0         |     0.245721 |     0        |                 0.166397  |
| shrunk  |                 26 |                26 | pbx1b_crispant__vs__pbx1b_pbx4_crispant |           18 |         12 | 0.5       |   0        |  0         |     0        |     0        |                 0         |
| shrunk  |                 26 |                26 | pbx1b_pbx4_crispant__vs__pbx4_crispant  |           18 |         12 | 0.5       |   0        |  0         |     0        |     0        |                 0         |
| shrunk  |                 26 |                26 | pbx1b_pbx4_crispant__vs__wik_ab         |           18 |         12 | 0.25      |   0        | -0.0963223 |     0        |    -0.150157 |                 0.0963223 |
| shrunk  |                 54 |                54 | inj_ctrl__vs__pbx1b_pbx4_crispant       |            5 |         12 | 0         |   0.245653 |  0         |     0.195822 |     0        |                 0.245653  |
| shrunk  |                 54 |                54 | pbx1b_crispant__vs__pbx1b_pbx4_crispant |            5 |         12 | 0.5       |   0        |  0         |     0        |     0        |                 0         |
| shrunk  |                 54 |                54 | pbx1b_pbx4_crispant__vs__pbx4_crispant  |            5 |         12 | 0.5       |   0        |  0         |     0        |     0        |                 0         |
| shrunk  |                 54 |                54 | pbx1b_pbx4_crispant__vs__wik_ab         |            5 |         12 | 0.333333  |   0        | -0.0906113 |     0        |    -0.127632 |                 0.0906113 |
| shrunk  |                 78 |                78 | inj_ctrl__vs__pbx1b_pbx4_crispant       |           12 |         10 | 0.0833333 |   0.386229 |  0         |     0.439512 |     0        |                 0.386229  |
| shrunk  |                 78 |                78 | pbx1b_crispant__vs__pbx1b_pbx4_crispant |           12 |         10 | 0.5       |   0        |  0         |     0        |     0        |                 0         |
| shrunk  |                 78 |                78 | pbx1b_pbx4_crispant__vs__pbx4_crispant  |           12 |         10 | 0.5       |   0        |  0         |     0        |     0        |                 0         |
| shrunk  |                 78 |                78 | pbx1b_pbx4_crispant__vs__wik_ab         |           12 |         10 | 0         |   0        | -0.486989  |     0        |    -0.510404 |                 0.486989  |