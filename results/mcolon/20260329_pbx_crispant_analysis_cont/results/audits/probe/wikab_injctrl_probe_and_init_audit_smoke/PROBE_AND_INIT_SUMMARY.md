# Wik_ab vs Inj_ctrl Probe And Init Audit

Late-time attribution target: 78.0 hpf

## Full-vector late-bin AUROC
| stage              |   time_bin_center |   n_rows |   n_inj_ctrl |   n_wik_ab |   full_vector_auroc |   full_vector_null_mean |   full_vector_null_std |   full_vector_pval |
|:-------------------|------------------:|---------:|-------------:|-----------:|--------------------:|------------------------:|-----------------------:|-------------------:|
| raw_coordinates    |                78 |       22 |           12 |         10 |            0.983333 |                0.596667 |               0.112515 |           0.166667 |
| shrunk_coordinates |                78 |       22 |           12 |         10 |            0.983333 |                0.596667 |               0.112515 |           0.166667 |

## Top raw late-bin probes
| probe                                   |   single_probe_auroc |   delta_full_minus_leave_one_out | is_direct_probe   |
|:----------------------------------------|---------------------:|---------------------------------:|:------------------|
| pbx1b_pbx4_crispant__vs__wik_ab         |             1        |                       0.05       | False             |
| pbx4_crispant__vs__wik_ab               |             1        |                       0.0333333  | False             |
| inj_ctrl__vs__pbx1b_crispant            |             0.616667 |                       0.00833333 | False             |
| pbx1b_crispant__vs__wik_ab              |             0.8      |                       0          | False             |
| pbx1b_crispant__vs__pbx1b_pbx4_crispant |             0.466667 |                       0          | False             |
| pbx1b_crispant__vs__pbx4_crispant       |             0.466667 |                       0          | False             |
| pbx1b_pbx4_crispant__vs__pbx4_crispant  |             0.466667 |                       0          | False             |
| inj_ctrl__vs__wik_ab                    |             0.441667 |                       0          | True              |
| inj_ctrl__vs__pbx1b_pbx4_crispant       |             0.891667 |                      -0.00833333 | False             |
| inj_ctrl__vs__pbx4_crispant             |             0.916667 |                      -0.0166667  | False             |

## Top shrunk late-bin probes
| probe                                   |   single_probe_auroc |   delta_full_minus_leave_one_out | is_direct_probe   |
|:----------------------------------------|---------------------:|---------------------------------:|:------------------|
| pbx1b_pbx4_crispant__vs__wik_ab         |             1        |                       0.05       | False             |
| pbx4_crispant__vs__wik_ab               |             1        |                       0.0333333  | False             |
| inj_ctrl__vs__pbx1b_crispant            |             0.616667 |                       0.00833333 | False             |
| pbx1b_crispant__vs__wik_ab              |             0.8      |                       0          | False             |
| pbx1b_crispant__vs__pbx1b_pbx4_crispant |             0.466667 |                       0          | False             |
| pbx1b_crispant__vs__pbx4_crispant       |             0.466667 |                       0          | False             |
| pbx1b_pbx4_crispant__vs__pbx4_crispant  |             0.466667 |                       0          | False             |
| inj_ctrl__vs__wik_ab                    |             0.441667 |                       0          | True              |
| inj_ctrl__vs__pbx1b_pbx4_crispant       |             0.891667 |                      -0.00833333 | False             |
| inj_ctrl__vs__pbx4_crispant             |             0.916667 |                      -0.0166667  | False             |

## Controlled AlignedUMAP init comparison
| variant              |   requested_anchor |   time_bin_center |   auroc_obs |   auroc_null_mean |     pval |
|:---------------------|-------------------:|------------------:|------------:|------------------:|---------:|
| direct_raw_only      |                 26 |                26 |    0.37963  |          0.510185 | 1        |
| direct_raw_only      |                 54 |                54 |    0.766667 |          0.253333 | 0.166667 |
| direct_raw_only      |                 78 |                78 |    0.25     |          0.511667 | 1        |
| full_pairwise_raw    |                 26 |                26 |    0.944444 |          0.62963  | 0.166667 |
| full_pairwise_raw    |                 54 |                54 |    0.9      |          0.376667 | 0.166667 |
| full_pairwise_raw    |                 78 |                78 |    1        |          0.571667 | 0.166667 |
| direct_shrunk_only   |                 26 |                26 |    0.657407 |          0.383333 | 0.166667 |
| direct_shrunk_only   |                 54 |                54 |    0.7      |          0.39     | 0.166667 |
| direct_shrunk_only   |                 78 |                78 |    0.266667 |          0.546667 | 0.833333 |
| full_pairwise_shrunk |                 26 |                26 |    0.939815 |          0.62963  | 0.166667 |
| full_pairwise_shrunk |                 54 |                54 |    0.883333 |          0.36     | 0.166667 |
| full_pairwise_shrunk |                 78 |                78 |    1        |          0.566667 | 0.166667 |