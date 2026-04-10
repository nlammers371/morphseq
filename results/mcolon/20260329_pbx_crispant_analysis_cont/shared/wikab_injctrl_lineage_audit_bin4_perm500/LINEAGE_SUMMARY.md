# Wik_ab vs Inj_ctrl Lineage Audit

## Pooled batch AUROC by stage
| stage                  |   auroc_obs |   auroc_null_mean |       pval |       qval |
|:-----------------------|------------:|------------------:|-----------:|-----------:|
| build06_embeddings     |    0.994669 |          0.499984 | 0.00199601 | 0.00199601 |
| binned_embeddings      |    0.98408  |          0.498969 | 0.00199601 | 0.00199601 |
| m_raw_injctrl_vs_wikab |    0.525508 |          0.497799 | 0.171657   | 0.199601   |
| raw_coordinates        |    0.899511 |          0.499102 | 0.00199601 | 0.00199601 |
| shrunk_coordinates     |    0.919988 |          0.49864  | 0.00199601 | 0.00199601 |

First stage with strong batch signal: `build06_embeddings`

## Interpretation
- The experiment separation is already present in the original build06 VAE embeddings, before classifier binning or pairwise coordinate assembly.

## Top probes by shrinkage-induced AUROC increase
| probe                                   |      raw |   shrunk |   delta_shrunk_minus_raw |
|:----------------------------------------|---------:|---------:|-------------------------:|
| inj_ctrl__vs__pbx1b_crispant            | 0.506854 | 0.558733 |                0.0518788 |
| inj_ctrl__vs__wik_ab                    | 0.492465 | 0.51592  |                0.0234542 |
| pbx1b_pbx4_crispant__vs__wik_ab         | 0.607519 | 0.628713 |                0.0211939 |
| inj_ctrl__vs__pbx4_crispant             | 0.620905 | 0.64177  |                0.0208657 |
| pbx4_crispant__vs__wik_ab               | 0.578134 | 0.598216 |                0.0200819 |
| inj_ctrl__vs__pbx1b_pbx4_crispant       | 0.631745 | 0.650411 |                0.0186661 |
| pbx1b_crispant__vs__pbx1b_pbx4_crispant | 0.497673 | 0.497673 |                0         |
| pbx1b_crispant__vs__pbx4_crispant       | 0.497673 | 0.497673 |                0         |
| pbx1b_pbx4_crispant__vs__pbx4_crispant  | 0.497673 | 0.497673 |                0         |
| pbx1b_crispant__vs__wik_ab              | 0.527349 | 0.524973 |               -0.0023758 |