# Wik_ab vs Inj_ctrl Lineage Audit

## Pooled batch AUROC by stage
| stage                  |   auroc_obs |   auroc_null_mean |      pval |      qval |
|:-----------------------|------------:|------------------:|----------:|----------:|
| build06_embeddings     |    0.994669 |          0.495786 | 0.0909091 | 0.0909091 |
| binned_embeddings      |    0.98408  |          0.513464 | 0.0909091 | 0.0909091 |
| m_raw_injctrl_vs_wikab |    0.525508 |          0.494465 | 0.181818  | 0.272727  |
| raw_coordinates        |    0.899511 |          0.499565 | 0.0909091 | 0.0909091 |
| shrunk_coordinates     |    0.919988 |          0.490532 | 0.0909091 | 0.0909091 |

No stage crossed the reporting threshold.

## Interpretation
- No stage crossed the current reporting threshold.

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