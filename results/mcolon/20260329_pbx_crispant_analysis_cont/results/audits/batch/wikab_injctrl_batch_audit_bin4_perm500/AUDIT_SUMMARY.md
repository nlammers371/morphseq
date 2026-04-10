# Wik_ab vs Inj_ctrl Batch Audit

## Support
| genotype   |   experiment_id |   n_rows |   n_embryos |   n_time_bins |
|:-----------|----------------:|---------:|------------:|--------------:|
| inj_ctrl   |        20260304 |      139 |          20 |            14 |
| inj_ctrl   |        20260306 |      144 |          12 |            14 |
| wik_ab     |        20260304 |      157 |          14 |            14 |
| wik_ab     |        20260306 |      134 |          12 |            14 |

## Aligned row counts
| representation   | stage             |   native_rows |   native_embryos |   aligned_rows |   aligned_embryos |
|:-----------------|:------------------|--------------:|-----------------:|---------------:|------------------:|
| pairwise_raw     | vector_space      |           574 |               58 |            574 |                58 |
| pairwise_raw     | aligned_umap_init |           574 |               58 |            574 |                58 |
| pairwise_raw     | condensed_final   |           574 |               58 |            574 |                58 |
| pairwise_shrunk  | vector_space      |           574 |               58 |            574 |                58 |
| pairwise_shrunk  | aligned_umap_init |           574 |               58 |            574 |                58 |
| pairwise_shrunk  | condensed_final   |           574 |               58 |            574 |                58 |
| multiclass       | vector_space      |           574 |               58 |            574 |                58 |
| multiclass       | aligned_umap_init |           574 |               58 |            574 |                58 |
| multiclass       | condensed_final   |           574 |               58 |            574 |                58 |

## Pooled batch predictability
| representation   | stage             |   n_rows |   auroc_obs |   auroc_null_mean |       pval |       qval |
|:-----------------|:------------------|---------:|------------:|------------------:|-----------:|-----------:|
| multiclass       | aligned_umap_init |      574 |    0.826269 |          0.497332 | 0.00199601 | 0.00199601 |
| multiclass       | condensed_final   |      574 |    0.905065 |          0.497921 | 0.00199601 | 0.00199601 |
| multiclass       | vector_space      |      574 |    0.847317 |          0.495486 | 0.00199601 | 0.00199601 |
| pairwise_raw     | aligned_umap_init |      574 |    0.810944 |          0.498075 | 0.00199601 | 0.00199601 |
| pairwise_raw     | condensed_final   |      574 |    0.831032 |          0.497808 | 0.00199601 | 0.00199601 |
| pairwise_raw     | vector_space      |      574 |    0.899511 |          0.499102 | 0.00199601 | 0.00199601 |
| pairwise_shrunk  | aligned_umap_init |      574 |    0.900265 |          0.497199 | 0.00199601 | 0.00199601 |
| pairwise_shrunk  | condensed_final   |      574 |    0.899013 |          0.49777  | 0.00199601 | 0.00199601 |
| pairwise_shrunk  | vector_space      |      574 |    0.919988 |          0.49864  | 0.00199601 | 0.00199601 |

## Pooled kNN mixing
| representation   | stage             |   same_experiment_fraction |   k_eff |
|:-----------------|:------------------|---------------------------:|--------:|
| multiclass       | aligned_umap_init |                   0.7      |      10 |
| multiclass       | condensed_final   |                   0.812544 |      10 |
| multiclass       | vector_space      |                   0.736934 |      10 |
| pairwise_raw     | aligned_umap_init |                   0.793206 |      10 |
| pairwise_raw     | condensed_final   |                   0.845296 |      10 |
| pairwise_raw     | vector_space      |                   0.78223  |      10 |
| pairwise_shrunk  | aligned_umap_init |                   0.808188 |      10 |
| pairwise_shrunk  | condensed_final   |                   0.846341 |      10 |
| pairwise_shrunk  | vector_space      |                   0.83223  |      10 |

## Decision summary
1. Is batch structure already visible in the saved vector spaces? Yes.
2. Does AlignedUMAP initialization increase experiment separation? No.
3. Does condensation further increase experiment separation? Yes.

Evidence:
- multiclass vector space AUROC=0.847, q=0.002
- multiclass final AUROC increase 0.079
- pairwise_raw vector space AUROC=0.900, q=0.002
- pairwise_shrunk vector space AUROC=0.920, q=0.002

Recommendation:
- tree work still blocked