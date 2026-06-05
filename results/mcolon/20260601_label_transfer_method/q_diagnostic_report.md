# Q Failure Diagnostic Report

## True-Label Rank

Pooled across HPF bins, true-label rank/coverage by method:

| method       | true_label    |    n |   argmax_acc |   coverage |   mean_set_size |   mean_q_true |   mean_true_label_rank |   rank1_rate |   rank2_or_better_rate |   rank4_rate |
|:-------------|:--------------|-----:|-------------:|-----------:|----------------:|--------------:|-----------------------:|-------------:|-----------------------:|-------------:|
| knn_q        | High_to_Low   | 2540 |        0.779 |      0.908 |           2.279 |         0.689 |                  1.406 |        0.779 |                  0.869 |        0.054 |
| knn_q        | Intermediate  |  723 |        0.133 |      0.52  |           2.609 |         0.167 |                  2.643 |        0.133 |                  0.414 |        0.189 |
| knn_q        | Low_to_High   | 2295 |        0.124 |      0.816 |           2.366 |         0.192 |                  2.211 |        0.124 |                  0.744 |        0.079 |
| knn_q        | Not Penetrant | 7504 |        0.895 |      0.962 |           2.305 |         0.799 |                  1.166 |        0.895 |                  0.955 |        0.015 |
| multiclass_q | High_to_Low   | 2540 |        0.754 |      0.925 |           2.168 |         0.622 |                  1.399 |        0.754 |                  0.885 |        0.038 |
| multiclass_q | Intermediate  |  723 |        0.272 |      0.781 |           2.675 |         0.277 |                  2.256 |        0.272 |                  0.625 |        0.154 |
| multiclass_q | Low_to_High   | 2295 |        0.311 |      0.793 |           2.531 |         0.328 |                  2.072 |        0.311 |                  0.706 |        0.088 |
| multiclass_q | Not Penetrant | 7504 |        0.778 |      0.934 |           2.244 |         0.628 |                  1.349 |        0.778 |                  0.909 |        0.036 |

## Neighbor Geometry

Mean first same-label neighbor rank and top-K same-label fraction:

| true_label    |    n |   rank_first_true_label_neighbor |   frac_true_label_neighbors_top15 |   frac_true_label_neighbors_top50 |   frac_true_label_neighbors_top200 |   frac_np_neighbors_top15 |
|:--------------|-----:|---------------------------------:|----------------------------------:|----------------------------------:|-----------------------------------:|--------------------------:|
| High_to_Low   | 2540 |                            4.23  |                             0.693 |                             0.645 |                              0.564 |                     0.118 |
| Intermediate  |  723 |                           22.305 |                             0.119 |                             0.108 |                              0.086 |                     0.454 |
| Low_to_High   | 2295 |                           11.929 |                             0.216 |                             0.208 |                              0.2   |                     0.567 |
| Not Penetrant | 7504 |                            1.689 |                             0.821 |                             0.794 |                              0.761 |                     0.821 |

## Rescue Groups

Counts for rare/mixed labels:

| true_label   | rescue_group               |    n |
|:-------------|:---------------------------|-----:|
| Intermediate | both_wrong                 |  466 |
| Intermediate | knn_wrong_multiclass_right |  161 |
| Intermediate | knn_right_multiclass_wrong |   60 |
| Intermediate | both_right                 |   36 |
| Low_to_High  | both_wrong                 | 1397 |
| Low_to_High  | knn_wrong_multiclass_right |  613 |
| Low_to_High  | knn_right_multiclass_wrong |  185 |
| Low_to_High  | both_right                 |  100 |

## Set Composition

Top conformal sets by method:

| method       | prediction_set                         |    n |   frac |
|:-------------|:---------------------------------------|-----:|-------:|
| knn_q        | Low_to_High|Not Penetrant              | 5784 |  0.443 |
| knn_q        | Low_to_High|High_to_Low|Not Penetrant  | 2600 |  0.199 |
| knn_q        | Low_to_High|High_to_Low                | 1494 |  0.114 |
| knn_q        | Low_to_High|Intermediate|Not Penetrant |  707 |  0.054 |
| knn_q        | Low_to_High|High_to_Low|Intermediate   |  556 |  0.043 |
| knn_q        | High_to_Low|Not Penetrant              |  508 |  0.039 |
| knn_q        | High_to_Low|Intermediate|Not Penetrant |  491 |  0.038 |
| knn_q        | Intermediate|Not Penetrant             |  432 |  0.033 |
| multiclass_q | Low_to_High|Not Penetrant              | 3438 |  0.263 |
| multiclass_q | Low_to_High|Intermediate|Not Penetrant | 2518 |  0.193 |
| multiclass_q | Low_to_High|High_to_Low|Not Penetrant  | 1366 |  0.105 |
| multiclass_q | Low_to_High|High_to_Low|Intermediate   | 1107 |  0.085 |
| multiclass_q | Not Penetrant                          |  925 |  0.071 |
| multiclass_q | High_to_Low|Intermediate|Not Penetrant |  665 |  0.051 |
| multiclass_q | High_to_Low                            |  638 |  0.049 |
| multiclass_q | Low_to_High|High_to_Low                |  618 |  0.047 |
