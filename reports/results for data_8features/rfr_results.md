# Model Parameters: 
|    | bootstrap   |   ccp_alpha | criterion     |   max_depth |   max_features | max_leaf_nodes   | max_samples   |   min_impurity_decrease |   min_samples_leaf |   min_samples_split |   min_weight_fraction_leaf | monotonic_cst   |   n_estimators | n_jobs   | oob_score   |   random_state |   verbose | warm_start   |
|---:|:------------|------------:|:--------------|------------:|---------------:|:-----------------|:--------------|------------------------:|-------------------:|--------------------:|---------------------------:|:----------------|---------------:|:---------|:------------|---------------:|----------:|:-------------|
|  0 | True        |           0 | squared_error |          18 |              1 |                  |               |                       0 |                  1 |                   2 |                          0 |                 |            200 |          | False       |             21 |         0 | False        |

# Results: 
|    | Metric   |   Train Set |   Test Set |
|---:|:---------|------------:|-----------:|
|  0 | R2       |    0.981223 |   0.842436 |
|  1 | MAE      |    3.60889  |  10.0585   |
|  2 | MAPE     |    4.49958  |  13.2267   |
|  3 | RMSE     |    5.54422  |  16.4389   |

# Results of 10-fold: 
|    | Fold    |       R2 |      MAE |     MAPE |    RMSE |
|---:|:--------|---------:|---------:|---------:|--------:|
|  0 | 1       | 0.780216 | 11.2217  | 0.117221 | 18.4635 |
|  1 | 2       | 0.926831 |  8.24805 | 0.111987 | 10.997  |
|  2 | 3       | 0.864031 |  9.90529 | 0.105653 | 16.1095 |
|  3 | 4       | 0.895639 |  8.25312 | 0.116958 | 10.5776 |
|  4 | 5       | 0.918464 |  7.30973 | 0.075032 | 11.5219 |
|  5 | 6       | 0.839899 | 11.7412  | 0.162913 | 17.9052 |
|  6 | 7       | 0.88391  | 11.7817  | 0.165028 | 16.5599 |
|  7 | 8       | 0.842646 | 13.1297  | 0.174381 | 17.2359 |
|  8 | 9       | 0.847831 |  9.68942 | 0.137152 | 13.9961 |
|  9 | 10      | 0.810929 |  8.98676 | 0.101587 | 12.991  |
| 10 | Average | 0.86104  | 10.0267  | 0.126791 | 14.6358 |