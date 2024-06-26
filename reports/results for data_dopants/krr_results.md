# Model Parameters: 
|    |   alpha |   coef0 |   degree |   gamma | kernel     | kernel_params   |
|---:|--------:|--------:|---------:|--------:|:-----------|:----------------|
|  0 |    1.04 |    1.52 |        1 |    1.02 | polynomial |                 |

# Results: 
|    | Metric   |   Train Set |   Test Set |
|---:|:---------|------------:|-----------:|
|  0 | R2       |    0.435135 |    0.44676 |
|  1 | MAE      |   22.8948   |   23.183   |
|  2 | MAPE     |   31.8184   |   31.754   |
|  3 | RMSE     |   30.4085   |   30.8036  |

# Results of 10-fold: 
|    | Fold    |        R2 |     MAE |     MAPE |    RMSE |
|---:|:--------|----------:|--------:|---------:|--------:|
|  0 | 1       | 0.0445803 | 25.8043 | 0.400129 | 38.4957 |
|  1 | 2       | 0.334563  | 24.6001 | 0.396727 | 33.1639 |
|  2 | 3       | 0.253925  | 29.0724 | 0.345082 | 37.7358 |
|  3 | 4       | 0.474009  | 19.544  | 0.281277 | 23.747  |
|  4 | 5       | 0.539988  | 20.2294 | 0.237297 | 27.3675 |
|  5 | 6       | 0.308137  | 28.193  | 0.465696 | 37.2214 |
|  6 | 7       | 0.590232  | 25.2458 | 0.382415 | 31.112  |
|  7 | 8       | 0.312263  | 28.5143 | 0.355359 | 36.0335 |
|  8 | 9       | 0.320652  | 21.6643 | 0.338396 | 29.5726 |
|  9 | 10      | 0.302424  | 19.1387 | 0.198129 | 24.9533 |
| 10 | Average | 0.348077  | 24.2007 | 0.340051 | 31.9403 |