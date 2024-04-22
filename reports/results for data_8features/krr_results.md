# Model Parameters: 
|    |   alpha |   coef0 |   degree |   gamma | kernel     | kernel_params   |
|---:|--------:|--------:|---------:|--------:|:-----------|:----------------|
|  0 |    1.04 |    6.49 |        2 |    1.05 | polynomial |                 |

# Results: 
|    | Metric   |   Train Set |   Test Set |
|---:|:---------|------------:|-----------:|
|  0 | R2       |    0.650618 |   0.605253 |
|  1 | MAE      |   17.8991   |  19.5648   |
|  2 | MAPE     |   23.715    |  25.1317   |
|  3 | RMSE     |   23.9152   |  26.0199   |

# Results of 10-fold: 
|    | Fold    |       R2 |     MAE |     MAPE |    RMSE |
|---:|:--------|---------:|--------:|---------:|--------:|
|  0 | 1       | 0.36786  | 20.6847 | 0.307173 | 31.3128 |
|  1 | 2       | 0.732403 | 17.7267 | 0.233627 | 21.0306 |
|  2 | 3       | 0.524248 | 23.023  | 0.309611 | 30.1337 |
|  3 | 4       | 0.570826 | 17.7376 | 0.240008 | 21.4505 |
|  4 | 5       | 0.688392 | 17.9553 | 0.229004 | 22.5245 |
|  5 | 6       | 0.640093 | 20.3291 | 0.292167 | 26.8459 |
|  6 | 7       | 0.650339 | 22.0361 | 0.342962 | 28.7398 |
|  7 | 8       | 0.468343 | 24.1137 | 0.289851 | 31.6819 |
|  8 | 9       | 0.568607 | 18.7301 | 0.279594 | 23.5657 |
|  9 | 10      | 0.330538 | 17.6251 | 0.1891   | 24.4453 |
| 10 | Average | 0.554165 | 19.9961 | 0.27131  | 26.1731 |