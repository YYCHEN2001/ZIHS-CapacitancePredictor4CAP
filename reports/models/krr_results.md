# Model Parameters: 
|    |   alpha |   coef0 |   degree |   gamma | kernel     | kernel_params   |
|---:|--------:|--------:|---------:|--------:|:-----------|:----------------|
|  0 |     0.8 |     7.7 |        2 |     0.1 | polynomial |                 |

# Results: 
|    | Metric   |   Train Set |   Test Set |
|---:|:---------|------------:|-----------:|
|  0 | R2       |    0.788259 |   0.673246 |
|  1 | MAE      |   13.6221   |  17.0513   |
|  2 | MAPE     |   16.9561   |  21.7902   |
|  3 | RMSE     |   18.6177   |  23.6731   |

# Results of 10-fold: 
|    | Fold    |       R2 |     MAE |     MAPE |    RMSE |
|---:|:--------|---------:|--------:|---------:|--------:|
|  0 | 1       | 0.554541 | 17.8891 | 0.232093 | 26.2857 |
|  1 | 2       | 0.821405 | 13.445  | 0.188115 | 17.1809 |
|  2 | 3       | 0.633923 | 19.7378 | 0.245438 | 26.4331 |
|  3 | 4       | 0.711973 | 14.0895 | 0.186452 | 17.5726 |
|  4 | 5       | 0.747494 | 16.0977 | 0.195935 | 20.2762 |
|  5 | 6       | 0.788838 | 14.9421 | 0.208206 | 20.5632 |
|  6 | 7       | 0.747964 | 18.1319 | 0.231956 | 24.4    |
|  7 | 8       | 0.696406 | 17.3477 | 0.183484 | 23.941  |
|  8 | 9       | 0.708766 | 14.6118 | 0.207293 | 19.3626 |
|  9 | 10      | 0.472498 | 15.7092 | 0.166809 | 21.6992 |
| 10 | Average | 0.688381 | 16.2002 | 0.204578 | 21.7715 |