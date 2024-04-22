# Model Parameters: 
|    |    C |   cache_size |   coef0 |   degree |   epsilon | gamma   | kernel   |   max_iter | shrinking   |   tol | verbose   |
|---:|-----:|-------------:|--------:|---------:|----------:|:--------|:---------|-----------:|:------------|------:|:----------|
|  0 | 6.97 |          200 |    8.11 |        2 |      0.46 | scale   | poly     |         -1 | True        | 0.001 | False     |

# Results: 
|    | Metric   |   Train Set |   Test Set |
|---:|:---------|------------:|-----------:|
|  0 | R2       |    0.563013 |   0.545365 |
|  1 | MAE      |   19.4238   |  20.6905   |
|  2 | MAPE     |   24.8967   |  26.6938   |
|  3 | RMSE     |   26.7459   |  27.924    |

# Results of 10-fold: 
|    | Fold    |       R2 |     MAE |     MAPE |    RMSE |
|---:|:--------|---------:|--------:|---------:|--------:|
|  0 | 1       | 0.33448  | 22.2297 | 0.276868 | 32.1289 |
|  1 | 2       | 0.659699 | 18.2134 | 0.237321 | 23.7161 |
|  2 | 3       | 0.395026 | 27.0873 | 0.337511 | 33.9806 |
|  3 | 4       | 0.535422 | 18.0219 | 0.254749 | 22.3177 |
|  4 | 5       | 0.572925 | 19.6817 | 0.234557 | 26.3695 |
|  5 | 6       | 0.566992 | 21.8116 | 0.311623 | 29.4463 |
|  6 | 7       | 0.683456 | 22.5311 | 0.318152 | 27.3449 |
|  7 | 8       | 0.382831 | 26.6518 | 0.319795 | 34.1348 |
|  8 | 9       | 0.472027 | 20.0821 | 0.287408 | 26.0705 |
|  9 | 10      | 0.358115 | 16.8564 | 0.173087 | 23.9365 |
| 10 | Average | 0.496097 | 21.3167 | 0.275107 | 27.9446 |