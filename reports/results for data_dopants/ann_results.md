# Model Parameters:
| Layer      | Output Shape   | Activation   |
|:-----------|:---------------|:-------------|
| InputLayer | N/A            | N/A          |
| Dense      | 128            | relu         |
| Dense      | 64             | relu         |
| Dense      | 32             | relu         |
| Dense      | 16             | relu         |
| Dense      | 1              | linear       |

# Performance Metrics:
| Metric   |    Train |      Test |
|:---------|---------:|----------:|
| R²       | 0.941293 |  0.828281 |
| MAE      | 5.94008  | 10.9777   |
| MAPE     | 6.33588  | 12.9544   |
| RMSE     | 9.80318  | 17.1615   |