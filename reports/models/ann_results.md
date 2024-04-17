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
| R²       | 0.943204 |  0.859937 |
| MAE      | 5.80827  |  9.54582  |
| MAPE     | 6.00833  | 10.7031   |
| RMSE     | 9.64234  | 15.4992   |