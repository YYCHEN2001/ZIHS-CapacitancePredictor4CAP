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
| R²       |  0.9276  |  0.810598 |
| MAE      |  7.24857 | 12.2135   |
| MAPE     |  7.54883 | 14.5564   |
| RMSE     | 10.8866  | 18.0235   |