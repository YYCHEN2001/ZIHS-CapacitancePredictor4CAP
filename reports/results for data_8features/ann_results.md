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
| R²       | 0.954316 |  0.807105 |
| MAE      | 5.12492  | 10.5433   |
| MAPE     | 5.57825  | 12.4824   |
| RMSE     | 8.64776  | 18.1889   |