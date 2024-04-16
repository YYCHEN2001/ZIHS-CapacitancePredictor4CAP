import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error


def calculate_metrics(y_true, y_pred):
    """
    Calculate and return model evaluation metrics.
    """
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)*100
    rmse = root_mean_squared_error(y_true, y_pred)
    return r2, mae, mape, rmse


def train_evaluate(model, x_train_scaled, y_train, x_test_scaled, y_test):
    """
    Train the model and evaluate it on both training and test sets.
    """
    # Train the model
    model.fit(x_train_scaled, y_train)

    # Predictions
    y_pred_train = model.predict(x_train_scaled)
    y_pred_test = model.predict(x_test_scaled)

    # Calculate metrics
    metrics_train = calculate_metrics(y_train, y_pred_train)
    metrics_test = calculate_metrics(y_test, y_pred_test)

    # Prepare and display results
    results = pd.DataFrame({
        'Metric': ['R2', 'MAE', 'MAPE', 'RMSE'],
        'Train Set': metrics_train,
        'Test Set': metrics_test
    })

    return results


def plot_actual_vs_predicted(y_train, y_pred_train, y_test, y_pred_test, savepath=None):
    """
    Plot the actual vs predicted values for both training and test sets,
    and plot y=x as the fit line.
    """
    # Set the font
    matplotlib.rcParams['font.family'] = 'Times New Roman'
    matplotlib.rcParams['font.size'] = 32
    matplotlib.rcParams['font.weight'] = 'bold'

    plt.figure(figsize=(10, 10), linewidth=3)
    plt.scatter(y_train, y_pred_train, color='green', label='Train', s=25, alpha=0.6)
    plt.scatter(y_test, y_pred_test, color='red', label='Test', s=25, alpha=0.6)

    # 绘制y=x的线
    y_combined = np.concatenate([y_train, y_pred_train, y_test, y_pred_test])
    min_val, max_val = y_combined.min(), y_combined.max()

    # 计算边缘缓冲
    padding = (max_val - min_val) * 0.05
    padded_min, padded_max = min_val - padding, max_val + padding

    font = {'family': 'Times New Roman',
            'color': 'black',
            'weight': 'bold',
            'size': 32,
            }

    plt.plot([padded_min, padded_max], [padded_min, padded_max], 'k--', lw=3, label='Regression Line')

    plt.title('Actual vs Predicted Values', fontdict=font)
    plt.xlabel('Actual Values', fontdict=font)
    plt.ylabel('Predicted Values', fontdict=font)
    plt.legend(prop={'family': 'Times New Roman', 'size': 32}, frameon=False, loc='upper left', facecolor='none')

    # 调整x轴和y轴刻度字体大小
    plt.tick_params(axis='both', which='major')

    plt.gca().spines['top'].set_visible(True)
    plt.gca().spines['top'].set_linewidth(2.5)
    plt.gca().spines['bottom'].set_visible(True)
    plt.gca().spines['bottom'].set_linewidth(2.5)
    plt.gca().spines['left'].set_visible(True)
    plt.gca().spines['left'].set_linewidth(2.5)
    plt.gca().spines['right'].set_visible(True)
    plt.gca().spines['right'].set_linewidth(2.5)

    # 设置坐标轴范围使得横纵坐标一致
    plt.axis('equal')
    plt.xlim([padded_min, padded_max])
    plt.ylim([padded_min, padded_max])
    plt.savefig(savepath, bbox_inches='tight', transparent=True)
    plt.show()
