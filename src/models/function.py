import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error

def load_data_dopants(filepath):
    # 读取数据
    data = pd.read_csv(filepath)
    # 对分类变量进行独热编码
    data_encoded = pd.get_dummies(data, columns=['Electrolyte', 'Current collector'])
    # 删除索引列
    data_encoded = data_encoded.drop('Index', axis=1)

    return data_encoded


def dataset_split_10class(data_encoded):
    # 将目标值分成10个等级
    data_encoded['target_class'] = pd.qcut(data_encoded['target'], q=10, labels=False)
    x = data_encoded.drop(['target', 'target_class'], axis=1)
    y = data_encoded['target']
    stratify_column = data_encoded['target_class']

    # 拆分训练和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=0.3, random_state=21,
                                                        stratify=stratify_column)

    # 初始化标准化器
    scaler = StandardScaler()

    # 使用训练集拟合标准化器
    scaler.fit(x_train)

    # 使用拟合过的标准化器来转换训练集和测试集
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # 将Series转换为DataFrame，这是为了kfold_cv函数的输入
    x_train_scaled = pd.DataFrame(x_train_scaled, columns=x.columns)
    x_test_scaled = pd.DataFrame(x_test_scaled, columns=x.columns)

    return x_train_scaled, x_test_scaled, y_train, y_test

def calculate_metrics(y_true, y_pred):
    """
    Calculate and return actual vs pred fig for data_dopants metrics.
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
    y_pred_train = model.predict(x_train_scaled).ravel()
    y_pred_test = model.predict(x_test_scaled).ravel()

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


def plot_actual_vs_predicted(y_train, y_pred_train, y_test, y_pred_test, figpath=None):
    """
    Plot the actual vs predicted values for both training and test sets,
    and plot y=x as the fit line.
    """
    # 设置全局字体为Times New Roman，字号为32，字体粗细为粗体
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 32,
        'font.weight': 'bold',
        'figure.figsize': (10, 10)  # 设置图像尺寸为10x10英寸
    })

    # 绘制训练集和测试集的散点图
    plt.scatter(y_train, y_pred_train, color='blue', label='Train', s=50, alpha=0.5)
    plt.scatter(y_test, y_pred_test, color='red', label='Test', s=50, alpha=0.5)

    # 计算合并数据的最小值和最大值，用于设置坐标轴范围和绘制y=x线
    y_pred_train = y_pred_train.ravel()
    y_pred_test = y_pred_test.ravel()
    y_combined = np.concatenate([y_train, y_pred_train, y_test, y_pred_test])
    min_val, max_val = np.min(y_combined), np.max(y_combined)
    padding = (max_val - min_val) * 0.05
    padded_min, padded_max = min_val - padding, max_val + padding

    # 绘制y=x的虚线，线宽为3
    plt.plot([padded_min, padded_max], [padded_min, padded_max], 'k--', lw=3, label='Regression Line')

    # 设置标题和轴标签，明确指定加粗
    plt.title('Actual vs Predicted Values', fontweight='bold')
    plt.xlabel('Actual Values', fontweight='bold')
    plt.ylabel('Predicted Values', fontweight='bold')

    # 设置图例，无边框，位于左上角
    plt.legend(frameon=False, loc='upper left', fontsize=28)

    # 设置坐标轴为相同比例，并且坐标轴范围一致
    plt.axis('equal')
    plt.xlim([padded_min, padded_max])
    plt.ylim([padded_min, padded_max])

    # 设置刻度线的长度和粗细
    plt.tick_params(axis='both', which='major', length=10, width=2, labelsize=32)

    # 检查并统一X轴和Y轴的刻度
    # 可以通过设置两个轴的相同刻度，或者根据数据自动选择刻度
    x_ticks = np.arange(0, max(y_combined) + 1, 50)  # 可以根据数据范围调整
    y_ticks = np.arange(0, max(y_combined) + 1, 50)  # 使得X和Y轴的刻度间隔相同

    plt.xticks(x_ticks)
    plt.yticks(y_ticks)

    # 设置图形边界的宽度和可见性
    for spine in plt.gca().spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2.5)

    # 保存图像，背景透明，紧凑布局
    plt.savefig(figpath, bbox_inches='tight', transparent=True)
    plt.show()


def kfold_cv(model, x, y, n_splits=10, random_state=21):
    """
    Perform K-Fold Cross Validation.

    Parameters:
    - model: The regression model to be evaluated.
    - X: Features DataFrame.
    - y: Target Series.
    - n_splits: Number of folds. Default is 10.
    - random_state: Random state for reproducibility.

    Returns:
    - metrics_df: A DataFrame containing the metrics for each fold and the averages.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    metrics_df = pd.DataFrame(columns=['Fold', 'R2', 'MAE', 'MAPE', 'RMSE'])
    rows = []

    for fold, (train_index, test_index) in enumerate(kf.split(x), start=1):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Fit the model
        model.fit(x_train, y_train)

        # Predict
        y_pred = model.predict(x_test)

        # Calculate and store metrics
        rows.append({
            'Fold': str(fold),
            'R2': r2_score(y_test, y_pred),
            'MAE': mean_absolute_error(y_test, y_pred),
            'MAPE': mean_absolute_percentage_error(y_test, y_pred),
            'RMSE': root_mean_squared_error(y_test, y_pred)
        })

    # Convert list of rows to DataFrame
    metrics_df = pd.DataFrame(rows)

    # Calculate average metrics and append
    average_metrics = metrics_df.mean(numeric_only=True)
    average_metrics['Fold'] = 'Average'
    metrics_df = pd.concat([metrics_df, pd.DataFrame([average_metrics])], ignore_index=True)

    return metrics_df


def model_results_to_md(model, results, kfold_df, md_path):
    # 提取KernelRidge模型的超参数
    hyperparameters = model.get_params()
    hyperparameters_df = pd.DataFrame([hyperparameters])

    # 使用to_markdown()函数将每个DataFrame转换为Markdown格式的字符串
    model_params_md = hyperparameters_df.to_markdown()
    results_md = results.to_markdown()
    kfold_df_md = kfold_df.to_markdown()

    # 将这些字符串写入到一个Markdown文件中
    with open(md_path, 'w') as f:
        f.write("# Model Parameters: \n")
        f.write(model_params_md)
        f.write("\n\n# Results: \n")
        f.write(results_md)
        f.write("\n\n# Results of 10-fold: \n")
        f.write(kfold_df_md)