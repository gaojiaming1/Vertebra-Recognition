import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
import os
import json

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 固定随机种子
np.random.seed(42)

# 获取当前路径
current_directory = os.getcwd()
train_path = os.path.join(current_directory, '体重分组训练集.xlsx')
test_path = os.path.join(current_directory, '体重分组测试集.xlsx')


def train_xgboost_model(X_train, y_train, X_val, y_val, model_name='xgboost'):
    """
    训练XGBoost回归模型，支持早停，返回模型和验证损失历史
    """
    # 确保y为一维
    if y_train.ndim > 1:
        y_train = y_train.ravel()
    if y_val.ndim > 1:
        y_val = y_val.ravel()

    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        early_stopping_rounds=50,
        eval_metric='rmse',
        random_state=42,
        verbosity=0
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # 获取验证损失历史
    evals_result = model.evals_result()
    val_loss = evals_result['validation_0']['rmse']  # 列表

    # 保存模型
    model.save_model(f'{model_name}.json')
    print(f"模型已保存为 {model_name}.json")

    return model, val_loss


def main():
    print("读取数据...")
    try:
        train_df = pd.read_excel(train_path)
        test_df = pd.read_excel(test_path)
    except Exception as e:
        print(f"读取文件失败: {e}")
        return

    # ---------- 特征列 ----------
    base_features = ['chine', 'coccyx', 'cocc']
    target_length = '体长(cm)'
    target_weight = '体重(g)'
    feature_columns = [target_length, target_weight] + base_features

    # 检查列是否存在
    missing = [col for col in feature_columns if col not in train_df.columns]
    if missing:
        print(f"数据缺少列: {missing}")
        return

    print("\n数据预处理...")

    # 提取训练数据
    train_data = train_df[feature_columns]

    # 准备特征和目标
    # 体长 -> 体重 (输入: 体长 + 骨骼, 输出: 体重)
    X_length = train_data[[target_length] + base_features].values
    y_weight = train_data[[target_weight]].values

    # 体重 -> 体长 (输入: 体重 + 骨骼, 输出: 体长)
    X_weight = train_data[[target_weight] + base_features].values
    y_length = train_data[[target_length]].values

    # 数据标准化（XGBoost对尺度不敏感，但保留以保持与RNN版本一致）
    scaler_length_x = StandardScaler()
    scaler_weight_y = StandardScaler()
    scaler_weight_x = StandardScaler()
    scaler_length_y = StandardScaler()

    X_length_scaled = scaler_length_x.fit_transform(X_length)
    y_weight_scaled = scaler_weight_y.fit_transform(y_weight)
    X_weight_scaled = scaler_weight_x.fit_transform(X_weight)
    y_length_scaled = scaler_length_y.fit_transform(y_length)

    # ---------- 随机划分训练/验证集 ----------
    X_len_train, X_len_val, y_w_train, y_w_val = train_test_split(
        X_length_scaled, y_weight_scaled, test_size=0.2, random_state=42
    )
    X_w_train, X_w_val, y_len_train, y_len_val = train_test_split(
        X_weight_scaled, y_length_scaled, test_size=0.2, random_state=42
    )

    # ---------- 训练 XGBoost 模型 ----------
    print("\n训练体长->体重的XGBoost模型...")
    model_len2w, val_loss_len2w = train_xgboost_model(
        X_len_train, y_w_train, X_len_val, y_w_val, model_name='xgboost_len2w'
    )

    print("\n训练体重->体长的XGBoost模型...")
    model_w2len, val_loss_w2len = train_xgboost_model(
        X_w_train, y_len_train, X_w_val, y_len_val, model_name='xgboost_w2len'
    )

    # ---------- 测试集预测 ----------
    print("\n在测试集上进行预测...")
    # 准备测试数据
    test_X_len = test_df[[target_length] + base_features].values
    test_X_w = test_df[[target_weight] + base_features].values

    test_X_len_scaled = scaler_length_x.transform(test_X_len)
    test_X_w_scaled = scaler_weight_x.transform(test_X_w)

    # 预测并逆标准化
    pred_w_scaled = model_len2w.predict(test_X_len_scaled)
    pred_w = scaler_weight_y.inverse_transform(pred_w_scaled.reshape(-1, 1))

    pred_len_scaled = model_w2len.predict(test_X_w_scaled)
    pred_len = scaler_length_y.inverse_transform(pred_len_scaled.reshape(-1, 1))

    # ---------- 结果汇总 ----------
    results = pd.DataFrame({
        '实际体长(cm)': test_df[target_length].values,
        '预测体长(cm)': pred_len.flatten(),
        '体长误差(cm)': np.abs(test_df[target_length].values - pred_len.flatten()),
        '实际体重(g)': test_df[target_weight].values,
        '预测体重(g)': pred_w.flatten(),
        '体重误差(g)': np.abs(test_df[target_weight].values - pred_w.flatten())
    })

    print("\n前10行预测结果：")
    print(results.head(10))

    # 计算指标
    mse_len = mean_squared_error(results['实际体长(cm)'], results['预测体长(cm)'])
    mse_w = mean_squared_error(results['实际体重(g)'], results['预测体重(g)'])
    r2_len = r2_score(results['实际体长(cm)'], results['预测体长(cm)'])
    r2_w = r2_score(results['实际体重(g)'], results['预测体重(g)'])
    mae_len = np.mean(results['体长误差(cm)'])
    mae_w = np.mean(results['体重误差(g)'])

    print(f"\n评估指标:")
    print(f"体长预测 - MSE: {mse_len:.4f}, MAE: {mae_len:.4f}, R²: {r2_len:.4f}")
    print(f"体重预测 - MSE: {mse_w:.4f}, MAE: {mae_w:.4f}, R²: {r2_w:.4f}")

    # ---------- 可视化 ----------
    plt.figure(figsize=(15, 10))

    # 1. 体长预测体重散点图
    plt.subplot(2, 3, 1)
    plt.scatter(results['实际体长(cm)'], results['预测体重(g)'], alpha=0.6)
    minv = min(results['实际体长(cm)'].min(), results['预测体重(g)'].min())
    maxv = max(results['实际体长(cm)'].max(), results['预测体重(g)'].max())
    plt.plot([minv, maxv], [minv, maxv], 'r--', alpha=0.5)
    plt.xlabel('实际体长(cm)')
    plt.ylabel('预测体重(g)')
    plt.title('体长→体重 (XGBoost)')

    # 2. 体重预测体长散点图
    plt.subplot(2, 3, 2)
    plt.scatter(results['实际体重(g)'], results['预测体长(cm)'], alpha=0.6)
    minv = min(results['实际体重(g)'].min(), results['预测体长(cm)'].min())
    maxv = max(results['实际体重(g)'].max(), results['预测体长(cm)'].max())
    plt.plot([minv, maxv], [minv, maxv], 'r--', alpha=0.5)
    plt.xlabel('实际体重(g)')
    plt.ylabel('预测体长(cm)')
    plt.title('体重→体长 (XGBoost)')

    # 3. XGBoost 验证损失曲线（体长→体重）
    plt.subplot(2, 3, 3)
    plt.plot(val_loss_len2w, label='验证集 RMSE')
    plt.xlabel('迭代轮次')
    plt.ylabel('RMSE')
    plt.legend()
    plt.title('体长→体重 验证损失')

    # 4. XGBoost 验证损失曲线（体重→体长）
    plt.subplot(2, 3, 4)
    plt.plot(val_loss_w2len, label='验证集 RMSE')
    plt.xlabel('迭代轮次')
    plt.ylabel('RMSE')
    plt.legend()
    plt.title('体重→体长 验证损失')

    # 5. 体长误差分布
    plt.subplot(2, 3, 5)
    plt.hist(results['体长误差(cm)'], bins=15, color='skyblue', edgecolor='black')
    plt.axvline(mae_len, color='red', linestyle='--', label=f'MAE={mae_len:.3f}')
    plt.xlabel('体长误差 (cm)')
    plt.ylabel('频数')
    plt.legend()
    plt.title('体长误差分布')

    # 6. 体重误差分布
    plt.subplot(2, 3, 6)
    plt.hist(results['体重误差(g)'], bins=15, color='lightgreen', edgecolor='black')
    plt.axvline(mae_w, color='red', linestyle='--', label=f'MAE={mae_w:.3f}')
    plt.xlabel('体重误差 (g)')
    plt.ylabel('频数')
    plt.legend()
    plt.title('体重误差分布')

    plt.tight_layout()
    plt.show()

    # ---------- 保存结果 ----------
    results.to_csv('XGBoost_yugu_predictions.csv', index=False, encoding='utf-8-sig')
    print("\n预测结果已保存至 'XGBoost_yugu_predictions.csv'")
    print("模型已保存为 'xgboost_len2w.json' 和 'xgboost_w2len.json'")


if __name__ == "__main__":
    main()