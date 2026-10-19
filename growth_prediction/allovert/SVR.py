import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import warnings
import os
import joblib  # 用于保存和加载 SVR 模型

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


def train_svr_model(X_train, y_train, X_val, y_val, model_name='svr'):
    """
    训练 SVR 模型，返回模型及训练/验证 MSE（用于绘图）
    """
    # 确保 y 为一维
    if y_train.ndim > 1:
        y_train = y_train.ravel()
    if y_val.ndim > 1:
        y_val = y_val.ravel()

    # 使用 RBF 核，默认参数（可尝试调整 C 和 gamma）
    model = SVR(kernel='rbf', C=1.0, gamma='scale', epsilon=0.1)

    # 训练模型
    model.fit(X_train, y_train)

    # 预测训练集和验证集
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    # 计算 MSE
    train_mse = mean_squared_error(y_train, y_train_pred)
    val_mse = mean_squared_error(y_val, y_val_pred)

    # 保存模型
    joblib.dump(model, f'{model_name}.pkl')
    print(f"模型已保存为 {model_name}.pkl")

    return model, train_mse, val_mse


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

    # 数据标准化（SVR 对尺度敏感，标准化至关重要）
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

    # ---------- 训练 SVR 模型 ----------
    print("\n训练体长->体重的 SVR 模型...")
    model_len2w, train_mse_len2w, val_mse_len2w = train_svr_model(
        X_len_train, y_w_train, X_len_val, y_w_val, model_name='svr_len2w'
    )

    print("\n训练体重->体长的 SVR 模型...")
    model_w2len, train_mse_w2len, val_mse_w2len = train_svr_model(
        X_w_train, y_len_train, X_w_val, y_len_val, model_name='svr_w2len'
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
    plt.title('体长→体重 (SVR)')

    # 2. 体重预测体长散点图
    plt.subplot(2, 3, 2)
    plt.scatter(results['实际体重(g)'], results['预测体长(cm)'], alpha=0.6)
    minv = min(results['实际体重(g)'].min(), results['预测体长(cm)'].min())
    maxv = max(results['实际体重(g)'].max(), results['预测体长(cm)'].max())
    plt.plot([minv, maxv], [minv, maxv], 'r--', alpha=0.5)
    plt.xlabel('实际体重(g)')
    plt.ylabel('预测体长(cm)')
    plt.title('体重→体长 (SVR)')

    # 3. 体长→体重：训练/验证 MSE 对比
    plt.subplot(2, 3, 3)
    bars = plt.bar(['训练集', '验证集'], [train_mse_len2w, val_mse_len2w], color=['steelblue', 'orange'])
    plt.ylabel('MSE')
    plt.title('体长→体重 训练/验证 MSE')
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=9)

    # 4. 体重→体长：训练/验证 MSE 对比
    plt.subplot(2, 3, 4)
    bars = plt.bar(['训练集', '验证集'], [train_mse_w2len, val_mse_w2len], color=['steelblue', 'orange'])
    plt.ylabel('MSE')
    plt.title('体重→体长 训练/验证 MSE')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=9)

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
    results.to_csv('SVR_yugu_predictions.csv', index=False, encoding='utf-8-sig')
    print("\n预测结果已保存至 'SVR_yugu_predictions.csv'")
    print("模型已保存为 'svr_len2w.pkl' 和 'svr_w2len.pkl'")


if __name__ == "__main__":
    main()