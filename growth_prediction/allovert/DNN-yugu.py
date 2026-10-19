import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
import os
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# 设置中文字体（确保字体存在）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 设置随机种子以保证结果可重现
torch.manual_seed(42)
np.random.seed(42)

# 检查GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 获取当前工作目录
current_directory = os.getcwd()

# 拼接当前路径和文件名
train_path = os.path.join(current_directory, '体重分组训练集.xlsx')
test_path = os.path.join(current_directory, '体重分组测试集.xlsx')


class DNNModel(nn.Module):
    """深度神经网络模型"""

    def __init__(self, input_dim, output_dim, hidden_dims=[128, 64, 32]):
        super(DNNModel, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def train_dnn_model(model, train_loader, val_loader, epochs=500):
    """训练DNN模型"""
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=30, factor=0.5)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch_x, batch_y = batch
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch_x, batch_y = batch
                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        if epoch % 100 == 0:
            print(f'DNN Epoch {epoch}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

    return train_losses, val_losses


def main():
    # 1. 读取数据
    print("读取数据...")
    try:
        # 原始数据
        original_data = pd.read_excel(train_path)

        # 测试集数据
        test_data = pd.read_excel(test_path)

    except Exception as e:
        print(f"读取数据时出错: {e}")
        return

    # 2. 数据预处理
    print("\n数据预处理...")

    # 选择用于训练的特征
    feature_columns = ['体长(cm)', '体重(g)', 'chine', 'coccyx', 'cocc']

    # 检查数据中是否存在这些列
    missing_columns = [col for col in feature_columns if col not in original_data.columns]
    if missing_columns:
        print(f"错误: 以下列在数据中不存在: {missing_columns}")
        print(f"数据中的列: {original_data.columns.tolist()}")
        return

    # 直接使用原始数据
    train_data = original_data[feature_columns]
    print(f"\n训练数据量: {len(train_data)}")

    # 3. 划分训练集和验证集（随机打乱）
    print("\n随机划分训练集和验证集...")
    # 生成索引并打乱分割
    indices = np.arange(len(train_data))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, random_state=42, shuffle=True
    )

    # 4. 准备两个任务的数据（先分割，后标准化）
    # 任务1：体长预测体重
    X_len = train_data[['体长(cm)', 'chine', 'coccyx', 'cocc']].values
    y_weight = train_data['体重(g)'].values.reshape(-1, 1)

    X_len_train, X_len_val = X_len[train_idx], X_len[val_idx]
    y_weight_train, y_weight_val = y_weight[train_idx], y_weight[val_idx]

    # 任务2：体重预测体长
    X_weight = train_data[['体重(g)', 'chine', 'coccyx', 'cocc']].values
    y_len = train_data['体长(cm)'].values.reshape(-1, 1)

    X_weight_train, X_weight_val = X_weight[train_idx], X_weight[val_idx]
    y_len_train, y_len_val = y_len[train_idx], y_len[val_idx]

    # 5. 数据标准化（在训练集上拟合，再转换训练集和验证集）
    # 任务1 的 scaler
    scaler_len_x = StandardScaler()
    scaler_weight_y = StandardScaler()

    X_len_train_scaled = scaler_len_x.fit_transform(X_len_train)
    X_len_val_scaled = scaler_len_x.transform(X_len_val)
    y_weight_train_scaled = scaler_weight_y.fit_transform(y_weight_train)
    y_weight_val_scaled = scaler_weight_y.transform(y_weight_val)

    # 任务2 的 scaler
    scaler_weight_x = StandardScaler()
    scaler_len_y = StandardScaler()

    X_weight_train_scaled = scaler_weight_x.fit_transform(X_weight_train)
    X_weight_val_scaled = scaler_weight_x.transform(X_weight_val)
    y_len_train_scaled = scaler_len_y.fit_transform(y_len_train)
    y_len_val_scaled = scaler_len_y.transform(y_len_val)

    # 6. 转换为张量并移至设备
    X_len_train_t = torch.FloatTensor(X_len_train_scaled).to(device)
    y_weight_train_t = torch.FloatTensor(y_weight_train_scaled).to(device)
    X_len_val_t = torch.FloatTensor(X_len_val_scaled).to(device)
    y_weight_val_t = torch.FloatTensor(y_weight_val_scaled).to(device)

    X_weight_train_t = torch.FloatTensor(X_weight_train_scaled).to(device)
    y_len_train_t = torch.FloatTensor(y_len_train_scaled).to(device)
    X_weight_val_t = torch.FloatTensor(X_weight_val_scaled).to(device)
    y_len_val_t = torch.FloatTensor(y_len_val_scaled).to(device)

    # 7. 创建数据加载器
    batch_size = 8
    train_dataset_len = TensorDataset(X_len_train_t, y_weight_train_t)
    val_dataset_len = TensorDataset(X_len_val_t, y_weight_val_t)
    train_dataset_weight = TensorDataset(X_weight_train_t, y_len_train_t)
    val_dataset_weight = TensorDataset(X_weight_val_t, y_len_val_t)

    train_loader_len = DataLoader(train_dataset_len, batch_size=batch_size, shuffle=True)
    val_loader_len = DataLoader(val_dataset_len, batch_size=batch_size)
    train_loader_weight = DataLoader(train_dataset_weight, batch_size=batch_size, shuffle=True)
    val_loader_weight = DataLoader(val_dataset_weight, batch_size=batch_size)

    # 8. 训练DNN模型
    print("\n训练体长预测体重的DNN模型...")
    model_len_to_weight = DNNModel(input_dim=4, output_dim=1).to(device)
    train_losses_len, val_losses_len = train_dnn_model(
        model_len_to_weight, train_loader_len, val_loader_len, epochs=1000
    )

    print("\n训练体重预测体长的DNN模型...")
    model_weight_to_len = DNNModel(input_dim=4, output_dim=1).to(device)
    train_losses_weight, val_losses_weight = train_dnn_model(
        model_weight_to_len, train_loader_weight, val_loader_weight, epochs=1000
    )

    # 9. 在测试集上进行预测
    print("\n在测试集上进行预测...")

    # 准备测试数据
    test_X_len = test_data[['体长(cm)', 'chine', 'coccyx', 'cocc']].values
    test_X_weight = test_data[['体重(g)', 'chine', 'coccyx', 'cocc']].values

    # 使用训练集上拟合的 scaler 进行标准化
    test_X_len_scaled = scaler_len_x.transform(test_X_len)
    test_X_weight_scaled = scaler_weight_x.transform(test_X_weight)

    # 转换为张量
    test_X_len_tensor = torch.FloatTensor(test_X_len_scaled).to(device)
    test_X_weight_tensor = torch.FloatTensor(test_X_weight_scaled).to(device)

    # 预测
    model_len_to_weight.eval()
    model_weight_to_len.eval()

    with torch.no_grad():
        # 体长预测体重
        pred_weight_scaled = model_len_to_weight(test_X_len_tensor)
        pred_weight = scaler_weight_y.inverse_transform(pred_weight_scaled.cpu().numpy())

        # 体重预测体长
        pred_len_scaled = model_weight_to_len(test_X_weight_tensor)
        pred_len = scaler_len_y.inverse_transform(pred_len_scaled.cpu().numpy())

    # 10. 输出结果
    print("\n" + "=" * 50)
    print("预测结果")
    print("=" * 50)

    results = pd.DataFrame({
        '实际体长(cm)': test_data['体长(cm)'].values,
        '预测体长(cm)': pred_len.flatten(),
        '体长预测误差(cm)': np.abs(test_data['体长(cm)'].values - pred_len.flatten()),
        '实际体重(g)': test_data['体重(g)'].values,
        '预测体重(g)': pred_weight.flatten(),
        '体重预测误差(g)': np.abs(test_data['体重(g)'].values - pred_weight.flatten())
    })

    print(results)

    # 计算评估指标
    mse_len = mean_squared_error(results['实际体长(cm)'], results['预测体长(cm)'])
    mse_weight = mean_squared_error(results['实际体重(g)'], results['预测体重(g)'])
    r2_len = r2_score(results['实际体长(cm)'], results['预测体长(cm)'])
    r2_weight = r2_score(results['实际体重(g)'], results['预测体重(g)'])

    # 计算平均绝对误差
    mae_len = np.mean(results['体长预测误差(cm)'])
    mae_weight = np.mean(results['体重预测误差(g)'])

    print(f"\n评估指标:")
    print(f"体长预测 - MSE: {mse_len:.4f}, MAE: {mae_len:.4f}, R²: {r2_len:.4f}")
    print(f"体重预测 - MSE: {mse_weight:.4f}, MAE: {mae_weight:.4f}, R²: {r2_weight:.4f}")

    # 11. 可视化结果
    plt.figure(figsize=(15, 10))

    # 体长预测体重
    plt.subplot(2, 3, 1)
    plt.scatter(results['实际体长(cm)'], results['预测体重(g)'], alpha=0.7)
    min_val = min(results['实际体长(cm)'].min(), results['预测体重(g)'].min())
    max_val = max(results['实际体长(cm)'].max(), results['预测体重(g)'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
    plt.xlabel('实际体长(cm)')
    plt.ylabel('预测体重(g)')
    plt.title('体长预测体重')

    # 体重预测体长
    plt.subplot(2, 3, 2)
    plt.scatter(results['实际体重(g)'], results['预测体长(cm)'], alpha=0.7)
    min_val = min(results['实际体重(g)'].min(), results['预测体长(cm)'].min())
    max_val = max(results['实际体重(g)'].max(), results['预测体长(cm)'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
    plt.xlabel('实际体重(g)')
    plt.ylabel('预测体长(cm)')
    plt.title('体重预测体长')

    # 训练损失 - 体长预测体重
    plt.subplot(2, 3, 3)
    plt.plot(train_losses_len, label='训练损失')
    plt.plot(val_losses_len, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('体长->体重训练损失')

    # 训练损失 - 体重预测体长
    plt.subplot(2, 3, 4)
    plt.plot(train_losses_weight, label='训练损失')
    plt.plot(val_losses_weight, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('体重->体长训练损失')

    # 预测误差分布 - 体长
    plt.subplot(2, 3, 5)
    plt.hist(results['体长预测误差(cm)'], bins=10, alpha=0.7, color='skyblue')
    plt.axvline(mae_len, color='red', linestyle='--', label=f'MAE: {mae_len:.3f}')
    plt.xlabel('体长预测误差(cm)')
    plt.ylabel('频数')
    plt.legend()
    plt.title('体长预测误差分布')

    # 预测误差分布 - 体重
    plt.subplot(2, 3, 6)
    plt.hist(results['体重预测误差(g)'], bins=10, alpha=0.7, color='lightgreen')
    plt.axvline(mae_weight, color='red', linestyle='--', label=f'MAE: {mae_weight:.3f}')
    plt.xlabel('体重预测误差(g)')
    plt.ylabel('频数')
    plt.legend()
    plt.title('体重预测误差分布')

    plt.tight_layout()
    plt.show()

    # 12. 保存预测结果
    results.to_csv('DNN-yugu预测结果.csv', index=False, encoding='utf-8-sig')
    print("\n预测结果已保存到 'DNN-yugu预测结果.csv'")

    # 13. 保存模型
    torch.save(model_len_to_weight.state_dict(), 'DNN-yugu-model_length_to_weight.pth')
    torch.save(model_weight_to_len.state_dict(), 'DNN-yugu-model_weight_to_length.pth')
    print("模型已保存为 'DNN-yugu-model_length_to_weight.pth' 和 'DNN-yugu-model_weight_to_length.pth'")


if __name__ == "__main__":
    main()