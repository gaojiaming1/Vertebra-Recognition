import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
import os

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

class CNNModel(nn.Module):
    """一维卷积神经网络模型"""

    def __init__(self, input_dim, output_dim):
        super(CNNModel, self).__init__()

        # 将输入重塑为适合1D CNN的形状 (batch_size, 1, input_dim)
        self.input_dim = input_dim

        # 第一个卷积块
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.MaxPool1d(kernel_size=2)
        )

        # 第二个卷积块
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(kernel_size=2)
        )

        # 计算卷积后的特征维度
        # 经过两次池化，特征维度变为 input_dim // 4
        conv_output_size = 32 * (input_dim // 4)

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(conv_output_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        # 重塑输入为 (batch_size, 1, input_dim)
        x = x.unsqueeze(1)

        # 卷积层
        x = self.conv1(x)
        x = self.conv2(x)

        # 展平
        x = x.view(x.size(0), -1)

        # 全连接层
        x = self.fc(x)
        return x


class SimpleCNNModel(nn.Module):
    """简化版CNN模型，适用于小数据集"""

    def __init__(self, input_dim, output_dim):
        super(SimpleCNNModel, self).__init__()

        self.input_dim = input_dim

        # 卷积层
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # 全局平均池化，输出大小为1
        )

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        # 重塑输入
        x = x.unsqueeze(1)

        # 卷积层
        x = self.conv_layers(x)

        # 展平
        x = x.view(x.size(0), -1)

        # 全连接层
        x = self.fc(x)
        return x


def train_cnn_model(model, train_loader, val_loader, epochs=500):
    """训练CNN模型"""
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=30, factor=0.5)

    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 50  # 早停耐心值

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

        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'best_cnn_model.pth')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"早停在第 {epoch} 轮")
            break

        scheduler.step(val_loss)

        if epoch % 100 == 0:
            print(f'CNN Epoch {epoch}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

    # 加载最佳模型
    model.load_state_dict(torch.load('best_cnn_model.pth'))

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

    # 3. 准备CNN训练数据
    print("\n准备CNN训练数据...")

    # 体长预测体重
    X_length = train_data[['体长(cm)', 'chine', 'coccyx', 'cocc']].values
    y_weight = train_data['体重(g)'].values.reshape(-1, 1)

    # 体重预测体长
    X_weight = train_data[['体重(g)', 'chine', 'coccyx', 'cocc']].values
    y_length = train_data['体长(cm)'].values.reshape(-1, 1)

    # 数据标准化
    scaler_length_x = StandardScaler()
    scaler_weight_y = StandardScaler()
    scaler_weight_x = StandardScaler()
    scaler_length_y = StandardScaler()

    X_length_scaled = scaler_length_x.fit_transform(X_length)
    y_weight_scaled = scaler_weight_y.fit_transform(y_weight)
    X_weight_scaled = scaler_weight_x.fit_transform(X_weight)
    y_length_scaled = scaler_length_y.fit_transform(y_length)

    # 分割训练集和验证集
    split_idx = int(0.8 * len(train_data))

    # 体长预测体重数据集
    X_length_train = torch.FloatTensor(X_length_scaled[:split_idx]).to(device)
    y_weight_train = torch.FloatTensor(y_weight_scaled[:split_idx]).to(device)
    X_length_val = torch.FloatTensor(X_length_scaled[split_idx:]).to(device)
    y_weight_val = torch.FloatTensor(y_weight_scaled[split_idx:]).to(device)

    # 体重预测体长数据集
    X_weight_train = torch.FloatTensor(X_weight_scaled[:split_idx]).to(device)
    y_length_train = torch.FloatTensor(y_length_scaled[:split_idx]).to(device)
    X_weight_val = torch.FloatTensor(X_weight_scaled[split_idx:]).to(device)
    y_length_val = torch.FloatTensor(y_length_scaled[split_idx:]).to(device)

    # 创建数据加载器
    train_dataset_length = TensorDataset(X_length_train, y_weight_train)
    val_dataset_length = TensorDataset(X_length_val, y_weight_val)
    train_dataset_weight = TensorDataset(X_weight_train, y_length_train)
    val_dataset_weight = TensorDataset(X_weight_val, y_length_val)

    train_loader_length = DataLoader(train_dataset_length, batch_size=4, shuffle=True)
    val_loader_length = DataLoader(val_dataset_length, batch_size=4)
    train_loader_weight = DataLoader(train_dataset_weight, batch_size=4, shuffle=True)
    val_loader_weight = DataLoader(val_dataset_weight, batch_size=4)

    # 4. 训练CNN模型
    print("\n训练体长预测体重的CNN模型...")
    # 使用简化版CNN，因为数据量小
    model_length_to_weight = SimpleCNNModel(input_dim=4, output_dim=1).to(device)
    train_losses_length, val_losses_length = train_cnn_model(
        model_length_to_weight, train_loader_length, val_loader_length, epochs=1000
    )

    print("\n训练体重预测体长的CNN模型...")
    model_weight_to_length = SimpleCNNModel(input_dim=4, output_dim=1).to(device)
    train_losses_weight, val_losses_weight = train_cnn_model(
        model_weight_to_length, train_loader_weight, val_loader_weight, epochs=1000
    )

    # 5. 在测试集上进行预测
    print("\n在测试集上进行预测...")

    # 准备测试数据
    test_X_length = test_data[['体长(cm)', 'chine', 'coccyx', 'cocc']].values
    test_X_weight = test_data[['体重(g)', 'chine', 'coccyx', 'cocc']].values

    # 标准化测试数据
    test_X_length_scaled = scaler_length_x.transform(test_X_length)
    test_X_weight_scaled = scaler_weight_x.transform(test_X_weight)

    # 转换为张量
    test_X_length_tensor = torch.FloatTensor(test_X_length_scaled).to(device)
    test_X_weight_tensor = torch.FloatTensor(test_X_weight_scaled).to(device)

    # 预测
    model_length_to_weight.eval()
    model_weight_to_length.eval()

    with torch.no_grad():
        # 体长预测体重
        pred_weight_scaled = model_length_to_weight(test_X_length_tensor)
        pred_weight = scaler_weight_y.inverse_transform(pred_weight_scaled.cpu().numpy())

        # 体重预测体长
        pred_length_scaled = model_weight_to_length(test_X_weight_tensor)
        pred_length = scaler_length_y.inverse_transform(pred_length_scaled.cpu().numpy())

    # 6. 输出结果
    print("\n" + "=" * 50)
    print("CNN模型预测结果")
    print("=" * 50)

    results = pd.DataFrame({
        '实际体长(cm)': test_data['体长(cm)'].values,
        '预测体长(cm)': pred_length.flatten(),
        '体长预测误差(cm)': np.abs(test_data['体长(cm)'].values - pred_length.flatten()),
        '实际体重(g)': test_data['体重(g)'].values,
        '预测体重(g)': pred_weight.flatten(),
        '体重预测误差(g)': np.abs(test_data['体重(g)'].values - pred_weight.flatten())
    })

    print(results)

    # 计算评估指标
    mse_length = mean_squared_error(results['实际体长(cm)'], results['预测体长(cm)'])
    mse_weight = mean_squared_error(results['实际体重(g)'], results['预测体重(g)'])
    r2_length = r2_score(results['实际体长(cm)'], results['预测体长(cm)'])
    r2_weight = r2_score(results['实际体重(g)'], results['预测体重(g)'])

    # 计算平均绝对误差
    mae_length = np.mean(results['体长预测误差(cm)'])
    mae_weight = np.mean(results['体重预测误差(g)'])

    # 计算相对误差百分比
    mape_length = np.mean(np.abs(results['体长预测误差(cm)'] / results['实际体长(cm)'])) * 100
    mape_weight = np.mean(np.abs(results['体重预测误差(g)'] / results['实际体重(g)'])) * 100

    print(f"\n评估指标:")
    print(f"体长预测 - MSE: {mse_length:.4f}, MAE: {mae_length:.4f}, R²: {r2_length:.4f}, MAPE: {mape_length:.2f}%")
    print(f"体重预测 - MSE: {mse_weight:.4f}, MAE: {mae_weight:.4f}, R²: {r2_weight:.4f}, MAPE: {mape_weight:.2f}%")

    # 7. 可视化结果
    plt.figure(figsize=(15, 12))

    # 体长预测体重
    plt.subplot(3, 3, 1)
    plt.scatter(results['实际体长(cm)'], results['预测体重(g)'], alpha=0.7)
    min_val = min(results['实际体长(cm)'].min(), results['预测体重(g)'].min())
    max_val = max(results['实际体长(cm)'].max(), results['预测体重(g)'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
    plt.xlabel('实际体长(cm)')
    plt.ylabel('预测体重(g)')
    plt.title('体长预测体重 (CNN)')

    # 体重预测体长
    plt.subplot(3, 3, 2)
    plt.scatter(results['实际体重(g)'], results['预测体长(cm)'], alpha=0.7)
    min_val = min(results['实际体重(g)'].min(), results['预测体长(cm)'].min())
    max_val = max(results['实际体重(g)'].max(), results['预测体长(cm)'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
    plt.xlabel('实际体重(g)')
    plt.ylabel('预测体长(cm)')
    plt.title('体重预测体长 (CNN)')

    # 训练损失 - 体长预测体重
    plt.subplot(3, 3, 3)
    plt.plot(train_losses_length, label='训练损失')
    plt.plot(val_losses_length, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('体长->体重训练损失 (CNN)')

    # 训练损失 - 体重预测体长
    plt.subplot(3, 3, 4)
    plt.plot(train_losses_weight, label='训练损失')
    plt.plot(val_losses_weight, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('体重->体长训练损失 (CNN)')

    # 预测误差分布 - 体长
    plt.subplot(3, 3, 5)
    plt.hist(results['体长预测误差(cm)'], bins=10, alpha=0.7, color='skyblue')
    plt.axvline(mae_length, color='red', linestyle='--', label=f'MAE: {mae_length:.3f}')
    plt.xlabel('体长预测误差(cm)')
    plt.ylabel('频数')
    plt.legend()
    plt.title('体长预测误差分布 (CNN)')

    # 预测误差分布 - 体重
    plt.subplot(3, 3, 6)
    plt.hist(results['体重预测误差(g)'], bins=10, alpha=0.7, color='lightgreen')
    plt.axvline(mae_weight, color='red', linestyle='--', label=f'MAE: {mae_weight:.3f}')
    plt.xlabel('体重预测误差(g)')
    plt.ylabel('频数')
    plt.legend()
    plt.title('体重预测误差分布 (CNN)')

    # 体长预测相对误差
    plt.subplot(3, 3, 7)
    relative_error_length = (results['体长预测误差(cm)'] / results['实际体长(cm)']) * 100
    plt.hist(relative_error_length, bins=10, alpha=0.7, color='orange')
    plt.axvline(mape_length, color='red', linestyle='--', label=f'MAPE: {mape_length:.2f}%')
    plt.xlabel('体长相对误差(%)')
    plt.ylabel('频数')
    plt.legend()
    plt.title('体长相对误差分布 (CNN)')

    # 体重预测相对误差
    plt.subplot(3, 3, 8)
    relative_error_weight = (results['体重预测误差(g)'] / results['实际体重(g)']) * 100
    plt.hist(relative_error_weight, bins=10, alpha=0.7, color='purple')
    plt.axvline(mape_weight, color='red', linestyle='--', label=f'MAPE: {mape_weight:.2f}%')
    plt.xlabel('体重相对误差(%)')
    plt.ylabel('频数')
    plt.legend()
    plt.title('体重相对误差分布 (CNN)')

    # 模型结构示意图
    plt.subplot(3, 3, 9)
    plt.text(0.1, 0.8, "CNN模型结构:", fontsize=12, fontweight='bold')
    plt.text(0.1, 0.6, "输入: (batch, 1, 4)", fontsize=10)
    plt.text(0.1, 0.5, "Conv1D(1->8, k=3)", fontsize=10)
    plt.text(0.1, 0.4, "MaxPool1D(k=2)", fontsize=10)
    plt.text(0.1, 0.3, "Conv1D(8->16, k=3)", fontsize=10)
    plt.text(0.1, 0.2, "GlobalAvgPool", fontsize=10)
    plt.text(0.1, 0.1, "FC(16->32->1)", fontsize=10)
    plt.axis('off')
    plt.title('模型结构示意图')

    plt.tight_layout()
    plt.show()

    # 8. 保存预测结果
    results.to_csv('CNN-yugu预测结果.csv', index=False, encoding='utf-8-sig')
    print("\n预测结果已保存到 'CNN-yugu预测结果.csv'")

    # 9. 保存模型
    torch.save(model_length_to_weight.state_dict(), 'CNN-yugu_model_length_to_weight.pth')
    torch.save(model_weight_to_length.state_dict(), 'CNN-yugu_model_weight_to_length.pth')
    print("CNN模型已保存为 'CNN-yugu_model_length_to_weight.pth' 和 'CNN-yugu_model_weight_to_length.pth'")


if __name__ == "__main__":
    main()