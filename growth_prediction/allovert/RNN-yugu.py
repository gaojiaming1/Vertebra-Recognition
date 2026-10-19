import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
import os

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 固定随机种子
torch.manual_seed(42)
np.random.seed(42)

# 检测设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 获取当前路径
current_directory = os.getcwd()
train_path = os.path.join(current_directory, '体重分组训练集.xlsx')
test_path = os.path.join(current_directory, '体重分组测试集.xlsx')


class SimpleRNNModel(nn.Module):
    """简化版RNN模型（适用于非序列数据）"""

    def __init__(self, input_dim, output_dim, hidden_dim=64, num_layers=1):
        super(SimpleRNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 使用GRU，比标准RNN更稳定
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        # 将每个样本视为长度为1的序列 (batch, 1, input_dim)
        x = x.unsqueeze(1)

        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        out, _ = self.rnn(x, h0)

        # 取最后一个时间步的输出
        out = out[:, -1, :]

        # 全连接层
        out = self.fc(out)
        return out


def train_rnn_model(model, train_loader, val_loader, epochs=1000, model_name='model'):
    """
    训练RNN模型，支持早停和模型保存
    model_name: 用于保存最佳模型的文件名前缀
    """
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=30, factor=0.5)

    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    patience = 50

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                val_loss += loss.item() * batch_x.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        # 早停逻辑
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), f'best_{model_name}.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"早停: 在第 {epoch} 轮停止，最佳轮次为 {best_epoch}，最佳验证损失 {best_val_loss:.6f}")
                break

        if epoch % 100 == 0:
            print(f'Epoch {epoch:4d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}')

    # 加载最佳模型
    model.load_state_dict(torch.load(f'best_{model_name}.pth'))
    return train_losses, val_losses


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

    # 数据标准化
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

    # 转换为张量
    X_len_train_t = torch.FloatTensor(X_len_train).to(device)
    y_w_train_t = torch.FloatTensor(y_w_train).to(device)
    X_len_val_t = torch.FloatTensor(X_len_val).to(device)
    y_w_val_t = torch.FloatTensor(y_w_val).to(device)

    X_w_train_t = torch.FloatTensor(X_w_train).to(device)
    y_len_train_t = torch.FloatTensor(y_len_train).to(device)
    X_w_val_t = torch.FloatTensor(X_w_val).to(device)
    y_len_val_t = torch.FloatTensor(y_len_val).to(device)

    # 创建DataLoader
    batch_size = 8
    train_loader_len = DataLoader(TensorDataset(X_len_train_t, y_w_train_t), batch_size=batch_size, shuffle=True)
    val_loader_len = DataLoader(TensorDataset(X_len_val_t, y_w_val_t), batch_size=batch_size)
    train_loader_w = DataLoader(TensorDataset(X_w_train_t, y_len_train_t), batch_size=batch_size, shuffle=True)
    val_loader_w = DataLoader(TensorDataset(X_w_val_t, y_len_val_t), batch_size=batch_size)

    # ---------- 训练模型 ----------
    print("\n训练体长->体重的RNN模型...")
    model_len2w = SimpleRNNModel(input_dim=4, output_dim=1, hidden_dim=64).to(device)
    losses_len2w_train, losses_len2w_val = train_rnn_model(
        model_len2w, train_loader_len, val_loader_len, epochs=1000, model_name='len2w'
    )

    print("\n训练体重->体长的RNN模型...")
    model_w2len = SimpleRNNModel(input_dim=4, output_dim=1, hidden_dim=64).to(device)
    losses_w2len_train, losses_w2len_val = train_rnn_model(
        model_w2len, train_loader_w, val_loader_w, epochs=1000, model_name='w2len'
    )

    # ---------- 测试集预测 ----------
    print("\n在测试集上进行预测...")
    # 准备测试数据
    test_X_len = test_df[[target_length] + base_features].values
    test_X_w = test_df[[target_weight] + base_features].values

    test_X_len_scaled = scaler_length_x.transform(test_X_len)
    test_X_w_scaled = scaler_weight_x.transform(test_X_w)

    test_X_len_t = torch.FloatTensor(test_X_len_scaled).to(device)
    test_X_w_t = torch.FloatTensor(test_X_w_scaled).to(device)

    model_len2w.eval()
    model_w2len.eval()
    with torch.no_grad():
        pred_w_scaled = model_len2w(test_X_len_t)
        pred_w = scaler_weight_y.inverse_transform(pred_w_scaled.cpu().numpy())

        pred_len_scaled = model_w2len(test_X_w_t)
        pred_len = scaler_length_y.inverse_transform(pred_len_scaled.cpu().numpy())

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

    # 体长预测体重
    plt.subplot(2, 3, 1)
    plt.scatter(results['实际体长(cm)'], results['预测体重(g)'], alpha=0.6)
    minv = min(results['实际体长(cm)'].min(), results['预测体重(g)'].min())
    maxv = max(results['实际体长(cm)'].max(), results['预测体重(g)'].max())
    plt.plot([minv, maxv], [minv, maxv], 'r--', alpha=0.5)
    plt.xlabel('实际体长(cm)')
    plt.ylabel('预测体重(g)')
    plt.title('体长→体重 (RNN)')

    # 体重预测体长
    plt.subplot(2, 3, 2)
    plt.scatter(results['实际体重(g)'], results['预测体长(cm)'], alpha=0.6)
    minv = min(results['实际体重(g)'].min(), results['预测体长(cm)'].min())
    maxv = max(results['实际体重(g)'].max(), results['预测体长(cm)'].max())
    plt.plot([minv, maxv], [minv, maxv], 'r--', alpha=0.5)
    plt.xlabel('实际体重(g)')
    plt.ylabel('预测体长(cm)')
    plt.title('体重→体长 (RNN)')

    # 训练损失 - 体长→体重
    plt.subplot(2, 3, 3)
    plt.plot(losses_len2w_train, label='Train Loss')
    plt.plot(losses_len2w_val, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.title('体长→体重损失曲线')

    # 训练损失 - 体重→体长
    plt.subplot(2, 3, 4)
    plt.plot(losses_w2len_train, label='Train Loss')
    plt.plot(losses_w2len_val, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.title('体重→体长损失曲线')

    # 体长误差分布
    plt.subplot(2, 3, 5)
    plt.hist(results['体长误差(cm)'], bins=15, color='skyblue', edgecolor='black')
    plt.axvline(mae_len, color='red', linestyle='--', label=f'MAE={mae_len:.3f}')
    plt.xlabel('体长误差 (cm)')
    plt.ylabel('频数')
    plt.legend()
    plt.title('体长误差分布')

    # 体重误差分布
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
    results.to_csv('RNN_yugu_predictions.csv', index=False, encoding='utf-8-sig')
    torch.save(model_len2w.state_dict(), 'RNN_model_len2w.pth')
    torch.save(model_w2len.state_dict(), 'RNN_model_w2len.pth')
    print("\n预测结果已保存至 'RNN_yugu_predictions.csv'")
    print("模型已保存为 'RNN_model_len2w.pth' 和 'RNN_model_w2len.pth'")


if __name__ == "__main__":
    main()