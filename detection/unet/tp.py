import numpy as np
import matplotlib.pyplot as plt


def plot_training_log(log_dir):
    # 加载损失数据
    loss_data = np.loadtxt(f"{log_dir}/loss_2023_10_10_12_30_45.txt")  # 替换为你的实际文件名
    epochs = loss_data[:, 0]
    train_loss = loss_data[:, 1]
    val_loss = loss_data[:, 2]

    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 如果有评估数据，加载并绘制准确率
    try:
        eval_data = np.loadtxt(f"{log_dir}/eval_2023_10_10_12_30_45.txt")  # 替换为实际文件名
        eval_epochs = eval_data[:, 0]
        miou = eval_data[:, 1]
        accuracy = eval_data[:, 2]

        plt.subplot(1, 2, 2)
        plt.plot(eval_epochs, miou, label='mIoU')
        plt.plot(eval_epochs, accuracy, label='Accuracy')
        plt.title('Evaluation Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend()
    except FileNotFoundError:
        print("未找到评估日志文件，请确保训练时启用了eval_flag")

    plt.tight_layout()
    plt.savefig(f'{log_dir}/training_metrics.png')
    plt.show()


# 使用示例
log_dir = 'logs/loss_2023_10_10_12_30_45'  # 替换为你的实际日志目录
plot_training_log(log_dir)