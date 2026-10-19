import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# 绘图字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# ==================== Configuration ====================
TRAIN_PATH = '体重分组训练集.xlsx'
TEST_PATH = '体重分组测试集.xlsx'
OUTPUT_DIR = './output_rf_logit'
MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42
VAL_SIZE = 0.2

# 原始目标
ORIGINAL_TARGETS = ['chine', 'coccyx', 'cocc']

# 分组定义（与之前一致）
LENGTH_BINS = [0, 30, 45, 51, np.inf]
LENGTH_LABELS = ['Age1', 'Age2', 'Age3', 'Age4']
WEIGHT_BINS = [0, 30, 60, 90, np.inf]
WEIGHT_LABELS = ['Very Light', 'Light', 'Medium', 'Heavy']

# ==================== Data Loading with Feature Engineering ====================
def add_biological_features(df):
    L = df['体长(cm)']
    W = df['体重(g)']
    df['log_L'] = np.log(L)
    df['log_W'] = np.log(W)
    df['L_W_ratio'] = L / W
    df['W_L2'] = W / (L ** 2)
    df['W_L3'] = W / (L ** 3)
    df['sqrt_W'] = np.sqrt(W)
    df['sqrt_L'] = np.sqrt(L)
    df['L2'] = L ** 2
    df['L3'] = L ** 3
    df['log_L_log_W'] = np.log(L) * np.log(W)
    return df

def load_and_add_features(filepath):
    df = pd.read_excel(filepath, sheet_name='Sheet1')
    required_cols = ['体重(g)', '体长(cm)', 'chine', 'coccyx', 'cocc']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    # 清洗零值
    cols_check = ['体重(g)', '体长(cm)', 'chine', 'coccyx', 'cocc']
    before = len(df)
    df = df[(df[cols_check] != 0).all(axis=1)].copy()
    print(f"  {os.path.basename(filepath)}: removed {before - len(df)} rows with zeros, remaining {len(df)} rows")
    if len(df) == 0:
        raise ValueError("No samples remain!")

    # 分组编码
    df['体长组'] = pd.cut(df['体长(cm)'], bins=LENGTH_BINS,
                          labels=LENGTH_LABELS, right=False)
    length_map = {label: i+1 for i, label in enumerate(LENGTH_LABELS)}
    df['体长组编码'] = df['体长组'].map(length_map).astype(float)

    df['体重组'] = pd.cut(df['体重(g)'], bins=WEIGHT_BINS,
                          labels=WEIGHT_LABELS, right=False)
    weight_map = {label: i+1 for i, label in enumerate(WEIGHT_LABELS)}
    df['体重组编码'] = df['体重组'].map(weight_map).astype(float)

    df.drop(['体长组', '体重组'], axis=1, inplace=True)
    df = add_biological_features(df)
    return df

print("Loading training data...")
train_df = load_and_add_features(TRAIN_PATH)
print(f"Training samples after cleaning: {len(train_df)}")
print("Loading test data...")
test_df = load_and_add_features(TEST_PATH)
print(f"Test samples after cleaning: {len(test_df)}")

# ==================== Feature Columns ====================
all_feature_cols = [
    '体重(g)', '体长(cm)',
    '体长组编码', '体重组编码',
    'log_L', 'log_W', 'L_W_ratio', 'W_L2', 'W_L3',
    'sqrt_W', 'sqrt_L', 'L2', 'L3', 'log_L_log_W'
]

X = train_df[all_feature_cols]
y_orig_df = train_df[ORIGINAL_TARGETS]   # DataFrame with original index

# 计算成分目标（Sum + 比例）
y_orig_values = y_orig_df.values
sum_all = y_orig_values.sum(axis=1, keepdims=True)
ratio_chine = y_orig_values[:, 0] / sum_all.flatten()
ratio_coccyx = y_orig_values[:, 1] / sum_all.flatten()
# ratio_cocc = 1 - ratio_chine - ratio_coccyx（不直接预测）

# Logit 变换（避免边界值）
def safe_logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

logit_chine = safe_logit(ratio_chine)
logit_coccyx = safe_logit(ratio_coccyx)

# 构造目标DataFrame
y_sum = pd.Series(sum_all.flatten(), index=train_df.index, name='Sum')
y_logit = pd.DataFrame(
    {'logit_chine': logit_chine, 'logit_coccyx': logit_coccyx},
    index=train_df.index
)

# ==================== 划分训练/验证集 ====================
X_train, X_val, y_sum_train, y_sum_val = train_test_split(
    X, y_sum, test_size=VAL_SIZE, random_state=RANDOM_STATE, shuffle=True
)
_, _, y_logit_train, y_logit_val = train_test_split(
    X, y_logit, test_size=VAL_SIZE, random_state=RANDOM_STATE, shuffle=True
)

# 保留原始目标用于评估
y_train_orig = y_orig_df.loc[y_sum_train.index].values
y_val_orig = y_orig_df.loc[y_sum_val.index].values

X_test = test_df[all_feature_cols]
y_test_orig = test_df[ORIGINAL_TARGETS].values

print(f"Train set: {X_train.shape[0]} samples, Validation set: {X_val.shape[0]} samples")

# ==================== 特征选择（针对 Sum 和 Logit 分别选择） ====================
print("\n--- Feature selection for Sum and Logit targets ---")
def select_features(X, y, n_features=8, random_state=42):
    rf = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
    rf.fit(X, y)
    importances = rf.feature_importances_
    idx_sorted = np.argsort(importances)[::-1][:n_features]
    selected_feats = X.columns[idx_sorted]
    return selected_feats, idx_sorted

# 为 Sum 选择特征
sum_feats, sum_idx = select_features(X_train, y_sum_train, n_features=8)
print(f"  Sum selected features: {list(sum_feats)}")

# 为 Logit 选择特征（使用两个logit的均值或总和作为重要性依据）
# 此处用 logit_chine + logit_coccyx 的均值作为合成目标，或分别选择再合并
# 简单起见，我们训练一个模型预测 logit_chine 和 logit_coccyx 的平均值
y_logit_avg = (y_logit_train['logit_chine'] + y_logit_train['logit_coccyx']) / 2
logit_feats, logit_idx = select_features(X_train, y_logit_avg, n_features=8)
print(f"  Logit selected features: {list(logit_feats)}")

# 提取训练/验证/测试对应特征
X_train_sum = X_train.iloc[:, sum_idx]
X_val_sum = X_val.iloc[:, sum_idx]
X_test_sum = X_test.iloc[:, sum_idx]

X_train_logit = X_train.iloc[:, logit_idx]
X_val_logit = X_val.iloc[:, logit_idx]
X_test_logit = X_test.iloc[:, logit_idx]

# ==================== 训练模型 ====================
print("\n--- Training Random Forest models ---")
# 模型1：预测 Sum
model_sum = RandomForestRegressor(
    n_estimators=200,
    min_samples_split=5,
    max_depth=10,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
model_sum.fit(X_train_sum, y_sum_train)
print("  Sum model trained.")

# 模型2：预测两个 Logit（多输出）
model_logit = MultiOutputRegressor(
    RandomForestRegressor(
        n_estimators=200,
        min_samples_split=5,
        max_depth=10,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),
    n_jobs=-1
)
model_logit.fit(X_train_logit, y_logit_train)
print("  Logit model trained.")

# ==================== 预测与还原 ====================
def predict_original(model_sum, model_logit, X_sum, X_logit):
    pred_sum = model_sum.predict(X_sum)
    pred_logit = model_logit.predict(X_logit)  # [logit_chine, logit_coccyx]
    # Sigmoid 还原比例
    pred_ratio_chine = 1 / (1 + np.exp(-pred_logit[:, 0]))
    pred_ratio_coccyx = 1 / (1 + np.exp(-pred_logit[:, 1]))
    pred_ratio_cocc = 1 - pred_ratio_chine - pred_ratio_coccyx
    # 确保比例非负且和为1（归一化）
    ratios = np.column_stack([pred_ratio_chine, pred_ratio_coccyx, pred_ratio_cocc])
    ratios = np.clip(ratios, 0, 1)
    ratios = ratios / ratios.sum(axis=1, keepdims=True)
    # 还原原始值
    pred_chine = pred_sum * ratios[:, 0]
    pred_coccyx = pred_sum * ratios[:, 1]
    pred_cocc = pred_sum * ratios[:, 2]
    return np.column_stack([pred_chine, pred_coccyx, pred_cocc])

# 预测验证集和测试集
y_val_pred_orig = predict_original(model_sum, model_logit, X_val_sum, X_val_logit)
y_test_pred_orig = predict_original(model_sum, model_logit, X_test_sum, X_test_logit)

# ==================== 评估函数 ====================
def evaluate(y_true, y_pred, target_names):
    results = {}
    for i, name in enumerate(target_names):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        mask = ~(np.isnan(yt) | np.isnan(yp))
        yt = yt[mask]
        yp = yp[mask]
        if len(yt) == 0:
            continue
        r2 = r2_score(yt, yp)
        rmse = np.sqrt(mean_squared_error(yt, yp))
        mae = mean_absolute_error(yt, yp)
        if np.any(yt == 0):
            mape = np.nan
        else:
            mape = np.mean(np.abs((yt - yp) / yt)) * 100
        spearman, _ = spearmanr(yt, yp)
        results[name] = {'R2': r2, 'RMSE': rmse, 'MAE': mae,
                         'MAPE': mape, 'Spearman': spearman}
    return results

# 验证集评估
val_r2 = r2_score(y_val_orig, y_val_pred_orig, multioutput='raw_values')
val_rmse = np.sqrt(mean_squared_error(y_val_orig, y_val_pred_orig, multioutput='raw_values'))
print("\nValidation R² (per original target):", val_r2)
print("Validation RMSE (per original target):", val_rmse)

print("\n--- Test set metrics (Random Forest + Logit) ---")
metrics = evaluate(y_test_orig, y_test_pred_orig, ORIGINAL_TARGETS)
for name, metric in metrics.items():
    print(f"\n{name}:")
    for k, v in metric.items():
        print(f"  {k}: {v:.4f}")

# 检查总和约束
pred_sum = y_test_pred_orig.sum(axis=1)
in_range = np.sum((pred_sum >= 133) & (pred_sum <= 166)) / len(pred_sum) * 100
print(f"\nPercentage of predicted sums in [133,166]: {in_range:.1f}%")

# ==================== 保存预测结果 ====================
test_df_with_pred = test_df.copy()
test_df_with_pred['Pred_chine'] = y_test_pred_orig[:, 0]
test_df_with_pred['Pred_coccyx'] = y_test_pred_orig[:, 1]
test_df_with_pred['Pred_cocc'] = y_test_pred_orig[:, 2]
test_df_with_pred['Pred_Sum'] = pred_sum
result_path = os.path.join(OUTPUT_DIR, 'RF_Logit_预测结果.xlsx')
test_df_with_pred.to_excel(result_path, index=False)
print(f"\nPredictions saved to: {result_path}")

# ==================== 可视化 ====================
def plot_predictions(y_true, y_pred, target_names, save_path=None):
    n = len(target_names)
    fig, axes = plt.subplots(2, n, figsize=(5*n, 8))
    if n == 1:
        axes = axes.reshape(2, 1)
    for i, name in enumerate(target_names):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        mask = ~(np.isnan(yt) | np.isnan(yp))
        yt = yt[mask]
        yp = yp[mask]
        if len(yt) == 0:
            continue
        ax = axes[0, i]
        ax.scatter(yt, yp, alpha=0.6)
        min_val = min(yt.min(), yp.min())
        max_val = max(yt.max(), yp.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1)
        ax.set_xlabel('True Value')
        ax.set_ylabel('Predicted Value')
        ax.set_title(f'{name}')
        ax = axes[1, i]
        residuals = yp - yt
        ax.scatter(yp, residuals, alpha=0.6)
        ax.axhline(y=0, color='r', linestyle='--', lw=1)
        ax.set_xlabel('Predicted Value')
        ax.set_ylabel('Residual')
        ax.set_title(f'{name} Residuals')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prediction plot saved: {save_path}")
    plt.show()

plot_predictions(y_test_orig, y_test_pred_orig, ORIGINAL_TARGETS,
                 save_path=os.path.join(OUTPUT_DIR, 'RF_Logit_预测图.png'))

# ==================== 保存评估报告 ====================
report_path = os.path.join(OUTPUT_DIR, 'RF_Logit_评估报告.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("Random Forest + Logit Compositional Model\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")
    f.write("Features used for Sum:\n")
    for feat in sum_feats:
        f.write(f"  - {feat}\n")
    f.write("\nFeatures used for Logit:\n")
    for feat in logit_feats:
        f.write(f"  - {feat}\n")
    f.write("\nModel parameters:\n")
    f.write("  Sum: n_estimators=200, min_samples_split=5, max_depth=10\n")
    f.write("  Logit: same (MultiOutputRegressor)\n")
    f.write("\nValidation metrics:\n")
    for i, name in enumerate(ORIGINAL_TARGETS):
        f.write(f"  {name}: R²={val_r2[i]:.4f}, RMSE={val_rmse[i]:.4f}\n")
    f.write("\nTest metrics:\n")
    for name, metric in metrics.items():
        f.write(f"\n{name}:\n")
        for k, v in metric.items():
            f.write(f"  {k}: {v:.4f}\n")
    f.write(f"\nSum constraint satisfaction: {in_range:.1f}% in [133,166]\n")
print(f"Report saved to: {report_path}")

# ==================== 保存模型 ====================
joblib.dump(model_sum, os.path.join(MODEL_DIR, 'rf_sum_model.pkl'))
joblib.dump(model_logit, os.path.join(MODEL_DIR, 'rf_logit_model.pkl'))
print(f"Models saved to: {MODEL_DIR}")

print("\nAll tasks completed!")