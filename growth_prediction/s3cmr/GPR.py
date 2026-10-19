import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr
import warnings

warnings.filterwarnings('ignore')

# ==================== Configuration ====================
TRAIN_PATH = '体重分组训练集.xlsx'
TEST_PATH = '体重分组测试集.xlsx'
OUTPUT_DIR = './output_gpr_stable'
MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42
ORIGINAL_TARGETS = ['chine', 'coccyx', 'cocc']

# 分组定义
LENGTH_BINS = [0, 30, 45, 51, np.inf]
LENGTH_LABELS = ['Age1', 'Age2', 'Age3', 'Age4']
WEIGHT_BINS = [0, 30, 60, 90, np.inf]
WEIGHT_LABELS = ['Very Light', 'Light', 'Medium', 'Heavy']


# ==================== 数据加载 ====================
def add_biological_features(df):
    L = df['体长(cm)'];
    W = df['体重(g)']
    df['log_L'] = np.log(L);
    df['log_W'] = np.log(W)
    df['L_W_ratio'] = L / W
    df['W_L2'] = W / (L ** 2)
    df['W_L3'] = W / (L ** 3)
    df['sqrt_W'] = np.sqrt(W);
    df['sqrt_L'] = np.sqrt(L)
    df['L2'] = L ** 2;
    df['L3'] = L ** 3
    df['log_L_log_W'] = np.log(L) * np.log(W)
    return df


def load_and_add_features(filepath):
    df = pd.read_excel(filepath, sheet_name='Sheet1')
    required_cols = ['体重(g)', '体长(cm)', 'chine', 'coccyx', 'cocc']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    cols_check = ['体重(g)', '体长(cm)', 'chine', 'coccyx', 'cocc']
    before = len(df)
    df = df[(df[cols_check] != 0).all(axis=1)].copy()
    print(f"  {os.path.basename(filepath)}: removed {before - len(df)} rows, remaining {len(df)}")
    if len(df) == 0:
        raise ValueError("No samples remain!")
    df['体长组'] = pd.cut(df['体长(cm)'], bins=LENGTH_BINS, labels=LENGTH_LABELS, right=False)
    length_map = {label: i + 1 for i, label in enumerate(LENGTH_LABELS)}
    df['体长组编码'] = df['体长组'].map(length_map).astype(float)
    df['体重组'] = pd.cut(df['体重(g)'], bins=WEIGHT_BINS, labels=WEIGHT_LABELS, right=False)
    weight_map = {label: i + 1 for i, label in enumerate(WEIGHT_LABELS)}
    df['体重组编码'] = df['体重组'].map(weight_map).astype(float)
    df.drop(['体长组', '体重组'], axis=1, inplace=True)
    df = add_biological_features(df)
    return df


print("Loading training data...")
train_df = load_and_add_features(TRAIN_PATH)
print(f"Training samples: {len(train_df)}")
print("Loading test data...")
test_df = load_and_add_features(TEST_PATH)
print(f"Test samples: {len(test_df)}")

# ==================== 数据可视化：检查特征与目标的关系 ====================
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for i, target in enumerate(ORIGINAL_TARGETS):
    # 体长 vs 目标
    axes[0, i].scatter(train_df['体长(cm)'], train_df[target], alpha=0.5, edgecolors='black', linewidth=0.5)
    axes[0, i].set_xlabel('体长(cm)')
    axes[0, i].set_ylabel(target)
    axes[0, i].set_title(f'{target} vs 体长')
    axes[0, i].grid(True, alpha=0.3)
    # 体重 vs 目标
    axes[1, i].scatter(train_df['体重(g)'], train_df[target], alpha=0.5, edgecolors='black', linewidth=0.5)
    axes[1, i].set_xlabel('体重(g)')
    axes[1, i].set_ylabel(target)
    axes[1, i].set_title(f'{target} vs 体重')
    axes[1, i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'data_scatter.png'), dpi=300, bbox_inches='tight')
plt.show()

# ==================== 特征选择：极简策略 ====================
FEATURE_SETS = {
    'minimal': ['体长(cm)', '体重(g)'],
    'with_group': ['体长(cm)', '体重(g)', '体长组编码'],
    'biological': ['体长(cm)', '体重(g)', 'log_L', 'L_W_ratio']
}

SELECTED_FEATURES = FEATURE_SETS['minimal']
print(f"\nUsing features: {SELECTED_FEATURES}")

X = train_df[SELECTED_FEATURES].values
y_orig = train_df[ORIGINAL_TARGETS].values
X_test = test_df[SELECTED_FEATURES].values
y_test = test_df[ORIGINAL_TARGETS].values

# ==================== 标准化 ====================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# ==================== 模型：Ridge回归（强基线） ====================
print("\n" + "=" * 60)
print("=== Ridge Regression Baseline ===")
print("=" * 60)

ridge_models = {}
ridge_cv_scores = {}  # 存储平均负RMSE

for i, target in enumerate(ORIGINAL_TARGETS):
    y = y_orig[:, i]
    fold_rmse = []
    for train_idx, val_idx in LeaveOneOut().split(X_scaled):
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_scaled[train_idx], y[train_idx])
        pred = ridge.predict(X_scaled[val_idx])
        # 计算RMSE（单样本，即绝对误差）
        rmse = np.sqrt(mean_squared_error(y[val_idx], pred))
        fold_rmse.append(rmse)
    mean_rmse = np.mean(fold_rmse)
    # 为了与之前选择逻辑兼容，使用负RMSE（越大越好）
    mean_neg_rmse = -mean_rmse
    ridge_cv_scores[target] = mean_neg_rmse
    print(f"{target}: LOOCV RMSE = {mean_rmse:.4f} (neg_RMSE = {mean_neg_rmse:.4f})")

    # 训练最终模型
    final_ridge = Ridge(alpha=1.0)
    final_ridge.fit(X_scaled, y)
    ridge_models[target] = final_ridge

# ==================== 可选：简化版GPR ====================
print("\n" + "=" * 60)
print("=== Simplified GPR (Optional) ===")
print("=" * 60)

gpr_models = {}
gpr_cv_scores = {}  # 存储平均负RMSE

for i, target in enumerate(ORIGINAL_TARGETS):
    y = y_orig[:, i]
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    fold_rmse = []
    failed = False

    for train_idx, val_idx in LeaveOneOut().split(X_scaled):
        try:
            gpr = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1.0,
                n_restarts_optimizer=0,
                normalize_y=True
            )
            gpr.fit(X_scaled[train_idx], y[train_idx])
            pred = gpr.predict(X_scaled[val_idx])
            # 检查预测是否有效
            if np.isnan(pred).any():
                failed = True
                break
            rmse = np.sqrt(mean_squared_error(y[val_idx], pred))
            fold_rmse.append(rmse)
        except Exception:
            failed = True
            break

    if failed or len(fold_rmse) == 0:
        print(f"{target}: GPR FAILED, using Ridge")
        gpr_models[target] = None
        gpr_cv_scores[target] = -np.inf
    else:
        mean_rmse = np.mean(fold_rmse)
        mean_neg_rmse = -mean_rmse
        gpr_cv_scores[target] = mean_neg_rmse
        print(f"{target}: LOOCV RMSE = {mean_rmse:.4f} (neg_RMSE = {mean_neg_rmse:.4f})")

        # 训练最终GPR
        final_gpr = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1.0,
            n_restarts_optimizer=0,
            normalize_y=True
        )
        final_gpr.fit(X_scaled, y)
        gpr_models[target] = final_gpr

# ==================== 选择最佳模型 ====================
best_models = {}
for target in ORIGINAL_TARGETS:
    if gpr_cv_scores[target] > ridge_cv_scores[target]:
        best_models[target] = ('gpr', gpr_models[target])
        print(f"\n{target}: Selected GPR (neg_RMSE={gpr_cv_scores[target]:.4f} vs Ridge={ridge_cv_scores[target]:.4f})")
    else:
        best_models[target] = ('ridge', ridge_models[target])
        print(f"\n{target}: Selected Ridge (neg_RMSE={ridge_cv_scores[target]:.4f})")

# ==================== 测试集预测 ====================
print("\n" + "=" * 60)
print("=== Test Set Prediction ===")
print("=" * 60)

y_test_pred = np.zeros_like(y_test)

for i, target in enumerate(ORIGINAL_TARGETS):
    model_type, model = best_models[target]
    pred = model.predict(X_test_scaled)
    y_test_pred[:, i] = pred

# ==================== 后处理（边界从训练集估计） ====================
def compute_sum_bounds(y_train, p_low=1, p_high=99):
    sums = y_train.sum(axis=1)
    return np.percentile(sums, p_low), np.percentile(sums, p_high)

LOW, HIGH = compute_sum_bounds(y_orig, 1, 99)
print(f"Sum bounds from training: [{LOW:.2f}, {HIGH:.2f}]")

def enforce_sum_constraint(y_pred, low, high):
    y_corrected = y_pred.copy()
    sums = y_pred.sum(axis=1)
    for i in range(len(sums)):
        s = sums[i]
        if s < low:
            y_corrected[i] = y_pred[i] * (low / s)
        elif s > high:
            y_corrected[i] = y_pred[i] * (high / s)
    return y_corrected

y_test_pred_scaled = enforce_sum_constraint(y_test_pred, LOW, HIGH)

# ==================== 评估 ====================
def evaluate(y_true, y_pred, names):
    results = {}
    for i, name in enumerate(names):
        yt, yp = y_true[:, i], y_pred[:, i]
        mask = ~(np.isnan(yt) | np.isnan(yp))
        yt, yp = yt[mask], yp[mask]
        if len(yt) == 0:
            continue
        r2 = r2_score(yt, yp)
        rmse = np.sqrt(mean_squared_error(yt, yp))
        mae = mean_absolute_error(yt, yp)
        mape = np.mean(np.abs((yt - yp) / yt)) * 100 if np.all(yt != 0) else np.nan
        sp, _ = spearmanr(yt, yp)
        results[name] = {'R2': r2, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'Spearman': sp}
    return results

print("\n--- Test metrics ---")
metrics = evaluate(y_test, y_test_pred_scaled, ORIGINAL_TARGETS)
for name, m in metrics.items():
    print(f"\n{name}:")
    for k, v in m.items():
        print(f"  {k}: {v:.4f}" if not np.isnan(v) else f"  {k}: nan")

pred_sums = y_test_pred_scaled.sum(axis=1)
in_range = np.sum((pred_sums >= LOW) & (pred_sums <= HIGH)) / len(pred_sums) * 100
print(f"\nSum constraint: {in_range:.1f}%")

# ==================== 保存 ====================
test_df_with_pred = test_df.copy()
for i, target in enumerate(ORIGINAL_TARGETS):
    test_df_with_pred[f'Pred_{target}'] = y_test_pred_scaled[:, i]
test_df_with_pred['Pred_Sum'] = pred_sums
test_df_with_pred.to_excel(os.path.join(OUTPUT_DIR, 'predictions.xlsx'), index=False)

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, target in enumerate(ORIGINAL_TARGETS):
    ax = axes[i]
    ax.scatter(y_test[:, i], y_test_pred_scaled[:, i], alpha=0.6, edgecolors='black', linewidth=0.5)
    mn = min(y_test[:, i].min(), y_test_pred_scaled[:, i].min())
    mx = max(y_test[:, i].max(), y_test_pred_scaled[:, i].max())
    ax.plot([mn, mx], [mn, mx], 'r--', lw=2)
    ax.set_xlabel('True');
    ax.set_ylabel('Predicted')
    model_type = best_models[target][0]
    ax.set_title(f'{target} ({model_type})\nR²={metrics[target]["R2"]:.3f}')
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'predictions.png'), dpi=300, bbox_inches='tight')
plt.show()

# 报告
with open(os.path.join(OUTPUT_DIR, 'report.txt'), 'w', encoding='utf-8') as f:
    f.write("STABLE MODEL REPORT (GPR Simplified)\n")
    f.write("=" * 60 + "\n\n")
    f.write("Model selection based on LOOCV negative RMSE (higher is better).\n")
    f.write("Features: " + str(SELECTED_FEATURES) + "\n\n")
    f.write("CV Results (neg_RMSE):\n")
    for t in ORIGINAL_TARGETS:
        f.write(f"  {t}: Ridge={ridge_cv_scores[t]:.4f}, GPR={gpr_cv_scores[t]:.4f}\n")
    f.write("\nTest Metrics:\n")
    for n, m in metrics.items():
        f.write(f"\n{n}:\n")
        for k, v in m.items():
            f.write(f"  {k}: {v:.4f}\n")

# 保存模型
for target, (model_type, model) in best_models.items():
    suffix = model_type
    joblib.dump(model, os.path.join(MODEL_DIR, f'{target}_{suffix}.pkl'))
joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))

print(f"\nSaved to {OUTPUT_DIR}")