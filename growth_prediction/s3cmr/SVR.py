import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from datetime import datetime
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
import warnings

warnings.filterwarnings('ignore')

# Set font for English (no Chinese needed)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']  # fallback fonts
plt.rcParams['axes.unicode_minus'] = False

# ==================== Configuration ====================
TRAIN_PATH = '体重分组训练集.xlsx'
TEST_PATH = '体重分组测试集.xlsx'
OUTPUT_DIR = './output_svr'          # results directory (distinct from raw)
MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# SVR parameters (can be tuned)
SVR_PARAMS = {
    'kernel': 'rbf',
    'C': 1.0,
    'gamma': 'scale',
    'epsilon': 0.1,
    'cache_size': 200
}
VAL_SIZE = 0.2          # validation set proportion from training data
RANDOM_STATE = 42

# ==================== Load Data ====================
def load_data(filepath):
    """Load Excel data, keep only required columns."""
    df = pd.read_excel(filepath, sheet_name='Sheet1')
    required_cols = ['体重(g)', '体长(cm)', 'chine', 'coccyx', 'cocc']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"File {filepath} missing columns: {missing}")
    df = df[required_cols].copy()
    # No data cleaning
    return df

print("Loading training data...")
train_df = load_data(TRAIN_PATH)
print(f"Training samples: {len(train_df)}")
print("Loading test data...")
test_df = load_data(TEST_PATH)
print(f"Test samples: {len(test_df)}")

# ==================== Features and Targets ====================
feature_cols = ['体重(g)', '体长(cm)']
target_cols = ['chine', 'coccyx', 'cocc']

X = train_df[feature_cols]
y = train_df[target_cols].values   # shape (n_samples, 3)

# Split train/validation (for monitoring, not used for tuning)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=VAL_SIZE, random_state=RANDOM_STATE, shuffle=True
)
print(f"Train set: {X_train.shape[0]} samples, Validation set: {X_val.shape[0]} samples")

X_test = test_df[feature_cols]
y_test = test_df[target_cols].values

# ==================== Train Model ====================
print("\nTraining SVR model (MultiOutputRegressor)...")
# Note: SVR does not support multi-output natively; use MultiOutputRegressor
base_svr = SVR(**SVR_PARAMS)
model = MultiOutputRegressor(base_svr, n_jobs=-1)  # use all cores
model.fit(X_train, y_train)
print("Model training completed!")

# ==================== Validation Evaluation ====================
y_val_pred = model.predict(X_val)
val_r2 = r2_score(y_val, y_val_pred, multioutput='raw_values')
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred, multioutput='raw_values'))
print(f"Validation R² (per target): {val_r2}")
print(f"Validation RMSE (per target): {val_rmse}")

# ==================== Test Set Prediction and Evaluation ====================
y_pred = model.predict(X_test)

def evaluate(y_true, y_pred, target_names):
    results = {}
    for i, name in enumerate(target_names):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        # Remove invalid values (NaNs) but no data filtering
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

print("\nTest set evaluation results:")
metrics = evaluate(y_test, y_pred, target_cols)
for name, metric in metrics.items():
    print(f"\n{name}:")
    for k, v in metric.items():
        print(f"  {k}: {v:.4f}")

# ==================== Save Predictions ====================
test_df_with_pred = test_df.copy()
test_df_with_pred['Predicted_chine'] = y_pred[:, 0]
test_df_with_pred['Predicted_coccyx'] = y_pred[:, 1]
test_df_with_pred['Predicted_cocc'] = y_pred[:, 2]
result_path = os.path.join(OUTPUT_DIR, 'SVR模型_预测结果.xlsx')
test_df_with_pred.to_excel(result_path, index=False)
print(f"\nPredictions saved to: {result_path}")

# ==================== Visualization ====================
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
        # Scatter plot
        ax = axes[0, i]
        ax.scatter(yt, yp, alpha=0.6)
        min_val = min(yt.min(), yp.min())
        max_val = max(yt.max(), yp.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1)
        ax.set_xlabel('True Value')
        ax.set_ylabel('Predicted Value')
        ax.set_title(f'{name}')
        # Residual plot
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

plot_predictions(y_test, y_pred, target_cols,
                 save_path=os.path.join(OUTPUT_DIR, 'SVR模型_预测图.png'))

# Feature importance is not available for SVR; skip or comment out
# If desired, one could compute permutation importance, but we skip here.
print("Note: SVR does not provide built-in feature importance; skipping feature importance plot.")

# ==================== Save Evaluation Report ====================
report_path = os.path.join(OUTPUT_DIR, 'SVR模型_评估报告.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("SVR Model Evaluation Report (MultiOutputRegressor, only length + weight)\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")
    f.write("Validation metrics:\n")
    for i, name in enumerate(target_cols):
        f.write(f"  {name}: R²={val_r2[i]:.4f}, RMSE={val_rmse[i]:.4f}\n")
    f.write("\nTest metrics:\n")
    for name, metric in metrics.items():
        f.write(f"\n{name}:\n")
        for k, v in metric.items():
            f.write(f"  {k}: {v:.4f}\n")
print(f"Evaluation report saved to: {report_path}")

# ==================== Save Model ====================
model_path = os.path.join(MODEL_DIR, 'svr_model.pkl')
joblib.dump(model, model_path)
print(f"Model saved to: {model_path}")

print("\nAll tasks completed!")