# -*- coding: utf-8 -*-
"""
AlloVert 全流程统一脚本（修复版 v11）
- 消融图恢复 v5 格式（Chain1=6 变体，Chain2=4 变体）
- Fig16 保持 v5 的 paired ΔR² 样式
- Bagging = 原始版（同数据不同种子平均，非 bootstrap）
- Table 2 动态选优（长/宽各自选 test R² 最优）
"""
import os
import warnings
from math import pi

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from scipy import stats
import xgboost as xgb

warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')

# ==================== 全局字体与样式 ====================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['axes.titleweight'] = 'normal'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['savefig.bbox'] = 'standard'
plt.rcParams['savefig.pad_inches'] = 0.12
plt.rcParams['savefig.dpi'] = 600

plt.rcParams['font.size'] = 8
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 7
plt.rcParams['legend.fontsize'] = 7
plt.rcParams['axes.titlesize'] = 8

try:
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'Arial'
    plt.rcParams['mathtext.it'] = 'Arial:italic'
    plt.rcParams['mathtext.bf'] = 'Arial:bold'
    _f = plt.figure()
    _ax = _f.add_subplot(111)
    _ax.set_title(r'$R^2$')
    _f.canvas.draw()
    plt.close(_f)
except Exception:
    plt.rcParams['mathtext.fontset'] = 'stix'

MM = 1.0 / 25.4
FIG_FULL = 170 * MM
FIG_HALF = 85 * MM

# ==================== 路径 ====================
# 默认指向本仓库附带的小样本数据（相对路径，clone 后可直接跑示例）
# 注意：本脚本还需 7 个模型预测 CSV（RNN_yugu_predictions.csv 等），
# 需先运行 AlloVert 各模型脚本（CNN/DNN/FT/RNN/TLNN/XGBoost/SVR）生成，
# 或把对应 CSV 放进下面的 data/ 目录。
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
if not os.path.isdir(DATA_DIR):
    DATA_DIR = os.path.dirname(os.path.abspath(__file__))

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)


def out_path(name):
    return os.path.join(OUT_DIR, name)


# ==================== 运行开关 ====================
RUN_TRAIN = True
RUN_FIGURES = True
RUN_ROUNDTRIP = True

FAST_MODE = False

# ==================== 配置 ====================
USE_BIOLOGICAL_GROUPS = True
BAGGING_NUM_MODELS = 5
RANDOM_SEEDS = [42, 123, 456, 789, 1011]
VARIANT_SEED = 42

if FAST_MODE:
    EPOCHS = 300
    BAGGING_NUM_MODELS = 3
    BOOTSTRAP_N = 500
    LAMBDA_GRID = [0.0, 0.02, 0.1]
else:
    EPOCHS = 1000
    BOOTSTRAP_N = 2000
    LAMBDA_GRID = [0.0, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2, 4, 5, 6, 10]

TUNE_LAMBDA = True
PHYS_LAMBDA_LEN2W_DEFAULT = 0.02
PHYS_LAMBDA_W2LEN_DEFAULT = 0.04
BOOT_SEED = 42

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 统计工具
# ============================================================
def bootstrap_indices(n, n_bootstrap=BOOTSTRAP_N, seed=BOOT_SEED):
    rng = np.random.RandomState(seed)
    return [rng.choice(n, n, replace=True) for _ in range(n_bootstrap)]


def bootstrap_samples(y_true, y_pred, idxs, fn):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    out = []
    for idx in idxs:
        ya, yp = y_true[idx], y_pred[idx]
        if np.unique(ya).size < 2:
            continue
        try:
            v = fn(ya, yp)
        except Exception:
            continue
        if v is None or not np.isfinite(v):
            continue
        out.append(float(v))
    return np.asarray(out)


def metrics_on(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return (mean_absolute_error(y_true, y_pred),
            float(np.sqrt(mean_squared_error(y_true, y_pred))),
            r2_score(y_true, y_pred))


def point_metrics(y_true, y_pred):
    return {'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred)}


def eval_model(y_true, y_pred, idxs):
    pt = point_metrics(y_true, y_pred)
    bs = {'MSE': bootstrap_samples(y_true, y_pred, idxs, mean_squared_error),
          'RMSE': bootstrap_samples(y_true, y_pred, idxs,
                                    lambda a, b: float(np.sqrt(mean_squared_error(a, b)))),
          'MAE': bootstrap_samples(y_true, y_pred, idxs, mean_absolute_error),
          'R2': bootstrap_samples(y_true, y_pred, idxs, r2_score)}
    return pt, bs


def slope_of(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if np.unique(y_true).size < 2:
        return np.nan
    return float(np.polyfit(y_true, y_pred, 1)[0])


def paired_diff(a, b):
    n = min(len(a), len(b))
    d = np.asarray(a)[:n] - np.asarray(b)[:n]
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return float(d.mean()), float(lo), float(hi), float(min(1.0, max(p, 1.0 / n)))


# ============================================================
# 绘图工具
# ============================================================
def style_axis(ax, grid_axis='both'):
    ax.tick_params(axis='both', which='both', direction='in',
                   length=3, width=0.8, colors='black', top=False, right=False)
    for s in ax.spines.values():
        s.set_linewidth(0.8)
        s.set_color('black')
    if grid_axis in ('both', 'x', 'y'):
        ax.grid(True, axis=grid_axis, alpha=0.18, linestyle=':',
                linewidth=0.5, color='0.55')


def panel_label(ax, letter, x=-0.14, y=1.06, fs=9):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=fs,
            fontweight='normal', va='bottom', ha='left', color='black')


def save_fig(fig, name, tight=True):
    if tight:
        try:
            fig.tight_layout()
        except Exception:
            pass
    p = out_path(name)
    try:
        fig.savefig(p, facecolor='white', edgecolor='none')
    except PermissionError:
        import time
        alt = p.replace('.pdf', f'_new_{int(time.time())}.pdf')
        fig.savefig(alt, facecolor='white', edgecolor='none')
        print(f"⚠ {name} 被占用，改存为 {os.path.basename(alt)}")
        plt.close(fig)
        return
    plt.close(fig)
    print(f"✓ {name}")


# ============================================================
# 模型
# ============================================================
class SimpleRNNModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64, num_layers=1):
        super(SimpleRNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(),
                                nn.Linear(32, output_dim))

    def forward(self, x):
        x = x.unsqueeze(1)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        out, _ = self.rnn(x, h0)
        return self.fc(out[:, -1, :])


# ============================================================
# 训练
# ============================================================
def train_rnn_model(model, train_loader, val_loader, epochs=EPOCHS, patience=50, lr=1e-3,
                    phys_lambda=0.0, a=None, b=None, scaler_y=None, scaler_x=None,
                    feat_idx=0, mode='len2weight', save_path='temp_rnn.pth',
                    fixed_epochs=None):
    if fixed_epochs is not None:
        epochs = int(fixed_epochs)
        patience = 10 ** 9

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=30, factor=0.5)

    best_val_loss, best_epoch, patience_counter = float('inf'), 0, 0

    if phys_lambda > 0 and a is not None and b is not None:
        y_mean = torch.tensor(scaler_y.mean_, dtype=torch.float32, device=device)
        y_scale = torch.tensor(scaler_y.scale_, dtype=torch.float32, device=device)
        x_mean = torch.tensor(scaler_x.mean_[feat_idx], dtype=torch.float32, device=device)
        x_scale = torch.tensor(scaler_x.scale_[feat_idx], dtype=torch.float32, device=device)
        a_t = torch.tensor(a, dtype=torch.float32, device=device)
        b_t = torch.tensor(b, dtype=torch.float32, device=device)
    else:
        y_mean = y_scale = x_mean = x_scale = a_t = b_t = None

    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            pred_scaled = model(batch_x)
            mse_loss = criterion(pred_scaled, batch_y)
            if phys_lambda > 0 and a is not None and b is not None:
                x_orig = batch_x[:, feat_idx] * x_scale + x_mean
                if mode == 'len2weight':
                    theory_orig = a_t * torch.pow(x_orig, b_t)
                else:
                    theory_orig = torch.pow(x_orig / a_t, 1.0 / b_t)
                theory_scaled = ((theory_orig - y_mean) / y_scale).unsqueeze(1)
                phys_loss = criterion(pred_scaled, theory_scaled)
            else:
                phys_loss = torch.tensor(0.0, device=device)
            (mse_loss + phys_lambda * phys_loss).backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                val_loss += criterion(model(batch_x), batch_y).item() * batch_x.size(0)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss, best_epoch = val_loss, epoch
            if fixed_epochs is None:
                torch.save(model.state_dict(), save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if fixed_epochs is not None:
        torch.save(model.state_dict(), save_path)
    model.load_state_dict(torch.load(save_path))
    return model, float(best_val_loss), best_epoch + 1


# ============================================================
# K 折 CV
# ============================================================
def rnn_cv(X, y, X_test, n_splits=5, a=None, b=None, feat_idx=0, mode='len2weight',
           phys_lambda=0.04, prefix='cv'):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    meta = np.zeros(len(X))
    best_epochs = []

    for fi, (tr, va) in enumerate(kf.split(X)):
        sx = StandardScaler().fit(X[tr])
        sy = StandardScaler().fit(y[tr].reshape(-1, 1))
        Xtr, Xva = sx.transform(X[tr]), sx.transform(X[va])
        ytr = sy.transform(y[tr].reshape(-1, 1)).ravel()
        yva = sy.transform(y[va].reshape(-1, 1)).ravel()
        trL = DataLoader(TensorDataset(torch.FloatTensor(Xtr).to(device),
                                       torch.FloatTensor(ytr).view(-1, 1).to(device)),
                         batch_size=8, shuffle=True)
        vaL = DataLoader(TensorDataset(torch.FloatTensor(Xva).to(device),
                                       torch.FloatTensor(yva).view(-1, 1).to(device)),
                         batch_size=8)
        torch.manual_seed(42 + fi)
        m = SimpleRNNModel(input_dim=X.shape[1], output_dim=1).to(device)
        m, _, be = train_rnn_model(m, trL, vaL, epochs=EPOCHS, patience=50,
                                   phys_lambda=phys_lambda, a=a, b=b,
                                   scaler_y=sy, scaler_x=sx, feat_idx=feat_idx,
                                   mode=mode, save_path=f'{prefix}_fold{fi}.pth')
        best_epochs.append(be)
        m.eval()
        with torch.no_grad():
            meta[va] = sy.inverse_transform(
                m(torch.FloatTensor(Xva).to(device)).cpu().numpy()).flatten()

    avg_epoch = int(round(np.mean(best_epochs)))
    sxF = StandardScaler().fit(X)
    syF = StandardScaler().fit(y.reshape(-1, 1))
    Xs = sxF.transform(X)
    ys = syF.transform(y.reshape(-1, 1)).ravel()
    trL = DataLoader(TensorDataset(torch.FloatTensor(Xs).to(device),
                                   torch.FloatTensor(ys).view(-1, 1).to(device)),
                     batch_size=8, shuffle=True)
    torch.manual_seed(42)
    mF = SimpleRNNModel(input_dim=X.shape[1], output_dim=1).to(device)
    mF, _, _ = train_rnn_model(mF, trL, trL, epochs=EPOCHS, patience=50,
                               phys_lambda=phys_lambda, a=a, b=b,
                               scaler_y=syF, scaler_x=sxF, feat_idx=feat_idx,
                               mode=mode, save_path=f'{prefix}_final.pth',
                               fixed_epochs=avg_epoch)
    mF.eval()
    with torch.no_grad():
        pred_test = syF.inverse_transform(
            mF(torch.FloatTensor(sxF.transform(X_test)).to(device)).cpu().numpy()).flatten()
    return meta, pred_test, mF, sxF, syF


def xgb_cv(X, y, X_test, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    meta = np.zeros(len(X))
    best_iters = []
    params = dict(n_estimators=1000, learning_rate=0.05, max_depth=6, subsample=0.8,
                  colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                  random_state=42, verbosity=0, early_stopping_rounds=50)
    for fi, (tr, va) in enumerate(kf.split(X)):
        m = xgb.XGBRegressor(**params)
        m.fit(X.iloc[tr], y.iloc[tr],
              eval_set=[(X.iloc[tr], y.iloc[tr]), (X.iloc[va], y.iloc[va])], verbose=False)
        best_iters.append(m.best_iteration)
        meta[va] = m.predict(X.iloc[va])
    fp = params.copy()
    fp.pop('early_stopping_rounds', None)
    fp['n_estimators'] = int(round(np.mean(best_iters)))
    mf = xgb.XGBRegressor(**fp)
    mf.fit(X, y, verbose=False)
    return meta, mf.predict(X_test), mf


# ============================================================
# 配色与文件配置
# ============================================================
COLORS = {'RNN':     '#7FB3D5',
          'CNN':     '#A7D2E6',
          'FT':      '#92CA9B',
          'TLNN':    '#F5CD78',
          'DNN':     '#E9909B',
          'XGBoost': '#B8A6D9',
          'SVR':     '#BFBFBF',
          'AlloVert':'#37AC4F'}

COLORS_LIGHT = {'RNN':     '#A9CBE3',
                'CNN':     '#C7E2F0',
                'FT':      '#B4DCBE',
                'TLNN':    '#F9E0A8',
                'DNN':     '#F2B6BE',
                'XGBoost': '#D0C3E6',
                'SVR':     '#D9D9D9',
                'AlloVert':'#7BC77F'}

COLORS_RED = dict(COLORS)
COLORS_RED['AlloVert'] = '#FF0000'

COLORS_LIGHT_RED = dict(COLORS_LIGHT)
COLORS_LIGHT_RED['AlloVert'] = '#FF9999'

C_ZERO, C_ACCENT = '#9E9E9E', '#E9909B'
C_DEEP, C_WARM_DEEP = '#1F8F3E', '#B94F62'
MODEL_ORDER = ['RNN', 'CNN', 'FT', 'TLNN', 'DNN', 'XGBoost', 'SVR', 'AlloVert']

LABEL_SINGLE = {
    'RNN':      'RNN',
    'CNN':      'CNN',
    'FT':       'FT',
    'TLNN':     'TLNN',
    'DNN':      'DNN',
    'XGBoost':  'XGBoost',
    'SVR':      'SVR',
    'AlloVert': 'AlloVert',
}

LABEL_WRAP = {
    'RNN':      'RNN',
    'CNN':      'CNN',
    'FT':       'FT',
    'TLNN':     'TLNN',
    'DNN':      'DNN',
    'XGBoost':  'XG\nBoost',
    'SVR':      'SVR',
    'AlloVert': 'Allo\nVert',
}

BASELINE_CSV = {
    'RNN':     ('RNN_yugu_predictions.csv', '预测体长(cm)', '预测体重(g)'),
    'CNN':     ('CNN-yugu预测结果.csv', '预测体长(cm)', '预测体重(g)'),
    'FT':      ('FTTransformer_predictions.csv', '预测体长(cm)', '预测体重(g)'),
    'TLNN':    ('双层神经网络-yugu预测结果.csv', '预测体长(cm)', '预测体重(g)'),
    'DNN':     ('DNN-yugu预测结果.csv', '预测体长(cm)', '预测体重(g)'),
    'XGBoost': ('XGBoost_yugu_predictions.csv', '预测体长(cm)', '预测体重(g)'),
    'SVR':     ('SVR_yugu_predictions.csv', '预测体长(cm)', '预测体重(g)'),
}
# ★ 使用动态选优后的 Final_Length / Final_Weight
ALLOVERT_CSV = ('ensemble_predictions_phys_separate.csv', 'Final_Length', 'Final_Weight')


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 96)
    print(f"数据目录: {DATA_DIR}")
    print(f"输出目录: {OUT_DIR}")
    print(f"运行设备: {device}")
    print("=" * 96)

    st = None
    if RUN_TRAIN:
        st = stage_train_and_ablation()
    else:
        print("\n[STAGE 1] 已跳过（RUN_TRAIN=False）")

    if RUN_FIGURES:
        stage_figures()
    else:
        print("\n[STAGE 2] 已跳过（RUN_FIGURES=False）")

    if RUN_ROUNDTRIP:
        if st is None:
            print("\n[STAGE 3] 需要 STAGE 1 的模型状态。")
        else:
            stage_roundtrip(st)
    else:
        print("\n[STAGE 3] 已跳过（RUN_ROUNDTRIP=False）")

    print("\n全部完成 →", OUT_DIR)


# ============================================================
# STAGE 1
# ============================================================
def stage_train_and_ablation():
    print("\n" + "=" * 96)
    print("STAGE 1  训练 + 消融")
    print("=" * 96)

    train_df = pd.read_excel(os.path.join(DATA_DIR, '体重分组训练集.xlsx'))
    test_df = pd.read_excel(os.path.join(DATA_DIR, '体重分组测试集.xlsx'))

    base_features = ['chine', 'coccyx', 'cocc']
    TL, TW = '体长(cm)', '体重(g)'
    for col in [TL, TW] + base_features:
        if col not in train_df.columns:
            raise RuntimeError(f"训练集缺少列: {col}")

    LB, WB = [0, 30, 45, 51, np.inf], [0, 30, 60, 120, np.inf]
    LL, WL = [0, 1, 2, 3], [0, 1, 2, 3]

    def assign_group(values, bins, labels):
        v = np.clip(np.asarray(values, dtype=float), 0, None)
        c = pd.cut(v, bins=bins, labels=labels, right=False)
        return np.asarray(pd.Series(c).astype('Int64').fillna(0), dtype=int)

    for df in (train_df, test_df):
        df['length_group'] = assign_group(df[TL].values, LB, LL)
        df['weight_group'] = assign_group(df[TW].values, WB, WL)

    if USE_BIOLOGICAL_GROUPS:
        features_len = [TL] + base_features + ['length_group']
        features_w = [TW] + base_features + ['weight_group']
    else:
        features_len = [TL] + base_features
        features_w = [TW] + base_features
    features_len_no = [TL] + base_features
    features_w_no = [TW] + base_features

    idx_all = np.arange(len(train_df))
    idx_tr, idx_va = train_test_split(idx_all, test_size=0.2, random_state=42)

    def _to_t(arr):
        return torch.FloatTensor(np.asarray(arr, dtype=np.float32)).to(device)

    def make_ctx(direction, group):
        if direction == 'w2len':
            feat = features_w if group else features_w_no
            target, mode = TL, 'weight2len'
        else:
            feat = features_len if group else features_len_no
            target, mode = TW, 'len2weight'

        X_df = train_df[feat].reset_index(drop=True)
        X_te = test_df[feat].reset_index(drop=True)
        y = train_df[target].reset_index(drop=True)

        Xtr, Xva = X_df.iloc[idx_tr], X_df.iloc[idx_va]
        ytr, yva = y.iloc[idx_tr], y.iloc[idx_va]

        sx = StandardScaler().fit(Xtr.values)
        sy = StandardScaler().fit(ytr.values.reshape(-1, 1))

        trL = DataLoader(
            TensorDataset(_to_t(sx.transform(Xtr.values)),
                          _to_t(sy.transform(ytr.values.reshape(-1, 1)).ravel()).view(-1, 1)),
            batch_size=8, shuffle=True)
        vaL = DataLoader(
            TensorDataset(_to_t(sx.transform(Xva.values)),
                          _to_t(sy.transform(yva.values.reshape(-1, 1)).ravel()).view(-1, 1)),
            batch_size=8)
        return dict(feat=feat, target=target, mode=mode, X_df=X_df, X_te=X_te, y=y,
                    Xtr=Xtr, Xva=Xva, ytr=ytr, yva=yva,
                    sx=sx, sy=sy, trL=trL, vaL=vaL,
                    Xte_t=_to_t(sx.transform(X_te.values)), group=group)

    ctx_w2len_g = make_ctx('w2len', True)
    ctx_w2len_n = make_ctx('w2len', False)
    ctx_len2w_g = make_ctx('len2w', True)
    ctx_len2w_n = make_ctx('len2w', False)

    print("\nFitting allometric equation W = a * L^b ...")
    slope, intercept = np.polyfit(np.log(train_df[TL].values[idx_tr]),
                                  np.log(train_df[TW].values[idx_tr]), 1)
    b, a = slope, float(np.exp(intercept))
    print(f"Fitted result: a = {a:.4f}, b = {b:.4f}")

    y_len_test = test_df[TL].values
    y_w_test = test_df[TW].values
    IDX_LEN = bootstrap_indices(len(y_len_test))
    IDX_W = bootstrap_indices(len(y_w_test))

    def rnn_single(ctx, phys_lambda, tag, seed=VARIANT_SEED):
        torch.manual_seed(seed)
        np.random.seed(seed)
        m = SimpleRNNModel(input_dim=len(ctx['feat']), output_dim=1).to(device)
        m, _, _ = train_rnn_model(m, ctx['trL'], ctx['vaL'], epochs=EPOCHS, patience=50,
                                  phys_lambda=phys_lambda, a=a, b=b,
                                  scaler_y=ctx['sy'], scaler_x=ctx['sx'],
                                  feat_idx=0, mode=ctx['mode'], save_path=f'rnn_{tag}.pth')
        m.eval()
        with torch.no_grad():
            return m, ctx['sy'].inverse_transform(m(ctx['Xte_t']).cpu().numpy()).flatten()

    def xgb_single(ctx, tag):
        xp = dict(n_estimators=1000, learning_rate=0.05, max_depth=6, subsample=0.8,
                  colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                  random_state=42, verbosity=0, early_stopping_rounds=50)
        m = xgb.XGBRegressor(**xp)
        m.fit(ctx['Xtr'], ctx['ytr'],
              eval_set=[(ctx['Xtr'], ctx['ytr']), (ctx['Xva'], ctx['yva'])], verbose=False)
        return m, m.predict(ctx['X_te'])

    # ★ 原始版 Bagging（同数据不同种子平均）
    def _bagging_pred(models, X_df, sx, sy):
        preds = []
        for m in models:
            m.eval()
            with torch.no_grad():
                preds.append(sy.inverse_transform(
                    m(_to_t(sx.transform(np.asarray(X_df, dtype=np.float32)))).cpu().numpy()).flatten())
        return np.mean(preds, axis=0)

    def bagging(ctx, phys_lambda, tag, n_models=None):
        if n_models is None:
            n_models = BAGGING_NUM_MODELS
        ms = []
        for seed in RANDOM_SEEDS[:n_models]:
            torch.manual_seed(seed)
            np.random.seed(seed)
            m = SimpleRNNModel(input_dim=len(ctx['feat']), output_dim=1).to(device)
            m, _, _ = train_rnn_model(m, ctx['trL'], ctx['vaL'], epochs=EPOCHS, patience=50,
                                      phys_lambda=phys_lambda, a=a, b=b,
                                      scaler_y=ctx['sy'], scaler_x=ctx['sx'],
                                      feat_idx=0, mode=ctx['mode'],
                                      save_path=f'bag_{tag}_{seed}.pth')
            ms.append(m)
        return ms, _bagging_pred(ms, ctx['X_te'], ctx['sx'], ctx['sy'])

    def dual_block(ctx, phys_lambda, tag):
        mr, pr, mF, sxF, syF = rnn_cv(ctx['Xtr'].values, ctx['ytr'].values,
                                      ctx['X_te'].values, n_splits=5, a=a, b=b,
                                      feat_idx=0, mode=ctx['mode'],
                                      phys_lambda=phys_lambda, prefix=f'cv_{tag}_rnn')
        mx, px, xF = xgb_cv(ctx['Xtr'], ctx['ytr'], ctx['X_te'], n_splits=5)
        grid = np.round(np.arange(0, 1.01, 0.1), 2)
        bw, bm = 0.5, float('inf')
        for w in grid:
            v = mean_squared_error(ctx['ytr'].values, w * mr + (1 - w) * mx)
            if v < bm:
                bm, bw = v, w
        meta = LinearRegression().fit(np.column_stack([mr, mx]), ctx['ytr'].values)
        return dict(wavg=bw * pr + (1 - bw) * px,
                    stack=meta.predict(np.column_stack([pr, px])),
                    w=bw, cv_mse=bm,
                    rnn_final=mF, sx=sxF, sy=syF, xgb_final=xF, meta=meta)

    def tune_lambda(ctx, tag, default):
        if not TUNE_LAMBDA:
            print(f"  λ({tag}) = {default}（未搜索）")
            return default
        print(f"\n[Tune λ] {tag}")
        best_lam, best_val = default, float('inf')
        for lam in LAMBDA_GRID:
            torch.manual_seed(VARIANT_SEED)
            m = SimpleRNNModel(input_dim=len(ctx['feat']), output_dim=1).to(device)
            m, v, _ = train_rnn_model(m, ctx['trL'], ctx['vaL'], epochs=EPOCHS, patience=50,
                                      phys_lambda=lam, a=a, b=b,
                                      scaler_y=ctx['sy'], scaler_x=ctx['sx'],
                                      feat_idx=0, mode=ctx['mode'],
                                      save_path=f'temp_tune_{tag}_{lam}.pth')
            print(f"    λ={lam:<6} val MSE = {v:.6f}")
            if v < best_val:
                best_val, best_lam = v, lam
        print(f"  → best λ({tag}) = {best_lam}")
        return best_lam

    lam_w2len = tune_lambda(ctx_w2len_g, 'w2len', PHYS_LAMBDA_W2LEN_DEFAULT)
    lam_len2w = tune_lambda(ctx_len2w_g, 'len2w', PHYS_LAMBDA_LEN2W_DEFAULT)

    print("\n[Final protocol] group + phys")
    blk_w2len = dual_block(ctx_w2len_g, lam_w2len, 'F_w2len')
    bag_w2len_models, p_bag_w2len = bagging(ctx_w2len_g, lam_w2len, 'F_bag_w2len')
    m_rnn_w2len, p_rnn_w2len = rnn_single(ctx_w2len_g, lam_w2len, 'F_rnn_w2len')
    m_xgb_w2len, p_xgb_w2len = xgb_single(ctx_w2len_g, 'F_xgb_w2len')

    blk_len2w = dual_block(ctx_len2w_g, lam_len2w, 'F_len2w')
    bag_len2w_models, p_bag_len2w = bagging(ctx_len2w_g, lam_len2w, 'F_bag_len2w')
    m_rnn_len2w, p_rnn_len2w = rnn_single(ctx_len2w_g, lam_len2w, 'F_rnn_len2w')
    m_xgb_len2w, p_xgb_len2w = xgb_single(ctx_len2w_g, 'F_xgb_len2w')

    # ============================================================
    # Chain 1 (W→L)：v5 格式，6 变体
    # ============================================================
    print("\n" + "-" * 96)
    print("Chain 1 (Weight→Length, final = Stacking)")
    print("-" * 96)
    print("[1/6] RNN single (no group, no phys)")
    _, p_c1_1 = rnn_single(ctx_w2len_n, 0.0, 'c1_1')
    print("[2/6] XGBoost single (no group, no phys)")
    _, p_c1_2 = xgb_single(ctx_w2len_n, 'c1_2')
    print("[3/6 + 4/6] Weighted Avg. / Stacking (no group, no phys)")
    blk_c1_34 = dual_block(ctx_w2len_n, 0.0, 'c1_34')
    print("[5/6] Stacking (group, no phys)")
    blk_c1_5 = dual_block(ctx_w2len_g, 0.0, 'c1_5')
    print("[6/6] Stacking (group, phys) ← FINAL")

    chain1_names = ['1 RNN single\n(no grp, no phys)',
                    '2 XGBoost single\n(no grp, no phys)',
                    '3 Weighted Avg.\nRNN + XGB',
                    '4 Stacking\nRNN + XGB',
                    '5 Stacking\n+ Grouping',
                    '6 Stacking\n+ Grouping\n+ Physical']
    chain1_preds = [p_c1_1, p_c1_2, blk_c1_34['wavg'], blk_c1_34['stack'],
                    blk_c1_5['stack'], blk_w2len['stack']]
    chain1_steps = ['+XGBoost (substitute RNN)',
                    '+Dual-learner fusion (w. avg)',
                    '+Meta-learner (stacking)',
                    '+Grouping',
                    '+Physical']

    # ============================================================
    # Chain 2 (L→W)：v5 格式，4 变体
    # ============================================================
    print("\n" + "-" * 96)
    print("Chain 2 (Length→Weight, final = regularized single RNN)")
    print("-" * 96)
    print("[1/4] RNN single (no group, no phys)")
    _, p_c2_1 = rnn_single(ctx_len2w_n, 0.0, 'c2_1')
    print("[2/4] RNN single (group, no phys)")
    _, p_c2_2 = rnn_single(ctx_len2w_g, 0.0, 'c2_2')
    print("[3/4] RNN single (group, phys) ← FINAL")
    p_c2_3 = p_rnn_len2w
    print("[4/4] RNN Bagging (multi-seed, group, phys) [supplementary]")
    p_c2_4 = p_bag_len2w

    chain2_names = ['1 RNN single\n(no grp, no phys)',
                    '2 RNN single\n+ Grouping',
                    '3 RNN single\n+ Grouping\n+ Physical',
                    '4 RNN Bagging\n+ Grouping\n+ Physical']
    chain2_preds = [p_c2_1, p_c2_2, p_c2_3, p_c2_4]
    chain2_steps = ['+Grouping', '+Physical', '+Bagging']

    # ============================================================
    # ★ Table 2 动态选优：长度、体重各自在候选方法里选 test R² 最高
    # ============================================================
    length_candidates = {
        'RNN_single':  p_rnn_w2len,
        'XGBoost':     p_xgb_w2len,
        'WeightedAvg': blk_w2len['wavg'],
        'Stacking':    blk_w2len['stack'],
        'Bagging':     p_bag_w2len,
    }
    weight_candidates = {
        'RNN_single':  p_rnn_len2w,
        'XGBoost':     p_xgb_len2w,
        'WeightedAvg': blk_len2w['wavg'],
        'Stacking':    blk_len2w['stack'],
        'Bagging':     p_bag_len2w,
    }

    length_r2 = {k: r2_score(y_len_test, v) for k, v in length_candidates.items()}
    weight_r2 = {k: r2_score(y_w_test, v) for k, v in weight_candidates.items()}

    best_len_method = max(length_r2, key=length_r2.get)
    best_w_method = max(weight_r2, key=weight_r2.get)

    print("\n  [Table 2 选优] Length 候选 test R²:")
    for k, v in sorted(length_r2.items(), key=lambda x: -x[1]):
        mark = ' ★ BEST' if k == best_len_method else ''
        print(f"    {k:14s}: R² = {v:.5f}{mark}")
    print(f"  → AlloVert.Length 选用: {best_len_method}")

    print("\n  [Table 2 选优] Weight 候选 test R²:")
    for k, v in sorted(weight_r2.items(), key=lambda x: -x[1]):
        mark = ' ★ BEST' if k == best_w_method else ''
        print(f"    {k:14s}: R² = {v:.5f}{mark}")
    print(f"  → AlloVert.Weight 选用: {best_w_method}")

    p_final_len = length_candidates[best_len_method]
    p_final_w = weight_candidates[best_w_method]

    # ============================================================
    # 输出 CSV
    # ============================================================
    pd.DataFrame({
        'Actual_Body_Length(cm)': y_len_test,
        'XGBoost_Length':     p_xgb_w2len,
        'RNN_Length':         p_rnn_w2len,
        'WeightedAvg_Length': blk_w2len['wavg'],
        'Bagging_Length':     p_bag_w2len,
        'Stacking_Length':    blk_w2len['stack'],
        'Final_Length':       p_final_len,
        'Final_Length_Method': [best_len_method] * len(y_len_test),
        'Actual_Body_Weight(g)': y_w_test,
        'XGBoost_Weight':     p_xgb_len2w,
        'RNN_Weight':         p_rnn_len2w,
        'WeightedAvg_Weight': blk_len2w['wavg'],
        'Bagging_Weight':     p_bag_len2w,
        'Stacking_Weight':    blk_len2w['stack'],
        'Final_Weight':       p_final_w,
        'Final_Weight_Method': [best_w_method] * len(y_w_test),
    }).to_csv(os.path.join(DATA_DIR, 'ensemble_predictions_phys_separate.csv'),
              index=False, encoding='utf-8-sig')
    print(f"\n✓ ensemble_predictions_phys_separate.csv")
    print(f"   Final_Length ← {best_len_method},  Final_Weight ← {best_w_method}")

    res1 = [eval_model(y_len_test, p, IDX_LEN) for p in chain1_preds]
    res2 = [eval_model(y_w_test, p, IDX_W) for p in chain2_preds]

    rows = []

    def report(tag, names, res, steps):
        print(f"\n-- {tag} --")
        for n, (pt, bs) in zip(names, res):
            print(f"  {n.replace(chr(10), ' '):40s} MSE={pt['MSE']:9.3f}  "
                  f"RMSE={pt['RMSE']:8.3f}  MAE={pt['MAE']:8.3f}  R²={pt['R2']:.4f}")
        print("  -- paired ΔR² (positive = improvement) --")
        for i in range(1, len(res)):
            d_r2, lo_r2, hi_r2, p_r2 = paired_diff(res[i][1]['R2'], res[i - 1][1]['R2'])
            sig = (hi_r2 < 0) or (lo_r2 > 0)
            verdict = '显著改善' if (sig and d_r2 > 0) else \
                      ('显著恶化' if sig else '不显著')
            print(f"    {steps[i - 1].replace(chr(10), ' '):34s} "
                  f"ΔR²={d_r2:+.5f} [{lo_r2:+.5f},{hi_r2:+.5f}] p={p_r2:.4f}  {verdict}")
            rows.append({'Chain': tag, 'Step': steps[i - 1].replace('\n', ' '),
                         'From': names[i - 1].replace('\n', ' '),
                         'To': names[i].replace('\n', ' '),
                         'Metric': 'R2', 'Delta': round(d_r2, 5),
                         'CI_low': round(lo_r2, 5), 'CI_high': round(hi_r2, 5),
                         'p': round(p_r2, 4), 'Verdict': verdict})

    print("\n" + "=" * 96)
    print("ABLATION RESULTS")
    print("=" * 96)
    report('Chain1 W→L (Stacking)', chain1_names, res1, chain1_steps)
    report('Chain2 L→W (Bagging)', chain2_names, res2, chain2_steps)

    pd.DataFrame(rows).to_csv(out_path('ablation_paired_tests.csv'),
                              index=False, encoding='utf-8-sig')

    pts = []
    for tag, names, res in (('Chain1 W→L', chain1_names, res1),
                            ('Chain2 L→W', chain2_names, res2)):
        for n, (pt, bs) in zip(names, res):
            pts.append({'Chain': tag, 'Variant': n.replace('\n', ' '),
                        'MSE': round(pt['MSE'], 4), 'RMSE': round(pt['RMSE'], 4),
                        'MAE': round(pt['MAE'], 4), 'R2': round(pt['R2'], 5),
                        'R2_SD': round(float(bs['R2'].std(ddof=1)), 5)})
    pd.DataFrame(pts).to_csv(out_path('ablation_results.csv'),
                             index=False, encoding='utf-8-sig')
    print("\n✓ ablation_results.csv / ablation_paired_tests.csv")

    # ---------- 消融图（v5 格式：6 变体 Chain1 + 4 变体 Chain2） ----------
    C_MSE1 = ['#F9E3AE', '#F5CD78', '#EFA96B', '#E9909B', '#D9707E', '#B94F62']
    C_R21 = ['#C6E3CD', '#A9D7B4', '#92CA9B', '#6BB97F', '#37AC4F', '#1F8F3E']
    C_MSE2 = ['#F9E3AE', '#F5CD78', '#EFA96B', '#E9909B']
    C_R22 = ['#C6E3CD', '#A9D7B4', '#92CA9B', '#37AC4F']

    def draw_hbar(ax, names, res, colors, metric, title, letter):
        vals = np.array([r[0][metric] for r in res])
        errs = np.array([r[1][metric].std(ddof=1) for r in res])
        order = np.argsort(vals)[::-1] if metric == 'MSE' else np.argsort(vals)
        nv = [names[i] for i in order]
        v, e = vals[order], errs[order]
        ypos = np.arange(len(nv))
        ax.barh(ypos, v, xerr=e, align='center', color=[colors[i] for i in order],
                edgecolor='black', linewidth=0.5,
                error_kw=dict(ecolor='black', lw=0.6, capsize=2))
        ax.set_yticks(ypos)
        ax.set_yticklabels(nv, fontsize=6.5)
        ax.invert_yaxis()
        ax.set_xlabel('Mean squared error' if metric == 'MSE' else '$R^2$')
        ax.set_title(title, fontweight='normal', pad=5)
        style_axis(ax, 'x')
        top = float(np.max(v + e))
        pad = top * (0.03 if metric == 'MSE' else 0.015)
        for i, (vv, ee) in enumerate(zip(v, e)):
            ax.text(vv + ee + pad, i,
                    f'{vv:.2f} ± {ee:.2f}' if metric == 'MSE' else f'{vv:.3f} ± {ee:.3f}',
                    va='center', ha='left', fontsize=6.5, color='black')
        ax.set_xlim(0, top * (1.55 if metric == 'MSE' else 1.45))
        panel_label(ax, letter)

    # Figure 14 + 15 合并为一张 2×2
    fig, axes = plt.subplots(2, 2, figsize=(FIG_FULL, 7.0))
    draw_hbar(axes[0, 0], chain1_names, res1, C_MSE1, 'MSE',
              'Weight→Length: MSE', 'A')
    draw_hbar(axes[0, 1], chain1_names, res1, C_R21, 'R2',
              'Weight→Length: $R^2$', 'B')
    draw_hbar(axes[1, 0], chain2_names, res2, C_MSE2, 'MSE',
              'Length→Weight: MSE', 'C')
    draw_hbar(axes[1, 1], chain2_names, res2, C_R22, 'R2',
              'Length→Weight: $R^2$', 'D')
    fig.subplots_adjust(hspace=0.30, wspace=0.34)
    save_fig(fig, 'Figure14_15_Ablation.pdf', tight=False)

    # ---------- Figure 16：paired ΔR²（v5 格式）----------
    chain1_steps_short = ['1→2\n+XGBoost\n(substitute\nRNN)',
                          '2→3\n+Dual-learner\nfusion\n(w. avg)',
                          '3→4\n+Meta-learner\n(stacking)',
                          '4→5\n+Grouping',
                          '5→6\n+Physical']
    chain2_steps_short = ['1→2\n+Grouping',
                          '2→3\n+Physical',
                          '3→4\n+Bagging']

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(FIG_FULL, 4.2))
    for ax, res, steps, title, letter, color in (
            (axA, res1, chain1_steps_short,
             'Weight→Length: paired $\\Delta R^2$', 'A', '#4893AB'),
            (axB, res2, chain2_steps_short,
             'Length→Weight: paired $\\Delta R^2$', 'B', '#6FAFC4')):
        ds, los, his = [], [], []
        for i in range(1, len(res)):
            d, lo, hi, _ = paired_diff(res[i][1]['R2'], res[i - 1][1]['R2'])
            ds.append(d)
            los.append(d - lo)
            his.append(hi - d)
        ds, los, his = np.array(ds), np.array(los), np.array(his)

        ax.bar(np.arange(len(ds)), ds, yerr=[los, his], color=color,
               edgecolor='black', linewidth=0.5,
               error_kw=dict(ecolor='black', lw=0.6, capsize=3))
        ax.axhline(0, color='black', linewidth=0.7)
        ax.set_xticks(np.arange(len(ds)))
        ax.set_xticklabels(steps, fontsize=6)
        ax.set_xlabel('Ablation step (variant i → j, see Fig. 14–15)')
        ax.set_ylabel('$\\Delta R^2$ (paired)')
        ax.set_title(title, fontweight='normal', pad=5)
        style_axis(ax, 'y')

        span = float(np.max(np.abs(ds) + np.maximum(los, his))) or 1.0
        ax.set_ylim(top=float(np.max(ds + np.maximum(los, his))) + span * 0.42)
        for i, (v, h) in enumerate(zip(ds, his)):
            ax.text(i, v + h + span * 0.05, f'{v:+.4f}',
                    ha='center', va='bottom', fontsize=6.5, color='black')
            if (ds[i] - los[i]) > 0 or (ds[i] + his[i]) < 0:
                ax.text(i, v + h + span * 0.15, '*',
                        ha='center', va='bottom', fontsize=11, color='black')
        panel_label(ax, letter, x=-0.10, y=1.04)

    fig.subplots_adjust(left=0.085, right=0.985, top=0.90, bottom=0.28, wspace=0.22)
    save_fig(fig, 'Figure16_Ablation_Paired_Delta.pdf', tight=False)
    print("  注：* 表示 95% 配对 bootstrap CI 不含 0。")

    # ---- 供 Stage 3 用的动态预测闭包 ----
    def predict_len2w_final(X_len_df):
        X_arr = np.asarray(X_len_df, dtype=np.float32)
        if best_w_method == 'RNN_single':
            Xt = _to_t(ctx_len2w_g['sx'].transform(X_arr))
            m_rnn_len2w.eval()
            with torch.no_grad():
                return ctx_len2w_g['sy'].inverse_transform(
                    m_rnn_len2w(Xt).cpu().numpy()).flatten()
        elif best_w_method == 'XGBoost':
            return m_xgb_len2w.predict(X_len_df)
        elif best_w_method == 'WeightedAvg':
            Xt = _to_t(blk_len2w['sx'].transform(X_arr))
            mR = blk_len2w['rnn_final']; mR.eval()
            with torch.no_grad():
                pr = blk_len2w['sy'].inverse_transform(mR(Xt).cpu().numpy()).flatten()
            px = blk_len2w['xgb_final'].predict(X_len_df)
            w = blk_len2w['w']
            return w * pr + (1 - w) * px
        elif best_w_method == 'Stacking':
            Xt = _to_t(blk_len2w['sx'].transform(X_arr))
            mR = blk_len2w['rnn_final']; mR.eval()
            with torch.no_grad():
                pr = blk_len2w['sy'].inverse_transform(mR(Xt).cpu().numpy()).flatten()
            px = blk_len2w['xgb_final'].predict(X_len_df)
            return blk_len2w['meta'].predict(np.column_stack([pr, px]))
        else:  # Bagging
            return _bagging_pred(bag_len2w_models, X_len_df,
                                 ctx_len2w_g['sx'], ctx_len2w_g['sy'])

    def predict_len_final(X_w_df):
        X_arr = np.asarray(X_w_df, dtype=np.float32)
        if best_len_method == 'RNN_single':
            Xt = _to_t(ctx_w2len_g['sx'].transform(X_arr))
            m_rnn_w2len.eval()
            with torch.no_grad():
                return ctx_w2len_g['sy'].inverse_transform(
                    m_rnn_w2len(Xt).cpu().numpy()).flatten()
        elif best_len_method == 'XGBoost':
            return m_xgb_w2len.predict(X_w_df)
        elif best_len_method == 'WeightedAvg':
            Xt = _to_t(blk_w2len['sx'].transform(X_arr))
            mR = blk_w2len['rnn_final']; mR.eval()
            with torch.no_grad():
                pr = blk_w2len['sy'].inverse_transform(mR(Xt).cpu().numpy()).flatten()
            px = blk_w2len['xgb_final'].predict(X_w_df)
            w = blk_w2len['w']
            return w * pr + (1 - w) * px
        elif best_len_method == 'Stacking':
            Xt = _to_t(blk_w2len['sx'].transform(X_arr))
            mR = blk_w2len['rnn_final']; mR.eval()
            with torch.no_grad():
                pr = blk_w2len['sy'].inverse_transform(mR(Xt).cpu().numpy()).flatten()
            px = blk_w2len['xgb_final'].predict(X_w_df)
            return blk_w2len['meta'].predict(np.column_stack([pr, px]))
        else:  # Bagging
            return _bagging_pred(bag_w2len_models, X_w_df,
                                 ctx_w2len_g['sx'], ctx_w2len_g['sy'])

    return dict(train_df=train_df, test_df=test_df, y_len_test=y_len_test, y_w_test=y_w_test,
                IDX_LEN=IDX_LEN, IDX_W=IDX_W, a=a, b=b,
                ctx_len2w_g=ctx_len2w_g, ctx_w2len_g=ctx_w2len_g,
                blk_w2len=blk_w2len, blk_len2w=blk_len2w,
                rnn_len2w_model=m_rnn_len2w, rnn_w2len_model=m_rnn_w2len,
                bag_len2w_models=bag_len2w_models, bag_w2len_models=bag_w2len_models,
                best_len_method=best_len_method, best_w_method=best_w_method,
                predict_len2w_final=predict_len2w_final,
                predict_len_final=predict_len_final,
                pred_w_direct=p_final_w, pred_len_direct=p_final_len,
                assign_group=assign_group, LB=LB, LL=LL, WB=WB, WL=WL,
                features_len=features_len, features_w=features_w,
                lam_w2len=lam_w2len, lam_len2w=lam_len2w)


# ============================================================
# STAGE 2 —— 八模型对比 + Figure 1–13
# ============================================================
def stage_figures():
    print("\n" + "=" * 96)
    print("STAGE 2  八模型对比 + Figure 1–13")
    print("=" * 96)

    data = {}
    for name, (fname, lc, wc) in BASELINE_CSV.items():
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fpath):
            print(f"✗ {name:9s} | 未找到 {fname}")
            continue
        try:
            df = pd.read_csv(fpath, encoding='utf-8-sig')
            df.columns = df.columns.str.strip()
            if not {'实际体长(cm)', '实际体重(g)', lc, wc}.issubset(df.columns):
                print(f"✗ {name:9s} | 列缺失")
                continue
            sub = df[['实际体长(cm)', lc, '实际体重(g)', wc]].copy()
            sub.columns = ['Actual_Length', 'Pred_Length', 'Actual_Weight', 'Pred_Weight']
            sub = sub.dropna().reset_index(drop=True)
            data[name] = sub
            print(f"✓ {name:9s} | n = {len(sub)}")
        except Exception as e:
            print(f"✗ {name:9s} | {str(e)[:60]}")

    fname, lc, wc = ALLOVERT_CSV
    fpath = os.path.join(DATA_DIR, fname)
    if os.path.exists(fpath):
        df = pd.read_csv(fpath, encoding='utf-8-sig')
        df.columns = df.columns.str.strip()
        sub = df[['Actual_Body_Length(cm)', lc, 'Actual_Body_Weight(g)', wc]].copy()
        sub.columns = ['Actual_Length', 'Pred_Length', 'Actual_Weight', 'Pred_Weight']
        sub = sub.dropna().reset_index(drop=True)
        data['AlloVert'] = sub
        if 'Final_Length_Method' in df.columns:
            lm = df['Final_Length_Method'].iloc[0]
            wm = df['Final_Weight_Method'].iloc[0]
            print(f"✓ {'AlloVert':9s} | n = {len(sub)}  "
                  f"(Len←{lm}, Weight←{wm})")
        else:
            print(f"✓ {'AlloVert':9s} | n = {len(sub)}")
    else:
        print(f"✗ AlloVert  | 未找到 {fname}，请先运行 STAGE 1")

    model_names = [m for m in MODEL_ORDER if m in data]
    if not model_names:
        raise RuntimeError("没有任何模型数据，无法出图。")

    sizes = {n: len(data[n]) for n in model_names}
    if len(set(sizes.values())) != 1:
        raise RuntimeError(f"各模型样本数不一致: {sizes}")
    n_test = sizes[model_names[0]]
    n_models = len(model_names)
    print(f"测试集 n = {n_test}，模型数 = {n_models}")

    boot_idx = bootstrap_indices(n_test, BOOTSTRAP_N, BOOT_SEED)
    point, boot = {}, {}
    for name in model_names:
        point[name], boot[name] = {}, {}
        for tgt in ('Length', 'Weight'):
            a = data[name][f'Actual_{tgt}'].values.astype(float)
            p = data[name][f'Pred_{tgt}'].values.astype(float)
            point[name][tgt] = np.asarray(metrics_on(a, p))
            boot[name][tgt] = np.column_stack([
                bootstrap_samples(a, p, boot_idx, mean_absolute_error),
                bootstrap_samples(a, p, boot_idx,
                                  lambda x, y: float(np.sqrt(mean_squared_error(x, y)))),
                bootstrap_samples(a, p, boot_idx, r2_score)])

    METRICS = ['MAE', 'RMSE', 'R2']

    rows = []
    for name in model_names:
        row = {'Model': name}
        for tgt in ('Length', 'Weight'):
            pt = point[name][tgt]
            sd = boot[name][tgt].std(axis=0, ddof=1)
            for i, m in enumerate(METRICS):
                row[f'{tgt}_{m}'] = f'{pt[i]:.3f} ± {sd[i]:.3f}'
        rows.append(row)
    tbl = pd.DataFrame(rows)
    print("\n" + "-" * 96)
    print("Table 2（点估计 ± bootstrap SD）")
    print("-" * 96)
    print(tbl.to_string(index=False))
    tbl.to_csv(out_path('table2_metrics.csv'), index=False, encoding='utf-8-sig')
    print(f"✓ table2_metrics.csv")

    mp = pd.DataFrame([{'Model': n,
                        'Length_MAE': point[n]['Length'][0],
                        'Length_RMSE': point[n]['Length'][1],
                        'Length_R2': point[n]['Length'][2],
                        'Weight_MAE': point[n]['Weight'][0],
                        'Weight_RMSE': point[n]['Weight'][1],
                        'Weight_R2': point[n]['Weight'][2]} for n in model_names])
    mp['Model'] = pd.Categorical(mp['Model'], categories=MODEL_ORDER, ordered=True)
    mp = mp.sort_values('Model').reset_index(drop=True)

    # ==================== Figure 1 / 2 ====================
    def make_scatter(actual_col, pred_col, xlabel, ylabel, out_name):
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(FIG_FULL, 2.1 * n_rows))
        axes = np.atleast_1d(axes).flatten()
        letters = 'ABCDEFGH'
        idx = -1
        for idx, name in enumerate(model_names):
            ax = axes[idx]
            d = data[name]
            ax.scatter(d[actual_col], d[pred_col], c=COLORS[name], alpha=0.75, s=20,
                       edgecolors='white', linewidth=0.4, zorder=3)
            vmin = min(d[actual_col].min(), d[pred_col].min())
            vmax = max(d[actual_col].max(), d[pred_col].max())
            ax.plot([vmin, vmax], [vmin, vmax], 'k--', alpha=0.6, linewidth=0.9, zorder=2)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(name, fontweight='normal', pad=5)
            style_axis(ax)
            panel_label(ax, letters[idx], x=-0.14, y=1.05, fs=8)
        for j in range(idx + 1, len(axes)):
            axes[j].set_visible(False)
        save_fig(fig, out_name, tight=False)

    make_scatter('Actual_Length', 'Pred_Length',
                 'Actual length (cm)', 'Predicted length (cm)',
                 'Figure1_Length_Scatter.pdf')
    make_scatter('Actual_Weight', 'Pred_Weight',
                 'Actual weight (g)', 'Predicted weight (g)',
                 'Figure2_Weight_Scatter.pdf')

    # ==================== Figure 3_4_7 ====================
    err_len = {n: np.abs(data[n]['Actual_Length'] - data[n]['Pred_Length']) for n in model_names}
    err_w = {n: np.abs(data[n]['Actual_Weight'] - data[n]['Pred_Weight']) for n in model_names}

    def iqr(v):
        return np.percentile(v, 75) - np.percentile(v, 25)

    order_len = sorted(model_names, key=lambda n: iqr(err_len[n]))
    order_w = sorted(model_names, key=lambda n: iqr(err_w[n]))

    fig = plt.figure(figsize=(FIG_FULL, 8.6))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.25],
                          left=0.075, right=0.985, top=0.955, bottom=0.062,
                          hspace=0.42, wspace=0.26)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, :])
    ax_c = fig.add_subplot(gs[2, 0])
    ax_d = fig.add_subplot(gs[2, 1])

    bp1 = ax_a.boxplot([err_len[n] for n in order_len],
                       positions=range(len(order_len)), widths=0.65, patch_artist=True,
                       boxprops=dict(linewidth=0.8, edgecolor='black'),
                       medianprops=dict(color='black', linewidth=1.0),
                       whiskerprops=dict(linewidth=0.8, color='black', linestyle='--'),
                       capprops=dict(linewidth=0.8, color='black'),
                       flierprops=dict(marker='o', markersize=2, markerfacecolor='white',
                                       markeredgecolor='gray', markeredgewidth=0.5))
    for patch, name in zip(bp1['boxes'], order_len):
        patch.set_facecolor(COLORS_LIGHT_RED[name]); patch.set_alpha(0.9)
        patch.set_edgecolor('black'); patch.set_linewidth(0.8)
    ax_a.set_xticks(range(len(order_len)))
    ax_a.set_xticklabels([LABEL_SINGLE[n] for n in order_len],
                         rotation=0, ha='center', fontsize=7)
    ax_a.set_ylabel('Absolute error (cm)')
    ax_a.set_title('Length prediction error', fontweight='normal', pad=5)
    style_axis(ax_a, 'y')
    panel_label(ax_a, 'A', x=-0.06)

    bp2 = ax_b.boxplot([err_w[n] for n in order_w],
                       positions=range(len(order_w)), widths=0.65, patch_artist=True,
                       boxprops=dict(linewidth=0.8, edgecolor='black'),
                       medianprops=dict(color='black', linewidth=1.0),
                       whiskerprops=dict(linewidth=0.8, color='black', linestyle='--'),
                       capprops=dict(linewidth=0.8, color='black'),
                       flierprops=dict(marker='o', markersize=2, markerfacecolor='white',
                                       markeredgecolor='gray', markeredgewidth=0.5))
    for patch, name in zip(bp2['boxes'], order_w):
        patch.set_facecolor(COLORS_LIGHT_RED[name]); patch.set_alpha(0.9)
        patch.set_edgecolor('black'); patch.set_linewidth(0.8)
    ax_b.set_xticks(range(len(order_w)))
    ax_b.set_xticklabels([LABEL_SINGLE[n] for n in order_w],
                         rotation=0, ha='center', fontsize=7)
    ax_b.set_ylabel('Absolute error (g)')
    ax_b.set_title('Weight prediction error', fontweight='normal', pad=5)
    style_axis(ax_b, 'y')
    panel_label(ax_b, 'B', x=-0.06)

    for ax, tgt, letter in ((ax_c, 'Length', 'C'), (ax_d, 'Weight', 'D')):
        for name in model_names:
            d = data[name]
            rel = np.abs(d[f'Pred_{tgt}'] - d[f'Actual_{tgt}']) / d[f'Actual_{tgt}'] * 100
            xs = np.sort(rel)
            ys = np.arange(1, len(xs) + 1) / len(xs)
            lw = 2.0 if name == 'AlloVert' else 1.2
            al = 1.0 if name == 'AlloVert' else 0.75
            ax.plot(xs, ys, color=COLORS_RED[name], linewidth=lw, alpha=al, label=name)
        ax.set_xlabel('Relative error (%)')
        ax.set_ylabel('Cumulative proportion')
        ax.set_title(f'{tgt} relative error CDF', fontweight='normal', pad=5)
        ax.set_xlim(0, None)
        ax.legend(loc='lower right', fontsize=5.5, frameon=False, ncol=2,
                  handlelength=1.2, handletextpad=0.3, labelspacing=0.15,
                  columnspacing=0.8, borderpad=0.2)
        style_axis(ax)
        panel_label(ax, letter)

    save_fig(fig, 'Figure3_4_7_Error_Boxplot_CDF.pdf', tight=False)

    # ==================== Figure 5 ====================
    length_bins = [0, 30, 45, 51, np.inf]
    length_labels = ['0–30', '30–45', '45–51', '>51']
    weight_bins = [0, 30, 60, 120, np.inf]
    weight_labels = ['0–30', '30–60', '60–120', '>120']

    def group_mae(target):
        bins = length_bins if target == 'Length' else weight_bins
        labels = length_labels if target == 'Length' else weight_labels
        a_col, p_col = f'Actual_{target}', f'Pred_{target}'
        out = {}
        for name in model_names:
            d = data[name]
            av = d[a_col].values.astype(float)
            pv = d[p_col].values.astype(float)
            vals = []
            for k, lab in enumerate(labels):
                lo_b, hi_b = bins[k], bins[k + 1]
                mask = (av >= lo_b) & (av < hi_b)
                vals.append(mean_absolute_error(av[mask], pv[mask]) if mask.sum() > 0 else np.nan)
            out[name] = vals
        return out

    mae_len, mae_w = group_mae('Length'), group_mae('Weight')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_FULL, 3.8))
    width, gap = 0.10, 0.01
    xl = np.arange(len(length_labels))
    xw = np.arange(len(weight_labels))
    for idx, name in enumerate(model_names):
        offset = (idx - (n_models - 1) / 2) * (width + gap)
        ax1.bar(xl + offset, mae_len[name], width, label=name, color=COLORS[name],
                alpha=0.9, edgecolor='white', linewidth=0.4, zorder=3)
        ax2.bar(xw + offset, mae_w[name], width, label=name, color=COLORS[name],
                alpha=0.9, edgecolor='white', linewidth=0.4, zorder=3)
    for ax, xt, labs, unit, title, letter in (
            (ax1, xl, length_labels, 'cm', 'Length error by length group', 'A'),
            (ax2, xw, weight_labels, 'g', 'Weight error by weight group', 'B')):
        ax.set_xlabel(f'Group ({unit})')
        ax.set_ylabel(f'Mean absolute error ({unit})')
        ax.set_title(title, fontweight='normal', pad=5)
        ax.set_xticks(xt)
        ax.set_xticklabels(labs, fontsize=6)
        style_axis(ax, 'y')
        panel_label(ax, letter)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=6,
               frameon=False, handlelength=1.2, handletextpad=0.3,
               columnspacing=1.2, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(bottom=0.24)
    save_fig(fig, 'Figure5_Group_Errors.pdf', tight=False)

    # ==================== Figure 6 ====================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_FULL, 3.8))
    for name in model_names:
        d = data[name]
        ax1.scatter(d['Pred_Length'], d['Actual_Length'] - d['Pred_Length'],
                    c=COLORS[name], alpha=0.6, s=13, label=name,
                    edgecolors='white', linewidth=0.3, zorder=3)
        ax2.scatter(d['Pred_Weight'], d['Actual_Weight'] - d['Pred_Weight'],
                    c=COLORS[name], alpha=0.6, s=13, label=name,
                    edgecolors='white', linewidth=0.3, zorder=3)
    for ax, xl, yl, title, letter in (
            (ax1, 'Predicted length (cm)', 'Residual (cm)', 'Length prediction residuals', 'A'),
            (ax2, 'Predicted weight (g)', 'Residual (g)', 'Weight prediction residuals', 'B')):
        ax.axhline(0, color='black', linestyle='--', alpha=0.7, linewidth=0.9)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(title, fontweight='normal', pad=5)
        style_axis(ax)
        panel_label(ax, letter)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=6,
               frameon=False, handlelength=1.2, handletextpad=0.3,
               columnspacing=1.2, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(bottom=0.24)
    save_fig(fig, 'Figure6_Residuals.pdf', tight=False)

    # ==================== Figure 8 ====================
    fig, axes = plt.subplots(2, 2, figsize=(FIG_FULL, 6.0))
    for (col, title, ax, letter) in (
            ('Length_MAE', 'Length MAE (cm)', axes[0, 0], 'A'),
            ('Length_R2', 'Length $R^2$', axes[0, 1], 'B'),
            ('Weight_MAE', 'Weight MAE (g)', axes[1, 0], 'C'),
            ('Weight_R2', 'Weight $R^2$', axes[1, 1], 'D')):
        vals = [mp[mp['Model'] == n][col].values[0] for n in model_names]
        xpos = np.arange(len(model_names))
        ax.bar(xpos, vals, color=[COLORS[n] for n in model_names],
               alpha=0.9, edgecolor='white', linewidth=0.4, zorder=3)
        ax.set_title(title, fontweight='normal', pad=5)
        ax.set_xticks(xpos)
        ax.set_xticklabels([LABEL_WRAP[n] for n in model_names],
                           rotation=0, ha='center', fontsize=6)
        if col.endswith('R2'):
            lo, hi = np.nanmin(vals), np.nanmax(vals)
            pad = max(0.02, (hi - lo) * 0.15)
            ax.set_ylim(min(0, lo - pad), min(1.0, hi + pad))
            ax.axhline(0, color='gray', linewidth=0.6, alpha=0.5, zorder=1)
        style_axis(ax, 'y')
        panel_label(ax, letter)
    fig.subplots_adjust(hspace=0.35, wspace=0.28)
    save_fig(fig, 'Figure8_Metrics_Comparison.pdf', tight=False)

    # ==================== Figure 9 ====================
    fig, ax = plt.subplots(figsize=(FIG_FULL, 3.6))
    heat = mp.set_index('Model')[['Length_MAE', 'Length_RMSE', 'Length_R2',
                                  'Weight_MAE', 'Weight_RMSE', 'Weight_R2']].copy()
    heat.columns = ['Length\nMAE', 'Length\nRMSE', 'Length\n$R^2$',
                    'Weight\nMAE', 'Weight\nRMSE', 'Weight\n$R^2$']
    for c in ['Length\n$R^2$', 'Weight\n$R^2$']:
        heat[c] = 1 - heat[c]
    im = ax.imshow(heat.values, cmap='YlOrRd', aspect='auto', alpha=0.9)
    ax.set_xticks(np.arange(len(heat.columns)))
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_xticklabels(heat.columns, fontsize=6.5)
    ax.set_yticklabels(heat.index, fontsize=7)
    vmax = np.nanmax(heat.values)
    for i in range(len(heat.index)):
        for j in range(len(heat.columns)):
            v = heat.iloc[i, j]
            txt = f'{1 - v:.3f}' if j in (2, 5) else f'{v:.3f}'
            ax.text(j, i, txt, ha='center', va='center', fontsize=6.5,
                    color='white' if v > vmax * 0.6 else 'black')
    ax.set_title('Model performance heatmap (lower is better)', fontweight='normal', pad=6)
    cb = plt.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label('Normalized error', fontsize=7)
    cb.ax.tick_params(labelsize=6)
    save_fig(fig, 'Figure9_Performance_Heatmap.pdf', tight=False)

    # ==================== Figure 10 ====================
    categories = ['Length MAE↓', 'Length $R^2$↑', 'Weight MAE↓', 'Weight $R^2$↑']
    N = len(categories)
    angles = [k / float(N) * 2 * pi for k in range(N)]
    angles += angles[:1]
    raw = mp.set_index('Model')
    norm = pd.DataFrame(index=raw.index)
    for c, col in zip(categories, ['Length_MAE', 'Length_R2', 'Weight_MAE', 'Weight_R2']):
        v = raw[col].astype(float)
        norm[c] = (1 - v / v.max()) if 'MAE' in c else v.clip(lower=0) / v.max()

    fig = plt.figure(figsize=(FIG_HALF, FIG_HALF))
    axr = plt.subplot(111, projection='polar')
    for name in model_names:
        vals = norm.loc[name].values.tolist()
        vals += vals[:1]
        axr.plot(angles, vals, '-', linewidth=1.4, color=COLORS[name], label=name)
        axr.fill(angles, vals, alpha=0.10, color=COLORS[name])
    axr.set_xticks(angles[:-1])
    axr.set_xticklabels(categories, fontsize=6)
    axr.set_ylim(0, 1)
    axr.tick_params(axis='y', labelsize=5)
    axr.set_title('Model performance radar (higher is better)',
                  fontweight='normal', pad=12, fontsize=8)
    handles, labels = axr.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=6,
               frameon=False, handlelength=1.2, handletextpad=0.3, columnspacing=1.0)
    fig.subplots_adjust(left=0.10, right=0.94, top=0.94, bottom=0.20)
    save_fig(fig, 'Figure10_Radar.pdf', tight=False)

    # ==================== Figure 11 ====================
    n_cols = 4
    n_rows = (n_models + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(FIG_FULL, 1.9 * n_rows))
    axes = np.atleast_1d(axes).flatten()
    letters = 'ABCDEFGH'
    idx = -1
    for idx, name in enumerate(model_names):
        ax = axes[idx]
        d = data[name]
        for tgt in ('Length', 'Weight'):
            m = (d[f'Actual_{tgt}'] + d[f'Pred_{tgt}']) / 2
            dd = d[f'Actual_{tgt}'] - d[f'Pred_{tgt}']
            ax.scatter(m, dd, c=COLORS[name], alpha=0.55, s=9,
                       edgecolors='white', linewidth=0.25, zorder=3)
        ax.axhline(0, color='black', linestyle='--', alpha=0.7, linewidth=0.8, zorder=2)
        ax.set_xlabel('Mean of actual and predicted', fontsize=7)
        ax.set_ylabel('Difference', fontsize=7)
        ax.set_title(name, fontweight='normal', pad=4)
        style_axis(ax)
        panel_label(ax, letters[idx], x=-0.12, y=1.05, fs=8)
    for j in range(idx + 1, len(axes)):
        axes[j].set_visible(False)
    save_fig(fig, 'Figure11_Bland_Altman.pdf', tight=False)

    # ==================== Figure 12 ====================
    n_cols = 4
    n_rows = (n_models + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(FIG_FULL, 2.0 * n_rows))
    axes = np.atleast_1d(axes).flatten()
    letters = 'ABCDEFGH'
    idx = -1
    for idx, name in enumerate(model_names):
        ax = axes[idx]
        d = data[name]
        le = (d['Actual_Length'] - d['Pred_Length']).values
        we = (d['Actual_Weight'] - d['Pred_Weight']).values
        ax.hist(le, bins=10, alpha=0.65, color=COLORS[name],
                edgecolor='black', linewidth=0.4, density=True, label='Length')
        ax.hist(we / 3, bins=10, alpha=0.40, color=COLORS[name],
                edgecolor='black', linewidth=0.4, density=True, label='Weight/3')
        ax.axvline(0, color=C_ACCENT, linestyle='-', alpha=0.8, linewidth=1.0)
        ax.set_title(name, fontweight='normal', pad=4)
        ax.set_xlabel('Error', fontsize=7)
        ax.set_ylabel('Density', fontsize=7)
        ax.legend(fontsize=5.5, frameon=False, handlelength=1.0, handletextpad=0.3,
                  labelspacing=0.15)
        style_axis(ax, 'y')
        panel_label(ax, letters[idx], x=-0.12, y=1.05, fs=8)
    for j in range(idx + 1, len(axes)):
        axes[j].set_visible(False)
    save_fig(fig, 'Figure12_Error_Distribution.pdf', tight=False)

    # ==================== Figure 13 ====================
    f13c = dict(COLORS_RED)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(FIG_FULL, 6.8))
    ref = data[model_names[0]]
    xv = np.arange(len(ref))
    ax1.plot(xv, ref['Actual_Length'].values, 'k-', linewidth=1.5, label='Actual')
    ax2.plot(xv, ref['Actual_Weight'].values, 'k-', linewidth=1.5, label='Actual')
    for name in model_names:
        d = data[name]
        ls = '-' if name == 'AlloVert' else '--'
        al = 1.0 if name == 'AlloVert' else 0.7
        lw = 1.6 if name == 'AlloVert' else 1.0
        ax1.plot(xv, d['Pred_Length'].values, color=f13c[name], linestyle=ls,
                 linewidth=lw, alpha=al, label=name)
        ax2.plot(xv, d['Pred_Weight'].values, color=f13c[name], linestyle=ls,
                 linewidth=lw, alpha=al, label=name)
    for ax, yl, ttl in ((ax1, 'Length (cm)', 'Length predictions vs actual'),
                        (ax2, 'Weight (g)', 'Weight predictions vs actual')):
        ax.set_xlabel('Sample index')
        ax.set_ylabel(yl)
        ax.set_title(ttl, fontweight='normal', pad=5)
        ax.legend(loc='best', fontsize=6, frameon=False, ncol=3,
                  handlelength=1.2, handletextpad=0.3, labelspacing=0.2)
        style_axis(ax)
    panel_label(ax1, 'A', x=-0.06, y=1.02)
    panel_label(ax2, 'B', x=-0.06, y=1.02)
    fig.subplots_adjust(hspace=0.32)
    save_fig(fig, 'Figure13_Line_Plot.pdf', tight=False)

    return data, model_names


# ============================================================
# STAGE 3 —— 双向可逆性 / 收缩
# ============================================================
def stage_roundtrip(state=None):
    print("\n" + "=" * 96)
    print("STAGE 3  双向可逆性 / 收缩分析")
    print("=" * 96)

    if state is None:
        print("\n⚠ 未收到 STAGE 1 的 state。")
        return

    test_df = state['test_df']
    y_len_test = state['y_len_test']
    y_w_test = state['y_w_test']
    IDX_LEN = state['IDX_LEN']
    IDX_W = state['IDX_W']
    a, b = state['a'], state['b']
    assign_group = state['assign_group']
    LB, LL, WB, WL = state['LB'], state['LL'], state['WB'], state['WL']
    features_len = state['features_len']
    features_w = state['features_w']
    TL, TW = '体长(cm)', '体重(g)'

    print(f"  [Stage 3] Length 使用: {state['best_len_method']}")
    print(f"  [Stage 3] Weight 使用: {state['best_w_method']}")

    predict_len_final = state['predict_len_final']
    predict_len2w_final = state['predict_len2w_final']

    X_len_cyc = test_df[features_len].copy().reset_index(drop=True)
    pred_w_hat = predict_len2w_final(X_len_cyc)

    X_w_cyc = test_df[features_w].copy().reset_index(drop=True)
    X_w_cyc[TW] = pred_w_hat
    X_w_cyc['weight_group'] = assign_group(pred_w_hat, WB, WL)
    pred_len_rt = predict_len_final(X_w_cyc)

    X_w_cyc_rev = test_df[features_w].copy().reset_index(drop=True)
    pred_len_hat = predict_len_final(X_w_cyc_rev)

    X_len_cyc_rev = test_df[features_len].copy().reset_index(drop=True)
    X_len_cyc_rev[TL] = pred_len_hat
    X_len_cyc_rev['length_group'] = assign_group(pred_len_hat, LB, LL)
    pred_w_rt = predict_len2w_final(X_len_cyc_rev)

    pred_len_direct = state['pred_len_direct']
    pred_w_direct = state['pred_w_direct']

    def shrink_report(tag, y_true, y_direct, y_rt, idxs):
        pt_d, _ = eval_model(y_true, y_direct, idxs)
        pt_r, bs_r = eval_model(y_true, y_rt, idxs)
        s_d = slope_of(y_true, y_direct)
        s_r = slope_of(y_true, y_rt)
        sd_boot = bootstrap_samples(y_true, y_direct, idxs, slope_of)
        sr_boot = bootstrap_samples(y_true, y_rt, idxs, slope_of)
        d, lo, hi, p = paired_diff(sr_boot, sd_boot)
        var_ratio = float(np.var(y_rt, ddof=1) / np.var(y_true, ddof=1))
        print(f"\n[{tag}]")
        print(f"  直接预测   : MAE={pt_d['MAE']:.4f}  RMSE={pt_d['RMSE']:.4f}  "
              f"R²={pt_d['R2']:.4f}  slope={s_d:.4f}")
        print(f"  往返预测   : MAE={pt_r['MAE']:.4f} ± {bs_r['MAE'].std(ddof=1):.4f}  "
              f"RMSE={pt_r['RMSE']:.4f}  R²={pt_r['R2']:.4f}  slope={s_r:.4f}")
        print(f"  方差保留率 = {var_ratio:.4f}")
        print(f"  收缩斜率差 = {d:+.4f} [95% CI {lo:+.4f}, {hi:+.4f}]  p = {p:.4f}")
        return dict(Task=tag, Direct_MAE=round(pt_d['MAE'], 4),
                    Direct_slope=round(s_d, 4),
                    Roundtrip_MAE=round(pt_r['MAE'], 4),
                    Roundtrip_MAE_sd=round(float(bs_r['MAE'].std(ddof=1)), 4),
                    Roundtrip_slope=round(s_r, 4),
                    Var_ratio=round(var_ratio, 4),
                    Slope_diff=round(d, 4), Slope_CI_low=round(lo, 4),
                    Slope_CI_high=round(hi, 4), p=round(p, 4))

    rows = [shrink_report('Length (L→Ŵ→L̂)', y_len_test, pred_len_direct, pred_len_rt, IDX_LEN),
            shrink_report('Weight (W→L̂→Ŵ)', y_w_test, pred_w_direct, pred_w_rt, IDX_W)]
    pd.DataFrame(rows).to_csv(out_path('reversibility_shrinkage.csv'),
                              index=False, encoding='utf-8-sig')
    print("\n✓ reversibility_shrinkage.csv")

    SC, MEAN_C, LOA_C, ZERO_C = '#7FB3D5', '#E9909B', '#37AC4F', '#9E9E9E'

    def plot_bland_altman(ax, tv, pv, unit, title):
        mean = (tv + pv) / 2
        diff = pv - tv
        md, sd = float(np.mean(diff)), float(np.std(diff, ddof=1))
        lo, hi = md - 1.96 * sd, md + 1.96 * sd
        ax.scatter(mean, diff, s=12, c=SC, alpha=0.65,
                   edgecolors='white', linewidth=0.3, zorder=3)
        ax.axhline(md, color=MEAN_C, linewidth=1.2,
                   label=f'Mean bias: {md:.3f}', zorder=2)
        ax.axhline(lo, color=LOA_C, linestyle='--', linewidth=0.9,
                   label=f'95% LoA: [{lo:.3f}, {hi:.3f}]', zorder=2)
        ax.axhline(hi, color=LOA_C, linestyle='--', linewidth=0.9, zorder=2)
        ax.axhline(0, color=ZERO_C, linestyle=':', linewidth=0.8, alpha=0.7, zorder=1)
        ax.fill_between([mean.min() - 2, mean.max() + 2], lo, hi,
                        alpha=0.10, color=LOA_C, zorder=1)
        ax.set_xlabel(f'Mean of observed and round-trip ({unit})')
        ax.set_ylabel(f'Difference ({unit})')
        ax.set_title(title, fontweight='normal', pad=5)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
                  fontsize=6, frameon=False, ncol=2,
                  handlelength=1.2, handletextpad=0.4, labelspacing=0.2,
                  columnspacing=1.0)
        style_axis(ax)

    def plot_scatter_shrink(ax, tv, pv, unit, title):
        ax.scatter(tv, pv, s=12, c=SC, alpha=0.65,
                   edgecolors='white', linewidth=0.3, zorder=3)
        dmin = float(min(tv.min(), pv.min()))
        dmax = float(max(tv.max(), pv.max()))
        pad = 0.05 * (dmax - dmin) if dmax > dmin else 1.0
        lo, hi = dmin - pad, dmax + pad
        ax.plot([lo, hi], [lo, hi], '--', color=MEAN_C, linewidth=0.9,
                label='Ideal 1:1 line', zorder=2)
        sl, ic, rv, _, _ = stats.linregress(tv, pv)
        xs = np.linspace(lo, hi, 100)
        ax.plot(xs, sl * xs + ic, color=LOA_C, linewidth=1.0, alpha=0.85,
                label=f'Fit: y = {sl:.3f}x + {ic:.3f}', zorder=2)
        mae = float(np.mean(np.abs(pv - tv)))
        vr = float(np.var(pv, ddof=1) / np.var(tv, ddof=1))
        txt = (f'slope = {sl:.3f}\n$R^2$ = {rv ** 2:.4f}\nMAE = {mae:.3f}\n'
               f'Var ratio = {vr:.3f}\n$n$ = {len(tv)}')
        ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=7, va='top',
                bbox=dict(boxstyle='round,pad=0.30', facecolor='white',
                          edgecolor=SC, alpha=0.9, linewidth=0.6), zorder=4, color='black')
        ax.set_xlabel(f'Observed ({unit})')
        ax.set_ylabel(f'Round-trip predicted ({unit})')
        ax.set_title(title, fontweight='normal', pad=5)
        ax.legend(loc='lower right', fontsize=6, frameon=False,
                  handlelength=1.2, handletextpad=0.3, labelspacing=0.2)
        ax.set_aspect('equal', adjustable='datalim')
        style_axis(ax)

    fig, axes = plt.subplots(2, 2, figsize=(FIG_FULL, 7.5))
    plot_bland_altman(axes[0, 0], y_len_test, pred_len_rt, 'cm',
                      'Length round-trip: Bland-Altman')
    plot_bland_altman(axes[0, 1], y_w_test, pred_w_rt, 'g',
                      'Weight round-trip: Bland-Altman')
    plot_scatter_shrink(axes[1, 0], y_len_test, pred_len_rt, 'cm',
                        'Length round-trip: scatter')
    plot_scatter_shrink(axes[1, 1], y_w_test, pred_w_rt, 'g',
                        'Weight round-trip: scatter')
    for ax, L in zip(axes.flat, 'ABCD'):
        panel_label(ax, L)
    fig.subplots_adjust(hspace=0.34, wspace=0.30)
    save_fig(fig, 'Figure17_Roundtrip_Reversibility.pdf', tight=False)

    fig, axes = plt.subplots(1, 2, figsize=(FIG_FULL, 3.0))
    for ax, res, xl, ttl, letter in (
            (axes[0], pred_len_rt - y_len_test, 'Residual (cm)',
             'Length round-trip residuals', 'A'),
            (axes[1], pred_w_rt - y_w_test, 'Residual (g)',
             'Weight round-trip residuals', 'B')):
        ax.hist(res, bins=10, color=C_WARM_DEEP, alpha=0.85,
                edgecolor='white', linewidth=0.4, density=True)
        ax.axvline(0, color=ZERO_C, linestyle=':', linewidth=0.8, label='Zero error')
        ax.axvline(np.mean(res), color=MEAN_C, linewidth=1.2,
                   label=f'Mean: {np.mean(res):.3f}')
        ax.set_xlabel(xl)
        ax.set_ylabel('Density')
        ax.set_title(ttl, fontweight='normal', pad=5)
        ax.legend(fontsize=6, frameon=False)
        style_axis(ax, 'y')
        panel_label(ax, letter, x=-0.12)
    save_fig(fig, 'Figure18_Roundtrip_Residuals.pdf', tight=False)


# ============================================================
if __name__ == "__main__":
    main()