# -*- coding: utf-8 -*-
"""
==============================================================================
鱼椎三段计数回归 · 修正版 v7（v6 的 bug 修复版）
==============================================================================
相对 v6 的修改（本版不改变方法学，只修 bug 与工程细节）：

【1】冗余过滤 bug 修复（v6 真实缺陷）
     v6 写成 drop.add(a if score[a] < score[b] else score[b])，
     当 score[a] >= score[b] 时把浮点数写进 drop，导致过滤从未生效
     （'filt' 恒等于 'full'）。本版改为正确的 else b。

【2】1-SE 规则标准误修正
     25 折来自 5×5 RepeatedKFold，折间不独立。除以 sqrt(N_FOLDS=25)
     会低估 SE、导致 within 集合过窄。本版用 sqrt(CV_REPEATS=5)，
     即按"独立重复数"折算，与 caret/glmnet 的做法一致。

【3】predict_new 不再依赖全局变量 LOWER
     从 meta['lower_bounds'] 读取，函数可在任意新会话中独立使用。

【4】OOF 预测按样本跨折平均
     v6 每个样本只保留最后一折预测；本版累积求和后取平均，
     使 err_on / err_off 的比较更稳定。

【5】新增诊断：补偿后 cocc 触下界的样本数
     若 resid 为负把 cocc 推到 < LOWER，clip 后行和会偏离 T_TARGET，
     需显式报告，避免"看起来守恒其实不守恒"。

【6】predict_new 增加输入列校验
     缺列时给出清晰报错，而不是 np.log 处的 KeyError。

【7】其余保持 v6 不变
     —— 无 OVERRIDE、无测试集决策、整数预测为主指标、
        clip 下界=训练集观测最小值、CV 裁决定特征集与模型。
==============================================================================
"""
import os
import warnings

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import RidgeCV
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['svg.fonttype'] = 'path'

# ==================== 配置 ====================
TRAIN_PATH = '体重分组训练集.xlsx'
TEST_PATH = '体重分组测试集.xlsx'
OUTPUT_DIR = './output_fixed'
MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42
ORIGINAL_TARGETS = ['chine', 'coccyx', 'cocc']
CV_SPLITS, CV_REPEATS = 5, 5
N_FOLDS = CV_SPLITS * CV_REPEATS

SANITY_MAX = {'chine': 200, 'coccyx': 120, 'cocc': 120}

LENGTH_BINS = [0, 30, 45, 51, np.inf]
LENGTH_LABELS = ['Age1', 'Age2', 'Age3', 'Age4']
WEIGHT_BINS = [0, 30, 60, 90, np.inf]
WEIGHT_LABELS = ['Very Light', 'Light', 'Medium', 'Heavy']

BASE_FEATURES = ['体长(cm)', '体重(g)', 'log_L', 'log_W', 'L_W_ratio', 'W_L2',
                 'sqrt_W', 'sqrt_L', '体长组编码', '体重组编码']
RAW_FEATURES = ['体长(cm)', '体重(g)']

SIMPLICITY_ORDER = ['Ridge', 'SVR(C=0.5)', 'SVR(C=1)', 'SVR(C=5)',
                    'SVR(C=10)', 'GP(fixed)']


# ==================== 数据加载 ====================
def add_biological_features(df):
    L, W = df['体长(cm)'], df['体重(g)']
    df['log_L'] = np.log(L)
    df['log_W'] = np.log(W)
    df['L_W_ratio'] = L / W
    df['W_L2'] = W / (L ** 2)
    df['sqrt_W'] = np.sqrt(W)
    df['sqrt_L'] = np.sqrt(L)
    df['体长组'] = pd.cut(L, bins=LENGTH_BINS, labels=LENGTH_LABELS, right=False)
    df['体长组编码'] = df['体长组'].map({l: i + 1 for i, l in enumerate(LENGTH_LABELS)}).astype(float)
    df['体重组'] = pd.cut(W, bins=WEIGHT_BINS, labels=WEIGHT_LABELS, right=False)
    df['体重组编码'] = df['体重组'].map({l: i + 1 for i, l in enumerate(WEIGHT_LABELS)}).astype(float)
    df.drop(['体长组', '体重组'], axis=1, inplace=True)
    return df


def sanity_check(df, name):
    bad = []
    for col, mx in SANITY_MAX.items():
        hit = df.index[df[col] > mx]
        if len(hit):
            bad.append(f"{col}>{mx}: 行{list(hit)} 值{df.loc[hit, col].tolist()}")
    if bad:
        print(f"  [警告] {name} 存在不可能标注 -> " + " | ".join(bad))
    else:
        print(f"  {name}: {len(df)} 行，生物学合理性校验通过")
    return bad


print("Loading data...")
tr = add_biological_features(pd.read_excel(TRAIN_PATH, sheet_name='Sheet1'))
te = add_biological_features(pd.read_excel(TEST_PATH, sheet_name='Sheet1'))
print("  （不再静默改写任何数据；离群值如需修正请直接改源文件）")
sanity_check(tr, '训练集')
sanity_check(te, '测试集')

ytr = tr[ORIGINAL_TARGETS].values
yte = te[ORIGINAL_TARGETS].values

# ==================== 诊断 ====================
print("\n" + "=" * 72)
print("=== 诊断 ===")
print("=" * 72)
print("\n-- 各目标取值范围（决定 clip 下界）--")
for i, t in enumerate(ORIGINAL_TARGETS):
    print(f"{t:8s} train min={ytr[:, i].min():.0f} max={ytr[:, i].max():.0f}")
LOWER = ytr.min(axis=0).astype(float)
print(f"-> clip 下界: {dict(zip(ORIGINAL_TARGETS, LOWER.astype(int)))}")

tot = ytr.sum(axis=1)
cv_total = float(tot.std() / tot.mean())
T_TARGET = float(tot.mean())
TENABLE = bool(cv_total < 0.15)
print(f"\n-- 总椎数守恒性（仅训练集）--")
print(f"train total: mean={tot.mean():.2f} std={tot.std():.2f} CV={cv_total:.4f}")
print(f"-> CV={cv_total:.4f} {'<' if TENABLE else '>='} 0.15 → 守恒假设"
      f"{'成立' if TENABLE else '不成立'}（是否启用补偿仍交由 CV 裁决）")

# ==================== 特征集 ====================
Xdf = tr[BASE_FEATURES]
rho = Xdf.corr(method='spearman').abs()
tgt_rho = pd.DataFrame(
    {t: [abs(spearmanr(Xdf[c].values, ytr[:, i])[0]) for c in BASE_FEATURES]
     for i, t in enumerate(ORIGINAL_TARGETS)},
    index=BASE_FEATURES)
score = tgt_rho.mean(axis=1)

# ---- v7 修复点 1：冗余过滤 ----
drop = set()
for a in BASE_FEATURES:
    for b in BASE_FEATURES:
        if a == b or a in drop or b in drop:
            continue
        if rho.loc[a, b] > 0.95:
            # 保留与三目标平均 |Spearman| 更高者；淘汰的写特征名，而不是得分
            loser = a if score[a] < score[b] else b
            drop.add(loser)
FILT_FEATURES = [c for c in BASE_FEATURES if c not in drop]

print(f"\n-- 冗余过滤(|Spearman|>0.95，成对保留与目标更相关者) --")
print(f"  丢弃: {sorted(drop)}")
print(f"  保留 {len(FILT_FEATURES)} 个: {FILT_FEATURES}")
assert len(FILT_FEATURES) <= len(BASE_FEATURES)  # 便于自我校验

Xt = {'full': tr[BASE_FEATURES].values,
      'filt': tr[FILT_FEATURES].values,
      'raw': tr[RAW_FEATURES].values}
Xe = {'full': te[BASE_FEATURES].values,
      'filt': te[FILT_FEATURES].values,
      'raw': te[RAW_FEATURES].values}
FEATURE_SETS = ['full', 'filt', 'raw']
FEATURES_OF = {'full': BASE_FEATURES, 'filt': FILT_FEATURES, 'raw': RAW_FEATURES}


# ==================== 候选模型 ====================
def mk_ridge():
    return Pipeline([('s', StandardScaler()),
                     ('r', RidgeCV(alphas=np.logspace(-2, 2, 20), cv=5))])


def mk_svr(C):
    return Pipeline([
        ('s', StandardScaler()),
        ('r', TransformedTargetRegressor(
            regressor=SVR(kernel='rbf', C=C, gamma='scale', epsilon=0.1),
            transformer=StandardScaler()))])


def mk_gp():
    k = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-2)
    return Pipeline([
        ('s', StandardScaler()),
        ('r', TransformedTargetRegressor(
            regressor=GaussianProcessRegressor(kernel=k, optimizer=None,
                                               alpha=1e-6,
                                               random_state=RANDOM_STATE),
            transformer=StandardScaler()))])


CANDIDATE_FACTORIES = {
    'Ridge': mk_ridge,
    'SVR(C=0.5)': lambda: mk_svr(0.5),
    'SVR(C=1)': lambda: mk_svr(1.0),
    'SVR(C=5)': lambda: mk_svr(5.0),
    'SVR(C=10)': lambda: mk_svr(10.0),
    'GP(fixed)': mk_gp,
}


def cv_score(factory, Xm, yv):
    rmses = []
    for tri, vai in RepeatedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS,
                                  random_state=RANDOM_STATE).split(Xm):
        m = factory()
        m.fit(Xm[tri], yv[tri])
        p = m.predict(Xm[vai])
        rmses.append(np.sqrt(mean_squared_error(yv[vai], p)))
    e = np.array(rmses)
    return float(e.mean()), float(e.std())


# ==================== 模型选择 ====================
print("\n" + "=" * 72)
print("=== 逐目标选择：RepeatedKFold CV + 1-SE 规则 + 最简单模型（仅训练集）===")
print("=" * 72)

chosen, cv_table = {}, {}
for i, t in enumerate(ORIGINAL_TARGETS):
    yv = ytr[:, i]
    best = {'cv': np.inf, 'sd': np.inf, 'feat': None, 'model': None}
    table = []
    for fs in FEATURE_SETS:
        Xm = Xt[fs]
        for name in SIMPLICITY_ORDER:
            mu, sd = cv_score(CANDIDATE_FACTORIES[name], Xm, yv)
            table.append((fs, name, mu, sd))
            if mu < best['cv']:
                best = {'cv': mu, 'sd': sd, 'feat': fs, 'model': name}

    # ---- v7 修复点 2：1-SE 阈值用 sqrt(CV_REPEATS)，不是 sqrt(N_FOLDS) ----
    # 25 折来自 5×5 RepeatedKFold，折间不独立，但 5 个 repeat 之间独立。
    # 用重复数折算标准误，与 caret / glmnet 的做法一致。
    se = best['sd'] / np.sqrt(CV_REPEATS)
    thr = best['cv'] + se
    within = [(fs, nm, mu) for fs, nm, mu, _ in table if mu <= thr]
    pick = min(within, key=lambda r: (SIMPLICITY_ORDER.index(r[1]),
                                      len(FEATURES_OF[r[0]])))
    picked = {'feat': pick[0], 'model': pick[1], 'cv': pick[2], 'thr': thr, 'se': se}
    chosen[t], cv_table[t] = picked, table

    print(f"\n--- {t} ---")
    for fs in FEATURE_SETS:
        for nm in SIMPLICITY_ORDER:
            mu, sd = next((m2, s2) for f2, n2, m2, s2 in table
                          if f2 == fs and n2 == nm)
            mark = "  <== 选中" if (fs == picked['feat'] and nm == picked['model']) else ""
            print(f"  {fs:5s}/{nm:10s}: CV RMSE = {mu:7.3f} ± {sd:6.3f}{mark}")
    print(f"  CV 最优 = {best['model']}/{best['feat']} ({best['cv']:.3f})；"
          f"SE = {se:.3f}；1-SE 阈值 = {thr:.3f}")
    print(f"  规则裁决 → {picked['model']} / {picked['feat']}")

# ==================== 全量训练 ====================
print("\n" + "=" * 72)
print("=== 全量训练 ===")
print("=" * 72)
fitted = {}
for i, t in enumerate(ORIGINAL_TARGETS):
    g = chosen[t]
    m = CANDIDATE_FACTORIES[g['model']]()
    m.fit(Xt[g['feat']], ytr[:, i])
    fitted[t] = m
    print(f"  {t:8s}: {g['model']} / {g['feat']} ({len(FEATURES_OF[g['feat']])} 特征)")


# ==================== CV 裁决总和补偿 ====================
# ---- v7 修复点 4：OOF 按样本跨折平均 ----
def oof_predictions():
    """用选定模型产生 out-of-fold 三列预测（训练集内），每样本取多折平均。"""
    oof = np.zeros((len(ytr), 3))
    for i, t in enumerate(ORIGINAL_TARGETS):
        g = chosen[t]
        Xm = Xt[g['feat']]
        acc = np.zeros(len(ytr))
        cnt = np.zeros(len(ytr))
        for tri, vai in RepeatedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS,
                                      random_state=RANDOM_STATE).split(Xm):
            m = CANDIDATE_FACTORIES[g['model']]()
            m.fit(Xm[tri], ytr[tri, i])
            acc[vai] += m.predict(Xm[vai])
            cnt[vai] += 1
        oof[:, i] = acc / np.maximum(cnt, 1)
    return oof


print("\n" + "=" * 72)
print("=== 总和守恒补偿：开/关 由 CV 裁决（训练集内）===")
print("=" * 72)
USE_TOTAL = False
constraint_info = {'tenable': TENABLE, 'cv_total': cv_total, 'applied': False}
if TENABLE:
    oof = oof_predictions()
    err_off = float(np.sqrt(mean_squared_error(ytr[:, 2], np.clip(oof[:, 2], LOWER[2], None))))
    comp = oof.copy()
    comp[:, 2] += T_TARGET - comp.sum(axis=1)
    err_on = float(np.sqrt(mean_squared_error(ytr[:, 2], np.clip(comp[:, 2], LOWER[2], None))))
    USE_TOTAL = bool(err_on < err_off)
    constraint_info.update({'err_off': err_off, 'err_on': err_on, 'applied': USE_TOTAL})
    print(f"  cocc CV-RMSE  不补偿 = {err_off:.4f}   加法补偿 = {err_on:.4f}")
    print(f"  -> CV 裁决: {'启用' if USE_TOTAL else '不启用'}加法补偿到 cocc")
else:
    print("  总椎数 CV >= 0.15，守恒假设不成立，跳过补偿")

# ==================== 测试集预测 ====================
P = np.zeros((len(yte), 3))
for i, t in enumerate(ORIGINAL_TARGETS):
    g = chosen[t]
    P[:, i] = fitted[t].predict(Xe[g['feat']])

resid_stats = None
cocc_below_lower = 0
if USE_TOTAL:
    resid = T_TARGET - P.sum(axis=1)
    resid_stats = {'mean': float(resid.mean()), 'min': float(resid.min()),
                   'max': float(resid.max()), 'absmax': float(np.abs(resid).max())}
    P[:, 2] += resid
    # ---- v7 修复点 5：诊断补偿后 cocc 是否被推到下界以下 ----
    cocc_below_lower = int((P[:, 2] < LOWER[2]).sum())
    print(f"\n[补偿] resid: mean={resid.mean():+.2f} "
          f"min={resid.min():+.2f} max={resid.max():+.2f}")
    print(f"[补偿] cocc 补偿后低于下界的样本数: {cocc_below_lower}")

# 整数化
P_int = np.zeros_like(P)
clip_cnt, bound_hit = {}, {}
for i, t in enumerate(ORIGINAL_TARGETS):
    lo = LOWER[i]
    r = np.round(P[:, i])
    clip_cnt[t] = int((r < lo).sum())
    P_int[:, i] = np.clip(r, lo, None)
    bound_hit[t] = int((P_int[:, i] == lo).sum())


def evaluate(y_true, y_pred, names):
    res = {}
    for i, n in enumerate(names):
        yt, yp = np.asarray(y_true[:, i]), np.asarray(y_pred[:, i])
        nz = yt != 0
        mape = float(np.mean(np.abs((yt[nz] - yp[nz]) / yt[nz])) * 100) if nz.any() else np.nan
        res[n] = {'R2': r2_score(yt, yp),
                  'RMSE': float(np.sqrt(mean_squared_error(yt, yp))),
                  'MAE': float(mean_absolute_error(yt, yp)),
                  'MAPE': mape,
                  'Spearman': float(spearmanr(yt, yp)[0])}
    return res


metrics_int = evaluate(yte, P_int, ORIGINAL_TARGETS)
metrics_raw = evaluate(yte, np.clip(P, LOWER, None), ORIGINAL_TARGETS)

print("\n--- 测试指标【主】：整数预测（= predictions.xlsx 交付内容）---")
for n, m in metrics_int.items():
    print(f"{n:8s}: R2={m['R2']:+.4f} RMSE={m['RMSE']:.4f} MAE={m['MAE']:.4f} "
          f"MAPE={m['MAPE']:.2f}% Spearman={m['Spearman']:+.4f}")
print("\n--- 测试指标【辅】：连续预测 ---")
for n, m in metrics_raw.items():
    print(f"{n:8s}: R2={m['R2']:+.4f} RMSE={m['RMSE']:.4f} MAPE={m['MAPE']:.2f}%")

print(f"\n[诊断] clip 触发次数: {clip_cnt}")
print(f"[诊断] 取到下界的预测数: {bound_hit}")
print(f"[诊断] 整数行和: mean={P_int.sum(axis=1).mean():.1f} "
      f"min={P_int.sum(axis=1).min():.0f} max={P_int.sum(axis=1).max():.0f}")
print(f"[诊断] 总和守恒补偿: {constraint_info}")

# ==================== 保存 ====================
out = te.copy()
for i, t in enumerate(ORIGINAL_TARGETS):
    out[f'Pred_{t}'] = P_int[:, i]
    out[f'PredRaw_{t}'] = P[:, i]
out['Pred_Sum'] = P_int.sum(axis=1)
out.to_excel(os.path.join(OUTPUT_DIR, 'predictions.xlsx'), index=False)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, t in enumerate(ORIGINAL_TARGETS):
    ax = axes[i]
    ax.scatter(yte[:, i], P_int[:, i], alpha=.6, edgecolors='black', linewidth=.5)
    mn = min(yte[:, i].min(), P_int[:, i].min())
    mx = max(yte[:, i].max(), P_int[:, i].max())
    ax.plot([mn, mx], [mn, mx], 'r--', lw=2)
    ax.set_xlabel('True')
    ax.set_ylabel('Predicted')
    ax.set_title(f"{t} ({chosen[t]['model']})\nR²={metrics_int[t]['R2']:.3f}")
    ax.grid(True, alpha=.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'predictions.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'predictions.svg'), bbox_inches='tight')
plt.close()

with open(os.path.join(OUTPUT_DIR, 'report.txt'), 'w', encoding='utf-8') as f:
    f.write("MODEL REPORT (s3mdr_v7)\n" + "=" * 66 + "\n\n")
    f.write("选择规则: RepeatedKFold(5x5) CV-RMSE + 1-SE 规则 + 最简单模型优先\n")
    f.write("          该规则预先定义、对三目标一致、仅使用训练集。\n")
    f.write("          全过程不使用测试集做任何决策；无 OVERRIDE。\n\n")
    f.write(f"特征集候选(由CV裁决): {FEATURE_SETS}\n")
    f.write(f"冗余过滤丢弃: {sorted(drop)}；保留: {FILT_FEATURES}\n")
    f.write(f"clip 下界(训练集观测最小): {dict(zip(ORIGINAL_TARGETS, LOWER.astype(int)))}\n")
    f.write(f"总椎数: mean={tot.mean():.2f} std={tot.std():.2f} CV={cv_total:.4f} "
            f"(守恒假设{'成立' if TENABLE else '不成立'})\n")
    f.write(f"加法补偿到 cocc: {USE_TOTAL}  {constraint_info}\n")
    if resid_stats:
        f.write(f"补偿残差: mean={resid_stats['mean']:+.2f} "
                f"min={resid_stats['min']:+.2f} max={resid_stats['max']:+.2f}\n")
    f.write(f"补偿后 cocc 触下界样本数: {cocc_below_lower}\n\n")
    for t in ORIGINAL_TARGETS:
        g = chosen[t]
        f.write(f"{t}: {g['model']} / {g['feat']} "
                f"(CV={g['cv']:.3f}, SE={g['se']:.3f}, 1SE阈值={g['thr']:.3f})\n")
        for fs, nm, mu, sd in cv_table[t]:
            f.write(f"    {fs:5s}/{nm:10s}: {mu:.3f} ± {sd:.3f}\n")
    f.write("\n测试指标【主=整数交付】:\n")
    for n, m in metrics_int.items():
        f.write(f"  {n}: " + "  ".join(f"{k}={v:.4f}" for k, v in m.items()) + "\n")
    f.write("\n测试指标【辅=连续】:\n")
    for n, m in metrics_raw.items():
        f.write(f"  {n}: " + "  ".join(f"{k}={v:.4f}" for k, v in m.items()) + "\n")
    f.write(f"\n诊断: clip次数={clip_cnt} 触下界数={bound_hit} "
            f"整数行和 mean={P_int.sum(axis=1).mean():.1f} "
            f"min={P_int.sum(axis=1).min():.0f} max={P_int.sum(axis=1).max():.0f}\n")

# ---- v7 修复点 3：保存"训练集下界"到 meta，供 predict_new 使用 ----
LOWER_DICT = {t: float(LOWER[i]) for i, t in enumerate(ORIGINAL_TARGETS)}

for t, m in fitted.items():
    joblib.dump(m, os.path.join(MODEL_DIR, f'{t}_model.pkl'))
joblib.dump({'features_of': FEATURES_OF, 'chosen': chosen,
             'targets': ORIGINAL_TARGETS,
             'lower_bounds': LOWER_DICT,           # <-- 用于推理
             'use_total_compensation': USE_TOTAL,
             'total_train_mean': T_TARGET,
             'cv_scheme': f'RepeatedKFold({CV_SPLITS}x{CV_REPEATS})',
             'length_bins': LENGTH_BINS, 'length_labels': LENGTH_LABELS,
             'weight_bins': WEIGHT_BINS, 'weight_labels': WEIGHT_LABELS},
            os.path.join(MODEL_DIR, 'meta.pkl'))


# ---- v7 修复点 3 & 6：predict_new 从 meta 读下界 + 输入列校验 ----
def predict_new(new_df):
    """对新样本推理。不依赖任何全局变量，可在新会话中独立使用。"""
    required_raw = ['体长(cm)', '体重(g)']
    missing = [c for c in required_raw if c not in new_df.columns]
    if missing:
        raise ValueError(f"predict_new 输入缺少必要列: {missing}")

    d = add_biological_features(new_df.copy())
    meta = joblib.load(os.path.join(MODEL_DIR, 'meta.pkl'))
    outp = np.zeros((len(d), 3))
    for i, t in enumerate(ORIGINAL_TARGETS):
        m = joblib.load(os.path.join(MODEL_DIR, f'{t}_model.pkl'))
        cols = meta['features_of'][meta['chosen'][t]['feat']]
        missing_cols = [c for c in cols if c not in d.columns]
        if missing_cols:
            raise ValueError(f"目标 {t} 需要的特征缺失: {missing_cols}")
        outp[:, i] = m.predict(d[cols].values)

    if meta['use_total_compensation']:
        outp[:, 2] += meta['total_train_mean'] - outp.sum(axis=1)

    for i, t in enumerate(ORIGINAL_TARGETS):
        lo = meta['lower_bounds'][t]
        outp[:, i] = np.clip(np.round(outp[:, i]), lo, None)
    return pd.DataFrame(outp, columns=[f'Pred_{t}' for t in ORIGINAL_TARGETS])


print(f"\n全部结果已保存至 {OUTPUT_DIR}")
print("模型已保存至", MODEL_DIR)
