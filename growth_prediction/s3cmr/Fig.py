import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr, wilcoxon
from scipy import stats
import os
import warnings

warnings.filterwarnings('ignore')

# ======================== 学术绘图风格（无框） ========================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 300          # 仅用于显示，不影响PDF
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['legend.frameon'] = False    # 全局去掉图例框
sns.set_style("ticks")
sns.set_context("paper", font_scale=1.2)

# ======================== 配置 ========================
TEST_PATH = '体重分组测试集.xlsx'
TARGETS = ['chine', 'coccyx', 'cocc']
OUTPUT_DIR = './figures_academic'
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_CONFIGS = [
    {'name': 'GPR+衍生 (Ours)', 'file': './output_fixed/predictions.xlsx',
     'cols': ['Pred_chine', 'Pred_coccyx', 'Pred_cocc'], 'highlight': True},
    {'name': 'MLP', 'file': './output_mlp/MLP模型_预测结果.xlsx',
     'cols': ['Predicted_chine', 'Predicted_coccyx', 'Predicted_cocc'], 'highlight': False},
    {'name': '随机森林', 'file': './output_rf/随机森林模型_预测结果.xlsx',
     'cols': ['Predicted_chine', 'Predicted_coccyx', 'Predicted_cocc'], 'highlight': False},
    {'name': 'SVR', 'file': './output_svr/SVR模型_预测结果.xlsx',
     'cols': ['Predicted_chine', 'Predicted_coccyx', 'Predicted_cocc'], 'highlight': False},
    {'name': 'XGBoost (原始)', 'file': './output_raw/原始模型_预测结果.xlsx',
     'cols': ['Predicted_chine', 'Predicted_coccyx', 'Predicted_cocc'], 'highlight': False},
    {'name': 'RF+Logit', 'file': './output_rf_logit/RF_Logit_预测结果.xlsx',
     'cols': ['Pred_chine', 'Pred_coccyx', 'Pred_cocc'], 'highlight': False},
    {'name': 'Gaussian Process', 'file': './output_gp/GP模型_预测结果.xlsx',
     'cols': ['Predicted_chine', 'Predicted_coccyx', 'Predicted_cocc'], 'highlight': False},
]

MODEL_COLORS = {
    'GPR+衍生 (Ours)': '#D62728',
    'MLP': '#1f77b4',
    '随机森林': '#ff7f0e',
    'SVR': '#2ca02c',
    'XGBoost (原始)': '#9467bd',
    'RF+Logit': '#8c564b',
    'Gaussian Process': '#e377c2'
}


# ======================== 加载并清洗真实值 ========================
def load_true_values_cleaned():
    df = pd.read_excel(TEST_PATH, sheet_name='Sheet1')
    required = ['体重(g)', '体长(cm)'] + TARGETS
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Test set missing column: {col}")
    mask = (df[required] != 0).all(axis=1)
    df_clean = df[mask].copy()
    kept_indices = df.index[mask].tolist()
    removed_indices = df.index[~mask].tolist()
    original_n_rows = len(df)
    print(f"原始测试集: {original_n_rows} 行，清洗后: {len(df_clean)} 行 (删除了 {len(removed_indices)} 行)")
    y_true = df_clean[TARGETS].values
    return y_true, kept_indices, original_n_rows


y_true, kept_indices, original_n_rows = load_true_values_cleaned()

# ======================== 加载预测并统一对齐 ========================
all_results = []
model_names = []
model_predictions = {}

for config in MODEL_CONFIGS:
    name = config['name']
    file_path = config['file']
    cols = config['cols']
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} not found, skipping {name}")
        continue

    df_pred = pd.read_excel(file_path)
    for col in cols:
        if col not in df_pred.columns:
            raise ValueError(f"File {file_path} missing column {col}")

    n_rows = len(df_pred)
    if n_rows == original_n_rows:
        y_pred = df_pred.iloc[kept_indices][cols].values
    elif n_rows == len(kept_indices):
        y_pred = df_pred[cols].values
    else:
        raise ValueError(
            f"Model {name}: prediction rows ({n_rows}) do not match expected ({len(kept_indices)} or {original_n_rows})")

    if y_pred.shape[0] != y_true.shape[0]:
        raise ValueError(f"After alignment, {name} has {y_pred.shape[0]} samples, but true has {y_true.shape[0]}")

    model_predictions[name] = y_pred

    # 计算每个目标的指标（保留原有功能）
    metrics_dict = {}
    for i, target in enumerate(TARGETS):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        mask = ~(np.isnan(yt) | np.isnan(yp))
        yt_clean = yt[mask]
        yp_clean = yp[mask]
        if len(yt_clean) == 0:
            metrics_dict[target] = {'R2': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'MAPE': np.nan, 'Spearman': np.nan}
            continue
        r2 = r2_score(yt_clean, yp_clean)
        rmse = np.sqrt(mean_squared_error(yt_clean, yp_clean))
        mae = mean_absolute_error(yt_clean, yp_clean)
        if np.any(yt_clean == 0):
            mape = np.nan
        else:
            mape = np.mean(np.abs((yt_clean - yp_clean) / yt_clean)) * 100
        spearman, _ = spearmanr(yt_clean, yp_clean)
        metrics_dict[target] = {'R2': r2, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'Spearman': spearman}

    df_metrics = pd.DataFrame(metrics_dict).T
    df_metrics['Model'] = name
    all_results.append(df_metrics)
    model_names.append(name)

df_all = pd.concat(all_results).reset_index().rename(columns={'index': 'Target'})
df_all.set_index(['Model', 'Target'], inplace=True)
print("\n=== All model metrics (aligned on cleaned test set) ===")
print(df_all.round(4))

excel_path = os.path.join(OUTPUT_DIR, 'model_comparison_table.xlsx')
df_all.reset_index().to_excel(excel_path, index=False)
print(f"\nComparison table saved to: {excel_path}")


# ======================== 构建长格式数据（含精确度） ========================
def build_long_df(y_true, model_predictions, targets):
    data = []
    for model_name, preds in model_predictions.items():
        for i, target in enumerate(targets):
            for j in range(len(y_true)):
                obs = y_true[j, i]
                pred = preds[j, i]
                abs_err = np.abs(pred - obs)
                pct_err = (abs_err / obs) * 100 if obs != 0 else np.nan
                accuracy = 1 - (abs_err / obs) if obs != 0 else np.nan   # 精确度
                data.append({
                    'Model': model_name,
                    'Target': target,
                    'Observed': obs,
                    'Predicted': pred,
                    'Residual': pred - obs,
                    'AbsError': abs_err,
                    'PctError': pct_err,
                    'Accuracy': accuracy          # 新增列
                })
    return pd.DataFrame(data)


df_long = build_long_df(y_true, model_predictions, TARGETS)
available_models = list(model_predictions.keys())


def get_model_color(name):
    return MODEL_COLORS.get(name, '#333333')


# ======================== 图1: 残差小提琴图 + 箱线图 + 抖动散点 (PDF) ========================
def plot_fig1_violin_residuals(df_long, save_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
    for idx, target in enumerate(TARGETS):
        ax = axes[idx]
        df_target = df_long[df_long['Target'] == target].copy()
        order = ['GPR+衍生 (Ours)'] + [m for m in available_models if m != 'GPR+衍生 (Ours)']
        sns.violinplot(data=df_target, x='Model', y='Residual', order=order,
                       palette=MODEL_COLORS, inner=None, alpha=0.5, linewidth=1, ax=ax)
        sns.boxplot(data=df_target, x='Model', y='Residual', order=order,
                    width=0.15, fliersize=2, linewidth=1.2, ax=ax,
                    boxprops=dict(alpha=0.9, zorder=3),
                    medianprops=dict(color='black', linewidth=2, zorder=4))
        df_sample = df_target.groupby('Model').apply(
            lambda x: x.sample(min(100, len(x)), random_state=42)
        ).reset_index(drop=True)
        sns.stripplot(data=df_sample, x='Model', y='Residual', order=order,
                      color='black', alpha=0.2, size=2.5, jitter=True, ax=ax, zorder=2)
        ax.axhline(0, color='#E94F37', linestyle='--', linewidth=1.5, zorder=1)
        ours_resid = df_target[df_target['Model'] == 'GPR+衍生 (Ours)']['Residual'].values
        y_max = df_target['Residual'].max()
        for j, model in enumerate(order[1:], 1):
            other_resid = df_target[df_target['Model'] == model]['Residual'].values
            if len(ours_resid) == len(other_resid):
                try:
                    stat, p = wilcoxon(np.abs(ours_resid), np.abs(other_resid))
                    stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                    if stars:
                        ax.plot([0, j], [y_max * 1.08, y_max * 1.08], 'k-', linewidth=0.8)
                        ax.text(j / 2, y_max * 1.12, stars, ha='center', fontsize=12, color='red')
                except:
                    pass
        ax.set_title(f'{target}', fontsize=14, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Residual (Predicted − Observed)', fontsize=11)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        sns.despine(ax=ax)
    plt.suptitle('Figure 1. Residual Distribution across Models (Violin + Box + Strip)',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/Fig1_Violin_Residuals.pdf', bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_dir}/Fig1_Violin_Residuals.pdf")


# ======================== 图2: 预测值 vs 真实值 (PDF) ========================
def plot_fig2_scatter_regression(df_long, save_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for idx, target in enumerate(TARGETS):
        ax = axes[idx]
        df_target = df_long[df_long['Target'] == target]
        obs_min = df_target['Observed'].min()
        obs_max = df_target['Observed'].max()
        ax.plot([obs_min, obs_max], [obs_min, obs_max], 'k--', linewidth=1.5,
                label='1:1 Line', alpha=0.7, zorder=1)
        key_models_for_reg = ['GPR+衍生 (Ours)', 'MLP', 'XGBoost (原始)', '随机森林']
        for model in df_target['Model'].unique():
            df_model = df_target[df_target['Model'] == model]
            color = get_model_color(model)
            z = 3 if model.startswith('GPR') else 2
            s = 50 if model.startswith('GPR') else 30
            a = 0.6 if model.startswith('GPR') else 0.35
            ax.scatter(df_model['Observed'], df_model['Predicted'],
                       alpha=a, s=s, color=color, edgecolors='white', linewidth=0.3,
                       label=model, zorder=z)
            if model in key_models_for_reg:
                sns.regplot(data=df_model, x='Observed', y='Predicted',
                            scatter=False, ax=ax, color=color,
                            line_kws={'linewidth': 2, 'alpha': 0.8},
                            ci=95, truncate=False, label='_nolegend_')
        df_ours = df_target[df_target['Model'] == 'GPR+衍生 (Ours)']
        r2 = r2_score(df_ours['Observed'], df_ours['Predicted'])
        rmse = np.sqrt(mean_squared_error(df_ours['Observed'], df_ours['Predicted']))
        mae = mean_absolute_error(df_ours['Observed'], df_ours['Predicted'])
        textstr = f'GPR+衍生 (Ours)\n$R^2$ = {r2:.4f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}'
        props = dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.9, edgecolor='gray')
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props, zorder=5)
        ax.set_xlabel('Observed', fontsize=12)
        ax.set_ylabel('Predicted', fontsize=12)
        ax.set_title(f'{target}', fontsize=14, fontweight='bold')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(alpha=0.3, linestyle='--')
        ax.legend(loc='lower right', fontsize=8, frameon=False, markerscale=1.5, title='Model')
        sns.despine(ax=ax)
    plt.suptitle('Figure 2. Observed vs. Predicted with Regression Lines (95% CI)',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/Fig2_Scatter_Regression.pdf', bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_dir}/Fig2_Scatter_Regression.pdf")


# ======================== 图3: Bland-Altman (PDF) ========================
def plot_fig3_bland_altman(df_long, save_dir):
    key_models = [m for m in ['GPR+衍生 (Ours)', 'MLP', 'XGBoost (原始)', '随机森林'] if m in available_models]
    if not key_models:
        return
    fig, axes = plt.subplots(len(TARGETS), len(key_models), figsize=(4 * len(key_models), 4 * len(TARGETS)))
    if len(TARGETS) == 1 and len(key_models) == 1:
        axes = np.array([[axes]])
    for i, target in enumerate(TARGETS):
        for j, model in enumerate(key_models):
            ax = axes[i, j] if len(TARGETS) > 1 else axes[j]
            df_sub = df_long[(df_long['Target'] == target) & (df_long['Model'] == model)]
            obs = df_sub['Observed'].values
            pred = df_sub['Predicted'].values
            mean = (obs + pred) / 2
            diff = pred - obs
            md = np.mean(diff)
            sd = np.std(diff, ddof=1)
            loa_lower = md - 1.96 * sd
            loa_upper = md + 1.96 * sd
            ax.scatter(mean, diff, alpha=0.5, s=40, c=get_model_color(model),
                       edgecolors='white', linewidth=0.5, zorder=3)
            ax.axhline(md, color='#E94F37', linestyle='-', linewidth=2, zorder=2)
            ax.axhline(loa_lower, color='#6A994E', linestyle='--', linewidth=1.5, zorder=2)
            ax.axhline(loa_upper, color='#6A994E', linestyle='--', linewidth=1.5, zorder=2)
            ax.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.7, zorder=1)
            ax.fill_between([mean.min() - 1, mean.max() + 1], loa_lower, loa_upper,
                            alpha=0.08, color='#6A994E', zorder=1)
            ax.text(mean.max() * 0.95, md + 0.3 * sd, f'Mean: {md:.3f}',
                    fontsize=9, color='#E94F37', fontweight='bold', ha='right')
            ax.text(mean.max() * 0.95, loa_upper - 0.3 * sd, f'+1.96SD: {loa_upper:.3f}',
                    fontsize=8, color='#6A994E', ha='right')
            ax.text(mean.max() * 0.95, loa_lower + 0.3 * sd, f'-1.96SD: {loa_lower:.3f}',
                    fontsize=8, color='#6A994E', ha='right')
            if i == 0:
                ax.set_title(model, fontsize=12, fontweight='bold')
            if j == 0:
                ax.set_ylabel(f'{target}\nDifference (Pred − Obs)', fontsize=10)
            if i == len(TARGETS) - 1:
                ax.set_xlabel('Mean of Observed and Predicted', fontsize=10)
            ax.grid(alpha=0.3, linestyle='--')
            sns.despine(ax=ax)
    plt.suptitle('Figure 3. Bland-Altman Analysis of Model Consistency',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/Fig3_Bland_Altman.pdf', bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_dir}/Fig3_Bland_Altman.pdf")


# ======================== 图4: 热力图 (PDF) ========================
def plot_fig4_heatmap(df_all, save_dir):
    metrics = ['R2', 'RMSE', 'MAE', 'Spearman']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        pivot = df_all.reset_index().pivot(index='Model', columns='Target', values=metric)
        order = ['GPR+衍生 (Ours)'] + [m for m in pivot.index if m != 'GPR+衍生 (Ours)']
        pivot = pivot.reindex(order)
        if metric in ['R2', 'Spearman']:
            cmap = 'YlGn'
            fmt = '.3f'
        else:
            cmap = 'YlOrRd_r'
            fmt = '.2f'
        sns.heatmap(pivot, annot=True, fmt=fmt, cmap=cmap, linewidths=0.5,
                    ax=ax, cbar_kws={'label': metric, 'shrink': 0.8},
                    annot_kws={'size': 10, 'weight': 'bold'})
        ax.set_title(f'{metric}', fontsize=13, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.suptitle('Figure 4. Comprehensive Performance Heatmap',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/Fig4_Heatmap.pdf', bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_dir}/Fig4_Heatmap.pdf")


# ======================== 图5: ECDF (PDF) ========================
def plot_fig5_ecdf(df_long, save_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, target in enumerate(TARGETS):
        ax = axes[idx]
        df_target = df_long[df_long['Target'] == target]
        for model in df_target['Model'].unique():
            df_model = df_target[df_target['Model'] == model]
            errors = df_model['AbsError'].dropna().sort_values()
            y_vals = np.arange(1, len(errors) + 1) / len(errors)
            lw = 2.8 if model.startswith('GPR') else 1.5
            ax.plot(errors, y_vals, linewidth=lw, color=get_model_color(model),
                    label=model, alpha=0.9)
        ax.set_xlabel('Absolute Error', fontsize=12)
        ax.set_ylabel('Cumulative Probability', fontsize=12)
        ax.set_title(f'{target}', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9, frameon=False)
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_xlim(left=0)
        sns.despine(ax=ax)
    plt.suptitle('Figure 5. Cumulative Distribution of Absolute Errors (ECDF)',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/Fig5_ECDF.pdf', bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_dir}/Fig5_ECDF.pdf")


# ======================== 图6: 箱线图+均值点 (PDF) ========================
def plot_fig6_boxplot_mean(df_long, save_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for idx, target in enumerate(TARGETS):
        ax = axes[idx]
        df_target = df_long[df_long['Target'] == target]
        order = ['GPR+衍生 (Ours)'] + [m for m in available_models if m != 'GPR+衍生 (Ours)']
        sns.boxplot(data=df_target, x='Model', y='AbsError', order=order,
                    palette=MODEL_COLORS, width=0.6, ax=ax,
                    medianprops=dict(color='black', linewidth=2))
        sns.pointplot(data=df_target, x='Model', y='AbsError', order=order,
                      color='red', markers='D', scale=0.8, ax=ax, ci=None, linestyles='')
        ax.set_title(f'{target}', fontsize=14, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Absolute Error', fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        sns.despine(ax=ax)
    plt.suptitle('Figure 6. Absolute Error Distribution across Models (Box + Mean)',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/Fig6_Boxplot_Mean.pdf', bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_dir}/Fig6_Boxplot_Mean.pdf")


# ======================== 执行绘图 ========================
print("\nGenerating academic figures (PDF vector format)...")
plot_fig1_violin_residuals(df_long, OUTPUT_DIR)
plot_fig2_scatter_regression(df_long, OUTPUT_DIR)
plot_fig3_bland_altman(df_long, OUTPUT_DIR)
plot_fig4_heatmap(df_all, OUTPUT_DIR)
plot_fig5_ecdf(df_long, OUTPUT_DIR)
plot_fig6_boxplot_mean(df_long, OUTPUT_DIR)

print("\nAll figures generated successfully as PDF in:", os.path.abspath(OUTPUT_DIR))

# ======================== 导出详细预测与误差（含个体精确度） ========================
detailed_path = os.path.join(OUTPUT_DIR, 'detailed_predictions_errors.xlsx')
df_long.to_excel(detailed_path, index=False)
print(f"\n详细预测、真实值及个体精确度已保存至: {detailed_path}")

# ======================== 计算并导出按目标分组的平均精确度 ========================
accuracy_by_target = df_long.groupby(['Model', 'Target'])['Accuracy'].mean().unstack()
accuracy_by_target.to_excel(os.path.join(OUTPUT_DIR, 'accuracy_summary_by_target.xlsx'))
print(f"按骨节平均精确度已保存至: {OUTPUT_DIR}/accuracy_summary_by_target.xlsx")

# ======================== 计算并导出整体平均精确度（所有目标合并） ========================
overall_accuracy = df_long.groupby('Model')['Accuracy'].mean()
overall_accuracy.to_frame(name='Overall_Accuracy').to_excel(os.path.join(OUTPUT_DIR, 'overall_accuracy.xlsx'))
print(f"整体平均精确度已保存至: {OUTPUT_DIR}/overall_accuracy.xlsx")

# ======================== 原有独立表格绘制（保存为PDF） ========================
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# ================== 数据准备 ==================
models = ['GPR+衍生 (Ours)', 'MLP', '随机森林', 'SVR',
          'XGBoost', 'RF+Logit', 'Gaussian Process']
metrics = ['R²', 'RMSE', 'MAE', 'MAPE (%)', 'Spearman']

# [model, target, metric]
data = np.array([
    # GPR+衍生 (Ours)
    [[-0.0428, 2.2566, 1.9450, 2.1860, 0.2431],
     [0.4357, 4.9352, 4.0992, 11.4369, 0.6604],
     [-0.0635, 8.8306, 6.9451, 32.5223, 0.2324]],
    # MLP
    [[-4.9748, 5.4015, 4.6751, 5.2399, 0.0808],
     [0.2758, 5.5912, 4.6223, 12.7976, 0.5460],
     [-0.1148, 9.0412, 7.2707, 32.3534, 0.1158]],
    # 随机森林
    [[-0.4850, 2.6929, 2.3226, 2.6097, 0.0769],
     [-0.0674, 6.7876, 5.1421, 13.7135, 0.6151],
     [-0.7203, 11.2311, 8.1085, 37.6789, 0.1284]],
    # SVR
    [[-0.1891, 2.4097, 1.9729, 2.2196, 0.0295],
     [0.4131, 5.0330, 4.1341, 11.7159, 0.6584],
     [0.0622, 8.2923, 6.6700, 31.7482, 0.2199]],
    # XGBoost
    [[-0.9471, 3.0835, 2.5187, 2.8202, -0.0829],
     [0.0818, 6.2956, 5.0662, 13.8540, 0.5837],
     [-0.3539, 9.9638, 7.8511, 35.1645, 0.1618]],
    # RF+Logit
    [[-0.4541, 2.6647, 2.2258, 2.5066, 0.0796],
     [0.1443, 6.0775, 4.6093, 12.2857, 0.6520],
     [-0.2973, 9.7534, 7.6086, 34.9162, 0.2073]],
    # Gaussian Process
    [[-0.1393, 2.3587, 1.9831, 2.2279, 0.1141],
     [0.4055, 5.0657, 4.1506, 11.7257, 0.7383],
     [-0.0526, 8.7855, 7.0530, 32.8895, 0.2199]]
])

best_dir = [1, -1, -1, -1, 1]

C_HEADER = '#2C3E50'
C_OURS = '#FFF3CD'
C_BEST = '#D4EDDA'
C_ALT = '#F8F9FA'
C_TEXT = '#212529'
C_BEST_TEXT = '#155724'

fig, axes = plt.subplots(3, 1, figsize=(10, 11), facecolor='white')
fig.subplots_adjust(hspace=0.45, top=0.93, bottom=0.06)

targets = ['chine', 'coccyx', 'cocc']
target_titles = ['CHINE', 'COCCYX', 'COCC']

for t_idx in range(3):
    ax = axes[t_idx]
    ax.axis('off')

    table_data = []
    for m_idx in range(len(models)):
        row = [models[m_idx]]
        for metric_idx in range(5):
            val = data[m_idx, t_idx, metric_idx]
            if metric_idx == 3:
                row.append(f"{val:.2f}")
            else:
                row.append(f"{val:.4f}")
        table_data.append(row)

    col_labels = ['Model'] + metrics
    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc='upper center',
        cellLoc='center',
        colColours=[C_HEADER] * len(col_labels),
        colWidths=[0.28, 0.12, 0.12, 0.12, 0.14, 0.12]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.9)

    for j in range(len(col_labels)):
        cell = table[(0, j)]
        cell.set_text_props(color='white', fontweight='bold', fontsize=10)
        cell.set_height(0.065)
        cell.set_facecolor(C_HEADER)

    best_indices = []
    for metric_idx in range(5):
        vals = data[:, t_idx, metric_idx]
        if best_dir[metric_idx] == 1:
            best_idx = np.argmax(vals)
        else:
            best_idx = np.argmin(vals)
        best_indices.append(best_idx)

    for i in range(len(models)):
        is_ours = 'Ours' in models[i]
        for j in range(len(col_labels)):
            cell = table[(i + 1, j)]
            metric_idx = j - 1
            if i % 2 == 1:
                base_color = C_ALT
            else:
                base_color = 'white'
            if is_ours:
                cell.set_facecolor(C_OURS)
            else:
                cell.set_facecolor(base_color)
            cell.set_text_props(color=C_TEXT, fontsize=9.5)
            if j > 0 and i == best_indices[metric_idx]:
                cell.set_text_props(fontweight='bold', color=C_BEST_TEXT, fontsize=9.5)
                if not is_ours:
                    cell.set_facecolor(C_BEST)

    ax.set_title(f'{target_titles[t_idx]} — Comparison of Predictive Models',
                 fontsize=13, fontweight='bold', color=C_TEXT, pad=15)

    for j in range(len(col_labels)):
        table[(0, j)].set_edgecolor('#FFFFFF')
        table[(0, j)].set_linewidth(0.5)
        table[(1, j)].set_edgecolor('#2C3E50')
        table[(1, j)].set_linewidth(1.5)
        table[(len(models), j)].set_edgecolor('#2C3E50')
        table[(len(models), j)].set_linewidth(1.2)

fig.suptitle('Table 1. Performance Comparison Across Three Vertebra Targets',
             fontsize=15, fontweight='bold', y=0.98, color=C_TEXT)

fig.text(0.5, 0.01,
         'Note: Best values in each column are shown in bold. Our proposed model is highlighted in gold. '
         '↑ indicates higher is better; ↓ indicates lower is better.',
         ha='center', fontsize=9, style='italic', color='#6C757D')

plt.savefig(os.path.join(OUTPUT_DIR, 'model_comparison_table.pdf'), bbox_inches='tight', facecolor='white')
plt.show()
print(f"\nTable 1 saved to: {OUTPUT_DIR}/model_comparison_table.pdf")

print("\nAll tasks completed.")