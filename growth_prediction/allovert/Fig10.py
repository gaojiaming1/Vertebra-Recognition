import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import warnings
import os

warnings.filterwarnings('ignore')

# ================== 全局样式设置 ==================
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['axes.titleweight'] = 'normal'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.15

plt.rcParams['font.size'] = 8
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 7
plt.rcParams['legend.fontsize'] = 7
plt.rcParams['axes.titlesize'] = 8

sns.set_style('white')

# ================== 原有色系的浅色版本（保留色相，降低饱和度） ==================
COLORS = {
    'chine':  '#5DADE2',   # 浅蓝
    'coccyx': '#EC7063',   # 浅砖红
    'cocc':   '#52BE80',   # 浅绿
    'true':   '#7F8C8D',   # 浅灰
    'ideal':  '#F1948A',   # 浅红
}


# ================== 通用坐标轴样式（去刻度、内置虚线网格） ==================
def style_axis_inner_grid(ax, grid_axis='both'):
    ax.tick_params(axis='both', which='both',
                   direction='in', top=True, right=True,
                   length=0, width=0.8)
    for s in ax.spines.values():
        s.set_linewidth(0.8)
        s.set_color('black')
    ax.grid(True, axis=grid_axis, alpha=0.40, linestyle='--',
            linewidth=0.5, color='#AAAAAA', zorder=0)


# ================== 1. 读取数据 ==================
file_path = '体长体重与骨骼测量数据-预测对比.xlsx'

if not os.path.exists(file_path):
    raise FileNotFoundError(f"找不到文件: {file_path}，请确认文件路径正确")

df_raw = pd.read_excel(file_path, sheet_name=0, header=None)
data_rows = df_raw.iloc[2:].copy()
data_rows.reset_index(drop=True, inplace=True)

col_names = ['sample_name', 'chine_pred', 'chine_true',
             'coccyx_pred', 'coccyx_true', 'cocc_pred', 'cocc_true']
data_rows.columns = col_names

for col in col_names[1:]:
    data_rows[col] = pd.to_numeric(data_rows[col], errors='coerce')

# ================== 2. 提取有效数据 ==================
metrics = ['chine', 'coccyx', 'cocc']
data_dict = {}

for m in metrics:
    pred_col = f'{m}_pred'
    true_col = f'{m}_true'
    sub_df = data_rows[[pred_col, true_col]].dropna()
    if len(sub_df) > 0:
        data_dict[m] = {
            'pred': sub_df[pred_col].values,
            'true': sub_df[true_col].values,
            'sample_names': data_rows.loc[sub_df.index, 'sample_name'].values
        }
    else:
        data_dict[m] = {'pred': np.array([]), 'true': np.array([]),
                        'sample_names': np.array([])}


# ================== 3. 辅助函数 ==================
def compute_metrics(true, pred):
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    r2 = r2_score(true, pred)
    mape = np.mean(np.abs((true - pred) / true)) * 100 if np.all(true != 0) else np.nan
    return mae, rmse, r2, mape


def add_stats_text(ax, true, pred, color, x=0.04, y=0.96):
    mae, rmse, r2, mape = compute_metrics(true, pred)
    textstr = f"$R^2$ = {r2:.3f}\nMAE = {mae:.2f}\nRMSE = {rmse:.2f}"
    if not np.isnan(mape):
        textstr += f"\nMAPE = {mape:.1f}%"
    props = dict(boxstyle='round,pad=0.30', facecolor='white',
                 edgecolor=color, alpha=0.9, linewidth=0.7)
    ax.text(x, y, textstr, transform=ax.transAxes, fontsize=7,
            verticalalignment='top', bbox=props, color='black')


def add_identity_line(ax, true, pred, color='gray'):
    min_val = min(true.min(), pred.min())
    max_val = max(true.max(), pred.max())
    pad = (max_val - min_val) * 0.05
    if pad == 0:
        pad = 1.0
    lim_low, lim_high = min_val - pad, max_val + pad

    ax.fill_between([lim_low, lim_high],
                    [lim_low * 0.9, lim_high * 0.9],
                    [lim_low * 1.1, lim_high * 1.1],
                    alpha=0.12, color=color,
                    label='±10% error band', linewidth=0)

    ax.plot([lim_low, lim_high], [lim_low, lim_high],
            color=COLORS['ideal'], linewidth=1.2, linestyle='--',
            label='Ideal 1:1 line', zorder=3)

    return lim_low, lim_high


# ================== 4. 绘图 ==================
fig = plt.figure(figsize=(7.2, 6.0))
gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.42,
              top=0.94, bottom=0.10, left=0.09, right=0.98)

labels_top = ['A', 'B', 'C']
labels_bot = ['D', 'E', 'F']
metrics_ch = {'chine': 'Chine', 'coccyx': 'Coccyx', 'cocc': 'Cocc'}

bar_legend_loc = {'chine': 'upper left', 'coccyx': 'upper left', 'cocc': 'upper left'}

# === 关键改动：x 轴刻度的显示间隔 ===
TICK_INTERVAL = 3   # 每隔 3 个显示一个标签（可改成 2 或 5）

for idx, m in enumerate(metrics):
    vals = data_dict[m]
    if len(vals['true']) == 0:
        continue
    pred = vals['pred']
    true = vals['true']
    names = vals['sample_names']
    color = COLORS[m]

    # ---- 第一行：散点图 ----
    ax1 = fig.add_subplot(gs[0, idx])
    ax1.scatter(true, pred, s=36, c=color, alpha=0.85,
                edgecolors='white', linewidth=0.7, zorder=4)

    lim_low, lim_high = add_identity_line(ax1, true, pred, color=color)
    add_stats_text(ax1, true, pred, color)

    ax1.set_xlim(lim_low, lim_high)
    ax1.set_ylim(lim_low, lim_high)
    ax1.set_aspect('equal', adjustable='box')

    tick_step = max(1, int((lim_high - lim_low) / 5))
    ticks = np.arange(np.floor(lim_low), np.ceil(lim_high) + 1, tick_step)
    ax1.set_xticks(ticks)
    ax1.set_yticks(ticks)

    ax1.set_xlabel('True value')
    ax1.set_ylabel('Predicted value')
    ax1.set_title(f'{metrics_ch[m]}', fontsize=8, fontweight='normal',
                  color='black', pad=6)

    style_axis_inner_grid(ax1, grid_axis='both')

    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
               ncol=2, frameon=False, fontsize=6,
               handlelength=1.5, handletextpad=0.3,
               columnspacing=1.0, labelspacing=0.2)

    ax1.text(-0.22, 1.08, labels_top[idx], transform=ax1.transAxes,
             fontsize=14, fontweight='normal', va='bottom', ha='left',
             color='black')

    # ---- 第二行：分组柱状图 ----
    ax2 = fig.add_subplot(gs[1, idx])
    display_n = min(15, len(names))
    indices = np.arange(display_n)
    width = 0.38

    ax2.bar(indices - width / 2, pred[:display_n], width,
            label='Predicted', color=color, alpha=0.9,
            edgecolor='white', linewidth=0.5, zorder=3)
    ax2.bar(indices + width / 2, true[:display_n], width,
            label='True', color=COLORS['true'], alpha=0.9,
            edgecolor='white', linewidth=0.5, zorder=3)

    data_max = max(pred[:display_n].max(), true[:display_n].max())
    ax2.set_ylim(0, data_max * 1.25)

    ax2.set_xlabel('Sample index')
    ax2.set_ylabel('Vertebra count')
    ax2.set_title(f'{metrics_ch[m]}: sample-wise comparison',
                  fontsize=8, fontweight='normal', color='black', pad=6)

    # === 关键改动：x 轴用整数索引，间隔若干显示，不旋转 ===
    ax2.set_xticks(indices)
    tick_labels = [str(i) if (i % TICK_INTERVAL == 0) else ''
                   for i in range(display_n)]
    ax2.set_xticklabels(tick_labels, rotation=0, ha='center', fontsize=7)

    ax2.legend(loc=bar_legend_loc[m], frameon=False, fontsize=6,
               handlelength=1.2, handletextpad=0.4, labelspacing=0.25)

    style_axis_inner_grid(ax2, grid_axis='y')

    ax2.text(-0.22, 1.08, labels_bot[idx], transform=ax2.transAxes,
             fontsize=14, fontweight='normal', va='bottom', ha='left',
             color='black')

# ================== 5. 保存（矢量图） ==================
output_filename = 'vertebrae_compare_optimized.pdf'
plt.savefig(output_filename, bbox_inches='tight',
            facecolor='white', edgecolor='none')
abs_path = os.path.abspath(output_filename)
print(f"✅ 矢量图（PDF）已保存至: {abs_path}")

plt.show()

# ================== 6. 输出统计信息 ==================
print("\n" + "=" * 60)
print("Summary Statistics")
print("=" * 60)
for m in metrics:
    vals = data_dict[m]
    if len(vals['true']) > 0:
        mae, rmse, r2, mape = compute_metrics(vals['true'], vals['pred'])
        print(f"\n{metrics_ch[m]}:")
        print(f"  Samples : {len(vals['true'])}")
        print(f"  R²      : {r2:.4f}")
        print(f"  MAE     : {mae:.4f}")
        print(f"  RMSE    : {rmse:.4f}")
        if not np.isnan(mape):
            print(f"  MAPE    : {mape:.2f}%")