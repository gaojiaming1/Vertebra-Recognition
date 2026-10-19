import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')

# ======================== 全局样式（ZR 格式） ========================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['axes.titleweight'] = 'normal'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

plt.rcParams['font.size'] = 8
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['axes.titlesize'] = 8
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 7
plt.rcParams['legend.fontsize'] = 7
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.15
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['legend.frameon'] = False

sns.set_style("white")

# ======================== 配置 ========================
# TEST_PATH = 'C:/Users/PS/Desktop/ZN黄鳝体长体重数据.xlsx'
TEST_PATH = '体重分组测试集.xlsx'
OUTPUT_DIR = './figures_academic'
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGETS = ['chine', 'coccyx', 'cocc']
GROUP_COL = '体重组'

# ======================== 原配色保留 ========================
MEASURE_COLORS = {
    'chine':  '#2c5f8a',
    'coccyx': '#b25b2c',
    'cocc':   '#2d6a4f',
}
GROUP_PALETTE = ['#3a5f8a', '#6a8caf', '#b27a5c', '#8a6e4b']


# ======================== 通用工具 ========================
def _style_axis(ax, grid_axis='both'):
    """不要刻度线；网格用虚线"""
    ax.tick_params(axis='both', which='both', direction='in',
                   top=True, right=True, length=0, width=0.8)
    for s in ax.spines.values():
        s.set_linewidth(0.8)
        s.set_color('black')
    ax.grid(True, axis=grid_axis, alpha=0.35, linestyle='--', linewidth=0.5)


def _panel_label(ax, letter, x=-0.20, y=1.06, fs=14):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=fs,
            fontweight='normal', va='bottom', ha='left', color='black')


def _headroom_for_legend(ax, ratio=1.55):
    """顶部预留更多空间，避免图例压住曲线"""
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * ratio)


# ======================== 读取并清洗数据 ========================
df = pd.read_excel(TEST_PATH, sheet_name='Sheet1')
required = ['体重(g)', '体长(cm)'] + TARGETS
mask = (df[required] != 0).all(axis=1)
df_clean = df[mask].copy()
print(f"清洗后保留 {len(df_clean)} 行")


# ======================== 图 7：纯 KDE 密度曲线 ========================
from matplotlib.ticker import MaxNLocator

fig7, axes = plt.subplots(1, 3, figsize=(8.5, 3.0))

for i, target in enumerate(TARGETS):
    ax = axes[i]
    data_sub = df_clean[target].dropna()
    color = MEASURE_COLORS[target]

    sns.kdeplot(data=data_sub, color=color, fill=True, alpha=0.30,
                linewidth=1.2, ax=ax)

    mean_val = data_sub.mean()
    std_val = data_sub.std()

    ax.axvline(mean_val, color='black', linestyle='--', linewidth=0.9,
               label=f'Mean = {mean_val:.1f}')
    ax.axvspan(mean_val - std_val, mean_val + std_val, alpha=0.08,
               color=color, label='±1 SD')

    ax.set_xlabel('Measurement (mm)')
    ax.set_ylabel('Density')
    ax.set_title(target.capitalize(), fontsize=8, fontweight='normal',
                 color='black', pad=5)

    # y 轴刻度精简到 4~5 个，避免过密
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

    _style_axis(ax, grid_axis='both')

    # 顶部只留一点点空隙（原来 1.55 太高）
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.20)

    # 图例放右上角，紧贴预留空白区，不压 KDE 曲线
    ax.legend(loc='upper right', fontsize=6, frameon=False,
              handlelength=1.2, handletextpad=0.3, labelspacing=0.2,
              borderaxespad=0.6)

    _panel_label(ax, 'ABC'[i], x=-0.20, y=1.06, fs=14)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/Fig7_KDE_Only.pdf', facecolor='white',
            edgecolor='none', bbox_inches='tight')
plt.show()
print(f"Saved: {OUTPUT_DIR}/Fig7_KDE_Only.pdf")

# ======================== 图 8：按体重组的分组分布 ========================
fig8, axes = plt.subplots(1, 3, figsize=(7.0, 3.4), sharey=False)

group_order = ['超轻体重', '轻体重', '中体重', '超重体重']

for i, target in enumerate(TARGETS):
    ax = axes[i]
    sub = df_clean[[GROUP_COL, target]].dropna()

    sns.violinplot(data=sub, x=GROUP_COL, y=target, order=group_order,
                   palette=GROUP_PALETTE, inner='box',
                   linewidth=0.6, saturation=1.0, ax=ax)
    sns.stripplot(data=sub, x=GROUP_COL, y=target, order=group_order,
                  color='#4a4a4a', alpha=0.30, size=1.6, jitter=0.15, ax=ax)

    ax.set_xlabel('Weight group')
    ax.set_ylabel('Measurement (mm)')
    ax.set_title(target.capitalize(), fontsize=8, fontweight='normal',
                 color='black', pad=5)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=6)
    _style_axis(ax, grid_axis='y')
    _panel_label(ax, 'ABC'[i], x=-0.20, y=1.06, fs=14)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/Fig8_Violin_by_Group.pdf', facecolor='white',
            edgecolor='none', bbox_inches='tight')
plt.show()
print(f"Saved: {OUTPUT_DIR}/Fig8_Violin_by_Group.pdf")


# ======================== 输出基本统计 ========================
print("\n===== Basic Statistics =====")
print(df_clean[TARGETS].describe())

print("\nAll distribution figures generated successfully.")