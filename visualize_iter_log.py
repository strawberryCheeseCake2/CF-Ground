import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# 경로 설정
# ------------------------------
dir = './attn_output/' + "0824_hoon"
csv_file = dir + '/iter_log.csv'
output_dir = dir + '/vis'
os.makedirs(output_dir, exist_ok=True)

# ------------------------------
# 데이터 읽기
# ------------------------------
data = pd.read_csv(csv_file)

# 숫자형 열 자동 변환
def to_numeric_safe(s):
    return pd.to_numeric(s, errors='coerce')

# 시각화 대상 열
columns_to_visualize = [
    'crop_time', 'num_crop', 'num_selected_crop', 's1_time',
    's1_flops_gflops', 's2_time', 's2_flops_gflops',
    'total_time', 'total_flops_gflops'
]

# 존재하는 열만 필터링
columns_to_visualize = [c for c in columns_to_visualize if c in data.columns]

# 숫자형으로 강제 캐스팅
for c in columns_to_visualize:
    data[c] = to_numeric_safe(data[c])

# 이모지 → bool 변환
def emoji_to_bool(series):
    mapping_true = {'☑️', '✅', 'true', 'True', '1', 1}
    mapping_false = {'🫥', '❌', 'false', 'False', '0', 0}
    out = []
    for v in series.astype(str):
        if v in mapping_true:
            out.append(True)
        elif v in mapping_false:
            out.append(False)
        else:
            # 숫자/기타 케이스
            if v.strip().lower() in ('true','1'):
                out.append(True)
            elif v.strip().lower() in ('false','0'):
                out.append(False)
            else:
                out.append(np.nan)
    return pd.Series(out, index=series.index, dtype='float').astype('float')

if 'early_exit' in data.columns:
    data['early_exit_flag'] = emoji_to_bool(data['early_exit']).fillna(0).astype(int)
else:
    data['early_exit_flag'] = 0

if 's1_hit' in data.columns:
    data['s1_hit_flag'] = emoji_to_bool(data['s1_hit']).fillna(0).astype(int)
else:
    data['s1_hit_flag'] = 0

if 's2_hit' in data.columns:
    data['s2_hit_flag'] = emoji_to_bool(data['s2_hit']).fillna(0).astype(int)
else:
    data['s2_hit_flag'] = 0

# ------------------------------
# 유틸: 히스토그램 저장
# ------------------------------
def auto_bins(x):
    x = x.dropna().values
    if len(x) < 2:
        return 10
    # Freedman–Diaconis
    q75, q25 = np.percentile(x, [75 ,25])
    iqr = max(q75 - q25, 1e-9)
    bin_width = 2 * iqr * (len(x) ** (-1/3))
    if bin_width <= 0:
        return 10
    bins = int(np.clip(np.ptp(x) / bin_width, 5, 60))
    return bins

def plot_hist(series, title, xlabel, path):
    x = series.dropna()
    if len(x) == 0:
        return
    bins = auto_bins(x)
    plt.figure(figsize=(6,4), dpi=150)
    plt.hist(x, bins=bins, edgecolor='black', alpha=0.75)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

# ------------------------------
# 1) 단변량 분포: 히스토그램
# ------------------------------
for col in columns_to_visualize:
    plot_hist(data[col], f"Distribution of {col}", col, os.path.join(output_dir, f"{col}__hist.png"))

# ------------------------------
# 2) 관계 플롯: 산점도
# ------------------------------
def scatter_xy(df, x, y, color_by, title, path):
    if x not in df.columns or y not in df.columns:
        return
    dx = df[[x, y, color_by]].dropna()
    if len(dx) == 0:
        return

    plt.figure(figsize=(6,4), dpi=150)
    # 색상: 0/1 기준
    colors = np.where(dx[color_by] > 0, 'tab:orange', 'tab:blue')
    plt.scatter(dx[x], dx[y], c=colors, s=18, alpha=0.7, linewidths=0.2, edgecolors='k')
    # 범례
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker='o', color='w', label=f'{color_by}=0', markerfacecolor='tab:blue', markersize=6, markeredgecolor='k'),
        Line2D([0], [0], marker='o', color='w', label=f'{color_by}=1', markerfacecolor='tab:orange', markersize=6, markeredgecolor='k'),
    ]
    plt.legend(handles=legend_elems, loc='best', frameon=True)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

scatter_xy(data, 'total_flops_gflops', 'total_time', 'early_exit_flag',
           'FLOPs vs Total Time (early_exit)', os.path.join(output_dir, 'scatter_totalflops_vs_totaltime_by_earlyexit.png'))

scatter_xy(data, 's1_flops_gflops', 's1_time', 's1_hit_flag',
           'S1 FLOPs vs S1 Time (s1_hit)', os.path.join(output_dir, 'scatter_s1flops_vs_s1time_by_s1hit.png'))

scatter_xy(data, 's2_flops_gflops', 's2_time', 's2_hit_flag',
           'S2 FLOPs vs S2 Time (s2_hit)', os.path.join(output_dir, 'scatter_s2flops_vs_s2time_by_s2hit.png'))

# ------------------------------
# 3) 박스플롯: 카테고리별 시간
# ------------------------------
def box_by_cat(df, xcat, yval, title, path):
    if xcat not in df.columns or yval not in df.columns:
        return
    d = df[[xcat, yval]].dropna()
    if len(d) == 0:
        return
    plt.figure(figsize=(6,4), dpi=150)
    # matplotlib boxplot은 x가 수치인 경우 그룹핑 필요
    groups = [g[yval].values for _, g in d.groupby(xcat)]
    labels = [str(k) for k in sorted(d[xcat].dropna().unique())]
    plt.boxplot(groups, labels=labels, showfliers=False)
    plt.title(title)
    plt.xlabel(xcat)
    plt.ylabel(yval)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

box_by_cat(data, 'num_crop', 'total_time', 'Total Time by num_crop',
           os.path.join(output_dir, 'box_total_time_by_num_crop.png'))

box_by_cat(data, 'num_selected_crop', 's1_time', 'S1 Time by num_selected_crop',
           os.path.join(output_dir, 'box_s1_time_by_num_selected_crop.png'))

# ------------------------------
# 4) ECDF: early_exit 별 total_time
# ------------------------------
def ecdf_plot(df, value_col, group_flag_col, title, path):
    if value_col not in df.columns or group_flag_col not in df.columns:
        return
    d = df[[value_col, group_flag_col]].dropna()
    if len(d) == 0:
        return

    plt.figure(figsize=(6,4), dpi=150)
    for flag, sub in d.groupby(group_flag_col):
        x = np.sort(sub[value_col].values)
        y = np.arange(1, len(x)+1) / len(x)
        label = f'{group_flag_col}={int(flag)}'
        plt.step(x, y, where='post', linewidth=1.8, label=label)
    plt.title(title)
    plt.xlabel(value_col)
    plt.ylabel("ECDF")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

ecdf_plot(data, 'total_time', 'early_exit_flag',
          'ECDF of total_time by early_exit', os.path.join(output_dir, 'ecdf_total_time_by_early_exit.png'))

# ------------------------------
# 5) 상관 행렬 히트맵
# ------------------------------
corr_cols = [c for c in columns_to_visualize if data[c].dtype.kind in 'fi']
corr = data[corr_cols].corr(method='pearson')
plt.figure(figsize=(0.8*len(corr_cols)+2, 0.8*len(corr_cols)+2), dpi=150)
im = plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.xticks(range(len(corr_cols)), corr_cols, rotation=45, ha='right')
plt.yticks(range(len(corr_cols)), corr_cols)
plt.title("Correlation Matrix (Pearson)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'corr_matrix.png'))
plt.close()

# ------------------------------
# 6) 요약 통계 + 상위/하위 케이스 저장
# ------------------------------
summary = data[columns_to_visualize].describe().T
summary.to_csv(os.path.join(output_dir, 'summary_stats.csv'))

# 가장 느린 케이스 / 가장 FLOPs 큰 케이스
slow_k = min(20, len(data))
flop_k = min(20, len(data))
data.sort_values('total_time', ascending=False).head(slow_k).to_csv(os.path.join(output_dir, 'top_slowest.csv'), index=False)
data.sort_values('total_flops_gflops', ascending=False).head(flop_k).to_csv(os.path.join(output_dir, 'top_flops.csv'), index=False)

print(f"[Done] Visualized at {output_dir} and generated summary_stats.csv / top_slowest.csv / top_flops.csv")
