import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# CSV 파일 읽기
df = pd.read_csv('../_results/results_all.csv')

# final_0919 데이터만 필터링
df_filtered = df[df['method'] == 'final_0918'].copy()

print("Filtered data:")
print(df_filtered[['method', 'resize_ratio', 'region_threshold', 'bbox_padding', 
                  'stage1_accuracy', 'stage2_accuracy', 'stage3_accuracy', 
                  'avg_stage1_tflops', 'avg_total_tflops']])

# resize_ratio별로 정렬
df_filtered = df_filtered.sort_values('resize_ratio')

# 그래프 설정
plt.figure(figsize=(15, 10))

# resize_ratio별로 고유한 색상 설정 (8개 resize ratio를 위해 충분한 색상)
resize_ratios = df_filtered['resize_ratio'].unique()
colors = plt.cm.Set1(np.linspace(0, 1, len(resize_ratios)))

# 하나의 그래프에 모든 Stage 표시
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# resize_ratio별로 그래프 그리기 (같은 resize끼리 이어서)
for i, ratio in enumerate(resize_ratios):
    data = df_filtered[df_filtered['resize_ratio'] == ratio]
    
    # 각 resize_ratio별로 Stage 1→2→3 순서로 점들을 연결
    x_values = [
        data['avg_stage1_tflops'].iloc[0],  # Stage 1 TFLOPs
        data['avg_total_tflops'].iloc[0],   # Stage 2 TFLOPs (total)
        data['avg_total_tflops'].iloc[0]    # Stage 3 TFLOPs (total)
    ]
    y_values = [
        data['stage1_accuracy'].iloc[0],    # Stage 1 Accuracy
        data['stage2_accuracy'].iloc[0],    # Stage 2 Accuracy  
        data['stage3_accuracy'].iloc[0]     # Stage 3 Accuracy
    ]
    
    # 선으로 연결하여 그리기 (모든 마커를 동그라미로 통일)
    ax.plot(x_values, y_values, 
            color=colors[i], marker='o', 
            linewidth=2, markersize=8,
            label=f'resize_ratio={ratio:.1f}')

ax.set_xlabel('TFLOPs', fontsize=14)
ax.set_ylabel('Accuracy (%)', fontsize=14)
ax.set_title('TFLOPs vs Accuracy by Stage and Resize Ratio', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig('../_results/combined_tflops_vs_accuracy.png', dpi=300, bbox_inches='tight')
plt.show()

# 데이터 요약 출력
print("\n=== Data Summary ===")
print(f"Total filtered records: {len(df_filtered)}")
print(f"Resize ratios: {sorted(resize_ratios)}")
print("\nDetailed data:")
for _, row in df_filtered.iterrows():
    print(f"Resize Ratio: {row['resize_ratio']:.1f} | "
          f"Stage1 TFLOPs: {row['avg_stage1_tflops']:.2f} | "
          f"Total TFLOPs: {row['avg_total_tflops']:.2f} | "
          f"Stage1 Acc: {row['stage1_accuracy']:.2f}% | "
          f"Stage2 Acc: {row['stage2_accuracy']:.2f}% | "
          f"Stage3 Acc: {row['stage3_accuracy']:.2f}%")