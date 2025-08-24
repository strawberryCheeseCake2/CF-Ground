import pandas as pd
import matplotlib.pyplot as plt

# CSV 파일 경로
csv_file = './attn_output/0824_hoon_1/iter_log.csv'

# 데이터 읽기
data = pd.read_csv(csv_file)

# 시각화할 열
columns_to_visualize = [
    'crop_time', 'num_crop', 'num_selected_crop', 's1_time', 
    's1_flops_gflops', 's2_time', 's2_flops_gflops', 'total_time', 'total_flops_gflops'
]

# 꺾은선 그래프
for column in columns_to_visualize:
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data[column], marker='o', label=column)
    plt.title(f'{column} Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel(column)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./attn_output/0824_hoon_1/{column}_line_plot.png')
    plt.close()

# 막대그래프
for column in columns_to_visualize:
    plt.figure(figsize=(10, 6))
    plt.bar(data.index, data[column], label=column, color='skyblue')
    plt.title(f'{column} Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel(column)
    plt.legend()
    plt.grid(axis='y')
    plt.savefig(f'./attn_output/0824_hoon_1/{column}_bar_plot.png')
    plt.close()

print("Visualization complete. Check the output directory for graphs.")
