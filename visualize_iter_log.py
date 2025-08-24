import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 파일 경로
dir = './attn_output/' + "0824_hoon"
csv_file = dir + '/iter_log.csv'

# 데이터 읽기
data = pd.read_csv(csv_file)

# 시각화할 열
columns_to_visualize = [
    'crop_time', 'num_crop', 'num_selected_crop', 's1_time', 
    's1_flops_gflops', 's2_time', 's2_flops_gflops', 'total_time', 'total_flops_gflops'
]

# x축이 정수인지 확인하는 함수
def is_integer_series(series):
    return np.all(series % 1 == 0)

# 시각화 수정
for column in columns_to_visualize:
    plt.figure(figsize=(10, 6))
    
    if is_integer_series(data[column]):
        # 정수일 경우 막대그래프
        plt.bar(data[column], data.index, label=column, color='skyblue')
        plt.xlabel(column)
        plt.ylabel('Frequency')
    else:
        # 정수가 아닐 경우 꺾은선 그래프
        plt.plot(data[column], data.index, label=column, color='skyblue', marker='o')
        plt.xlabel(column)
        plt.ylabel('Frequency')

    plt.title(f'{column} Distribution')
    plt.legend()
    plt.grid(axis='both')
    plt.savefig(f'{dir}/{column}_plot.png')
    plt.close()

print("Visualization complete. Check the output directory for updated graphs.")
