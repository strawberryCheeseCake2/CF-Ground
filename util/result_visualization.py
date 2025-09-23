import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# CSV 파일 경로 설정
csv_path = '/home/ubuntu/hoon/CF-Ground/_results/results_all.csv'

def load_and_process_data(csv_path):
    """
    CSV 파일을 로드하고 필요한 열만 추출
    """
    df = pd.read_csv(csv_path)
    
    # 필요한 열만 선택
    columns_needed = ['method', 'resize_ratio', 'region_threshold', 'bbox_padding', 
                     'stage3_accuracy', 'avg_total_tflops']
    
    # 선택한 열이 모두 존재하는지 확인
    missing_cols = [col for col in columns_needed if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    
    df_filtered = df[columns_needed].copy()
    
    # NaN 값 제거
    df_filtered = df_filtered.dropna()

    # 제일 잘 나오는 값들만 남기기
    df_filtered = df_filtered[df_filtered['region_threshold'].isin([0, 0.12, 88, 99])]
    # df_filtered = df_filtered[df_filtered['bbox_padding'].isin([20,30,40,50])]
    df_filtered = df_filtered[df_filtered['bbox_padding'].isin([0])]

    df_filtered = df_filtered[(df_filtered['method'] == 'vanilla') | (df_filtered['method'] == 'final_0918')]

    # vanilla 제외
    # df_filtered = df_filtered[~(df_filtered['method'] == 'vanilla') & ~(df_filtered['method'] == 'v2') & ~(df_filtered['method'] == 'stage1')]
    # df_filtered = df_filtered[~(df_filtered['method'] == 'qwen25vl')]


    #! =========================================================================

    return df_filtered

def create_hyperparameter_visualization(csv_path, color_by='threshold', save_path=None):
    """
    하이퍼파라미터 최적화를 위한 시각화 생성
    
    Args:
        csv_path (str): CSV 파일 경로
        color_by (str): 색상 그룹화 방식 ('threshold' 또는 'padding')
        save_path (str, optional): 그래프 저장 경로
    """
    # 데이터 로드
    df = load_and_process_data(csv_path)
    
    # region_threshold와 bbox_padding 조합으로 그룹화
    df['hyperparameter_combo'] = df['region_threshold'].astype(str) + '_' + df['bbox_padding'].astype(str)
    
    # 그래프 설정
    plt.figure(figsize=(12, 8))
    
    # 색상 그룹화 방식에 따라 색상 지정
    if color_by == 'threshold':
        # region_threshold 값별로 색상 지정
        unique_values = sorted(df['region_threshold'].unique())
        color_column = 'region_threshold'
        title_suffix = "region_threshold"
    elif color_by == 'padding':
        # bbox_padding 값별로 색상 지정
        unique_values = sorted(df['bbox_padding'].unique())
        color_column = 'bbox_padding'
        title_suffix = "bbox_padding"
    else:
        raise ValueError("color_by must be 'threshold' or 'padding'")
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_values)))
    color_map = {value: color for value, color in zip(unique_values, colors)}
    
    # 각 하이퍼파라미터 조합별로 표시
    unique_combos = df['hyperparameter_combo'].unique()
    
    for i, combo in enumerate(unique_combos):
        combo_data = df[df['hyperparameter_combo'] == combo].copy()
        
        # resize_ratio 순으로 정렬 (선 연결을 위해)
        combo_data = combo_data.sort_values('resize_ratio')
        
        # region_threshold와 bbox_padding 값 추출
        region_thresh = combo_data['region_threshold'].iloc[0]
        bbox_pad = combo_data['bbox_padding'].iloc[0]
        
        # 라벨 생성
        label = f'region_thresh={region_thresh}, bbox_pad={bbox_pad}'
        
        # 색상 그룹화 방식에 따른 색상 선택
        if color_by == 'threshold':
            color = color_map[region_thresh]
        else:  # color_by == 'padding'
            color = color_map[bbox_pad]
        
        # 꺾은선 그래프 그리기
        plt.plot(combo_data['avg_total_tflops'], combo_data['stage3_accuracy'], 
                marker='o', color=color, label=label, linewidth=2, markersize=6)
        
        # 각 점에 resize_ratio 값 표시
        for _, row in combo_data.iterrows():
            plt.annotate(f'{row["resize_ratio"]:.2f}', 
                        (row['avg_total_tflops'], row['stage3_accuracy']),
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, alpha=0.7)
    
    # 그래프 꾸미기
    plt.xlabel('Average Total TFLOPs', fontsize=12, fontweight='bold')
    plt.ylabel('Stage3 Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title(f'Hyperparameter Optimization: TFLOPs vs Accuracy\n(Same color for same {title_suffix})', 
              fontsize=14, fontweight='bold')
    
    # y축 범위 설정 (데이터 범위에 맞춰 조정)
    y_min = df['stage3_accuracy'].min()
    y_max = df['stage3_accuracy'].max()
    y_margin = (y_max - y_min) * 0.05  # 5% 여백
    plt.ylim(y_min - y_margin, y_max + y_margin)
    
    # 범례 설정
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # 그리드 추가
    plt.grid(True, alpha=0.3)
    
    # 레이아웃 조정
    plt.tight_layout()
    
    # 그래프 저장
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"그래프가 {save_path}에 저장되었습니다.")
    
    # 그래프 표시
    plt.show()
    
    return df

def create_all_colors_visualization(csv_path, save_path=None):
    """
    모든 하이퍼파라미터 조합을 다른 색상으로 시각화
    
    Args:
        csv_path (str): CSV 파일 경로
        save_path (str, optional): 그래프 저장 경로
    """
    # 데이터 로드
    df = load_and_process_data(csv_path)
    
    # region_threshold와 bbox_padding 조합으로 그룹화
    df['hyperparameter_combo'] = df['region_threshold'].astype(str) + '_' + df['bbox_padding'].astype(str)
    
    # 그래프 설정
    plt.figure(figsize=(14, 10))
    
    # 각 하이퍼파라미터 조합별로 다른 색상 지정
    unique_combos = sorted(df['hyperparameter_combo'].unique())
    
    # 색상 팔레트 확장 (더 많은 색상 사용)
    if len(unique_combos) <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_combos)))
    elif len(unique_combos) <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_combos)))
    else:
        # 더 많은 조합이 있는 경우 여러 컬러맵 조합
        colors1 = plt.cm.tab10(np.linspace(0, 1, 10))
        colors2 = plt.cm.Set3(np.linspace(0, 1, len(unique_combos) - 10))
        colors = np.vstack((colors1, colors2))
    
    color_map = {combo: colors[i] for i, combo in enumerate(unique_combos)}
    
    for i, combo in enumerate(unique_combos):
        combo_data = df[df['hyperparameter_combo'] == combo].copy()
        
        # resize_ratio 순으로 정렬 (선 연결을 위해)
        combo_data = combo_data.sort_values('resize_ratio')
        
        # region_threshold와 bbox_padding 값 추출
        region_thresh = combo_data['region_threshold'].iloc[0]
        bbox_pad = combo_data['bbox_padding'].iloc[0]
        
        # 라벨 생성
        label = f'thresh={region_thresh}, pad={bbox_pad}'
        
        # 고유한 색상 선택
        color = color_map[combo]
        
        # 꺾은선 그래프 그리기
        plt.plot(combo_data['avg_total_tflops'], combo_data['stage3_accuracy'], 
                marker='o', color=color, label=label, linewidth=2, markersize=6)
        
        # 각 점에 resize_ratio 값 표시
        for _, row in combo_data.iterrows():
            plt.annotate(f'{row["resize_ratio"]:.2f}', 
                        (row['avg_total_tflops'], row['stage3_accuracy']),
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=7, alpha=0.7)
    
    # 그래프 꾸미기
    plt.xlabel('Average Total TFLOPs', fontsize=12, fontweight='bold')
    plt.ylabel('Stage3 Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title('Hyperparameter Optimization: TFLOPs vs Accuracy\n(Each combination has different color)', 
              fontsize=14, fontweight='bold')
    
    # y축 범위 설정 (데이터 범위에 맞춰 조정)
    y_min = df['stage3_accuracy'].min()
    y_max = df['stage3_accuracy'].max()
    y_margin = (y_max - y_min) * 0.05  # 5% 여백
    plt.ylim(y_min - y_margin, y_max + y_margin)
    
    # 범례 설정 (작은 폰트로 조정)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1)
    
    # 그리드 추가
    plt.grid(True, alpha=0.3)
    
    # 레이아웃 조정
    plt.tight_layout()
    
    # 그래프 저장
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"그래프가 {save_path}에 저장되었습니다.")
    
    # 그래프 표시
    plt.show()
    
    return df

def analyze_best_hyperparameters(df, efficiency_weight=0.5):
    """
    효율성을 고려한 최적 하이퍼파라미터 분석
    
    Args:
        df (DataFrame): 처리된 데이터프레임
        efficiency_weight (float): 효율성(낮은 TFLOPs) 가중치 (0-1)
    """
    print("=== 하이퍼파라미터 성능 분석 ===\n")
    
    # 정규화 (0-1 범위로)
    df_norm = df.copy()
    df_norm['accuracy_norm'] = (df['stage3_accuracy'] - df['stage3_accuracy'].min()) / (df['stage3_accuracy'].max() - df['stage3_accuracy'].min())
    df_norm['tflops_norm'] = (df['avg_total_tflops'] - df['avg_total_tflops'].min()) / (df['avg_total_tflops'].max() - df['avg_total_tflops'].min())
    
    # 효율성 점수 계산 (높은 정확도, 낮은 TFLOPs가 좋음)
    df_norm['efficiency_score'] = (1 - efficiency_weight) * df_norm['accuracy_norm'] + efficiency_weight * (1 - df_norm['tflops_norm'])
    
    # 상위 10개 결과 출력
    top_results = df_norm.nlargest(10, 'efficiency_score')
    
    print("Top 10 효율적인 하이퍼파라미터 조합:")
    print("-" * 80)
    for i, (_, row) in enumerate(top_results.iterrows(), 1):
        print(f"{i:2d}. region_threshold={row['region_threshold']:.2f}, bbox_padding={row['bbox_padding']:2.0f}, "
              f"resize_ratio={row['resize_ratio']:.2f}")
        print(f"    → Accuracy: {row['stage3_accuracy']:.2f}%, TFLOPs: {row['avg_total_tflops']:.2f}, "
              f"Score: {row['efficiency_score']:.3f}")
        print()
    
    # 하이퍼파라미터별 평균 성능
    print("\n=== 하이퍼파라미터별 평균 성능 ===")
    hyperparameter_analysis = df.groupby(['region_threshold', 'bbox_padding']).agg({
        'stage3_accuracy': ['mean', 'std'],
        'avg_total_tflops': ['mean', 'std'],
        'resize_ratio': 'count'
    }).round(2)
    
    hyperparameter_analysis.columns = ['accuracy_mean', 'accuracy_std', 'tflops_mean', 'tflops_std', 'count']
    hyperparameter_analysis = hyperparameter_analysis.sort_values('accuracy_mean', ascending=False)
    
    print(hyperparameter_analysis)
    
    return top_results

def main():
    """
    메인 실행 함수 - region_threshold 기준과 bbox_padding 기준 두 가지 시각화 모두 생성
    """
    
    try:
        print("데이터를 로드하고 두 가지 시각화를 생성합니다...")
        
        # 1. region_threshold 기준 시각화
        print("\n=== Region Threshold 기준 시각화 ===")
        save_path_threshold = '/home/ubuntu/hoon/CF-Ground/_results/hyperparameter_optimization_threshold.png'
        df = create_hyperparameter_visualization(csv_path, 'threshold', save_path_threshold)
        
        print(f"\n총 {len(df)}개의 데이터 포인트를 분석했습니다.")
        print(f"resize_ratio 범위: {df['resize_ratio'].min():.2f} - {df['resize_ratio'].max():.2f}")
        print(f"stage3_accuracy 범위: {df['stage3_accuracy'].min():.2f}% - {df['stage3_accuracy'].max():.2f}%")
        print(f"avg_total_tflops 범위: {df['avg_total_tflops'].min():.2f} - {df['avg_total_tflops'].max():.2f}")
        
        # 2. bbox_padding 기준 시각화
        print("\n=== Bbox Padding 기준 시각화 ===")
        save_path_padding = '/home/ubuntu/hoon/CF-Ground/_results/hyperparameter_optimization_padding.png'
        create_hyperparameter_visualization(csv_path, 'padding', save_path_padding)
        
        # 3. 모든 조합을 다른 색으로 하는 시각화
        print("\n=== 모든 조합별 다른 색상 시각화 ===")
        save_path_all = '/home/ubuntu/hoon/CF-Ground/_results/hyperparameter_optimization_all_colors.png'
        create_all_colors_visualization(csv_path, save_path_all)
        
        # 최적 하이퍼파라미터 분석 (한 번만)
        print("\n" + "="*60)
        analyze_best_hyperparameters(df)
        
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {csv_path}")
        print("파일 경로를 확인해주세요.")
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main()