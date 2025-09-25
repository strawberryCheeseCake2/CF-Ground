import pandas as pd

# CSV 파일을 읽어옵니다.
try:
    df = pd.read_csv('iter_log_web.csv')

    # s2_hit이 '❌'이고 s3_hit이 '✅'인 행을 필터링합니다.
    filtered_df = df[(df['s2_hit'] == '❌') & (df['s3_hit'] == '✅')]

    # 필터링된 행의 'idx'를 출력합니다.
    if not filtered_df.empty:
        print("s2_hit이 '❌'이고 s3_hit이 '✅'인 행의 idx:")
        for idx in filtered_df['idx']:
            print(idx)
    else:
        print("해당 조건을 만족하는 행이 없습니다.")

except FileNotFoundError:
    print("파일을 찾을 수 없습니다. 'iter_log_web.csv' 파일이 현재 디렉토리에 있는지 확인해주세요.")