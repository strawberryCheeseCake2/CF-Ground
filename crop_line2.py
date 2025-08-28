import numpy as np
from PIL import Image

'''
stripe_h (기본값: 3 px)
→ 훑는 가로 스트립 두께를 늘리면, 한 행에서 더 넓은 영역의 변화량을 평균내므로 노이즈는 줄어들고 에너지 값이 부드러워집니다. 다만, 너무 크게 하면 국부적인 변화가 희석되어 컷 위치가 덜 정확해질 수 있습니다.

smooth_radius (기본값: 5)
→ 1D 스무딩 커널의 크기를 키우면, 행별 에너지 값이 더 많이 평균화되어 노이즈는 줄지만 너무 부드러워져 지역적인 변화가 사라질 수 있습니다.

sample_stride (기본값: 1)
→ x축 샘플링 보폭을 늘리면 계산 속도는 빨라지지만, 세부적인 변화 포착 능력이 떨어져 컷 위치의 정밀도가 낮아질 수 있습니다. 반대로 낮추면 정밀도는 향상되지만 계산 비용이 증가합니다.
'''

def find_multi_cuts(
    img: Image.Image,
    stripe_h: int = 50,              # 훑는 가로 스트립 두께(px)
    smooth_radius: int = 5,          # 행 점수 1D 스무딩 크기
    sample_stride: int = 20,         # x축 샘플링 보폭(>1이면 속도 ↑, 정밀도 ↓)
    h_cuts: int = 1,                 # 수평으로 자르는 횟수 (라인 개수)
    v_cuts: int = 1,                 # 수직으로 자르는 횟수 (라인 개수)
    h_tolerance: float = 0.07,       # 수평 자를 때 허용 범위 비율 (±7%)
    v_tolerance: float = 0.07,       # 수직 자를 때 허용 범위 비율 (±7%)
    allow_vertical_cut: bool = True  # 수직 컷 허용 여부
):
    """
    멀티 컷팅: 수평/수직으로 여러 번 자르는 함수
    반환값: (horizontal_cuts, vertical_cuts)
    horizontal_cuts: 수평 컷 y좌표 리스트
    vertical_cuts: 수직 컷 x좌표 리스트
    """
    if img.mode != "RGB":
        img = img.convert("RGB")

    arr = np.asarray(img, dtype=np.float32)  # HxWx3
    H, W, _ = arr.shape
    
    horizontal_cuts = []
    vertical_cuts = []
    
    # 수평 컷 찾기 (y 좌표들)
    if h_cuts > 0:
        # h_cuts개의 라인으로 (h_cuts+1)개 구간 생성
        # 예: h_cuts=3이면 25%, 50%, 75% 지점에 라인
        for i in range(1, h_cuts + 1):
            target_ratio = i / (h_cuts + 1)  # 1/4, 2/4, 3/4
            target_y = int(H * target_ratio)
            
            # 허용 범위 계산
            tolerance_px = int(H * h_tolerance)
            y_min = max(0, target_y - tolerance_px)
            y_max = min(H-1, target_y + tolerance_px)
            
            # 해당 범위에서 x축 방향 RGB 차이가 가장 작은 지점 찾기
            diffx = np.abs(np.diff(arr[y_min:y_max+1, ::sample_stride, :], axis=1))
            row_energy = diffx.mean(axis=2).sum(axis=1)
            
            # 스트립 높이 평균
            if stripe_h > 1 and len(row_energy) >= stripe_h:
                k = np.ones(min(stripe_h, len(row_energy)), dtype=np.float32)
                k = k / k.sum()
                row_energy = np.convolve(row_energy, k, mode="same")
            
            # 스무딩
            if smooth_radius > 1 and len(row_energy) >= smooth_radius:
                k2 = np.ones(min(smooth_radius, len(row_energy)), dtype=np.float32)
                k2 = k2 / k2.sum()
                row_energy = np.convolve(row_energy, k2, mode="same")
            
            # 가장 변화가 적은 지점 선택
            best_idx = np.argmin(row_energy)
            y_cut = y_min + best_idx
            horizontal_cuts.append(y_cut)
    
    # 수직 컷 찾기 (x 좌표들)
    if v_cuts > 0 and allow_vertical_cut:
        for i in range(1, v_cuts + 1):
            target_ratio = i / (v_cuts + 1)
            target_x = int(W * target_ratio)
            
            # 허용 범위 계산
            tolerance_px = int(W * v_tolerance)
            x_min = max(0, target_x - tolerance_px)
            x_max = min(W-1, target_x + tolerance_px)
            
            # 해당 범위에서 y축 방향 RGB 차이가 가장 작은 지점 찾기
            diffy = np.abs(np.diff(arr[::sample_stride, x_min:x_max+1, :], axis=0))
            col_energy = diffy.mean(axis=2).sum(axis=0)
            
            # 스트립 높이 평균 (수직의 경우 폭)
            if stripe_h > 1 and len(col_energy) >= stripe_h:
                k = np.ones(min(stripe_h, len(col_energy)), dtype=np.float32)
                k = k / k.sum()
                col_energy = np.convolve(col_energy, k, mode="same")
            
            # 스무딩
            if smooth_radius > 1 and len(col_energy) >= smooth_radius:
                k2 = np.ones(min(smooth_radius, len(col_energy)), dtype=np.float32)
                k2 = k2 / k2.sum()
                col_energy = np.convolve(col_energy, k2, mode="same")
            
            # 가장 변화가 적은 지점 선택
            best_idx = np.argmin(col_energy)
            x_cut = x_min + best_idx
            vertical_cuts.append(x_cut)
    
    return horizontal_cuts, vertical_cuts


def generate_grid_crops(
    img: Image.Image,
    **kwargs
):
    """
    find_multi_cuts로 수평/수직 컷을 찾은 뒤 격자 형태로 이미지를 분할.
    반환: (bboxes, crop_imgs)
    bboxes: 각 격자 영역의 bbox 리스트 [[L, T, R, B], ...]
    crop_imgs: 각 격자 영역의 크롭된 이미지 리스트
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    W, H = img.size

    horizontal_cuts, vertical_cuts = find_multi_cuts(img, **kwargs)
    
    # y 좌표들 (0, horizontal_cuts..., H)
    y_coords = [0] + sorted(horizontal_cuts) + [H]
    
    # x 좌표들 (0, vertical_cuts..., W)  
    x_coords = [0] + sorted(vertical_cuts) + [W]
    
    bboxes = []
    crop_imgs = []
    
    # 격자 형태로 bbox 생성
    for i in range(len(y_coords) - 1):
        for j in range(len(x_coords) - 1):
            L = x_coords[j]
            T = y_coords[i]
            R = x_coords[j + 1]
            B = y_coords[i + 1]
            
            bbox = [L, T, R, B]
            bboxes.append(bbox)
            
            crop_img = img.crop(tuple(bbox))
            crop_imgs.append(crop_img)
    
    return bboxes, crop_imgs
