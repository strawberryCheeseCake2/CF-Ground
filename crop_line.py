import numpy as np
from PIL import Image

'''
stripe_h (기본값: 3 px)
→ 훑는 가로 스트립 두께를 늘리면, 한 행에서 더 넓은 영역의 변화량을 평균내므로 노이즈는 줄어들고 에너지 값이 부드러워집니다. 다만, 너무 크게 하면 국부적인 변화가 희석되어 컷 위치가 덜 정확해질 수 있습니다.

margin_ratio (기본값: 0.2)
→ 상하 금지 영역의 비율을 높이면, 이미지 상하의 일정 부분을 컷 후보에서 제외하게 되어 컷은 더 중앙에 위치하게 됩니다. 너무 크게 하면 유효 영역이 줄어들어 부적절한 컷 위치를 선택할 수 있습니다.

min_side (기본값: 28 px)
→ 최소 조각 높이를 높이면, 너무 작은(얇은) 조각은 생기지 않도록 제한합니다. 값이 크면 컷 위치가 중앙에 몰릴 수 있고, 조각 분할이 덜 세밀해질 수 있습니다.

fusion (좋은값: "sum")
→ "sum"과 "prod"는 합성 방식의 종류이므로 값 자체를 키우는 개념은 아니지만, 선택에 따라 컷 결정 로직이 달라집니다. "prod"는 에너지와 중앙 근접도를 곱해 전체 점수가 낮은 곳을 컷 후보로, "sum"은 두 값을 가중 평균합니다.

alpha (좋은값: 0.9)
→ fusion이 "sum"일 때만 사용됩니다. alpha를 높이면 에너지 값의 영향력이 커지고, 낮추면 중앙 근접도(거리)의 영향력이 커집니다. 즉, 에너지와 중앙 근접도 중 어느 쪽에 더 민감하게 반응할지 조정할 수 있습니다.

smooth_radius (기본값: 5)
→ 1D 스무딩 커널의 크기를 키우면, 행별 에너지 값이 더 많이 평균화되어 노이즈는 줄지만 너무 부드러워져 지역적인 변화가 사라질 수 있습니다.

sample_stride (기본값: 1)
→ x축 샘플링 보폭을 늘리면 계산 속도는 빨라지지만, 세부적인 변화 포착 능력이 떨어져 컷 위치의 정밀도가 낮아질 수 있습니다. 반대로 낮추면 정밀도는 향상되지만 계산 비용이 증가합니다.
'''

def find_safe_horizontal_cut(
    img: Image.Image,
    stripe_h: int = 50,              # 훑는 가로 스트립 두께(px)
    margin_ratio: float = 0.2,     # 상하 금지 영역 비율
    min_side: int = 28,             # 최소 조각 높이(px)
    fusion: str = "sum",           # "sum" 또는 "prod"
    alpha: float = 0.9,             # fusion="sum"일 때 가중치
    smooth_radius: int = 5,         # 행 점수 1D 스무딩 크기
    sample_stride: int = 20          # x축 샘플링 보폭(>1이면 속도 ↑, 정밀도 ↓)
):
    """
    가로 스트립 색변화(∂/∂x) + 중앙 근접도를 융합해 수평 컷 y좌표를 찾는 함수.
    반환값은 컷 기준 y 인덱스(int). 컷은 [0..y_cut) | [y_cut..H) 로 분할.
    """
    if img.mode != "RGB":
        img = img.convert("RGB")

    arr = np.asarray(img, dtype=np.float32)  # HxWx3
    H, W, _ = arr.shape

    # 최소 높이 보장 + 상하 여백 금지 영역
    top_guard    = max(int(H * margin_ratio), min_side)
    bottom_guard = max(int(H * margin_ratio), min_side)

    # x축 방향 미분(절대값) → 행별 총 변화량
    # sample_stride 적용으로 속도-정밀도 트레이드오프
    diffx = np.abs(np.diff(arr[:, ::sample_stride, :], axis=1))   # H x W' x 3
    row_energy = diffx.mean(axis=2).sum(axis=1)                   # 길이 H

    # 스트립 높이 평균(세로 방향 박스 필터)
    if stripe_h > 1:
        k = np.ones(stripe_h, dtype=np.float32) / float(stripe_h)
        row_energy = np.convolve(row_energy, k, mode="same")

    # 중앙 근접도
    ys = np.arange(H, dtype=np.float32)
    center = (H - 1) / 2.0
    center_dist = np.abs(ys - center) / max(center, 1.0)  # [0,1]

    # 에너지 정규화(로버스트, p95 스케일)
    p95 = np.percentile(row_energy, 95) if H >= 10 else (row_energy.max() + 1e-6)
    e_norm = np.clip(row_energy / (p95 + 1e-6), 0.0, 1.0)

    # 1D 스무딩으로 톱니 제거
    if smooth_radius > 1:
        k2 = np.ones(smooth_radius, dtype=np.float32) / float(smooth_radius)
        e_norm = np.convolve(e_norm, k2, mode="same")

    # 점수 융합
    if fusion == "sum":
        # alpha * 에너지 + (1-alpha) * 중앙근접
        score = alpha * e_norm + (1.0 - alpha) * center_dist
    else:
        # 곱 융합 → 둘 다 낮아야 최소
        score = (1e-6 + e_norm) * (1e-6 + center_dist)

    # 금지 영역 마스킹
    score[:top_guard] = np.inf
    score[H - bottom_guard:] = np.inf

    # 최적 y 선택 및 최소 높이 보정
    y_cut = int(np.argmin(score))
    y_cut = max(y_cut, min_side)
    y_cut = min(y_cut, H - min_side)

    return y_cut


def force_horizontal_split(
    img: Image.Image,
    **kwargs
):
    """
    find_safe_horizontal_cut으로 y_cut 찾은 뒤 상/하 bbox와 이미지 크롭 반환.
    반환: (top_bbox, bottom_bbox, top_img, bottom_img)
    bbox 형식: [L, T, R, B]
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    W, H = img.size

    y_cut = find_safe_horizontal_cut(img, **kwargs)

    top_bbox    = [0, 0, W, y_cut]
    bottom_bbox = [0, y_cut, W, H]

    top_img    = img.crop(tuple(top_bbox))
    bottom_img = img.crop(tuple(bottom_bbox))

    return top_bbox, bottom_bbox, top_img, bottom_img
