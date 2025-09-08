import numpy as np
from collections import deque
from typing import List, Dict, Tuple, Optional

def get_connected_region_bboxes_from_scores(
    image_result: Dict,
    threshold: Optional[float] = 0.5,
    min_patches: int = 3,
    connectivity: int = 8,
    percentile: Optional[float] = None,
) -> List[Dict]:
    """
    어텐션 스코어 맵에서 threshold를 적용해 마스크를 만들고,
    연결 성분(BFS)으로 묶어서 각 성분의 정규화 bbox와 점수를 반환한다.

    Args:
        image_result: prediction_results['per_image']의 단일 아이템
                      반드시 'attn_scores', 'n_width', 'n_height' 포함
        threshold: 0~1 사이면 max(attn_scores)*threshold로 임계값을 잡음.
                   None이면 percentile을 사용(기본 85p)하거나
                   자동 규칙(mean+std)으로 결정.
        min_patches: 너무 작은 성분 제거를 위한 최소 패치 수
        connectivity: 4 또는 8 연결
        percentile: None이 아니면 해당 분위수(예: 85.0)로 임계값 설정

    Returns:
        List[Dict]: 각 성분에 대해
            {
              "bbox": [l, t, r, b],  # 정규화(0~1)
              "patch_bbox": [x_min, y_min, x_max, y_max],  # 패치 좌표
              "size": 정점 수(패치 수),
              "score_sum": 성분 내 점수 합,
              "score_mean": 성분 내 점수 평균(0~1 정규화 아님),
              "score_norm": 성분 내 점수합 / 전체 max 합 기반 간단 정규화
            }
          형태의 리스트(점수가 높은 순으로 정렬)
    """
    # 1) 입력 파싱
    attn_scores_1d = np.array(image_result["attn_scores"][0], dtype=np.float32)
    n_w = int(image_result["n_width"])
    n_h = int(image_result["n_height"])

    if attn_scores_1d.size != n_w * n_h:
        raise ValueError(f"attn_scores size {attn_scores_1d.size} != n_w*n_h {n_w*n_h}")

    attn = attn_scores_1d.reshape(n_h, n_w)  # [y, x]

    # 2) 임계값 결정
    vmax = float(attn.max()) if attn.size > 0 else 0.0
    if percentile is not None:
        thr_val = float(np.percentile(attn, percentile))
    elif threshold is not None:
        # threshold∈(0,1]이면 max 기반, 1보다 크면 절대값으로 간주
        thr_val = float(vmax * threshold) if threshold <= 1.0 else float(threshold)
    else:
        # 자동: mean + 1*std와 max*0.5 중 더 낮은 쪽을 사용
        mean, std = float(attn.mean()), float(attn.std())
        thr_val = min(mean + std, vmax * 0.5)

    # 3) 마스크 생성
    mask = (attn >= thr_val)

    # 4) BFS로 연결 성분 추출
    visited = np.zeros_like(mask, dtype=bool)
    regions: List[List[Tuple[int, int]]] = []

    if connectivity == 4:
        nbrs = [(1,0),(-1,0),(0,1),(0,-1)]
    else:  # 8-연결
        nbrs = [(di, dj) for di in (-1,0,1) for dj in (-1,0,1) if not (di==0 and dj==0)]

    for y in range(n_h):
        for x in range(n_w):
            if not mask[y, x] or visited[y, x]:
                continue
            q = deque([(y, x)])
            visited[y, x] = True
            region = [(y, x)]
            while q:
                cy, cx = q.popleft()
                for dy, dx in nbrs:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < n_h and 0 <= nx < n_w and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        q.append((ny, nx))
                        region.append((ny, nx))
            if len(region) >= min_patches:
                regions.append(region)

    # 5) 각 성분의 bbox 및 점수 계산
    out: List[Dict] = []
    eps = 1e-9
    for region in regions:
        ys = [p[0] for p in region]
        xs = [p[1] for p in region]
        y_min, y_max = min(ys), max(ys)
        x_min, x_max = min(xs), max(xs)

        # 정규화 bbox (픽셀 좌표가 아니라 패치 그리드 기준)
        l = x_min / n_w
        t = y_min / n_h
        r = (x_max + 1) / n_w
        b = (y_max + 1) / n_h

        # 점수 집계
        region_scores = attn[ys, xs]
        score_sum = float(region_scores.sum())
        score_mean = float(region_scores.mean())
        score_norm = float(score_sum / (vmax * (len(region) + eps) + eps))  # 간단 정규화

        out.append({
            "bbox": [l, t, r, b],
            "patch_bbox": [int(x_min), int(y_min), int(x_max), int(y_max)],
            "size": int(len(region)),
            "score_sum": score_sum,
            "score_mean": score_mean,
            "score_norm": score_norm,
        })

    # 6) 점수 기준 정렬 (합이 큰 순)
    # out.sort(key=lambda d: (d["score_sum"], d["size"]), reverse=True)
    return out
