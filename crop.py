'''
멀티 컷팅 크롭 시스템

- 각 컷 위치에서 허용 범위 내에서 RGB 차이가 가장 적은 지점 선택
- 격자 형태로 이미지 분할하여 여러 개의 크롭 생성
'''

from PIL import Image, ImageDraw
import numpy as np

import os
from time import time 
import json

#! Hyper Parameter

device_type = "mobile"
# device_type = "web"
# device_type = "desktop"

#! ================================================================================================


def find_horizontal_cuts(
    img: Image.Image,
    h_cuts: int,                 # 수평으로 자르는 횟수 (라인 개수)
    h_tolerance: float,       # 수평 자를 때 허용 범위 비율 (예 50±7%)
    stripe_h: int = 20,              # 훑는 가로 스트립 두께(px)
    smooth_radius: int = 5,          # 행 점수 1D 스무딩 크기
    sample_stride: int = 20,         # x축 샘플링 보폭(>1이면 속도 ↑, 정밀도 ↓)
):
    """
    수평 컷팅: 수평으로 여러 번 자르는 함수
    반환값: horizontal_cuts (수평 컷 y좌표 리스트)
    """

    arr = np.asarray(img, dtype=np.float32)  # HxWx3
    H, W, _ = arr.shape
    
    horizontal_cuts = []
    
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
            
            # target_ratio에 가까울수록 점수를 더 좋게 주기
            # 기준점까지의 거리를 정규화하여 가중치로 사용
            target_idx = target_y - y_min  # 범위 내에서의 목표 인덱스
            distance_from_target = np.abs(np.arange(len(row_energy)) - target_idx)
            max_distance = max(1, len(row_energy) - 1)
            
            # 거리가 가까울수록 작은 값 (0~1), 멀수록 큰 값
            distance_penalty = distance_from_target / max_distance
            
            # RGB 변화량을 정규화하고 거리 가중치 추가 (둘 다 0~1 스케일로 맞춤)
            if row_energy.max() > 0:
                normalized_energy = row_energy / row_energy.max()
            else:
                normalized_energy = row_energy

            # 최종 점수: RGB 변화량(80%) + 거리 페널티(20%)
            combined_score = 0.8 * normalized_energy + 0.2 * distance_penalty

            # 가장 점수가 낮은 지점 선택
            best_idx = np.argmin(combined_score)
            y_cut = y_min + best_idx
            horizontal_cuts.append(y_cut)
    
    return horizontal_cuts


def crop_img(orig_img:Image.Image, output_image_path=None, save_visualization=False, print_latency=False, gt_bboxes=None):
    """
    멀티 컷팅으로 이미지를 격자 형태로 분할하여 결과 리스트 반환
    
    Args:
        image_path: 입력 이미지 경로
        output_image_path: 이미지 저장 경로 (None이면 저장 안함)
        save_visualization: 시각화 이미지 저장 여부
        print_latency: 실행 시간 출력 여부
        stripe_h: 스트립 두께
        smooth_radius: 스무딩 반경
        sample_stride: 샘플링 보폭
        h_cuts: 수평 컷 횟수 (None이면 가로세로 비율로 자동 결정)
        v_cuts: 수직 컷 횟수
        h_tolerance: 수평 허용 범위
        v_tolerance: 수직 허용 범위
        gt_bboxes: 정답 bbox 리스트 [[x, y, w, h], ...] 형식
    
    Returns:
        results_for_grounding: grounding용 crop 결과 리스트
    """

    start = time()

    orig_w, orig_h = orig_img.size

    #! 높이의 pixel에 따라서 나누는 횟수 지정
    if orig_h < 1000:  # 
        h_cuts = 1
        h_tolerance = 0.20
    elif orig_h < 1440:  # 중간화질 -> 3등분
        h_cuts = 2
        h_tolerance = 0.12
    else:  # 고화질이나 세로화면은 4등분
        h_cuts = 3
        h_tolerance = 0.08

    #! ======================================

    # 수평 컷만 실행
    horizontal_cuts = find_horizontal_cuts(orig_img, h_cuts = h_cuts, h_tolerance = h_tolerance)
    
    # 수직 컷은 현재 사용하지 않음
    vertical_cuts = []
    
    # 격자 bbox 생성
    # y 좌표들 (0, horizontal_cuts..., orig_h)
    y_coords = [0] + sorted(horizontal_cuts) + [orig_h]
    
    # x 좌표들 (0, vertical_cuts..., orig_w)  
    x_coords = [0] + sorted(vertical_cuts) + [orig_w]
    
    bboxes = []
    crop_imgs = []
    
    # 격자 형태로 bbox 생성 및 이미지 크롭
    for i in range(len(y_coords) - 1):
        for j in range(len(x_coords) - 1):
            L = x_coords[j]
            T = y_coords[i]
            R = x_coords[j + 1]
            B = y_coords[i + 1]
            
            bbox = [L, T, R, B]
            bboxes.append(bbox)
            
            crop_img_part = orig_img.crop(tuple(bbox))
            crop_imgs.append(crop_img_part)

    # 결과 리스트 구성
    results_for_grounding = []
    
    # 썸네일 (전체 이미지)
    results_for_grounding.append({
        "img": orig_img.copy(), 
        "id": 0, 
        "bbox": [0, 0, orig_w, orig_h],
        "recursion_depth": 0, 
        "fail": False, 
        "filename": None
    })
    
    # 크롭된 이미지들
    for idx, (bbox, crop_img_part) in enumerate(zip(bboxes, crop_imgs)):
        results_for_grounding.append({
            "img": crop_img_part, 
            "id": idx + 1, 
            "bbox": bbox,
            "recursion_depth": 0, 
            "fail": False, 
            "filename": None
        })

    end = time()

    if print_latency:
        print(f"[Time] {end - start:.3f}s", end = " | ")
        print(f"✂️ Crops : {len(results_for_grounding)-1}", end = "")  # 썸네일 제외 개수 출력

    if not save_visualization or output_image_path is None:
        print()
        return results_for_grounding
    
    #! ========================= 시각화 =========================

    vis_img = orig_img.copy()
    draw = ImageDraw.Draw(vis_img)

    palette = [
        (255, 0, 0),    # 빨간색
        (0, 255, 0),    # 초록색
        (0, 0, 255),    # 파란색
        (255, 165, 0),  # 주황색
        (255, 0, 255),  # 마젠타
        (0, 255, 255),  # 시안
        (255, 255, 0),  # 노란색
        (128, 0, 128),  # 보라색
    ]
    line_w = max(2, int(min(orig_w, orig_h) * 0.003))

    # 수평 컷 라인들 그리기
    for y in horizontal_cuts:
        draw.line([(0, y), (orig_w, y)], fill=(255, 0, 255), width=line_w)
    
    # 수직 컷 라인들 그리기  
    for x in vertical_cuts:
        draw.line([(x, 0), (x, orig_h)], fill=(255, 0, 255), width=line_w)
    
    # 각 격자 영역 외곽선 그리기
    for idx, bbox in enumerate(bboxes):
        color = palette[idx % len(palette)]
        draw.rectangle(bbox, outline=color, width=line_w)

    # 정답 bbox 그리기 (있는 경우)
    if gt_bboxes:
        gt_line_w = max(3, int(min(orig_w, orig_h) * 0.005))  # 더 두꺼운 선
        for gt_bbox in gt_bboxes:
            if len(gt_bbox) >= 4:
                x, y, w, h = gt_bbox[:4]
                # [x, y, w, h] → [L, T, R, B] 변환
                gt_rect = [x, y, x + w, y + h]
                # 빨간색으로 정답 bbox 그리기
                draw.rectangle(gt_rect, outline=(255, 255, 0), width=gt_line_w)  # 노란색

    vis_img.save(output_image_path)

    if print_latency:
        print(f" | [SAVE] {output_image_path}")

    return results_for_grounding

#! ================================================================================================

if __name__ == '__main__':
    # 테스트용 main
    data_path = "./data/screenspotv2_image/"

    output_path = f"./crop_test/{device_type}"
    os.makedirs(output_path, exist_ok=True)
    jsonlist = json.load(open(f"./data/screenspot_{device_type}_v2.json"))

    target_imgs = sorted(set(item["img_filename"] for item in jsonlist if "img_filename" in item))

    # 파일별 정답 bbox 딕셔너리 생성
    gt_bbox_dict = {}
    for item in jsonlist:
        if "img_filename" in item and "bbox" in item:
            fname = item["img_filename"]
            if fname not in gt_bbox_dict:
                gt_bbox_dict[fname] = []
            gt_bbox_dict[fname].append(item["bbox"])

    for fname in target_imgs:
        # 해당 파일의 정답 bbox들 가져오기
        gt_bboxes = gt_bbox_dict.get(fname, [])
        
        image_path=os.path.join(data_path, fname)
        orig_img = Image.open(image_path).convert("RGB")
        
        # 멀티 컷팅 테스트 실행
        crop_img(
            orig_img=orig_img,
            output_image_path=f"{output_path}/{fname}",
            save_visualization=True,
            print_latency=True,
            gt_bboxes=gt_bboxes,  # 정답 bbox 전달
        )
