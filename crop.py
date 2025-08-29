'''
멀티 컷팅 크롭 시스템
crop_line.py의 find_multi_cuts와 generate_grid_crops를 사용하여
수평/수직으로 여러 번 자르는 방식으로 이미지를 분할

- 수평 컷 횟수와 수직 컷 횟수를 지정 가능
- 각 컷 위치에서 허용 범위 내에서 RGB 차이가 가장 적은 지점 선택
- 격자 형태로 이미지 분할하여 여러 개의 크롭 생성
'''

from PIL import Image, ImageDraw
from crop_line import find_multi_cuts, generate_grid_crops

import os
from time import time 
import json

#! Hyper Parameter
# 멀티 컷팅 파라미터
DEFAULT_STRIPE_H = 50           # 스트립 두께
DEFAULT_SMOOTH_RADIUS = 5       # 스무딩 반경
DEFAULT_SAMPLE_STRIDE = 20      # 샘플링 보폭
DEFAULT_H_CUTS = 3             # 수평 컷 횟수 (4개 구간)
DEFAULT_V_CUTS = 1             # 수직 컷 횟수 (2개 구간)
DEFAULT_H_TOLERANCE = 0.12     # 수평 허용 범위
DEFAULT_V_TOLERANCE = 0.4     # 수직 허용 범위

#! ================================================================================================


def crop_img(image_path, output_image_path=None, save_visualization=False, print_latency=False,
             stripe_h=DEFAULT_STRIPE_H, smooth_radius=DEFAULT_SMOOTH_RADIUS, 
             sample_stride=DEFAULT_SAMPLE_STRIDE, h_cuts=DEFAULT_H_CUTS, v_cuts=DEFAULT_V_CUTS,
             h_tolerance=DEFAULT_H_TOLERANCE, v_tolerance=DEFAULT_V_TOLERANCE,
             gt_bboxes=None  # 정답 bbox 리스트 추가
             ):
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
        h_cuts: 수평 컷 횟수
        v_cuts: 수직 컷 횟수
        h_tolerance: 수평 허용 범위
        v_tolerance: 수직 허용 범위
        gt_bboxes: 정답 bbox 리스트 [[x, y, w, h], ...] 형식
    
    Returns:
        results_for_grounding: grounding용 crop 결과 리스트
    """

    start = time()

    # 원본 이미지 로드
    orig_img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = orig_img.size

    # 세로 화면 감지 및 수직 컷 제어
    # allow_vertical_cut = orig_w > orig_h * 1.5
    allow_vertical_cut = False

    time0 = time()
    if print_latency:
        print(f"[Crop] [Load] {time0 - start:.3f}s", end = " | ")

    # 멀티 컷팅 실행
    bboxes, crop_imgs = generate_grid_crops(
        orig_img,
        stripe_h=stripe_h,
        smooth_radius=smooth_radius,
        sample_stride=sample_stride,
        h_cuts=h_cuts,
        v_cuts=v_cuts,
        h_tolerance=h_tolerance,
        v_tolerance=v_tolerance,
        allow_vertical_cut=allow_vertical_cut
    )
    
    # 컷 라인들 가져오기 (시각화용)
    horizontal_cuts, vertical_cuts = find_multi_cuts(
        orig_img,
        stripe_h=stripe_h,
        smooth_radius=smooth_radius,
        sample_stride=sample_stride,
        h_cuts=h_cuts,
        v_cuts=v_cuts,
        h_tolerance=h_tolerance,
        v_tolerance=v_tolerance,
        allow_vertical_cut=allow_vertical_cut
    )

    time1 = time()
    if print_latency:
        print(f"[Multi-Cut] {time1 - time0:.3f}s", end = " | ")

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
    for idx, (bbox, crop_img) in enumerate(zip(bboxes, crop_imgs)):
        results_for_grounding.append({
            "img": crop_img, 
            "id": idx + 1, 
            "bbox": bbox,
            "recursion_depth": 0, 
            "fail": False, 
            "filename": None
        })

    end = time()

    if print_latency:
        print(f"[Total] {end - start:.3f}s", end = " | ")
        print(f"✂️ Crops : {len(results_for_grounding)-1}", end = "")  # 썸네일 제외 개수 출력

    if not save_visualization or output_image_path is None:
        print()
        return results_for_grounding
    
    # ---------------------------- 시각화 ----------------------------

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

    # device_type = "mobile"
    # device_type = "web"
    device_type = "desktop"

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
        
        # 멀티 컷팅 테스트 실행
        crop_img(
            image_path=os.path.join(data_path, fname),
            output_image_path=f"{output_path}/{fname}",
            save_visualization=True,
            print_latency=True,
            gt_bboxes=gt_bboxes,  # 정답 bbox 전달
            # 커스텀 파라미터 (필요시 조정)
            stripe_h=50,
            smooth_radius=5,
            sample_stride=20,
            h_cuts=3,           # 수평 3개 라인 (4개 구간)
            v_cuts=1,           # 수직 1개 라인 (2개 구간)
            h_tolerance=0.07,   # 수평 허용범위 ±7%
            v_tolerance=0.07,   # 수직 허용범위 ±7%
        )
