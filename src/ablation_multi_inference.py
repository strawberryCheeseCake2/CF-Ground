import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('gpu', type=int, help='GPU number')
parser.add_argument('--r', type=float, nargs=2, metavar=('MIN_RESIZE', 'MAX_RESIZE'), help='Stage 1 Resize ratio range (min max)')
parser.add_argument('--e', type=float, help='Stage 1 Ensemble ratio')
parser.add_argument('--v', action='store_true', help='Whether to save visualization images')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)  # 몇번 GPU 사용할지 argument로 지정 : run_gui_actor.py 2 -> 2번 GPU

#! Hyperparameter =====================================================================================

ATTN_IMPL = "eager"  # attention implement "eager" "sdpa" "flash" "efficient"

# Image Resize Ratios

MIN_RESIZE = args.r[0] if args.r else 0.50  # DYNAMIC_RESIZE 비율 최소값
MAX_RESIZE = args.r[1] if args.r else 0.50  # DYNAMIC_RESIZE 비율 최대값

# Crop Limitations
MAX_CROPS = 3  # 생성할 수 있는 최대 crop 개수
SELECT_THRESHOLD = 0.50  #! score >= tau * max_score 인 모든 crop select
CROP_WIDTH = 1176  # 크롭할 직사각형 가로 크기 (아이폰 전체 가로가 1170px)
CROP_HEIGHT = 602  # 크롭할 직사각형 세로 크기

# Ensemble Hyperparameters
# TODO: 이것도 resize처럼 동적으로 측정해서 변경 가능하도록
STAGE1_ENSEMBLE_RATIO = args.e if args.e else 0.50  # Stage1 attention 가중치
STAGE2_ENSEMBLE_RATIO = 1 - STAGE1_ENSEMBLE_RATIO  # Stage2 crop 가중치
ENSEMBLE_TOP_PATCHES = 100                         # Stage2에서 앙상블에 사용할 상위 패치 개수

# 최대 PIXELS 제한
MAX_PIXELS = 3211264  # Process단에서 적용

# csv에 기록할 method 이름
# method = "dynamic_resize"
method = "multi_fixed_resize"

memo = f"resize_{MIN_RESIZE:.2f}~{MAX_RESIZE:.2f}_ensemble{STAGE1_ENSEMBLE_RATIO}_crop{CROP_WIDTH}x{CROP_HEIGHT}"

#! Argument ==========================================================================================

SEED = 0

# Dataset & Model
MLLM_PATH = "microsoft/GUI-Actor-3B-Qwen2.5-VL"
SCREENSPOT_IMGS = "../data/screenspotv2_image"  # input image 경로
SCREENSPOT_JSON = "../data"  # input image json파일 경로
TASKS = ["mobile", "web", "desktop"]
# TASKS = ["mobile"]
# TASKS = ["web"]
# TASKS = ["desktop"]
SAMPLE_RANGE = slice(None)
# SAMPLE_RANGE = slice(0,2)

# Visualize & Logging
VISUALIZE = args.v if args.v else False
VIS_ONLY_WRONG = False  # True면 틀린 것만 시각화, False면 모든 것 시각화
TFOPS_PROFILING = True
MEMORY_EVAL = True
MEMORY_VIS = False

# Save Path
SAVE_DIR = f"../attn_output/" + method + "/" + memo

#! ==================================================================================================

# Standard Library
import os
import sys
import time
import threading
import re
import json
import logging
logging.disable(logging.CRITICAL)  # 모든 로깅 호출 무력화

# Third-Party Libraries
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoProcessor, AutoTokenizer, set_seed

# Project-Local Modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util.iter_logger import init_iter_logger, append_iter_log  # log csv 기록 파일
from gui_actor.modeling_qwen25vl import Qwen2_5_VLForConditionalGenerationWithPointer
from gui_actor.inference import inference
from gui_actor.multi_image_inference import multi_image_inference
from util.visualize_util import visualize_stage1_attention_crops, visualize_stage2_merged_attention, visualize_stage2_multi_attention, visualize_stage3_ensemble_attention, visualize_stage3_point_ensemble
from util.sharpness_util import get_fft_blur_score
if TFOPS_PROFILING:
    from deepspeed.profiling.flops_profiler import FlopsProfiler

#! ==============================================================================================

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super(NpEncoder, self).default(obj)

def resize_image(image, resize_to_pixels):
    image_width, image_height = image.size
    if (resize_to_pixels is not None) and ((image_width * image_height) > resize_to_pixels):
        resize_ratio = (resize_to_pixels / (image_width * image_height)) ** 0.5
        image_width_resized, image_height_resized = int(image_width * resize_ratio), int(image_height * resize_ratio)
        image = image.resize((image_width_resized, image_height_resized))
        print(f"🔧 Resized image: {image_width}x{image_height} -> {image_width_resized}x{image_height_resized} (ratio: {resize_ratio:.3f})")
    return image

def warm_up_model(model, tokenizer, processor):
    print("🏋️‍♂️ Warming up the model...")
    dummy_instruction = "This is a dummy instruction for warm-up."
    dummy_image = Image.new("RGB", (1000, 1000), color=(255, 255, 255))  # 1000x1000 흰색 이미지
    dummy_msgs = create_conversation_stage1(image=dummy_image, instruction=dummy_instruction, resize_ratio=1.0)
    
    # 예열용 inference 실행
    for _ in range(3):  # 3번 반복
        with torch.no_grad():
            _ = inference(dummy_msgs, model, tokenizer, processor, use_placeholder=True, topk=3)
    print("🏋️‍♂️ Warm-up complete!")

def create_conversation_stage1(image, instruction, resize_ratio):
    conversation = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    # "text": (
                    #     "You are a GUI agent. Given a screenshot of the current GUI and a human instruction, "
                    #     "your task is to locate the screen element that corresponds to the instruction. "
                    #     "You should output a PyAutoGUI action that performs a click on the correct position. "
                    #     "To indicate the click location, we will use some special tokens, which is used to refer to a visual patch later. "
                    #     "For example, you can output: pyautogui.click(<your_special_token_here>)."
                    # ),
                    "text": (
                        f"This is a resized screenshot of the whole GUI, scaled by {resize_ratio}. "
                        "You are a GUI agent. Given a screenshot of the current GUI and a human instruction, "
                        "your task is to locate the screen element that corresponds to the instruction. "
                        "You should output a PyAutoGUI action that performs a click on the correct position. "
                        "To indicate the click location, we will use some special tokens, which is used to refer to a visual patch later. "
                        "For example, you can output: pyautogui.click(<your_special_token_here>)."
                    ),
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": instruction,
                }
            ],
        },
    ]
    return conversation

def create_conversation_stage2(image, instruction, crop_cnt):
    conversation = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    # "text": (
                    #     "You are a GUI agent. Given a screenshot of the current GUI and a human instruction, "
                    #     "your task is to locate the screen element that corresponds to the instruction. "
                    #     "You should output a PyAutoGUI action that performs a click on the correct position. "
                    #     "To indicate the click location, we will use some special tokens, which is used to refer to a visual patch later. "
                    #     "For example, you can output: pyautogui.click(<your_special_token_here>)."
                    # ),
                    "text": (
                        f"This image is a vertical collage of {crop_cnt} cropped regions that were selected as promising. "
                        "Each crop has the same width and fixed height, separated by a red horizontal line. "
                        "You are a GUI agent. Given this collage image and a human instruction, "
                        "your task is to locate the screen element that corresponds to the instruction. "
                        "You should output a PyAutoGUI action that performs a click on the correct position. "
                        "To indicate the click location, we will use some special tokens, which is used to refer to a visual patch later. "
                        "For example, you can output: pyautogui.click(<your_special_token_here>)."
                    ),
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": instruction,
                }
            ],
        },
    ]
    return conversation

def run_stage1_attention_inference(original_image, instruction):
    """Stage 1: 리사이즈하고 inference"""

    orig_w, orig_h = original_image.size
    # 이미지 고정 리사이즈
    if MIN_RESIZE == MAX_RESIZE:
        resize_ratio = MIN_RESIZE
        resized_w, resized_h = int(orig_w * resize_ratio), int(orig_h * resize_ratio)
        print(f"🔧 Fixed Resized image: {orig_w}x{orig_h} -> {resized_w}x{resized_h} (ratio: {resize_ratio:.3f})")
    
    # 이미지 동적 리사이즈
    else:
        downsampled = original_image.resize((int(orig_w*0.5), int(orig_h*0.5)))
        resize_ratio = get_fft_blur_score(downsampled, min_resize=MIN_RESIZE, max_resize=MAX_RESIZE)
        resized_w, resized_h = int(orig_w * resize_ratio), int(orig_h * resize_ratio)
        print(f"🔧 Dynamic Resized image: {orig_w}x{orig_h} -> {resized_w}x{resized_h} (ratio: {resize_ratio:.3f})")
    
    # 리사이즈된 이미지로 inference
    resized_image = original_image.resize((resized_w, resized_h))
    conversation = create_conversation_stage1(resized_image, instruction, resize_ratio)
    pred = inference(conversation, model, tokenizer, processor, use_placeholder=True, topk=1)
    
    # 결과에 리사이즈 정보 추가
    pred['resize_ratio'] = resize_ratio
    pred['original_size'] = (orig_w, orig_h)
    pred['resized_size'] = resized_image.size
    
    return pred, resized_image

def find_attention_peaks(pred_result, resized_image, resize_ratio):
    """어텐션에서 고점들 찾기 (bbox 생성 후 중심점 거리로 중복 제거)"""
    
    attn_scores = np.array(pred_result['attn_scores'][0])
    n_width = pred_result['n_width'] 
    n_height = pred_result['n_height']
    
    resized_w, resized_h = resized_image.size
    
    # 패치를 픽셀 좌표로 변환하기 위한 비율
    patch_w = resized_w / n_width
    patch_h = resized_h / n_height
    
    # 점수가 높은 순서로 정렬
    sorted_indices = np.argsort(attn_scores)[::-1]
    
    peaks = []
    used_bbox_centers = []  # 이미 사용된 bbox 중심점들
    
    for idx in sorted_indices:
        # 패치 좌표 계산
        patch_y = idx // n_width
        patch_x = idx % n_width
        
        # 패치 중심점의 픽셀 좌표 (리사이즈된 이미지 기준)
        center_x = (patch_x + 0.5) * patch_w
        center_y = (patch_y + 0.5) * patch_h
        
        # 원본 이미지 기준으로 변환
        orig_center_x = center_x / resize_ratio
        orig_center_y = center_y / resize_ratio
        
        # 원본 이미지에서 bbox 계산 (실제 크롭이 생성될 위치)
        orig_w = resized_w / resize_ratio
        orig_h = resized_h / resize_ratio
        
        left = max(0, int(orig_center_x - CROP_WIDTH/2))
        top = max(0, int(orig_center_y - CROP_HEIGHT/2))
        right = min(orig_w, int(orig_center_x + CROP_WIDTH/2))
        bottom = min(orig_h, int(orig_center_y + CROP_HEIGHT/2))
        
        # 경계에서 잘렸을 경우 조정
        if right - left < CROP_WIDTH and right < orig_w:
            right = min(orig_w, left + int(CROP_WIDTH))
        if right - left < CROP_WIDTH and left > 0:
            left = max(0, right - int(CROP_WIDTH))
        if bottom - top < CROP_HEIGHT and bottom < orig_h:
            bottom = min(orig_h, top + int(CROP_HEIGHT))
        if bottom - top < CROP_HEIGHT and top > 0:
            top = max(0, bottom - int(CROP_HEIGHT))
        
        # 실제 bbox 중심점 계산
        bbox_center_x = (left + right) / 2
        bbox_center_y = (top + bottom) / 2
        
        # 이미 사용된 bbox와 중복되는지 확인
        skip = False
        for used_center in used_bbox_centers:
            x_distance = abs(bbox_center_x - used_center[0])
            y_distance = abs(bbox_center_y - used_center[1])
            # bbox 중심점이 너무 가까우면 스킵 (크롭 크기의 30% 이내)
            if x_distance < CROP_WIDTH * 0.3 and y_distance < CROP_HEIGHT * 0.3:  #! 크롭
                skip = True
                break
        
        if not skip:
            peaks.append({
                'center_x': center_x,  # 리사이즈된 이미지 기준 (기존 호환성)
                'center_y': center_y,
                'score': float(attn_scores[idx]),
                'patch_x': patch_x,
                'patch_y': patch_y,
                'bbox_center': (bbox_center_x, bbox_center_y)  # 실제 bbox 중심점
            })
            
            used_bbox_centers.append((bbox_center_x, bbox_center_y))
    
    return peaks

def filter_by_threshold(peaks, threshold=0.7, max_crops=MAX_CROPS):
    """1등의 threshold% 이상인 peak들만 남기고, 최대 개수 제한 적용"""
    
    if not peaks:
        return []
    
    max_score = peaks[0]['score']  # 이미 정렬되어 있음
    min_score = max_score * threshold
    
    # threshold 조건으로 먼저 필터링
    filtered_peaks = [peak for peak in peaks if peak['score'] >= min_score]
    
    # 최대 개수 제한 적용 (상위 점수순으로)
    if len(filtered_peaks) > max_crops:
        filtered_peaks = filtered_peaks[:max_crops]
    
    print(f"🎯 Found {len(peaks)} peaks, filtered to {len(filtered_peaks)} (threshold: {threshold}, max_crops: {max_crops})")
    
    return filtered_peaks

def create_crops_from_attention_peaks(peaks, original_image, resize_ratio):
    """Attention peaks를 기반으로 원본 이미지에서 직접 crop"""
    
    if not peaks:
        return []
    
    crops = []
    orig_w, orig_h = original_image.size
    
    for i, peak in enumerate(peaks):
        # 리사이즈된 이미지에서의 center를 원본 크기로 변환
        orig_center_x = peak['center_x'] / resize_ratio
        orig_center_y = peak['center_y'] / resize_ratio
        
        # 원본 이미지에서의 bbox 계산
        left = max(0, int(orig_center_x - CROP_WIDTH/2))
        top = max(0, int(orig_center_y - CROP_HEIGHT/2))
        right = min(orig_w, int(orig_center_x + CROP_WIDTH/2))
        bottom = min(orig_h, int(orig_center_y + CROP_HEIGHT/2))
        
        # 경계에서 잘렸을 경우 조정
        if right - left < CROP_WIDTH and right < orig_w:
            right = min(orig_w, left + int(CROP_WIDTH))
        if right - left < CROP_WIDTH and left > 0:
            left = max(0, right - int(CROP_WIDTH))
        if bottom - top < CROP_HEIGHT and bottom < orig_h:
            bottom = min(orig_h, top + int(CROP_HEIGHT))
        if bottom - top < CROP_HEIGHT and top > 0:
            top = max(0, bottom - int(CROP_HEIGHT))
        
        bbox = [left, top, right, bottom]
        crop_img = original_image.crop(bbox)
        
        crops.append({
            'img': crop_img,
            'bbox': bbox,
            'score': peak['score'],
            'id': i + 1
        })
        
        # print(f"🔧 Crop {i+1}: center=({orig_center_x:.1f}, {orig_center_y:.1f}), bbox={bbox}, size={CROP_WIDTH}x{CROP_HEIGHT}")
    
    return crops

def create_merged_image_for_stage2(crops):
    """Stage 2용: crop들을 세로로 합치기 (빨간색 구분선 이미지로 분리) - bbox y좌표 순으로 정렬"""

    if not crops:
        return None, []
    
    # bbox의 y좌표(top) 순으로 정렬 (위에서 아래로)
    sorted_crops = sorted(crops, key=lambda crop: crop['bbox'][1])
    
    # 세로로 합칠 이미지들과 구분선들을 별도로 준비
    separator_height = 28  # 빨간색 구분선 두께
    max_width = max(crop['img'].width for crop in sorted_crops)
    
    # 합칠 이미지들 리스트 (crop 이미지 + 구분선 이미지들)
    images_to_merge = []
    crop_y_mappings = []
    current_y = 0
    
    for i, crop in enumerate(sorted_crops):
        # crop 이미지 추가
        images_to_merge.append(crop['img'])
        
        # 매핑 정보 저장: (merged_y_start, merged_y_end) -> (original_bbox)
        paste_x = (max_width - crop['img'].width) // 2
        crop_y_mappings.append({
            'merged_y_start': current_y,
            'merged_y_end': current_y + crop['img'].height,
            'original_bbox': crop['bbox'],
            'paste_x': paste_x
        })
        
        current_y += crop['img'].height
        
        # 마지막이 아니면 빨간색 구분선 이미지 추가
        if i < len(sorted_crops) - 1:
            separator_img = Image.new('RGB', (max_width, separator_height), color=(256, 0, 0))
            images_to_merge.append(separator_img)
            current_y += separator_height
    
    # 총 높이 계산
    total_height = current_y
    
    # 합쳐진 이미지 생성
    merged_img = Image.new('RGB', (max_width, total_height), color=(0, 0, 0))
    
    # 이미지들을 순서대로 붙이기
    paste_y = 0
    image_idx = 0
    
    for i, crop in enumerate(sorted_crops):
        # crop 이미지 붙이기 (중앙 정렬)
        crop_img = images_to_merge[image_idx]
        paste_x = (max_width - crop_img.width) // 2
        merged_img.paste(crop_img, (paste_x, paste_y))
        paste_y += crop_img.height
        image_idx += 1
        
        # 구분선 이미지 붙이기 (마지막이 아닌 경우)
        if i < len(sorted_crops) - 1:
            separator_img = images_to_merge[image_idx]
            merged_img.paste(separator_img, (0, paste_y))
            paste_y += separator_img.height
            image_idx += 1
    
    return merged_img, crop_y_mappings


def create_multi_image_msgs(crop_list, instruction):
    user_content = []
    for crop in crop_list:
        img = crop["img"]  # "resized_img" -> "img" 수정
        user_content.append({"type": "image", "image": img})

    user_content.append({
        "type": "text",
        "text": instruction,
    })
    conversation = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are a GUI agent. Given a screenshot of the current GUI and a human instruction, "
                        "your task is to locate the screen element that corresponds to the instruction. "
                        "You should output a PyAutoGUI action that performs a click on the correct position. "
                        "To indicate the click location, we will use some special tokens, which is used to refer to a visual patch later. "
                        "For example, you can output: pyautogui.click(<your_special_token_here>)."
                    ),
                }
            ]
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]
    return conversation

def run_stage2_multi_image_inference(crop_list, instruction):
    """Stage 2: multi image inference - 각 crop별로 개별 inference"""
    
    # multi image inference용 대화 생성
    conversation = create_multi_image_msgs(crop_list, instruction)
    
    # multi image inference 실행 (각 이미지별 결과 반환)
    pred = multi_image_inference(conversation, model, tokenizer, processor, use_placeholder=True, topk=10)
    
    return pred

def convert_multi_image_results_to_original(multi_pred, crop_list):
    """multi_image_inference 결과를 원본 이미지 좌표로 변환하고 앙상블"""
    
    if not multi_pred.get('per_image') or not crop_list:
        return None, []
    
    # 각 crop별 결과를 원본 좌표로 변환
    converted_results = []
    all_candidates = []
    
    for img_idx, img_result in enumerate(multi_pred['per_image']):
        if img_idx >= len(crop_list):
            continue
            
        crop_info = crop_list[img_idx]
        crop_bbox = crop_info['bbox']  # [left, top, right, bottom]
        crop_width = crop_bbox[2] - crop_bbox[0]
        crop_height = crop_bbox[3] - crop_bbox[1]
        
        # 해당 이미지의 topk 결과들을 원본 좌표로 변환
        crop_candidates = []
        for point_idx, (point, score) in enumerate(zip(img_result['topk_points'], img_result['topk_values'])):
            # 정규화된 좌표를 crop 내 픽셀 좌표로 변환
            crop_x = point[0] * crop_width
            crop_y = point[1] * crop_height
            
            # crop 좌표를 원본 이미지 좌표로 변환
            original_x = crop_bbox[0] + crop_x
            original_y = crop_bbox[1] + crop_y
            
            candidate = {
                'point': [original_x, original_y],
                'score': score,
                'crop_id': crop_info['id'],
                'crop_bbox': crop_bbox,
                'rank_in_crop': point_idx
            }
            crop_candidates.append(candidate)
            all_candidates.append(candidate)
        
        converted_results.append({
            'crop_id': crop_info['id'],
            'crop_bbox': crop_bbox,
            'candidates': crop_candidates
        })
    
    # 모든 후보들을 점수순으로 정렬하여 최종 앙상블 결과 생성
    all_candidates.sort(key=lambda x: x['score'], reverse=True)
    
    # 최고 점수 후보를 최종 예측으로 선택
    best_candidate = all_candidates[0] if all_candidates else None
    
    return best_candidate, all_candidates

def create_stage2_attention_to_original_multi(multi_pred, crop_list, original_size):
    """multi_image_inference 결과로부터 원본 이미지 크기의 어텐션 맵 생성"""
    
    if not multi_pred.get('per_image') or not crop_list:
        return None
    
    orig_w, orig_h = original_size
    stage2_attention_map = np.zeros((orig_h, orig_w), dtype=np.float32)
    overlap_count_map = np.zeros((orig_h, orig_w), dtype=np.int32)
    
    # 각 crop의 상위 100개 패치를 원본 이미지에 매핑
    for img_idx, img_result in enumerate(multi_pred['per_image']):
        if img_idx >= len(crop_list):
            continue
            
        crop_info = crop_list[img_idx]
        crop_bbox = crop_info['bbox']  # [left, top, right, bottom]
        crop_width = crop_bbox[2] - crop_bbox[0]
        crop_height = crop_bbox[3] - crop_bbox[1]
        
        # 해당 이미지의 어텐션 스코어와 그리드 정보
        attn_scores = np.array(img_result['attn_scores'][0])
        n_width = img_result['n_width']
        n_height = img_result['n_height']
        
        # 상위 100개 패치 선택
        top_indices = np.argsort(attn_scores)[-100:][::-1]  # 상위 100개
        
        for patch_idx in top_indices:
            # 패치 좌표 계산 (crop 내에서)
            patch_y = patch_idx // n_width
            patch_x = patch_idx % n_width
            
            # 패치 중심점의 정규화된 좌표 (crop 내에서)
            norm_x = (patch_x + 0.5) / n_width
            norm_y = (patch_y + 0.5) / n_height
            
            # crop 내 픽셀 좌표로 변환
            crop_pixel_x = norm_x * crop_width
            crop_pixel_y = norm_y * crop_height
            
            # 원본 이미지 좌표로 변환
            orig_pixel_x = crop_bbox[0] + crop_pixel_x
            orig_pixel_y = crop_bbox[1] + crop_pixel_y
            
            # 원본 이미지 범위 내로 클리핑
            orig_pixel_x = max(0, min(orig_w - 1, orig_pixel_x))
            orig_pixel_y = max(0, min(orig_h - 1, orig_pixel_y))
            
            # 정수 좌표로 변환
            x_int = int(orig_pixel_x)
            y_int = int(orig_pixel_y)
            
            # 어텐션 스코어 누적 및 겹침 카운트
            stage2_attention_map[y_int, x_int] += attn_scores[patch_idx]
            overlap_count_map[y_int, x_int] += 1
    
    # 겹치는 부분은 겹치는 횟수로 나누어 가중평균 처리
    valid_mask = overlap_count_map > 0
    stage2_attention_map[valid_mask] = stage2_attention_map[valid_mask] / overlap_count_map[valid_mask]
    
    return stage2_attention_map

def run_stage1_attention_based(original_image, instruction, gt_bbox):
    """새로운 간단한 Stage 1: Attention 기반 crop 생성"""
    
    # 1. 리사이즈하고 inference
    print("🔍 Stage 1: Running attention-based inference...")
    s1_pred, resized_image = run_stage1_attention_inference(original_image, instruction)
    
    # 2. GT bbox도 리사이즈 비율에 맞춰 조정
    resize_ratio = s1_pred['resize_ratio']
    scaled_gt_bbox = [coord * resize_ratio for coord in gt_bbox]
    
    # 3. Attention 고점들 찾기
    peaks = find_attention_peaks(s1_pred, resized_image, resize_ratio)
    
    if not peaks:
        print("⚠️ No attention peaks found")
        return s1_pred, [], 0, resized_image, scaled_gt_bbox
    
    # 4. 1등의 70% 이상만 남기고 최대 개수 제한 적용
    filtered_peaks = filter_by_threshold(peaks, threshold=SELECT_THRESHOLD, max_crops=MAX_CROPS)
    
    if not filtered_peaks:
        print("⚠️ No peaks passed threshold")
        return s1_pred, [], 0, resized_image, scaled_gt_bbox
    
    # 5. 원본 이미지에서 직접 crop 생성
    crops = create_crops_from_attention_peaks(filtered_peaks, original_image, resize_ratio)
    
    num_crops = len(crops)
    
    return s1_pred, crops, num_crops, resized_image, scaled_gt_bbox

def get_stage1_score_at_point(point, s1_attn_scores, s1_n_width, s1_n_height, original_size, resize_ratio):
    """특정 점에서의 Stage1 어텐션 점수를 계산"""
    
    orig_w, orig_h = original_size
    point_x, point_y = point
    
    # 원본 좌표를 리사이즈된 좌표로 변환
    resized_x = point_x * resize_ratio
    resized_y = point_y * resize_ratio
    
    # 리사이즈된 좌표를 패치 좌표로 변환
    resized_w = orig_w * resize_ratio
    resized_h = orig_h * resize_ratio
    
    patch_x = int((resized_x / resized_w) * s1_n_width)
    patch_y = int((resized_y / resized_h) * s1_n_height)
    
    # 패치 좌표가 유효한 범위 내인지 확인
    patch_x = max(0, min(patch_x, s1_n_width - 1))
    patch_y = max(0, min(patch_y, s1_n_height - 1))
    
    # 해당 패치의 어텐션 점수 반환
    patch_idx = patch_y * s1_n_width + patch_x
    if patch_idx < len(s1_attn_scores):
        return float(s1_attn_scores[patch_idx])
    else:
        return 0.0

def point_in_bbox(point, bbox):
    """점이 bbox 안에 있는지 확인"""
    if point is None or bbox is None:
        return False
    return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]

def monitor_memory(interval=0.1):
    start = time.time()
    while not stop_flag:
        mem = torch.cuda.memory_allocated() / 1024**3
        now = time.time() - start
        mem_log.append(mem)
        time_log.append(now)
        time.sleep(interval)

#! ================================================================================================

if __name__ == '__main__':

    set_seed(SEED)

    # Model Import (NVIDIA CUDA)
    model = Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
        MLLM_PATH, torch_dtype="auto", attn_implementation=ATTN_IMPL,
        device_map="balanced",
        # max_memory=max_memory, 
        low_cpu_mem_usage=True
    )
    # Model Import (Mac)
    # model = Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
    #     MLLM_PATH, torch_dtype="auto", attn_implementation=ATTN_IMPL,
    #     device_map="mps", # Mac
    #     # max_memory=max_memory, 
    #     low_cpu_mem_usage=False
    # )
    tokenizer = AutoTokenizer.from_pretrained(MLLM_PATH)
    processor = AutoProcessor.from_pretrained(MLLM_PATH, max_pixels=MAX_PIXELS)
    prof = FlopsProfiler(model)

    warm_up_model(model, tokenizer, processor)

    if TFOPS_PROFILING:
        prof.start_profile()

    # save_dir 폴더명이 이미 존재하면 고유한 이름 생성 (save_dir -> save_dir_1 -> save_dir_2)
    save_dir = SAVE_DIR
    suffix = 0
    while os.path.exists(save_dir):
        suffix += 1
        save_dir = f"{SAVE_DIR}_{suffix}"
    os.makedirs(save_dir)

    # Process
    for task in TASKS:
        # 각 task별로 별도의 로그 파일 생성
        init_iter_logger(  
            save_dir=save_dir,
            csv_name=f"iter_log_{task}.csv",
            md_name=f"iter_log_{task}.md",
            headers=[  # 순서 그대로 들어감
                "idx", "orig_w", "orig_h", "resize_ratio",
                "num_crop", "crop_hit",
                "s1_time", "s1_tflops", "s1_hit", 
                "s2_time", "s2_tflops", "s2_hit", 
                "s3_time", "s3_hit",
                "total_time", "total_tflops", "peak_memory_gb", 
                "crop_acc_uptonow", "s1_acc_uptonow", "s2_acc_uptonow", "s3_acc_uptonow",
                "filename", "instruction"
            ],
            write_md=False, use_fsync=True, use_lock=True
        )
        task_res = dict()
        dataset = "screenspot_" + task + "_v2.json"
        screenspot_data = json.load(open(os.path.join(SCREENSPOT_JSON, dataset), 'r'))
        screenspot_data = screenspot_data[SAMPLE_RANGE]

        print("Num of sample: " + str(len(screenspot_data)), flush=True)

        # 통계 변수 초기화
        task_res = []
        num_action = 0
        s1_time_sum = s2_time_sum = s3_time_sum = 0.0
        s1_tflops_sum = s2_tflops_sum = 0.0
        stage1_success_count = stage2_success_count = stage3_success_count = 0
        crop_success_count = 0  # 새로운 crop 성공 카운터 추가
        peak_memory_sum = 0.0  # 피크 메모리 합계 추가
        
        # data_source별 통계 변수 초기화
        data_source_stats = {}

        if MEMORY_VIS:
            memory_dir = os.path.join(save_dir, "gpu_usage", task)
            os.makedirs(memory_dir, exist_ok=True)

        for j, item in tqdm(enumerate(screenspot_data)):

            if MEMORY_EVAL:
                mem_log = []
                time_log = []
                stop_flag = False
                monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
                monitor_thread.start()
                torch.cuda.reset_peak_memory_stats()

            s1_tflops = s2_tflops = 0.0

            print("\n\n----------------------\n")

            num_action += 1
            
            # 파일 및 데이터 로드
            filename = item["img_filename"]
            filename_wo_ext, ext = os.path.splitext(filename)
            img_path = os.path.join(SCREENSPOT_IMGS, filename)
            if not os.path.exists(img_path):
                continue

            original_image = Image.open(img_path).convert("RGB")
            instruction = item["instruction"]
            original_bbox = item["bbox"]
            original_bbox = [original_bbox[0], original_bbox[1], 
                           original_bbox[0] + original_bbox[2], original_bbox[1] + original_bbox[3]]

            orig_w, orig_h = original_image.size

            # data_source 정보 추출 (없으면 "unknown"으로 기본값 설정)
            data_source = item.get("data_source", "unknown")

            #! ==================================================================
            #! Stage 1 | Attention-based Crop Generation
            #! ==================================================================

            if TFOPS_PROFILING:
                prof.reset_profile()

            s1_start = time.time()
            
            # 새로운 attention 기반 방식 (원본 이미지 사용)
            s1_pred, s1_crop_list, num_crops, resized_image, scaled_gt_bbox = run_stage1_attention_based(
                original_image=original_image,
                instruction=instruction,
                gt_bbox=original_bbox
            )
            
            s1_infence_end = time.time()
            s1_time = s1_infence_end - s1_start

            if TFOPS_PROFILING:
                s1_tflops = prof.get_total_flops() / 1e12

            # Stage1 Grounding 성공 여부 확인 (실제 예측 결과)
            s1_success = False
            s1_original_point = None
            if s1_pred and "topk_points" in s1_pred and s1_pred["topk_points"]:
                s1_predicted_point = s1_pred["topk_points"][0]  # 정규화된 좌표 (0~1)
                # 정규화된 좌표를 원본 이미지 픽셀 좌표로 변환
                s1_original_point = [
                    s1_predicted_point[0] * original_image.size[0],
                    s1_predicted_point[1] * original_image.size[1]
                ]
                s1_success = point_in_bbox(s1_original_point, original_bbox)
            
            s1_hit = "✅" if s1_success else "❌"
            if s1_success:
                stage1_success_count += 1

            # Crop 생성 성공 여부 확인 (GT가 생성된 crop들 중 하나라도 포함되는지 확인)
            crop_success = False
            if num_crops > 0:
                gt_center = [(original_bbox[0] + original_bbox[2])/2, (original_bbox[1] + original_bbox[3])/2]
                for crop in s1_crop_list:
                    crop_bbox = crop["bbox"]
                    if point_in_bbox(gt_center, crop_bbox):
                        crop_success = True
                        break
            
            crop_hit = "✅" if crop_success else "❌"
            if crop_success:
                crop_success_count += 1

            #! ==================================================================
            #! [Stage 2] Merged Crop Inference
            #! ==================================================================
            
            s2_tflops = 0.0
            s2_pred = s2_corrected_point = None  # 시각화용 변수 초기화 (multi-image 방식)
            stage2_success = False

            s2_inference_start = time.time()
            
            if TFOPS_PROFILING:
                prof.reset_profile()
            
            # 멀티 이미지로 inference
            s2_pred = run_stage2_multi_image_inference(s1_crop_list, instruction)

            # Stage2 multi-image 결과를 원본 좌표로 변환
            s2_best_candidate, s2_all_candidates = convert_multi_image_results_to_original(s2_pred, s1_crop_list)
            
            # Stage2 성공 여부 확인
            if s2_best_candidate:
                s2_corrected_point = s2_best_candidate['point']
                stage2_success = point_in_bbox(s2_corrected_point, original_bbox)
            else:
                s2_corrected_point = [0, 0]
                stage2_success = False


            s2_inference_end = time.time()
            s2_time = s2_inference_end - s2_inference_start
            
            if TFOPS_PROFILING:
                s2_tflops = prof.get_total_flops() / 1e12

            s2_hit = "✅" if stage2_success else "❌"
            if stage2_success:
                stage2_success_count += 1

            #! ==================================================================
            #! [Stage 3] Ensemble Processing
            #! ==================================================================
            
            s3_ensemble_point = None
            stage3_success = False
            
            # 크롭 면적이 원본 이미지 면적의 50%를 넘는지 확인
            crop_area = CROP_WIDTH * CROP_HEIGHT
            original_area = orig_w * orig_h
            crop_area_ratio = crop_area / original_area
            
            s3_start = time.time()
            # Multi-image 앙상블 방법: Stage2 multi-image 결과와 해당 위치의 Stage1 점수 조합
            # Stage1 어텐션 정보
            s1_attn_scores = np.array(s1_pred['attn_scores'][0])
            s1_n_width = s1_pred['n_width']
            s1_n_height = s1_pred['n_height']
            s1_resize_ratio = s1_pred['resize_ratio']
            
            # Stage1 attention 점수들 정규화 (1등 기준)
            s1_max_score = float(max(s1_attn_scores)) if len(s1_attn_scores) > 0 else 1.0
            
            # Stage2에서 topk 후보들만 선별 (run_gui_actor와 동일)
            if s2_all_candidates:
                # 점수 상위 10개만 선택 (run_gui_actor의 topk=10과 동일)
                s2_topk_candidates = sorted(s2_all_candidates, key=lambda x: x['score'], reverse=True)[:10]
                s2_topk_scores = [candidate['score'] for candidate in s2_topk_candidates]
                
                # topk 점수들만으로 정규화 (run_gui_actor와 동일)
                if s2_topk_scores:
                    s2_max_score = max(s2_topk_scores)
                    if s2_max_score > 0:
                        s2_normalized_scores = [score / s2_max_score for score in s2_topk_scores]
                    else:
                        s2_normalized_scores = [0.0] * len(s2_topk_scores)
                else:
                    s2_normalized_scores = []
            else:
                s2_topk_candidates = []
                s2_normalized_scores = []
            
            # 각 Stage2 topk 점에 대해 앙상블 점수 계산
            ensemble_candidates = []
            
            for i, candidate in enumerate(s2_topk_candidates):
                s2_original_point = candidate['point']
                
                # 해당 점에서의 Stage1 점수 계산 (정규화된 값)
                s1_raw_score = get_stage1_score_at_point(
                    s2_original_point, s1_attn_scores, s1_n_width, s1_n_height, 
                    original_image.size, s1_resize_ratio
                )
                s1_score = s1_raw_score / s1_max_score if s1_max_score > 0 else 0.0
                
                # Stage2 점수는 정규화된 점수 사용 (run_gui_actor와 동일)
                s2_score = s2_normalized_scores[i] if i < len(s2_normalized_scores) else 0.0
                
                # 앙상블 점수 계산
                ensemble_score = STAGE1_ENSEMBLE_RATIO * s1_score + STAGE2_ENSEMBLE_RATIO * s2_score
                
                ensemble_candidates.append({
                    'point': s2_original_point,
                    'score': ensemble_score,
                    's1_score': s1_score,
                    's2_score': s2_score,
                    'crop_id': candidate['crop_id'],
                    'rank_in_crop': candidate['rank_in_crop'],
                    's2_rank': i + 1  # topk 내에서의 순위
                })
            
            # 최고 점수를 가진 점 선택
            if ensemble_candidates:
                best_candidate = max(ensemble_candidates, key=lambda x: x['score'])
                s3_ensemble_point = best_candidate['point']
            else:
                # Stage2 결과가 없으면 Stage1 결과를 사용
                if s1_original_point:
                    s3_ensemble_point = s1_original_point
                else:
                    s3_ensemble_point = [0, 0]

            s3_end = time.time()
            s3_time = s3_end - s3_start
            
            # 디버그 정보 출력 (상위 3개만) - run_gui_actor와 동일한 형태
            if ensemble_candidates:
                print(f"🎯 Stage3 Ensemble Candidates (Top 3):")
                for i, candidate in enumerate(sorted(ensemble_candidates, key=lambda x: x['score'], reverse=True)[:3]):
                    print(f"  Rank {i+1}: S2_rank={candidate['s2_rank']}, S1={candidate['s1_score']:.3f}, S2={candidate['s2_score']:.3f}, Ensemble={candidate['score']:.3f}")
            else:
                print(f"🎯 Stage3: No ensemble candidates, using Stage1 result")
            
            # 시각화를 위해 후보들 저장
            s3_ensemble_candidates = ensemble_candidates
            
            # 앙상블 결과로 성공 여부 확인
            stage3_success = point_in_bbox(s3_ensemble_point, original_bbox)
            
            s3_hit = "✅" if stage3_success else "❌"
            if stage3_success:
                stage3_success_count += 1

            #! ==================================================================
            #! [Common Processing]
            #! ==================================================================
            
            # 공통 통계 업데이트
            s1_time_sum += s1_time
            s2_time_sum += s2_time
            s3_time_sum += s3_time
            s1_tflops_sum += s1_tflops
            s2_tflops_sum += s2_tflops
                
            # 성능 로깅
            total_time = s1_time + s2_time
            if TFOPS_PROFILING:
                total_tflops_this = s1_tflops + s2_tflops  # Stage3는 FLOPs 제외

            #! ==================================================================
            #! [Visualization - After Time Measurement]
            #! ==================================================================
            
            # 시각화용 디렉토리 설정 (stage3 결과에 따라)
            if VISUALIZE and (not VIS_ONLY_WRONG or not stage3_success):
                inst_dir_name = re.sub(r'\W+', '_', instruction).strip('_')
                result_folder = "correct" if stage3_success else "incorrect"
                inst_dir = os.path.join(save_dir, f"{task}_visualize_{result_folder}", f"{num_action}_{inst_dir_name}")

                # Stage1 시각화
                visualize_stage1_attention_crops(
                    s1_pred=s1_pred,
                    resized_image=resized_image, 
                    crop_list=s1_crop_list,
                    original_image=original_image,
                    save_dir=inst_dir,
                    instruction=instruction,
                    gt_bbox=original_bbox,
                    s1_predicted_point=s1_original_point
                )
                
                # Stage2 Multi-Image 시각화
                if s2_pred and s1_crop_list:  # Stage2 결과가 있을 때만 시각화
                    visualize_stage2_multi_attention(
                        s2_pred=s2_pred,
                        crop_list=s1_crop_list,
                        original_image=original_image,
                        save_dir=inst_dir,
                        instruction=instruction,
                        predicted_point=s2_corrected_point
                    )
                
                # Stage3 앙상블 시각화
                visualize_stage3_point_ensemble(
                    s3_ensemble_candidates=s3_ensemble_candidates if 's3_ensemble_candidates' in locals() else [],
                    original_image=original_image,
                    crop_list=s1_crop_list,
                    original_bbox=original_bbox,
                    s3_ensemble_point=s3_ensemble_point,
                    s2_corrected_point=s2_corrected_point,
                    s1_original_point=s1_original_point,
                    stage1_ratio=STAGE1_ENSEMBLE_RATIO,
                    stage2_ratio=STAGE2_ENSEMBLE_RATIO,
                    save_dir=inst_dir,
                    vis_only_wrong=VIS_ONLY_WRONG,
                    stage3_success=stage3_success
                )


            num_attention_crops = len(s1_crop_list)
            print(f"✂️  Attention Crops : {num_attention_crops}")
            print(f"🕖 Times - S1: {s1_time:.2f}s | S2: {s2_time:.2f}s | Total: {total_time:.2f}s")
            if TFOPS_PROFILING:
                print(f"🔥 FLOPs - S1: {s1_tflops:.2f} | S2: {s2_tflops:.2f} | Total: {total_tflops_this:.2f} TFLOPs")
            print(f"{'✅ Success' if stage3_success else '❌🎯 Fail'}")

            #! ==================================================================
            #! [End]
            #! ==================================================================

            if MEMORY_EVAL:
                time.sleep(0.1)
                stop_flag = True
                monitor_thread.join()

                # 피크 메모리 계산 (GB 단위, 소수점 3자리)
                peak_memory = max(mem_log) if mem_log else 0.0
                peak_memory_gb = round(peak_memory, 3)

                if MEMORY_VIS:
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(10, 4))
                    plt.plot(time_log, mem_log)
                    plt.xlabel("Time (s)")
                    plt.ylabel("GPU Memory Allocated (GB)")
                    plt.title("GPU Memory Usage Over Time")
                    plt.grid(True)
                    plt.savefig(f"{memory_dir}/{num_action}_{filename}")
                    plt.close()  # 메모리 누수 방지를 위해 close 추가

            if MEMORY_EVAL:
                peak_memory_sum += peak_memory_gb

            # data_source별 통계 업데이트
            if data_source not in data_source_stats:
                data_source_stats[data_source] = {
                    'num_action': 0,
                    's1_time_sum': 0.0,
                    's2_time_sum': 0.0,
                    's3_time_sum': 0.0,
                    's1_tflops_sum': 0.0,
                    's2_tflops_sum': 0.0,
                    'total_tflops': 0.0,
                    'stage1_success_count': 0,
                    'crop_success_count': 0,
                    'stage2_success_count': 0,
                    'stage3_success_count': 0,
                    'peak_memory_sum': 0.0
                }
            
            stats = data_source_stats[data_source]
            stats['num_action'] += 1
            stats['s1_time_sum'] += s1_time
            stats['s2_time_sum'] += s2_time
            stats['s3_time_sum'] += s3_time
            if TFOPS_PROFILING:
                stats['s1_tflops_sum'] += s1_tflops
                stats['s2_tflops_sum'] += s2_tflops
                stats['total_tflops'] += total_tflops_this
            if MEMORY_EVAL:
                stats['peak_memory_sum'] += peak_memory_gb
            if s1_success:
                stats['stage1_success_count'] += 1
            if crop_success:
                stats['crop_success_count'] += 1
            if stage2_success:
                stats['stage2_success_count'] += 1
            if stage3_success:
                stats['stage3_success_count'] += 1

            up2now_s1_score = stage1_success_count / num_action * 100
            up2now_crop_score = crop_success_count / num_action * 100
            up2now_s2_score = stage2_success_count / num_action * 100
            up2now_s3_ensemble_score = stage3_success_count / num_action * 100
            # print(f"Up2Now Crop Accuracy: {up2now_crop_score:.2f}%")
            print(f"Up2Now Stage1 Accuracy: {up2now_s1_score:.2f}%")
            print(f"Up2Now Stage2 Accuracy: {up2now_s2_score:.2f}%")
            print(f"Up2Now Stage3 Ensemble Accuracy: {up2now_s3_ensemble_score:.2f}%")

            # Iter log - 개선된 로깅
            append_iter_log(
                idx=j+1,
                orig_w=original_image.size[0],
                orig_h=original_image.size[1],
                resize_ratio=s1_pred['resize_ratio'],
                num_crop=num_attention_crops,
                crop_hit=crop_hit,
                s1_time=f"{s1_time:.3f}",
                s1_tflops=f"{s1_tflops:.2f}",
                s1_hit=s1_hit,
                s2_time=f"{s2_time:.3f}",
                s2_tflops=f"{s2_tflops:.2f}",
                s2_hit=s2_hit,
                s3_time=f"{s3_time:.3f}",
                s3_hit=s3_hit,
                total_time=f"{total_time:.3f}",
                total_tflops=f"{total_tflops_this:.2f}",
                peak_memory_gb=f"{peak_memory_gb:.3f}" if MEMORY_EVAL else "N/A",
                crop_acc_uptonow=f"{up2now_crop_score:.2f}",
                s1_acc_uptonow=f"{up2now_s1_score:.2f}",
                s2_acc_uptonow=f"{up2now_s2_score:.2f}",
                s3_acc_uptonow=f"{up2now_s3_ensemble_score:.2f}",
                filename=filename_wo_ext,
                instruction=instruction[:50] + "..." if len(instruction) > 50 else instruction
            )

            # JSON 기록 - 핵심 정보만
            item_res = {
                'filename': filename,
                'orig_w': original_image.size[0],
                'orig_h': original_image.size[1],
                'instruction': instruction,
                'gt_bbox': original_bbox,
                'data_source': data_source,
                'num_crop': num_attention_crops,
                'crop_success': crop_success,
                'stage1_success': s1_success,
                'stage2_success': stage2_success,
                'stage3_success': stage3_success,
                's1_hit': s1_hit,
                'crop_hit': crop_hit,
                's2_hit': s2_hit,
                's3_hit': s3_hit,
                's3_ensemble_point': s3_ensemble_point,
                's1_original_point': s1_original_point,
                's2_original_point': s2_corrected_point,
                's1_time': s1_time,
                's2_time': s2_time,
                's3_time': s3_time,
                'total_time': total_time,
                's1_tflops': s1_tflops,
                's2_tflops': s2_tflops,
                'total_tflops': s1_tflops+s2_tflops,
                'peak_memory_gb': peak_memory_gb if MEMORY_EVAL else None,
                'ensemble_config': {
                    'attention_ratio': STAGE1_ENSEMBLE_RATIO,
                    'crop_ratio': STAGE2_ENSEMBLE_RATIO
                }
            }
            task_res.append(item_res)

        #! ==================================================
        # 결과 Json 정리
        os.makedirs(os.path.join(save_dir, "json"), exist_ok=True)
        with open(os.path.join(save_dir, "json", dataset), "w") as f:
            json.dump(task_res, f, indent=4, ensure_ascii=False, cls=NpEncoder)

        # 최종 성능 메트릭 계산
        metrics = {
            "task": task,
            "total_samples": num_action,
            "crop_accuracy": crop_success_count / num_action * 100,
            "stage1_accuracy": stage1_success_count / num_action * 100,
            "stage2_accuracy": stage2_success_count / num_action * 100,
            "stage3_accuracy": stage3_success_count / num_action * 100,
            "avg_times": {
                "stage1": s1_time_sum / num_action,
                "stage2": s2_time_sum / num_action,
                "stage3": s3_time_sum / num_action,
                "total": (s1_time_sum + s2_time_sum + s3_time_sum) / num_action
            },
            "avg_flops_tflops": {
                "stage1": s1_tflops_sum / num_action,
                "stage2": s2_tflops_sum / num_action,
                "total": (s1_tflops_sum + s2_tflops_sum) / num_action
            },
            "avg_peak_memory_gb": round(peak_memory_sum / num_action, 3) if MEMORY_EVAL else None,
            "hyperparameters": {
                "select_threshold": SELECT_THRESHOLD,
                "attn_impl": ATTN_IMPL,
                "STAGE1_ensemble_ratio": STAGE1_ENSEMBLE_RATIO,
                "STAGE2_ensemble_ratio": STAGE2_ENSEMBLE_RATIO
            }
        }

        with open(os.path.join(save_dir, f"results_{task}.json"), "w") as mf:
            json.dump(metrics, mf, ensure_ascii=False, indent=4)

        # data_source별 메트릭 저장
        data_source_metrics = {}
        for ds, stats in data_source_stats.items():
            if stats['num_action'] > 0:
                data_source_metrics[ds] = {
                    "task": task,
                    "data_source": ds,
                    "total_samples": stats['num_action'],
                    "crop_accuracy": stats['crop_success_count'] / stats['num_action'] * 100,
                    "stage1_accuracy": stats['stage1_success_count'] / stats['num_action'] * 100,
                    "stage2_accuracy": stats['stage2_success_count'] / stats['num_action'] * 100,
                    "stage3_accuracy": stats['stage3_success_count'] / stats['num_action'] * 100,
                    "avg_times": {
                        "stage1": stats['s1_time_sum'] / stats['num_action'],
                        "stage2": stats['s2_time_sum'] / stats['num_action'],
                        "stage3": stats['s3_time_sum'] / stats['num_action'],
                        "total": (stats['s1_time_sum'] + stats['s2_time_sum'] + stats['s3_time_sum']) / stats['num_action']
                    },
                    "avg_flops_tflops": {
                        "stage1": stats['s1_tflops_sum'] / stats['num_action'],
                        "stage2": stats['s2_tflops_sum'] / stats['num_action'],
                        "total": (stats['s1_tflops_sum'] + stats['s2_tflops_sum']) / stats['num_action']
                    },
                    "avg_peak_memory_gb": round(stats['peak_memory_sum'] / stats['num_action'], 3) if MEMORY_EVAL else None,
                    "hyperparameters": {
                        "select_threshold": SELECT_THRESHOLD,
                        "attn_impl": ATTN_IMPL,
                        "STAGE1_ensemble_ratio": STAGE1_ENSEMBLE_RATIO,
                        "STAGE2_ensemble_ratio": STAGE2_ENSEMBLE_RATIO
                    }
                }
        
        with open(os.path.join(save_dir, f"source_results_{task}.json"), "w") as dsf:
            json.dump(data_source_metrics, dsf, ensure_ascii=False, indent=4)

        # 전체 결과를 CSV 파일에 한 줄 추가
        results_csv_path = "../_results"
        os.makedirs(results_csv_path, exist_ok=True)
        csv_file_path = os.path.join(results_csv_path, f"results_{task}.csv")
        
        # CSV 헤더 정의
        csv_headers = [
            "method",
            "min_resize", "max_resize", "select_threshold", "stage1_ensemble_ratio", "crop_width", "crop_height",
            "total_samples", "crop_accuracy", "stage1_accuracy", "stage2_accuracy", "stage3_accuracy",
            "avg_stage1_time", "avg_stage2_time", "avg_stage3_time", "avg_total_time",
            "avg_stage1_tflops", "avg_stage2_tflops", "avg_total_tflops", "avg_peak_memory_gb",
            "timestamp"
        ]
        
        # CSV 데이터 행 생성
        import datetime
        csv_row = [
            method,
            MIN_RESIZE, MAX_RESIZE, SELECT_THRESHOLD, STAGE1_ENSEMBLE_RATIO, CROP_WIDTH, CROP_HEIGHT,
            num_action, 
            round(metrics['crop_accuracy'], 2),
            round(metrics['stage1_accuracy'], 2),
            round(metrics['stage2_accuracy'], 2), 
            round(metrics['stage3_accuracy'], 2),
            round(metrics['avg_times']['stage1'], 4),
            round(metrics['avg_times']['stage2'], 4),
            round(metrics['avg_times']['stage3'], 4),
            round(metrics['avg_times']['total'], 4),
            round(metrics['avg_flops_tflops']['stage1'], 2),
            round(metrics['avg_flops_tflops']['stage2'], 2),
            round(metrics['avg_flops_tflops']['total'], 2),
            metrics['avg_peak_memory_gb'] if metrics['avg_peak_memory_gb'] else 0.0,
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ]
        
        # CSV 파일이 없으면 헤더와 함께 생성, 있으면 데이터 행만 추가
        import csv
        file_exists = os.path.exists(csv_file_path)
        
        with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # 파일이 없거나 비어있으면 헤더 추가
            if not file_exists or os.path.getsize(csv_file_path) == 0:
                writer.writerow(csv_headers)
            
            # 데이터 행 추가
            writer.writerow(csv_row)
        
        print(f"📝 Results saved to CSV: {csv_file_path}")

        # 최종 결과 출력
        print("=" * 60)
        print(f"📊 Final Results for {task}:")
        print(f"Total Samples: {num_action}")
        print(f"Crop Accuracy: {metrics['crop_accuracy']:.2f}%")
        print(f"Stage1 Accuracy: {metrics['stage1_accuracy']:.2f}%")
        print(f"Stage2 Accuracy: {metrics['stage2_accuracy']:.2f}%")
        print(f"Stage3 Ensemble Accuracy: {metrics['stage3_accuracy']:.2f}%")
        print(f"Avg Times: S1 {metrics['avg_times']['stage1']:.3f}s | S2 {metrics['avg_times']['stage2']:.3f}s | S3 {metrics['avg_times']['stage3']:.3f}s | Total {metrics['avg_times']['total']:.3f}s")
        print(f"Avg FLOPs: S1 {metrics['avg_flops_tflops']['stage1']:.2f} | S2 {metrics['avg_flops_tflops']['stage2']:.2f} | Total {metrics['avg_flops_tflops']['total']:.2f} TFLOPs")
        if MEMORY_EVAL and metrics['avg_peak_memory_gb'] is not None:
            print(f"Avg Peak Memory: {metrics['avg_peak_memory_gb']:.3f} GB")
        print(f"Ensemble Config: Attention {STAGE1_ENSEMBLE_RATIO:.1f}, Crop {STAGE2_ENSEMBLE_RATIO:.1f}")
        
        print("=" * 60)