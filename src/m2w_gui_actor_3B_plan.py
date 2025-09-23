'''
Final version
'''

# _query 이걸로 모델 추론하면 되는거

DIR_MAX_LEN = 100

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('gpu', type=int, default=0, help='GPU number')
parser.add_argument('--r', type=float, default=0.50, help='Stage 1 Resize ratio')
parser.add_argument('--th', type=float, default=0.11, help='Stage 1 Crop threshold')
parser.add_argument('--p', type=int, default=20, help='Stage 1 Crop Padding')
parser.add_argument('--v', action='store_true', help='Whether to save visualization images')
args = parser.parse_args()


# dataset = load_dataset("demisama/UGround-Offline-Evaluation") 이걸로 블락 불러오기



os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

#! Hyperparameter =====================================================================================

ATTN_IMPL = "eager"                      # attention implement "eager" "sdpa" "flash" "efficient"

# Image Resize Ratios
RESIZE_RATIO = args.r

# Crop Limitations
MAX_CROPS = 3  # 최대 crop 개수

# Connected Region Based Cropping
REGION_THRESHOLD = args.th              # 연결된 영역 검출을 위한 임계값 (0~1)  # TODO: 0.1 ~ 0.5 중 최적 찾기
MIN_PATCHES = 1                         # 최소 패치 수 (너무 작은 영역 제거)
BBOX_PADDING = args.p                   # bbox 상하좌우로 확장할 픽셀  # TODO: 0 ~ 50 중 최적 찾기

# Ensemble Hyperparameters
STAGE1_ENSEMBLE_RATIO = 0.50                        # Stage1 attention 가중치
STAGE2_ENSEMBLE_RATIO = 1 - STAGE1_ENSEMBLE_RATIO   # Stage2 crop 가중치
ENSEMBLE_TOP_PATCHES = 100                          # Stage2에서 앙상블에 사용할 상위 패치 개수 (Qwen2.5VL용)

# 최대 PIXELS 제한
MAX_PIXELS = 3211264  # Process단에서 적용

# csv에 기록할 method 이름
method = "mind2web_0921"

memo = f"resize{RESIZE_RATIO:.2f}_region_thresh{REGION_THRESHOLD:.2f}_pad{BBOX_PADDING}"

#! Argument ==========================================================================================

SEED = 0

# Dataset & Model
MLLM_PATH = "microsoft/GUI-Actor-3B-Qwen2.5-VL"
SCREENSPOT_IMGS = "/data/hj/UGround/offline_evaluation/Multimodal-Mind2Web/data/sample/cross_"       # input image 경로
SCREENSPOT_JSON = "../data"                          # input image json파일 경로
TASKS = ["domain", "website", "task"]
SAMPLE_RANGE = slice(None)

# Visualize & Logging
VISUALIZE = args.v if args.v else True  # 시각화 여부
VIS_ONLY_WRONG = False                                # True면 틀린 것만 시각화, False면 모든 것 시각화
TFOPS_PROFILING = True
MEMORY_VIS = False

# Save Path
SAVE_DIR = f"../attn_output/" + method + "/" + memo + "mind2web_plan_real_final_gpt4_turbo"  # 결과 저장 경로

#! ==================================================================================================

# Standard Library
import os
import sys
import time
import re
import json
import logging
logging.disable(logging.CRITICAL)  # 모든 로깅 호출 무력화
from typing import Dict, List
from collections import deque

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
from util.visualize_util_mind2web  import visualize_stage1_attention_crops, visualize_stage2_multi_attention, visualize_stage3_point_ensemble
if TFOPS_PROFILING:
    from deepspeed.profiling.flops_profiler import FlopsProfiler

from datasets import load_dataset, DatasetDict

#! ==============================================================================================

# def resize_image(image, resize_to_pixels=MAX_PIXELS):
#     image_width, image_height = image.size
#     if (resize_to_pixels is not None) and ((image_width * image_height) != resize_to_pixels):
#         resize_ratio = (resize_to_pixels / (image_width * image_height)) ** 0.5
#         image_width_resized, image_height_resized = int(image_width * resize_ratio), int(image_height * resize_ratio)
#         image = image.resize((image_width_resized, image_height_resized))
#     return image, image_width_resized, image_height_resized

def resize_images(croplist, resize_to_pixels=MAX_PIXELS):

    total_size = 0
    # 1. 전체 픽셀 수 계산
    for crop in croplist:
        img = crop['img']
        image_width, image_height = img.size
        total_size += image_width * image_height

    # 2. 픽셀 수가 최대치를 초과할 경우에만 리사이즈 수행
    if total_size > resize_to_pixels:
        # 리사이즈 비율 계산 (면적 비율의 제곱근)
        resize_ratio = (resize_to_pixels / total_size) ** 0.5
        
        resized_croplist = []
        for crop in croplist:
            original_img = crop['img']
            original_width, original_height = original_img.size
            
            # 새로운 크기로 이미지 리사이즈
            resized_img = original_img.resize(
                (int(original_width * resize_ratio), int(original_height * resize_ratio))
            )
            
            # 원본 crop 딕셔너리의 복사본을 만들고 'img' 키 값을 업데이트
            new_crop = crop.copy()
            new_crop['img'] = resized_img
            resized_croplist.append(new_crop)
            
        return resized_croplist # ✅ FIX: 리사이즈된 이미지가 포함된 새 리스트를 반환
    
    # 3. 리사이즈가 필요 없으면 원본 그대로 반환
    return croplist


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
                    "text": (
                        # 추가 content
                        f"This is a resized screenshot of the whole GUI, scaled by {resize_ratio}. "
                        # 기존 content
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

def create_conversation_stage2(crop_list, instruction):
    user_content = []
    for crop in crop_list:
        user_content.append({"type": "image", "image": crop["img"]})
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
                        # 추가 content
                        f"This is a list of {len(crop_list)} cropped screenshots of the GUI, each showing a part of the GUI. "
                        # 기존 content
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



# 
def truncate_to_bytes(filename: str, max_bytes: int) -> str:
    """
    문자열을 지정된 바이트 길이를 넘지 않도록 자릅니다.
    멀티바이트 문자가 깨지지 않도록 안전하게 처리합니다.
    """
    # 1. 문자열을 UTF-8 바이트로 인코딩
    encoded_bytes = filename.encode('utf-8')

    # 2. 바이트 길이가 최대 길이를 넘으면 자르기
    if len(encoded_bytes) <= max_bytes:
        return filename

    truncated_bytes = encoded_bytes[:max_bytes]

    # 3. 디코딩 에러가 발생하지 않을 때까지 마지막 바이트를 제거하며 시도
    while True:
        try:
            # 디코딩 성공 시, 해당 문자열을 반환하고 종료
            return truncated_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # 디코딩 실패 시 (문자 중간에 잘린 경우), 마지막 바이트를 하나 제거
            truncated_bytes = truncated_bytes[:-1]

# 

def get_connected_region_bboxes_from_scores(
    image_result: Dict,
    threshold: float,
    min_patches: int = 1
) -> List[Dict]:
    '''
    간단한 버전: attention * threshold 넘는 부분들을 8방향 연결로 합쳐서 box 생성
    '''
    # 1) 입력 파싱 및 임계값 계산
    attn_scores_1d = np.array(image_result["attn_scores"][0], dtype=np.float32)
    n_w = int(image_result["n_width"])
    n_h = int(image_result["n_height"])
    attn = attn_scores_1d.reshape(n_h, n_w)
    
    vmax = float(attn.max()) if attn.size > 0 else 0.0
    thr_val = float(vmax * threshold) if threshold <= 1.0 else float(threshold)
    
    # 2) 기준 넘는 패치들 마스크 생성
    mask = (attn >= thr_val)
    
    # 3) BFS로 연결된 영역들 찾기
    visited = np.zeros_like(mask, dtype=bool)
    regions = []
    neighbors = [(di, dj) for di in (-1,0,1) for dj in (-1,0,1) if not (di==0 and dj==0)]  # 8방향
    # neighbors =   # TODO: 4방향 비교
    
    for y in range(n_h):
        for x in range(n_w):
            if not mask[y, x] or visited[y, x]:
                continue
                
            # BFS로 연결된 영역 찾기
            region = [(y, x)]
            queue = deque([(y, x)])
            visited[y, x] = True
            
            while queue:
                cy, cx = queue.popleft()
                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if (0 <= ny < n_h and 0 <= nx < n_w and 
                        mask[ny, nx] and not visited[ny, nx]):
                        visited[ny, nx] = True
                        queue.append((ny, nx))
                        region.append((ny, nx))
            
            if len(region) >= min_patches:
                regions.append(region)
    
    # 4) 각 영역의 bbox와 점수 계산
    out = []
    for region in regions:
        ys = [p[0] for p in region]
        xs = [p[1] for p in region]
        y_min, y_max = min(ys), max(ys)
        x_min, x_max = min(xs), max(xs)
        
        # 정규화된 bbox
        l = x_min / n_w
        t = y_min / n_h  
        r = (x_max + 1) / n_w
        b = (y_max + 1) / n_h
        
        # 점수 계산
        region_scores = attn[ys, xs]
        score_sum = float(region_scores.sum())
        score_mean = float(region_scores.mean())
        
        out.append({
            "bbox": [l, t, r, b],
            "patch_bbox": [int(x_min), int(y_min), int(x_max), int(y_max)],
            "size": int(len(region)),
            "score_sum": score_sum,
            "score_mean": score_mean,
            "score_norm": score_sum / (vmax * len(region) + 1e-9),
        })
    
    # 5) 점수순 정렬
    out.sort(key=lambda x: x["score_sum"], reverse=True)
    return out

def run_stage1_attention_inference(original_image, instruction):
    """Stage 1: 리사이즈하고 inference"""

    orig_w, orig_h = original_image.size
    # 이미지 고정 리사이즈
    resize_ratio = RESIZE_RATIO
    resized_w, resized_h = int(orig_w * resize_ratio), int(orig_h * resize_ratio)
    print(f"🔧 Resized image: {orig_w}x{orig_h} -> {resized_w}x{resized_h} (ratio: {resize_ratio:.3f})")
    
    # 리사이즈된 이미지로 inference
    resized_image = original_image.resize((resized_w, resized_h))
    conversation = create_conversation_stage1(resized_image, instruction, resize_ratio)
    pred = inference(conversation, model, tokenizer, processor, use_placeholder=True, topk=1)
    
    # 결과에 리사이즈 정보 추가
    pred['resize_ratio'] = resize_ratio
    pred['original_size'] = (orig_w, orig_h)
    pred['resized_size'] = resized_image.size
    
    return pred, resized_image

def find_connected_regions(pred_result, resized_image, resize_ratio):
    """어텐션에서 연결된 영역들 찾기"""

    regions = get_connected_region_bboxes_from_scores(
        image_result=pred_result,
        threshold=REGION_THRESHOLD,
        min_patches=MIN_PATCHES
    )
    
    resized_w, resized_h = resized_image.size
    orig_w = resized_w / resize_ratio
    orig_h = resized_h / resize_ratio
    
    # 각 영역을 원본 이미지 크기로 변환하고 정보 구성
    connected_regions = []
    for i, region in enumerate(regions):
        # 정규화된 bbox를 리사이즈된 이미지 픽셀 좌표로 변환
        l, t, r, b = region["bbox"]  # 정규화된 좌표 (0~1)
        
        # 리사이즈된 이미지에서의 픽셀 좌표
        resized_left = l * resized_w
        resized_top = t * resized_h
        resized_right = r * resized_w
        resized_bottom = b * resized_h
        
        # 원본 이미지 크기로 변환
        orig_left = resized_left / resize_ratio
        orig_top = resized_top / resize_ratio
        orig_right = resized_right / resize_ratio
        orig_bottom = resized_bottom / resize_ratio
        
        # bbox에 패딩 적용
        padded_left = max(0, int(orig_left - BBOX_PADDING))
        padded_top = max(0, int(orig_top - BBOX_PADDING))
        padded_right = min(orig_w, int(orig_right + BBOX_PADDING))
        padded_bottom = min(orig_h, int(orig_bottom + BBOX_PADDING))
        
        # 영역 중심점 계산 (패딩 적용 전 bbox 기준)
        center_x = (orig_left + orig_right) / 2
        center_y = (orig_top + orig_bottom) / 2
        
        connected_regions.append({
            'center_x': center_x,
            'center_y': center_y,
            'score': region["score_sum"],  # 영역 내 점수 합
            'score_mean': region["score_mean"],  # 영역 내 점수 평균
            'size': region["size"],  # 패치 수
            'bbox_original': [int(orig_left), int(orig_top), int(orig_right), int(orig_bottom)],  # 패딩 전 bbox
            'bbox_padded': [padded_left, padded_top, padded_right, padded_bottom],  # 패딩 후 bbox (실제 크롭용)
            'region_info': region  # 원본 영역 정보
        })
    
    # 점수 합이 높은 순서로 정렬
    connected_regions.sort(key=lambda x: x['score'], reverse=True)
    
    return connected_regions

def create_crops_from_connected_regions(regions, original_image):
    """연결된 영역들을 기반으로 원본 이미지에서 직접 crop"""
    
    if not regions:
        return []
    
    crops = []
    
    for i, region in enumerate(regions):
        bbox = region['bbox_padded']  # 패딩이 적용된 bbox 사용
        crop_img = original_image.crop(bbox)
        
        crops.append({
            'img': crop_img,
            'bbox': bbox,
            'score': region['score'],
            'id': i + 1,
            'region_info': region  # 원본 영역 정보 포함
        })
    
    return crops

def run_stage2_multi_image_inference(crop_list, instruction):
    """Stage 2: multi image inference - 각 crop별로 개별 inference"""


    
    # multi image inference용 대화 생성
    conversation = create_conversation_stage2(crop_list, instruction)
    
    # multi image inference 실행 (각 이미지별 결과 반환)
    pred = multi_image_inference(conversation, model, tokenizer, processor, use_placeholder=True, topk=10)
    
    return pred

def convert_multi_image_results_to_original(multi_pred, crop_list):
    """multi_image_inference 결과를 원본 이미지 좌표로 변환"""
    
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
    
    # 모든 후보들을 점수순으로 정렬
    all_candidates.sort(key=lambda x: x['score'], reverse=True)
    
    return all_candidates

def run_stage1_attention_based(original_image, instruction, gt_bbox):
    """새로운 간단한 Stage 1: 연결된 영역 기반 crop 생성"""
    
    # 1. 리사이즈하고 inference
    print("🔍 Stage 1: Running attention-based inference...")
    s1_pred, resized_image = run_stage1_attention_inference(original_image, instruction)
    
    # 2. GT bbox도 리사이즈 비율에 맞춰 조정
    resize_ratio = s1_pred['resize_ratio']
    scaled_gt_bbox = [coord * resize_ratio for coord in gt_bbox] if gt_bbox else None
    
    # 3. 연결된 영역들 찾기
    regions = find_connected_regions(s1_pred, resized_image, resize_ratio)

    regions = regions[:MAX_CROPS]
    
    # 5. 원본 이미지에서 직접 crop 생성
    crops = create_crops_from_connected_regions(regions, original_image)
    
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

    # 전체 task 통계 변수 초기화
    total_samples = 0
    total_crop_success = 0
    total_stage1_success = 0
    total_stage2_success = 0
    total_stage3_success = 0
    total_s1_time = 0.0
    total_s2_time = 0.0
    total_s3_time = 0.0
    total_s1_tflops = 0.0
    total_s2_tflops = 0.0

    # CSV 헤더 정의 (모든 task에서 공통 사용)
    csv_headers = [
        "method",
        "resize_ratio", "region_threshold", "bbox_padding",
        "total_samples", "crop_accuracy", "stage1_accuracy", "stage2_accuracy", "stage3_accuracy",
        "avg_stage1_time", "avg_stage2_time", "avg_stage3_time", "avg_total_time",
        "avg_stage1_tflops", "avg_stage2_tflops", "avg_total_tflops",
        "timestamp"
    ]

# import json

# data = []
# # 'cross_domain_plan.jsonl' 파일을 'r'(읽기) 모드로 엽니다.
# # utf-8 인코딩을 명시해주는 것이 좋습니다.
# with open('cross_domain_plan.jsonl', 'r', encoding='utf-8') as f:
#     for line in f:
#         # 각 줄(line)을 JSON 객체로 변환하여 data 리스트에 추가합니다.
#         data.append(json.loads(line))

# # 처음 5개의 데이터를 출력하여 확인합니다.
# for item in data[:5]:
#     print(item)

# # 전체 데이터의 개수를 확인합니다.
# print(f"총 {len(data)}개의 데이터가 로드되었습니다.")


    # Data loading setup
    data_folder = '/data/hj/UGround/offline_evaluation/Multimodal-Mind2Web/data/gpt-4o_results/'
    data_folder = '/data/hj/UGround/offline_evaluation/Multimodal-Mind2Web/data/gpt-4-turbo_results/'

    
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
                "total_time", "total_tflops",
                "crop_acc_uptonow", "s1_acc_uptonow", "s2_acc_uptonow", "s3_acc_uptonow",
                "filename", "instruction"
            ],
            write_md=False, use_fsync=True, use_lock=True
        )
        task_res = dict()
        dataset = "screenspot_" + task + "_v2.json"



        #data 불러오기
        data = []
        target = data_folder + 'cross_' + task + '_query.jsonl'
        with open(target, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))

        screenspot_data = data[SAMPLE_RANGE]

        print("Num of sample: " + str(len(screenspot_data)), flush=True)

        # 통계 변수 초기화
        task_res = []
        num_action = 0
        s1_time_sum = s2_time_sum = s3_time_sum = 0.0
        s1_tflops_sum = s2_tflops_sum = 0.0
        crop_success_count = stage1_success_count = stage2_success_count = stage3_success_count = 0
        
        # data_source별 통계 변수 초기화
        data_source_stats = {}

        if MEMORY_VIS:
            memory_dir = os.path.join(save_dir, "gpu_usage", task)
            os.makedirs(memory_dir, exist_ok=True)

        for j, item in tqdm(enumerate(screenspot_data)):

            s1_tflops = s2_tflops = 0.0
            num_action += 1

            print("\n\n----------------------\n")
            
            # 파일 및 데이터 로드
            filename = item["image"]
            filename_wo_ext, ext = os.path.splitext(filename)

            blockpath='/data/hj/UGround/offline_evaluation/Multimodal-Mind2Web/data/blocks_dir/'

            img_path = os.path.join(blockpath, filename)

            ###HERE!!
            if not os.path.exists(img_path):
                print("없는데요", item)
                continue

            original_image = Image.open(img_path).convert("RGB")
            instruction = item["description"]

            pos_candidates_list = []
            raw_bbox = item['bbox']
            ground_truth_bboxes = [ [x[0], x[1], x[0]+x[2], x[1]+x[3]] for x in raw_bbox]
            
            orig_w, orig_h = original_image.size

            # data_source 정보 추출 (없으면 "unknown"으로 기본값 설정)
            data_source = item.get("data_source", "unknown")

            if not ground_truth_bboxes:
                ground_truth_bboxes.append([0,0,0,0])  # 임시로 막아둠.
                print("No GT bbox!!", j)
            if len(ground_truth_bboxes) > 1:
                print("Multiple GT bbox!!", j)

            #! ==================================================================
            #! Stage 1 | Attention-based Crop Generation
            #! ==================================================================

            if TFOPS_PROFILING:
                prof.reset_profile()

            s1_start = time.time()
            
            original_bbox = ground_truth_bboxes[0] # 임시로 막아둠.
            
            s1_pred, s1_crop_list, num_crops, resized_image, scaled_gt_bbox = run_stage1_attention_based(
                original_image=original_image,
                instruction=instruction,
                gt_bbox=original_bbox
            )

            s1_end = time.time()
            s1_time = s1_end - s1_start

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
                s1_success = any(point_in_bbox(s1_original_point, bbox) for bbox in ground_truth_bboxes)
            
            
            s1_hit = "✅" if s1_success else "❌"
            if s1_success:
                stage1_success_count += 1

            # GT bbox와 crop bbox가 겹치는지 확인 (교집합이 있으면 성공)
            crop_success = False
            for crop in s1_crop_list:
                if crop_success: break # 이미 성공했으면 더 확인할 필요 없음
                crop_bbox = crop["bbox"]
                for gt_bbox in ground_truth_bboxes:
                    left = max(crop_bbox[0], gt_bbox[0])
                    top = max(crop_bbox[1], gt_bbox[1])
                    right = min(crop_bbox[2], gt_bbox[2])
                    bottom = min(crop_bbox[3], gt_bbox[3])
                    if left < right and top < bottom:
                        crop_success = True
                        break
            
            crop_hit = "✅" if crop_success else "❌"
            if crop_success:
                crop_success_count += 1

            #! ==================================================================
            #! [Stage 2] Crop Inference
            #! ==================================================================
            
            # 이미지 리사이즈 # 이거 너무커서
            print(s1_crop_list)
            # s1_crop_list = resize_images(s1_crop_list, resize_to_pixels=MAX_PIXELS)

            s2_tflops = 0.0
            if TFOPS_PROFILING:
                prof.reset_profile()

            
            
            s2_inference_start = time.time()

            
            
            # 멀티 이미지로 inference
            s2_pred = run_stage2_multi_image_inference(s1_crop_list, instruction)



            # Stage2 multi-image 결과를 원본 좌표로 변환
            s2_all_candidates = convert_multi_image_results_to_original(s2_pred, s1_crop_list)
            
            # Stage2 성공 여부 확인
            s2_corrected_point = s2_all_candidates[0]['point'] if s2_all_candidates else None
            stage2_success = any(point_in_bbox(s2_corrected_point, bbox) for bbox in ground_truth_bboxes)


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
            
            s3_start = time.time()
            # Stage1 어텐션 정보
            s1_attn_scores = np.array(s1_pred['attn_scores'][0])
            s1_n_width = s1_pred['n_width']
            s1_n_height = s1_pred['n_height']
            s1_resize_ratio = s1_pred['resize_ratio']
            
            # Stage1 attention 최고점수 구하기
            s1_max_score = float(max(s1_attn_scores)) if len(s1_attn_scores) > 0 else 1.0
            
            # Stage2에서 topk 후보 최고점수 구하기
            s2_topk_scores = [candidate['score'] for candidate in s2_all_candidates]
            s2_max_score = max(s2_topk_scores)

            # 각 Stage2 topk 점에 대해 앙상블 점수 계산
            ensemble_candidates = []
            
            for i, candidate in enumerate(s2_all_candidates):
                s2_original_point = candidate['point']
                
                # 해당 점에서의 Stage1 점수 계산 (정규화된 값)
                s1_raw_score = get_stage1_score_at_point(
                    s2_original_point, s1_attn_scores, s1_n_width, s1_n_height, 
                    original_image.size, s1_resize_ratio
                )

                # 각 점수 최고점 기준으로 정규화
                s1_score = s1_raw_score / s1_max_score
                s2_score = candidate['score'] / s2_max_score
                
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
            best_candidate = max(ensemble_candidates, key=lambda x: x['score'])
            s3_ensemble_point = best_candidate['point']

            s3_end = time.time()
            s3_time = s3_end - s3_start
            
            # 시각화를 위해 후보들 저장
            s3_ensemble_candidates = ensemble_candidates
            
            # 앙상블 결과로 성공 여부 확인
            stage3_success = any(point_in_bbox(s3_ensemble_point, bbox) for bbox in ground_truth_bboxes)
            
            s3_hit = "✅" if stage3_success else "❌"
            if stage3_success:
                stage3_success_count += 1

            #! ==================================================================
            #! [Visualization - After Time Measurement]
            #! ==================================================================
            
            if VISUALIZE and (not VIS_ONLY_WRONG or not stage3_success):
                inst_dir_name = re.sub(r'\W+', '_', instruction).strip('_')
                inst_dir_name =  truncate_to_bytes(inst_dir_name,DIR_MAX_LEN)
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
                    gt_bboxes=ground_truth_bboxes,
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
                    original_bboxes=ground_truth_bboxes,
                    s3_ensemble_point=s3_ensemble_point,
                    s2_corrected_point=s2_corrected_point,
                    s1_original_point=s1_original_point,
                    stage1_ratio=STAGE1_ENSEMBLE_RATIO,
                    stage2_ratio=STAGE2_ENSEMBLE_RATIO,
                    save_dir=inst_dir,
                    vis_only_wrong=VIS_ONLY_WRONG,
                    stage3_success=stage3_success
                )

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

            num_attention_crops = len(s1_crop_list)
            print(f"✂️  Attention Crops : {num_attention_crops}")
            print(f"🕖 Times - S1: {s1_time:.2f}s | S2: {s2_time:.2f}s | Total: {total_time:.2f}s")
            if TFOPS_PROFILING:
                print(f"🔥 FLOPs - S1: {s1_tflops:.2f} | S2: {s2_tflops:.2f} | Total: {total_tflops_this:.2f} TFLOPs")
            print(f"{'✅ Success' if stage3_success else '❌🎯 Fail'}")

            #! ==================================================================
            #! [Statistics & Logging]
            #! ==================================================================

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
                    'stage3_success_count': 0
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
            "hyperparameters": {
                "region_threshold": REGION_THRESHOLD,
                "bbox_padding": BBOX_PADDING,
                "min_patches": MIN_PATCHES,
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
                    "hyperparameters": {
                        "region_threshold": REGION_THRESHOLD,
                        "bbox_padding": BBOX_PADDING,
                        "min_patches": MIN_PATCHES,
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
        
        # CSV 데이터 행 생성
        import datetime
        csv_row = [
            method,
            RESIZE_RATIO, REGION_THRESHOLD, BBOX_PADDING,
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

        # 전체 task 통계에 누적
        total_samples += num_action
        total_crop_success += crop_success_count
        total_stage1_success += stage1_success_count
        total_stage2_success += stage2_success_count
        total_stage3_success += stage3_success_count
        total_s1_time += s1_time_sum
        total_s2_time += s2_time_sum
        total_s3_time += s3_time_sum
        total_s1_tflops += s1_tflops_sum
        total_s2_tflops += s2_tflops_sum

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
        print(f"Ensemble Config: Attention {STAGE1_ENSEMBLE_RATIO:.1f}, Crop {STAGE2_ENSEMBLE_RATIO:.1f}")
        print(f"Region Config: threshold={REGION_THRESHOLD}, padding={BBOX_PADDING}px, min_patches={MIN_PATCHES}")
        
        print("=" * 60)

    print("\n📊 All Task Done!")

    # 전체 결과 계산 및 저장
    total_crop_success_rate = total_crop_success / total_samples
    total_stage1_success_rate = total_stage1_success / total_samples
    total_stage2_success_rate = total_stage2_success / total_samples
    total_stage3_success_rate = total_stage3_success / total_samples
    
    # 전체 평균 시간
    avg_s1_time = total_s1_time / total_samples
    avg_s2_time = total_s2_time / total_samples
    avg_s3_time = total_s3_time / total_samples
    avg_total_time = (total_s1_time + total_s2_time + total_s3_time) / total_samples
    
    # 전체 평균 TFLOPS
    avg_s1_tflops = total_s1_tflops / total_samples
    avg_s2_tflops = total_s2_tflops / total_samples
    avg_total_tflops = (total_s1_tflops + total_s2_tflops) / total_samples
    
    print(f"Total Sample num: {total_samples}")
    print(f"Total Crop Success Rate: {total_crop_success_rate:.4f}")
    print(f"Total Stage1 Success Rate: {total_stage1_success_rate:.4f}")
    print(f"Total Stage2 Success Rate: {total_stage2_success_rate:.4f}")
    print(f"Total Stage3 Success Rate: {total_stage3_success_rate:.4f}")
    print(f"Total avg Stage1 time: {avg_s1_time:.4f}s")
    print(f"Total avg Stage2 time: {avg_s2_time:.4f}s")
    print(f"Total avg Stage3 time: {avg_s3_time:.4f}s")
    print(f"Total avg All Stage time: {avg_total_time:.4f}s")
    print(f"Total avg Stage1 TFLOPS: {avg_s1_tflops:.4f}")
    print(f"Total avg Stage2 TFLOPS: {avg_s2_tflops:.4f}")
    print(f"Total avg All Stage TFLOPS: {avg_total_tflops:.4f}")
    
    # 전체 결과를 CSV로 저장
    cumulative_csv_path = os.path.join("../_results", "results_all.csv")
    
    # 전체 결과 CSV 행 생성
    cumulative_csv_row = [
        method,
        RESIZE_RATIO, REGION_THRESHOLD, BBOX_PADDING,
        total_samples,
        round(total_crop_success_rate * 100, 2),
        round(total_stage1_success_rate * 100, 2),
        round(total_stage2_success_rate * 100, 2),
        round(total_stage3_success_rate * 100, 2),
        round(avg_s1_time, 4),
        round(avg_s2_time, 4),
        round(avg_s3_time, 4),
        round(avg_total_time, 4),
        round(avg_s1_tflops, 2),
        round(avg_s2_tflops, 2),
        round(avg_total_tflops, 2),
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ]
    
    # CSV 파일이 없으면 헤더와 함께 생성, 있으면 데이터 행만 추가
    file_exists = os.path.exists(cumulative_csv_path)
    
    with open(cumulative_csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # 파일이 없거나 비어있으면 헤더 추가
        if not file_exists or os.path.getsize(cumulative_csv_path) == 0:
            writer.writerow(csv_headers)
        
        # 전체 결과 행 추가
        writer.writerow(cumulative_csv_row)

    print(f"📝 Total Results : {cumulative_csv_path}")