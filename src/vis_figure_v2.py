'''
Final version
'''

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('gpu', type=int, default=0, help='GPU number')
parser.add_argument('--r', type=float, default=0.50, help='Stage 1 Resize ratio')
parser.add_argument('--th', type=float, default=0.12, help='Stage 1 Crop threshold')
parser.add_argument('--p', type=int, default=0, help='Stage 1 Crop Padding')
parser.add_argument('--v', action='store_true', help='Whether to save visualization images')
parser.add_argument('--mac', action='store_true', help='Whether to run on Mac (MPS)')
args = parser.parse_args()

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
STAGE1_ENSEMBLE_RATIO = 0.70              # Stage1 attention weight
STAGE2_ENSEMBLE_RATIO = 1 - STAGE1_ENSEMBLE_RATIO  # Stage2 crop weight
ENSEMBLE_TOP_PATCHES = 100                # Stage2에서 앙상블에 사용할 상위 패치 개수 (Qwen2.5VL용)

# 최대 PIXELS 제한
MAX_PIXELS = 3211264  # Process단에서 적용
# MAX_PIXELS = 1280*28*28  # Process단에서 적용

# csv에 기록할 method 이름
method = "figure_visualize_mg"

memo = f"resize{RESIZE_RATIO:.2f}_region_thresh{REGION_THRESHOLD:.2f}_pad{BBOX_PADDING}"

# INDEX_START = 
#! Argument ==========================================================================================

SEED = 0

# Dataset & Model
MLLM_PATH = "microsoft/GUI-Actor-3B-Qwen2.5-VL"
SCREENSPOT_IMGS = "../data/screenspotv2_image"      # input image 경로
SCREENSPOT_JSON = "../data"                         # input image json파일 경로
# TASKS = ["mobile", "web", "desktop"]

TASKS = ["mobile"]

# SAMPLE_RANGE = slice(None)

SAMPLE_RANGE = slice(437, 442)  # Main Figure
# SAMPLE_RANGE = slice(235, 237)  # Main Figure

# Visualize & Logging
VISUALIZE = True
VIS_ONLY_WRONG = False                          # True면 틀린 것만 시각화, False면 모든 것 시각화
TFOPS_PROFILING = True
MEMORY_VIS = False
# Save Path
SAVE_DIR = f"../attn_output/" + method + "/" + memo

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
from util.visualize_figure_util import visualize_stage1_attention_crops, visualize_stage2_multi_attention, visualize_stage3_point_ensemble
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
                        # Additional prompt
                        f"This is a resized screenshot of the whole GUI, scaled by {resize_ratio}. "
                        # previous prompt
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
                        # Additional prompt
                        f"This is a list of {len(crop_list)} cropped screenshots of the GUI, each showing a part of the GUI. "
                        # previous prompt
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
    """Stage 2: multi image inference"""
    
    conversation = create_conversation_stage2(crop_list, instruction)
    pred = multi_image_inference(conversation, model, tokenizer, processor, use_placeholder=True, topk=10)
    
    return pred

def convert_multi_image_results_to_original(multi_pred, crop_list):
    """multi_image_inference 결과를 원본 이미지 좌표로 변환"""
    
    converted_results = []
    all_candidates = []
    
    for img_idx, img_result in enumerate(multi_pred['per_image']):
        if img_idx >= len(crop_list):
            continue
            
        crop_info = crop_list[img_idx]
        crop_bbox = crop_info['bbox']  # [left, top, right, bottom]
        crop_width = crop_bbox[2] - crop_bbox[0]
        crop_height = crop_bbox[3] - crop_bbox[1]
        
        crop_candidates = []
        for point_idx, (point, score) in enumerate(zip(img_result['topk_points'], img_result['topk_values'])):
            crop_x = point[0] * crop_width
            crop_y = point[1] * crop_height
            
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
    
    all_candidates.sort(key=lambda x: x['score'], reverse=True)
    
    return all_candidates

def run_stage1_attention_based(original_image, instruction, gt_bbox):
    """새로운 간단한 Stage 1: 연결된 영역 기반 crop 생성"""
    
    s1_pred, resized_image = run_stage1_attention_inference(original_image, instruction)
    resize_ratio = s1_pred['resize_ratio']
    scaled_gt_bbox = [coord * resize_ratio for coord in gt_bbox]
    regions = find_connected_regions(s1_pred, resized_image, resize_ratio)
    regions = regions[:MAX_CROPS]
    crops = create_crops_from_connected_regions(regions, original_image)
    num_crops = len(crops)
    
    return s1_pred, crops, num_crops, resized_image, scaled_gt_bbox

def get_stage1_score_at_point(point, s1_attn_scores, s1_n_width, s1_n_height, original_size, resize_ratio):
    """특정 점에서의 Stage1 어텐션 점수를 계산"""
    
    orig_w, orig_h = original_size
    point_x, point_y = point
    
    resized_x = point_x * resize_ratio
    resized_y = point_y * resize_ratio
    
    resized_w = orig_w * resize_ratio
    resized_h = orig_h * resize_ratio
    
    patch_x = int((resized_x / resized_w) * s1_n_width)
    patch_y = int((resized_y / resized_h) * s1_n_height)
    
    patch_x = max(0, min(patch_x, s1_n_width - 1))
    patch_y = max(0, min(patch_y, s1_n_height - 1))
    
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

    # Model Import
    device_map = "mps" if args.mac else "balanced"

    model = Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
        MLLM_PATH, torch_dtype="auto", attn_implementation=ATTN_IMPL,
        device_map=device_map,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MLLM_PATH)
    processor = AutoProcessor.from_pretrained(MLLM_PATH, max_pixels=MAX_PIXELS)
    
    if TFOPS_PROFILING:
        prof = FlopsProfiler(model)

    # warm_up_model(model, tokenizer, processor)

    if TFOPS_PROFILING:
        prof.start_profile()

    save_dir = SAVE_DIR
    suffix = 0
    while os.path.exists(save_dir):
        suffix += 1
        save_dir = f"{SAVE_DIR}_{suffix}"
    os.makedirs(save_dir)

    # ... (rest of the script setup)
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

    csv_headers = [
        "method", "resize_ratio", "region_threshold", "bbox_padding",
        "total_samples", "crop_accuracy", "stage1_accuracy", "stage2_accuracy", "stage3_accuracy",
        "avg_stage1_time", "avg_stage2_time", "avg_stage3_time", "avg_total_time",
        "avg_stage1_tflops", "avg_stage2_tflops", "avg_total_tflops", "timestamp"
    ]

    for task in TASKS:
        init_iter_logger(
            save_dir=save_dir,
            csv_name=f"iter_log_{task}.csv",
            md_name=f"iter_log_{task}.md",
            headers=[
                "idx", "orig_w", "orig_h", "resize_ratio", "num_crop", "crop_hit",
                "s1_time", "s1_tflops", "s1_hit", "s2_time", "s2_tflops", "s2_hit",
                "s3_time", "s3_hit", "total_time", "total_tflops",
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

        task_res = []
        num_action = 0
        s1_time_sum = s2_time_sum = s3_time_sum = 0.0
        s1_tflops_sum = s2_tflops_sum = 0.0
        crop_success_count = stage1_success_count = stage2_success_count = stage3_success_count = 0
        data_source_stats = {}

        for j, item in tqdm(enumerate(screenspot_data)):
            s1_tflops = s2_tflops = 0.0
            num_action += 1
            print("\n\n----------------------\n")
            
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
            data_source = item.get("data_source", "unknown")

            #! ==================================================================
            #! Stage 1 | Attention-based Crop Generation
            #! ==================================================================
            if TFOPS_PROFILING:
                prof.reset_profile()
            s1_start = time.time()
            s1_pred, s1_crop_list, num_crops, resized_image, scaled_gt_bbox = run_stage1_attention_based(
                original_image=original_image, instruction=instruction, gt_bbox=original_bbox
            )
            s1_end = time.time()
            s1_time = s1_end - s1_start
            if TFOPS_PROFILING:
                s1_tflops = prof.get_total_flops() / 1e12

            s1_success = False
            s1_original_point = None
            if s1_pred and "topk_points" in s1_pred and s1_pred["topk_points"]:
                s1_predicted_point = s1_pred["topk_points"][0]
                s1_original_point = [
                    s1_predicted_point[0] * original_image.size[0],
                    s1_predicted_point[1] * original_image.size[1]
                ]
                s1_success = point_in_bbox(s1_original_point, original_bbox)
            
            s1_hit = "✅" if s1_success else "❌"
            if s1_success:
                stage1_success_count += 1

            crop_success = False
            for crop in s1_crop_list:
                crop_bbox = crop["bbox"]
                left = max(crop_bbox[0], original_bbox[0])
                top = max(crop_bbox[1], original_bbox[1])
                right = min(crop_bbox[2], original_bbox[2])
                bottom = min(crop_bbox[3], original_bbox[3])
                if left < right and top < bottom:
                    crop_success = True
                    break
            
            crop_hit = "✅" if crop_success else "❌"
            if crop_success:
                crop_success_count += 1

            #! ==================================================================
            #! [Stage 2] Crop Inference
            #! ==================================================================
            s2_tflops = 0.0
            if TFOPS_PROFILING:
                prof.reset_profile()
            s2_inference_start = time.time()
            s2_pred = run_stage2_multi_image_inference(s1_crop_list, instruction)
            s2_all_candidates = convert_multi_image_results_to_original(s2_pred, s1_crop_list)
            
            s2_corrected_point = s2_all_candidates[0]['point']
            stage2_success = point_in_bbox(s2_corrected_point, original_bbox)
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
            s1_attn_scores = np.array(s1_pred['attn_scores'][0])
            s1_n_width = s1_pred['n_width']
            s1_n_height = s1_pred['n_height']
            s1_resize_ratio = s1_pred['resize_ratio']
            s1_max_score = float(max(s1_attn_scores)) if len(s1_attn_scores) > 0 else 1.0
            s2_topk_scores = [candidate['score'] for candidate in s2_all_candidates]
            s2_max_score = max(s2_topk_scores)
            ensemble_candidates = []
            
            for i, candidate in enumerate(s2_all_candidates):
                s2_original_point = candidate['point']
                s1_raw_score = get_stage1_score_at_point(
                    s2_original_point, s1_attn_scores, s1_n_width, s1_n_height, 
                    original_image.size, s1_resize_ratio
                )
                s1_score = s1_raw_score / s1_max_score
                s2_score = candidate['score'] / s2_max_score
                ensemble_score = STAGE1_ENSEMBLE_RATIO * s1_score + STAGE2_ENSEMBLE_RATIO * s2_score
                
                ensemble_candidates.append({
                    'point': s2_original_point, 'score': ensemble_score, 's1_score': s1_score,
                    's2_score': s2_score, 'crop_id': candidate['crop_id'],
                    'rank_in_crop': candidate['rank_in_crop'], 's2_rank': i + 1
                })
            
            best_candidate = max(ensemble_candidates, key=lambda x: x['score'])
            s3_ensemble_point = best_candidate['point']
            s3_end = time.time()
            s3_time = s3_end - s3_start
            s3_ensemble_candidates = ensemble_candidates
            stage3_success = point_in_bbox(s3_ensemble_point, original_bbox)
            s3_hit = "✅" if stage3_success else "❌"
            if stage3_success:
                stage3_success_count += 1

            #! ==================================================================
            #! [Visualization - After Time Measurement]
            #! ==================================================================
            
            if VISUALIZE and (not VIS_ONLY_WRONG or not stage3_success):
                inst_dir_name = re.sub(r'\W+', '_', instruction).strip('_')
                result_folder = "correct" if stage3_success else "incorrect"
                inst_dir = os.path.join(save_dir, f"{task}_visualize_{result_folder}", f"{num_action}_{inst_dir_name}")

                # <<< [수정] 모델 입력 이미지 저장을 위한 코드 시작 >>>
                # 모델 입력 이미지를 저장할 별도 디렉토리 생성
                model_input_dir = os.path.join(inst_dir, "model_input_images")
                os.makedirs(model_input_dir, exist_ok=True)

                # Stage 1 모델 입력 이미지 저장 (리사이즈된 이미지)
                resized_image.save(os.path.join(model_input_dir, "stage1_input.png"))

                # Stage 2 모델 입력 이미지 저장 (크롭된 이미지들)
                if s1_crop_list:
                    for crop in s1_crop_list:
                        crop_img = crop['img']
                        crop_id = crop['id']
                        crop_img.save(os.path.join(model_input_dir, f"stage2_input_crop_{crop_id}.png"))
                # <<< [수정] 모델 입력 이미지 저장을 위한 코드 종료 >>>


                # Stage1 시각화
                visualize_stage1_attention_crops(
                    s1_pred=s1_pred, resized_image=resized_image, 
                    crop_list=s1_crop_list, original_image=original_image,
                    save_dir=inst_dir, instruction=instruction,
                    gt_bbox=original_bbox, s1_predicted_point=s1_original_point
                )
                
                # Stage2 Multi-Image 시각화
                if s2_pred and s1_crop_list:  # Stage2 결과가 있을 때만 시각화
                    visualize_stage2_multi_attention(
                        s2_pred=s2_pred, crop_list=s1_crop_list,
                        original_image=original_image, save_dir=inst_dir,
                        instruction=instruction, predicted_point=s2_corrected_point
                    )
                
                # Stage3 앙상블 시각화
                visualize_stage3_point_ensemble(
                    s3_ensemble_candidates=s3_ensemble_candidates if 's3_ensemble_candidates' in locals() else [],
                    original_image=original_image, crop_list=s1_crop_list,
                    original_bbox=original_bbox, s3_ensemble_point=s3_ensemble_point,
                    s2_corrected_point=s2_corrected_point, s1_original_point=s1_original_point,
                    stage1_ratio=STAGE1_ENSEMBLE_RATIO, stage2_ratio=STAGE2_ENSEMBLE_RATIO,
                    save_dir=inst_dir, vis_only_wrong=VIS_ONLY_WRONG, stage3_success=stage3_success
                )

            # ... (rest of the script for logging and saving results)
            # This part is left unchanged as it correctly handles the statistics.
            s1_time_sum += s1_time; s2_time_sum += s2_time; s3_time_sum += s3_time
            s1_tflops_sum += s1_tflops; s2_tflops_sum += s2_tflops
            total_time = s1_time + s2_time
            if TFOPS_PROFILING: total_tflops_this = s1_tflops + s2_tflops
            num_attention_crops = len(s1_crop_list)
            print(f"Task: {task}, Image: {filename} {orig_w}x{orig_h} (Resize Ratio : {s1_pred['resize_ratio']})")
            print(f"Attention Crops : {num_attention_crops}")
            print(f"Times - S1: {s1_time:.2f}s | S2: {s2_time:.2f}s | Total: {total_time:.2f}s")
            if TFOPS_PROFILING: print(f"FLOPs - S1: {s1_tflops:.2f} | S2: {s2_tflops:.2f} | Total: {total_tflops_this:.2f} TFLOPs")
            print(f"{'✅ Success' if stage3_success else '❌🎯 Fail'}")

            # Update stats... (rest of the original script logic follows)


    # ... The rest of the script for final aggregation and saving remains the same