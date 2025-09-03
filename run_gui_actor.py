# run_gui_actor.py
import setproctitle
setproctitle.setproctitle('CF_ground_gui_actor')


import os
import argparse
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "3"  # 몇번 GPU 사용할지 ("0,1", "2" 등)

parser = argparse.ArgumentParser()
parser.add_argument('--no_early_exit', action='store_true', help='Disable early exit')
parser.add_argument('--max_pixels', type=int, help='Maximum pixels for image resize')
args = parser.parse_args()

#! Hyperparameter =====================================================================================

ATTN_IMPL = "eager"  # attention implement "eager" "sdpa" "flash" "efficient"

# Image Resize Ratios
# MAX_PIXELS = None
# MAX_PIXELS = 1280 * 28 * 28
# MAX_PIXELS = 3211264
MAX_PIXELS = args.max_pixels if args.max_pixels else None
S1_RESIZE_RATIO = 0.35  # Stage 1 crop resize ratio
S2_RESIZE_RATIO = 0.60  # Stage 2 crop resize ratio
THUMBNAIL_RESIZE_RATIO = 0.10  # Thumbnail resize ratio

SELECT_THRESHOLD = 0.7  # score >= tau * max_score 인 모든 crop select
# EARLY_EXIT 설정: --no_early_exit이면 False, 기본값 True

EARLY_EXIT = False if args.no_early_exit else True
EARLY_EXIT_THRE = 0.6  # 1등 attention * thre > 2등 attention이라면 early exit

is_ee = "ee" if EARLY_EXIT else "not_ee"
SAVE_DIR = f"./attn_output/" + is_ee + "_" + str(MAX_PIXELS) + "_" + \
    str(S1_RESIZE_RATIO) + "_" + str(S2_RESIZE_RATIO) + "_" + "MaxPixelProcessor"  #! Save Path (특징이 있다면 적어주세요)

#! Argument ==========================================================================================

SEED = 0

# Dataset & Model
MLLM_PATH = "microsoft/GUI-Actor-3B-Qwen2.5-VL"
SCREENSPOT_IMGS = "./data/screenspotv2_image"  # input image 경로
SCREENSPOT_JSON = "./data"  # json파일 경로
TASKS = ["mobile", "web", "desktop"]
SAMPLE_RANGE = slice(None)  #! 샘플 범위 지정 (3번 샘플이면 3,4 / 5~9번 샘플이면 5,10 / 전체 사용이면 None)
# SAMPLE_RANGE = slice(0, 3)

# Visualize & Logging
STAGE0_VIS = False
STAGE1_VIS = False
STAGE2_VIS = False
ITER_LOG = True  # csv, md
TFOPS_PROFILING = True
MEMORY_EVAL = True
MEMORY_VIS = True

# Question
# QUESTION_TEMPLATE="""Where should you tap to {task_prompt}?"""
QUESTION_TEMPLATE="""
You are an assistant trained to navigate the android phone. Given a
task instruction, a screen observation, guess where should you tap.
# Intruction
{task_prompt}"""

#! ==================================================================================================

# Standard Library
import json
import re
import sys
sys.setrecursionlimit(3000)  # DeepSpeed logging
import time
from copy import deepcopy
from typing import List
from math import sqrt
import matplotlib.pyplot as plt
import threading

# Third-Party Libraries
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoProcessor, AutoTokenizer, set_seed
if TFOPS_PROFILING:
    from deepspeed.profiling.flops_profiler import FlopsProfiler

# Project-Local Modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from iter_logger import init_iter_logger, append_iter_log  # log csv 기록 파일
from gui_actor.modeling_qwen25vl import Qwen2_5_VLForConditionalGenerationWithPointer
from gui_actor.multi_image_inference import multi_image_inference
from visualize_util import get_highest_attention_patch_bbox, _visualize_early_exit_results, _visualize_stage1_results, _visualize_stage2_results, visualize_crop
from crop import crop_img  #! 어떤 crop 파일 사용?

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
    
def resize_image(image, resize_to_pixels=MAX_PIXELS):
    image_width, image_height = image.size
    if (resize_to_pixels is not None) and ((image_width * image_height) != resize_to_pixels):
        resize_ratio = (resize_to_pixels / (image_width * image_height)) ** 0.5
        image_width_resized, image_height_resized = int(image_width * resize_ratio), int(image_height * resize_ratio)
        image = image.resize((image_width_resized, image_height_resized))
    return image, image_width_resized, image_height_resized 


def warm_up_model(model, tokenizer, processor):
    print("🔄 Warming up the model...")
    dummy_instruction = "This is a dummy instruction for warm-up."
    dummy_image = Image.new("RGB", (1000, 1000), color=(255, 255, 255))  # 1000x1000 흰색 이미지
    dummy_crop = {
        "img": dummy_image,
        "resized_img": dummy_image,
        "id": 1,
        "bbox": [0, 0, 1000, 1000]
    }
    dummy_crop_list = [dummy_crop]
    dummy_msgs = create_guiactor_msgs(crop_list=dummy_crop_list, instruction=dummy_instruction)
    
    # 예열용 inference 실행
    for _ in range(3):  # 3번 반복
        with torch.no_grad():
            if TFOPS_PROFILING:
                prof.start_profile()
            _ = multi_image_inference(dummy_msgs, model, tokenizer, processor, use_placeholder=True, topk=3)
            if TFOPS_PROFILING:
                prof.stop_profile()
                prof.get_total_flops()
    print("✅ Warm-up complete!")


def create_guiactor_msgs(crop_list, instruction):
    user_content = []
    for crop in crop_list:
        img = crop["resized_img"]
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

def resize_crop_list(crop_list: List, ratio: float):
    stage_crop_list = []

    for crop in crop_list:
        crop_id = crop.get("id")

        # 썸네일(id=0)은 thumbnail 비율, 나머지는 crop 비율 사용
        if crop_id == 0:
            crop_ratio = THUMBNAIL_RESIZE_RATIO
        else:
            crop_ratio = ratio

        new_crop = deepcopy(crop)
        crop_img = deepcopy(crop["img"])

        # 이미지 리사이즈
        crop_width, crop_height = crop_img.size
        crop_img = crop_img.resize((int(crop_width * crop_ratio), int(crop_height * crop_ratio)))
        new_crop["resized_img"] = crop_img
        stage_crop_list.append(new_crop)

    return stage_crop_list

def select_crop(crop_list, tau):
    """
    score >= tau * max_score 인 모든 crop의 id만 반환 (id==0은 무시)
    """
    filtered = [crop for crop in crop_list if crop.get("id") != 0]
    scores = [float(c["s1_att_sum"]) for c in filtered]
    max_score = max(scores)
    threshold = max_score * float(tau)
    out_ids = []
    for crop in filtered:
        cid = crop.get("id")
        if float(crop["s1_att_sum"]) >= threshold:
            out_ids.append(cid)
    return out_ids

def check_gt_in_selected_crops(top_q_bboxes: List, gt_bbox: List):
    def rect_intersects(a, b):
        # a, b: [left, top, right, bottom]
        al, at, ar, ab = a
        bl, bt, br, bb = b
        # 교집합 영역 계산
        inter_left = max(al, bl)
        inter_top = max(at, bt)
        inter_right = min(ar, br)
        inter_bottom = min(ab, bb)
        return (inter_right > inter_left) and (inter_bottom > inter_top)
    return any(rect_intersects(gt_bbox, bbox) for bbox in top_q_bboxes)

def compute_attention_scores(crop_list, per_image_outputs):
    """각 crop의 attention score 계산"""
    for i, crop in enumerate(crop_list):
        crop_att_scores_np = np.array(per_image_outputs[i]['attn_scores'][0])
        total_att_score = np.sum(crop_att_scores_np)
        
        # 면적의 제곱근으로 normalize
        bbox = crop.get('bbox')
        if bbox is not None:
            left, top, right, bottom = bbox
            area = max(1, (right - left) * (bottom - top))
        else:
            area = 1
        crop['s1_att_sum'] = total_att_score / sqrt(area)

def find_top_crop_for_early_exit(crop_list, per_image_outputs):
    """Early Exit용 최고 점수 crop과 point 찾기"""
    top_score = -1
    top_point = None
    top_crop_id = -1
    
    for i, crop in enumerate(crop_list):
        if crop.get("id") == 0:  # 썸네일은 스킵
            continue
            
        crop_att_scores_np = np.array(per_image_outputs[i]['attn_scores'][0])
        crop_top_point = per_image_outputs[i]['topk_points'][0]
        crop_top_score = np.max(crop_att_scores_np)
        
        crop['top_point'] = {'point': crop_top_point, 'score': crop_top_score}
        
        if top_score < crop_top_score:
            top_score = crop_top_score
            top_point = crop_top_point
            top_crop_id = crop['id']
    
    return top_point, top_crop_id

def get_highest_attention_patch_bbox(image_result: dict) -> list:
    """
    per_image 결과에서 어텐션 스코어가 가장 높은 패치를 찾아 
    해당 패치의 정규화된 바운딩 박스 좌표를 반환
    """
    # 1. 입력 데이터 추출
    attn_scores = np.array(image_result['attn_scores'][0])
    n_width = image_result['n_width']
    n_height = image_result['n_height']

    # 2. 어텐션 스코어가 가장 높은 패치의 1차원 인덱스 찾기
    highest_score_idx = np.argmax(attn_scores)

    # 3. 1차원 인덱스를 2차원 패치 그리드 좌표 (patch_x, patch_y)로 변환
    # (patch_x는 가로 인덱스, patch_y는 세로 인덱스)
    patch_y = highest_score_idx // n_width
    patch_x = highest_score_idx % n_width

    # 4. 패치 그리드 좌표를 정규화된 바운딩 박스 좌표로 계산
    # 각 패치의 정규화된 너비와 높이
    patch_norm_width = 1.0 / n_width
    patch_norm_height = 1.0 / n_height
    
    # 바운딩 박스 계산
    left = patch_x * patch_norm_width
    top = patch_y * patch_norm_height
    right = (patch_x + 1) * patch_norm_width
    bottom = (patch_y + 1) * patch_norm_height
    
    return [left, top, right, bottom]


def get_highest_attention_patch_info(image_result: dict) -> dict:
    """
    per_image 결과에서 어텐션 스코어가 가장 높은 패치와 두 번째로 높은 패치를 찾아
    해당 패치들의 정규화된 바운딩 박스 좌표와 스코어를 딕셔너리로 반환합니다.

    Args:
        image_result (dict): prediction_results['per_image'] 리스트의 단일 아이템.

    Returns:
        dict: 아래와 같은 구조의 딕셔너리
              {
                  'highest': {'bbox': [l,t,r,b], 'score': float}, 
                  'second_highest': {'bbox': [l,t,r,b], 'score': float}
              }
    """
    # 1. 입력 데이터 추출
    attn_scores = np.array(image_result['attn_scores'][0])
    n_width = image_result['n_width']
    n_height = image_result['n_height']

    # 2. 어텐션 스코어를 정렬하여 가장 높은 두 개의 인덱스 찾기
    sorted_indices = np.argsort(attn_scores)
    highest_score_idx = sorted_indices[-1]
    second_highest_score_idx = sorted_indices[-2]

    # 3. 인덱스를 사용하여 실제 스코어 값 가져오기
    highest_score = attn_scores[highest_score_idx]
    second_highest_score = attn_scores[second_highest_score_idx]

    # 4. 바운딩 박스 계산을 위한 헬퍼 함수 정의
    def _calculate_bbox(index: int) -> list:
        """1차원 인덱스로부터 정규화된 바운딩 박스 좌표를 계산합니다."""
        patch_y = index // n_width
        patch_x = index % n_width

        patch_norm_width = 1.0 / n_width
        patch_norm_height = 1.0 / n_height
        
        left = patch_x * patch_norm_width
        top = patch_y * patch_norm_height
        right = (patch_x + 1) * patch_norm_width
        bottom = (patch_y + 1) * patch_norm_height
        
        return [left, top, right, bottom]

    # 5. 가장 높은 스코어와 두 번째로 높은 스코어의 바운딩 박스 계산
    highest_bbox = _calculate_bbox(highest_score_idx)
    second_highest_bbox = _calculate_bbox(second_highest_score_idx)
    
    # 6. 딕셔너리 형태로 결과 반환 (스코어 포함)
    return {
        'highest': {
            'bbox': highest_bbox,
            'score': float(highest_score) # numpy float을 일반 float으로 변환
        },
        'second_highest': {
            'bbox': second_highest_bbox,
            'score': float(second_highest_score) # numpy float을 일반 float으로 변환
        }
    }

def check_early_exit_condition(top_point, top_crop_id, crop_list, per_image_outputs, gt_bbox, original_image):
    """Early Exit 조건 확인"""
    if top_point is None or top_crop_id == -1:
        return False, False, None
    
    should_exit_early = False
    early_exit_success = False
    corrected_point = None

    ori_w, ori_h = original_image.size

    # 썸네일의 최고 attention patch 찾기
    thumb_res = next((res for res in per_image_outputs if res['index'] == 0), None)
    # thumb_top_patch_bbox = get_highest_attention_patch_bbox(thumb_res)
 
    top2_patches = get_highest_attention_patch_info(thumb_res)
    top1 = top2_patches['highest']
    top2 = top2_patches['second_highest']

    top1_score = top1['score']
    top2_score = top2['score']

    if  EARLY_EXIT_THRE * top1_score > top2_score:  # early exit

        l, t, r, b = top1['bbox']
        denorm_thumb_top_patch_bbox = [l*ori_w, t*ori_h, r*ori_w, b*ori_h]
        
        # 좌표 보정
        top_crop = next((c for c in crop_list if c['id'] == top_crop_id), None)
        top_crop_bbox = top_crop["bbox"]
        corrected_point = denormalize_crop_point(
            point_in_crop=top_point, 
            crop_size=top_crop['img'].size,
            crop_bbox=top_crop_bbox
        )
        
        should_exit_early = point_in_bbox(corrected_point, denorm_thumb_top_patch_bbox)

        # Early Exit 맞았는가?
        early_exit_success = point_in_bbox(corrected_point, gt_bbox)
    elif STAGE1_VIS:

        top_crop = next((c for c in crop_list if c['id'] == top_crop_id), None)
        top_crop_bbox = top_crop["bbox"]
        corrected_point = denormalize_crop_point(
            point_in_crop=top_point, 
            crop_size=top_crop['img'].size,
            crop_bbox=top_crop_bbox
        )
    
    return should_exit_early, early_exit_success, corrected_point

def run_selection_pass_with_guiactor(msgs, crop_list, gt_bbox: List, attn_vis_dir: str, original_image, img_path, instruction):
    """Stage 1 inference 및 Early Exit 판단"""
    
    # Inference 수행
    pred = multi_image_inference(msgs, model, tokenizer, processor, use_placeholder=True, topk=3)
    per_image_outputs = pred["per_image"]
    
    # Attention scores 계산
    compute_attention_scores(crop_list, per_image_outputs)
    
    # Early Exit 체크
    should_exit_early, early_exit_success = False, False
    
    if EARLY_EXIT:
        top_point, top_crop_id = find_top_crop_for_early_exit(crop_list, per_image_outputs)
        should_exit_early, early_exit_success, corrected_top_point = check_early_exit_condition(
            top_point, top_crop_id, crop_list, per_image_outputs, gt_bbox, original_image
        )    # Early Exit하면 select_crop 스킵
    if should_exit_early:
        top_q_crop_ids = []
        top_q_bboxes = []
    else:
        # Select crop: score >= tau * max_score인 crops 선택
        top_q_crop_ids = select_crop(crop_list, tau=SELECT_THRESHOLD)
        top_q_bboxes = [crop["bbox"] for crop in crop_list if crop.get("id") in top_q_crop_ids]
    
    # 시각화 (필요시)
    if STAGE1_VIS and EARLY_EXIT and should_exit_early:
        _visualize_early_exit_results(crop_list, pred, corrected_top_point, gt_bbox, attn_vis_dir, instruction, img_path)
    elif STAGE1_VIS and not should_exit_early:
        _visualize_stage1_results(crop_list, pred, attn_vis_dir, instruction)
    
    return top_q_crop_ids, top_q_bboxes, crop_list, should_exit_early, early_exit_success

def denormalize_crop_point(point_in_crop, crop_size, crop_bbox):
    crop_w, crop_h = crop_size

    scaled_point = [point_in_crop[0] * crop_w, point_in_crop[1] * crop_h]
    corrected_point = [scaled_point[0] + crop_bbox[0], scaled_point[1] + crop_bbox[1]] 

    return corrected_point

def find_best_crop_point(crop_list, per_image_outputs):
    """가장 높은 점수의 crop과 point 찾기"""
    top_score = -1
    top_point = None
    top_crop_id = -1

    for i, crop in enumerate(crop_list):
        if crop.get("id") == 0:  # 썸네일 스킵
            continue
            
        crop_att_scores_np = np.array(per_image_outputs[i]['attn_scores'][0])
        crop_top_point = per_image_outputs[i]['topk_points'][0]
        crop_top_score = np.max(crop_att_scores_np)

        crop['top_point'] = {'point': crop_top_point, 'score': crop_top_score}

        if top_score < crop_top_score:
            top_score = crop_top_score
            top_point = crop_top_point
            top_crop_id = crop['id']
    
    return top_point, top_crop_id

def run_refinement_pass_with_guiactor(crop_list: List, instruction: str, original_image: Image, save_dir: str, gt_bbox: List, img_path: str):
    """Stage 2: 선택된 crop들로 최종 grounding 수행"""
    
    # Stage 2 용 리사이즈
    s2_resized_crop_list = resize_crop_list(crop_list=crop_list, ratio=S2_RESIZE_RATIO)
    s2_msgs = create_guiactor_msgs(crop_list=s2_resized_crop_list, instruction=instruction)

    # Inference
    pred = multi_image_inference(s2_msgs, model, tokenizer, processor, use_placeholder=True, topk=3)
    per_image_outputs = pred["per_image"]
    
    # 최고 점수 crop 찾기
    top_point, top_crop_id = find_best_crop_point(s2_resized_crop_list, per_image_outputs)
    
    if top_point is None:
        return False
    
    # 원본 crop에서 bbox 정보 가져오기
    top_crop = next((c for c in crop_list if c['id'] == top_crop_id), None)
    if top_crop is None:
        return False
        
    top_crop_bbox = top_crop["bbox"]
    
    # 좌표 보정 및 성공 여부 판단
    corrected_point = denormalize_crop_point(
        point_in_crop=top_point, 
        crop_size=top_crop['img'].size, 
        crop_bbox=top_crop_bbox
    )
    is_success = point_in_bbox(corrected_point, gt_bbox)

    # 시각화 (필요시)
    if STAGE2_VIS:
        _visualize_stage2_results(save_dir, s2_resized_crop_list, pred, gt_bbox, corrected_point, instruction, img_path)
        
    return is_success

def point_in_bbox(point, bbox):
    """
    point: (x, y)
    bbox: (left, top, right, bottom)
    경계 포함
    """
    x, y = point
    l, t, r, b = bbox
    return (l <= x <= r) and (t <= y <= b)

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
        device_map="balanced",  # NVIDIA GPU
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

    if TFOPS_PROFILING:
        prof = FlopsProfiler(model)

    # save_dir 폴더명이 이미 존재하면 고유한 이름 생성 (save_dir -> save_dir_1 -> save_dir_2)
    save_dir = SAVE_DIR
    suffix = 0
    while os.path.exists(save_dir):
        suffix += 1
        save_dir = f"{SAVE_DIR}_{suffix}"
    os.makedirs(save_dir)

    warm_up_model(model, tokenizer, processor)

    # Process
    for task in TASKS:
        # 각 task별로 별도의 로그 파일 생성
        init_iter_logger(  
            save_dir=save_dir,
            csv_name=f"iter_log_{task}.csv",
            md_name=f"iter_log_{task}.md",
            headers=[  # 순서 그대로 들어감
                "idx", "crop_time", "num_crop", "early_exit", "num_selected_crop", 
                "s1_time", "s1_tflops", "s1_hit", "s2_time", "s2_tflops", "s2_hit", 
                "total_time", "total_flops_tflops", "peak_memory_gb", "acc_uptonow", 
                "filename", "instruction"
            ],
            write_md=True, use_fsync=True, use_lock=True
        )
        task_res = dict()
        dataset = "screenspot_" + task + "_v2.json"
        screenspot_data = json.load(open(os.path.join(SCREENSPOT_JSON, dataset), 'r'))
        screenspot_data = screenspot_data[SAMPLE_RANGE]

        print("Num of sample: " + str(len(screenspot_data)), flush=True)

        # 통계 변수 초기화
        task_res = []
        num_action = 0
        s0_time_sum = s1_time_sum = s2_time_sum = total_flops = 0.0
        early_exit_count = early_exit_success_count = final_success_count = stage1_success_count = 0
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

            s1_tflops = s2_tflops = total_flops_this = 0.0

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
            
            # 이미지 리사이즈 처리
            # if MAX_PIXELS is not None and orig_w * orig_h > MAX_PIXELS:
            #     resized_image, w_resized, h_resized = resize_image(original_image)
            #     # bbox도 리사이즈 비율에 맞춰 스케일링
            #     resize_ratio = (w_resized * h_resized) ** 0.5 / (orig_w * orig_h) ** 0.5
            #     scaled_bbox = [int(coord * resize_ratio) for coord in original_bbox]
            # else:
            # 리사이즈 X
            resized_image = original_image
            w_resized, h_resized = orig_w, orig_h
            resize_ratio = 1.0
            scaled_bbox = original_bbox
                
            # data_source 정보 추출 (없으면 "unknown"으로 기본값 설정)
            data_source = item.get("data_source", "unknown")
            
            # data_source별 처리 옵션 -> 폰 (세로화면) = crop할 때 세로분할 안함
            skip_vertical_split = data_source in ["ios", "android"]
            
            # 디렉토리 설정 (시각화용 - 필요시에만)
            if any([STAGE0_VIS, STAGE1_VIS, STAGE2_VIS]):
                inst_dir_name = re.sub(r'\W+', '_', instruction).strip('_')
                inst_dir = os.path.join(save_dir, "seg", filename_wo_ext, inst_dir_name)
                s1_dir = os.path.join(inst_dir, "stage1") 
                s2_dir = os.path.join(inst_dir, "stage2")
                os.makedirs(s1_dir, exist_ok=True)
                os.makedirs(s2_dir, exist_ok=True)
            else:
                inst_dir = s1_dir = s2_dir = None

            #! ==================================================================
            #! Stage 0 | Segmentation
            #! ==================================================================

            seg_start = time.time()

            # resize 하지 않은 원본의 화질을 기준으로 crop 횟수 정하기
            if orig_h < 960:  # 저화질이나 가로화면 -> 2등분
                h_cuts = 1
                h_tolerance = 0.20
            elif orig_h < 1440:  # 중간화질 -> 3등분
                h_cuts = 2
                h_tolerance = 0.12
            else:  # 고화질이나 세로화면 -> 4등분
                h_cuts = 3
                h_tolerance = 0.08
            crop_list = crop_img(orig_img=resized_image, h_cuts=h_cuts, h_tolerance=h_tolerance)  # resize된걸로 crop
            s0_crop_list = resize_crop_list(crop_list=crop_list, ratio=S1_RESIZE_RATIO)
            seg_end = time.time()
            seg_time = seg_end - seg_start

            if STAGE0_VIS and inst_dir:
                all_crops_bboxes = [crop["bbox"] for crop in s0_crop_list]
                visualize_crop(save_dir=inst_dir, gt_bbox=scaled_bbox, top_q_bboxes=all_crops_bboxes,
                                instruction=instruction, filename="s1_all_crop.png", img_path=img_path, click_point=None)

            #! ==================================================================
            #! Stage 1 | Find Top Q + Inference
            #! ==================================================================

            # Calculate Stage 1 FLOPs

            if TFOPS_PROFILING:
                prof.start_profile()

            s1_start = time.time()

            s1_msgs = create_guiactor_msgs(crop_list=s0_crop_list, instruction=instruction)

            s1_top_q_crop_ids, s1_top_q_bboxes, s0_crop_list_out, should_exit_early, early_exit_success = run_selection_pass_with_guiactor(
                msgs=s1_msgs,
                crop_list=s0_crop_list,
                gt_bbox=scaled_bbox,  # 스케일된 bbox 사용
                attn_vis_dir=s1_dir or "",
                original_image=resized_image,
                img_path=img_path,
                instruction=instruction
            )
            s1_infence_end = time.time()
            s1_time = s1_infence_end - s1_start

            if TFOPS_PROFILING:
                prof.stop_profile()
                s1_tflops = prof.get_total_flops()
                s1_tflops /= 1e12

            if should_exit_early:
                print(f"✂️  Crops : 1 | 🚀 Early Exit")
                early_exit_count +=1
                if early_exit_success:
                    s1_hit = "✅🚀"
                    early_exit_success_count += 1
                else:
                    s1_hit = "❌🚀"
                stage1_success = False  # Early exit이므로 stage1 성공 여부 미정의

            else:  # GT가 안에 들어가는지 체크
                s1_hit = "✅" if check_gt_in_selected_crops(s1_top_q_bboxes, scaled_bbox) else "❌"
                stage1_success = check_gt_in_selected_crops(s1_top_q_bboxes, scaled_bbox)
                if stage1_success:
                    stage1_success_count += 1

            # 불필요한 딕셔너리 연산 제거 - 결과 저장용도만
            # res_board_dict는 사실상 미사용
            
            #! ==================================================================
            #! [Stage 2] Attention Refinement Pass
            #! ==================================================================
            
            # Early Exit
            s2_tflops = 0.0
            if should_exit_early:
                final_success = early_exit_success
                s2_time = 0.0
                s2_tflops = 0.0
            else:
                original_crop_map = {c['id']: c for c in crop_list}
                s2_input_crop_ids = set()
                if 0 in original_crop_map:
                    s2_input_crop_ids.add(0)
                for crop_id in s1_top_q_crop_ids:
                    s2_input_crop_ids.add(crop_id)
                s2_input_crops = [original_crop_map[cid] for cid in s2_input_crop_ids if cid in original_crop_map]

                # Calculate Stage 2 FLOPs
                s2_resized_crops = resize_crop_list(crop_list=s2_input_crops, ratio=S2_RESIZE_RATIO)
                s2_msgs = create_guiactor_msgs(crop_list=s2_resized_crops, instruction=instruction)

                if TFOPS_PROFILING:
                    prof.start_profile()
                s2_inference_start = time.time()

                final_success = run_refinement_pass_with_guiactor(
                    crop_list=s2_input_crops,
                    instruction=instruction,
                    original_image=resized_image,
                    save_dir=s2_dir or "",
                    gt_bbox=scaled_bbox,  # 스케일된 bbox 사용
                    img_path=img_path
                )
                s2_inference_end = time.time()
                s2_time = s2_inference_end - s2_inference_start
                if TFOPS_PROFILING:
                    prof.stop_profile()
                    s2_tflops = prof.get_total_flops()
                    s2_tflops /= 1e12
        

            #! ==================================================================
            #! [Common Processing]
            #! ==================================================================
            
            # 공통 통계 업데이트
            s0_time_sum += seg_time
            s1_time_sum += s1_time
                
            # 성능 로깅
            total_time = seg_time + s1_time + s2_time
            if TFOPS_PROFILING:
                total_flops_this = s1_tflops + (s2_tflops if not should_exit_early else 0)
                total_flops += total_flops_this

            if len(s0_crop_list) != 2 and not should_exit_early:
                print(f"✂️  Crops : {len(s0_crop_list_out)-1} | Select Crops : {len(s1_top_q_crop_ids)}")
            print(f"🕖 Times - Seg: {seg_time:.2f}s | S1: {s1_time:.2f}s | S2: {s2_time:.2f}s | Total: {total_time:.2f}s")
            if TFOPS_PROFILING:
                print(f"🔥 FLOPs - S1: {s1_tflops:.2f} | S2: {s2_tflops:.2f} | Total: {total_flops_this:.2f} TFLOPs")
            print(f"{'✅🚨 Early Exit Success' if should_exit_early and early_exit_success else '❌🚨 Early Exit Fail' if should_exit_early else '✅ Grounding Success' if final_success else '❌ Grounding Fail'}")

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
                    plt.figure(figsize=(10, 4))
                    plt.plot(time_log, mem_log)
                    plt.xlabel("Time (s)")
                    plt.ylabel("GPU Memory Allocated (GB)")
                    plt.title("GPU Memory Usage Over Time")
                    plt.grid(True)
                    plt.savefig(f"{memory_dir}/{num_action}_{filename}")
                    plt.close()  # 메모리 누수 방지를 위해 close 추가

            s2_time_sum += s2_time
            final_success_count += final_success
            if MEMORY_EVAL:
                peak_memory_sum += peak_memory_gb

            # data_source별 통계 업데이트
            if data_source not in data_source_stats:
                data_source_stats[data_source] = {
                    'num_action': 0,
                    's0_time_sum': 0.0,
                    's1_time_sum': 0.0,
                    's2_time_sum': 0.0,
                    'total_flops': 0.0,
                    'early_exit_count': 0,
                    'early_exit_success_count': 0,
                    'final_success_count': 0,
                    'stage1_success_count': 0,
                    'peak_memory_sum': 0.0  # 피크 메모리 합계 추가
                }
            
            stats = data_source_stats[data_source]
            stats['num_action'] += 1
            stats['s0_time_sum'] += seg_time
            stats['s1_time_sum'] += s1_time
            stats['s2_time_sum'] += s2_time
            if TFOPS_PROFILING:
                stats['total_flops'] += total_flops_this
            if MEMORY_EVAL:
                stats['peak_memory_sum'] += peak_memory_gb
            if should_exit_early:
                stats['early_exit_count'] += 1
                if early_exit_success:
                    stats['early_exit_success_count'] += 1
            else:
                # Early exit이 아닐 때만 stage1 성공 카운트
                if stage1_success:
                    stats['stage1_success_count'] += 1
            if final_success:
                stats['final_success_count'] += 1

            up2now_gt_score = final_success_count / num_action * 100
            print(f"Up2Now Grounding Accuracy: {up2now_gt_score}%")

            # Iter log - 개선된 로깅
            append_iter_log(
                idx=j+1,
                filename=filename_wo_ext,
                instruction=instruction[:50] + "..." if len(instruction) > 50 else instruction,
                crop_time=f"{seg_time:.3f}",
                num_crop=len(s0_crop_list)-1,
                early_exit="☑️" if should_exit_early else "🫥",
                num_selected_crop=len(s1_top_q_crop_ids) if not should_exit_early else 0,
                s1_time=f"{s1_time:.3f}",
                s1_tflops=f"{s1_tflops:.2f}",
                s1_hit=s1_hit,
                s2_time=f"{s2_time:.3f}",
                s2_tflops=f"{s2_tflops:.2f}" if not should_exit_early else "0.00",
                s2_hit="✅" if final_success else "❌",
                total_time=f"{total_time:.3f}",
                total_flops_tflops=f"{total_flops_this:.2f}",
                peak_memory_gb=f"{peak_memory_gb:.3f}" if MEMORY_EVAL else "N/A",
                acc_uptonow=f"{up2now_gt_score:.2f}"
            )

            # JSON 기록 - 핵심 정보만
            item_res = {
                'filename': filename,
                'instruction': instruction,
                'gt_bbox': original_bbox,
                'data_source': data_source,
                'num_crop': len(s0_crop_list) - 1,
                'early_exit': should_exit_early,
                'early_exit_success': early_exit_success,
                'stage1_success': stage1_success if not should_exit_early else None,
                's1_hit': s1_hit,
                's2_hit': final_success,
                'seg_time': seg_time,
                's1_time': s1_time,
                's2_time': s2_time,
                'total_time': total_time,
                's1_tflops': s1_tflops,
                's2_tflops': s2_tflops if not should_exit_early else 0,
                'total_flops': total_flops_this,
                'peak_memory_gb': peak_memory_gb if MEMORY_EVAL else None
            }
            task_res.append(item_res)

        #! ==================================================
        # Json 정리
        with open(os.path.join(save_dir, dataset), "w") as f:
            json.dump(task_res, f, indent=4, ensure_ascii=False, cls=NpEncoder)

        # 최종 성능 메트릭 계산
        non_early_exit_samples = num_action - early_exit_count
        metrics = {
            "task": task,
            "total_samples": num_action,
            "stage1_accuracy": stage1_success_count / non_early_exit_samples * 100 if non_early_exit_samples > 0 else 0,
            "accuracy": final_success_count / num_action * 100,
            "early_exit_rate": early_exit_count / num_action * 100,
            "early_exit_success_rate": early_exit_success_count / early_exit_count * 100 if early_exit_count > 0 else 0,
            "avg_times": {
                "segmentation": s0_time_sum / num_action,
                "stage1": s1_time_sum / num_action,
                "stage2": s2_time_sum / num_action,
                "total": (s0_time_sum + s1_time_sum + s2_time_sum) / num_action
            },
            "avg_flops_tflops": total_flops / num_action,
            "avg_peak_memory_gb": round(peak_memory_sum / num_action, 3) if MEMORY_EVAL else None,
            "hyperparameters": {
                "max_pixels": MAX_PIXELS,
                "select_threshold": SELECT_THRESHOLD,
                "early_exit": EARLY_EXIT,
                "early_exit_thre": EARLY_EXIT_THRE,
                "s1_resize_ratio": S1_RESIZE_RATIO,
                "s2_resize_ratio": S2_RESIZE_RATIO,
                "thumbnail_resize_ratio" : THUMBNAIL_RESIZE_RATIO,
                "attn_impl" : ATTN_IMPL
            }
        }

        with open(os.path.join(save_dir, f"results_{task}.json"), "w") as mf:
            json.dump(metrics, mf, ensure_ascii=False, indent=4)

        # data_source별 메트릭 저장
        data_source_metrics = {}
        for ds, stats in data_source_stats.items():
            if stats['num_action'] > 0:
                non_early_exit_samples = stats['num_action'] - stats['early_exit_count']
                data_source_metrics[ds] = {
                    "task": task,
                    "data_source": ds,
                    "total_samples": stats['num_action'],
                    "stage1_accuracy": stats['stage1_success_count'] / non_early_exit_samples * 100 if non_early_exit_samples > 0 else 0,
                    "accuracy": stats['final_success_count'] / stats['num_action'] * 100,
                    "early_exit_rate": stats['early_exit_count'] / stats['num_action'] * 100,
                    "early_exit_success_rate": stats['early_exit_success_count'] / stats['early_exit_count'] * 100 if stats['early_exit_count'] > 0 else 0,
                    "avg_times": {
                        "segmentation": stats['s0_time_sum'] / stats['num_action'],
                        "stage1": stats['s1_time_sum'] / stats['num_action'],
                        "stage2": stats['s2_time_sum'] / stats['num_action'],
                        "total": (stats['s0_time_sum'] + stats['s1_time_sum'] + stats['s2_time_sum']) / stats['num_action']
                    },
                    "avg_flops_tflops": stats['total_flops'] / stats['num_action'],
                    "avg_peak_memory_gb": round(stats['peak_memory_sum'] / stats['num_action'], 3) if MEMORY_EVAL else None
                }
        
        with open(os.path.join(save_dir, f"results_{task}_source.json"), "w") as dsf:
            json.dump(data_source_metrics, dsf, ensure_ascii=False, indent=4)

        # 최종 결과 출력
        print("=" * 60)
        print(f"📊 Final Results for {task}:")
        print(f"Total Samples: {num_action}")
        print(f"Stage1 Accuracy: {metrics['stage1_accuracy']:.2f}%")
        print(f"Accuracy: {metrics['accuracy']:.2f}%")
        print(f"Early Exit Rate: {metrics['early_exit_rate']:.2f}%")
        print(f"Early Exit Success Rate: {metrics['early_exit_success_rate']:.2f}%") 
        print(f"Avg Times: Seg {metrics['avg_times']['segmentation']:.3f}s, S1 {metrics['avg_times']['stage1']:.3f}s, S2 {metrics['avg_times']['stage2']:.3f}s, Total {metrics['avg_times']['total']:.3f}s")
        print(f"Avg FLOPs: {metrics['avg_flops_tflops']:.2f} TFLOPs")
        if MEMORY_EVAL and metrics['avg_peak_memory_gb'] is not None:
            print(f"Avg Peak Memory: {metrics['avg_peak_memory_gb']:.3f} GB")
        
        # data_source별 결과 출력
        print("\n📊 Results by Data Source:")
        for ds, ds_metrics in data_source_metrics.items():
            memory_str = f", Mem: {ds_metrics['avg_peak_memory_gb']:.3f}GB" if MEMORY_EVAL and ds_metrics['avg_peak_memory_gb'] is not None else ""
            print(f"  {ds}: {ds_metrics['total_samples']} samples, S1 Acc: {ds_metrics['stage1_accuracy']:.2f}%, "
                  f"Acc: {ds_metrics['accuracy']:.2f}%, Early Exit: {ds_metrics['early_exit_rate']:.2f}%, "
                  f"TFLOPs: {ds_metrics['avg_flops_tflops']:.2f}{memory_str}")
        print("=" * 60)