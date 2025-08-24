# run_gui_actor.py

#! Argument =======================
SEED = 0

# Enviroment
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "1"  # 몇번 GPU 사용할지 ("0,1", "2" 등)

# Dataset & Model
MLLM_PATH = "microsoft/GUI-Actor-3B-Qwen2.5-VL"
SCREENSPOT_IMGS = "./data/screenspotv2_imgs"  # input image 경로
SCREENSPOT_JSON = "./data"  # json파일 경로
TASKS = ["mobile"]
SAMPLE_RANGE = slice(None)  #! 샘플 범위 지정 (3번 샘플이면 3,4 / 5~9번 샘플이면 5,10 / 전체 사용이면 None)
SAVE_DIR = "./attn_output/" + "0824_hoon_not_early_exit"  #! 결과 저장 경로 (방법을 바꾼다면 바꿔서 기록하기)

# Visualize
STAGE0_VIS = False
STAGE1_VIS = False
STAGE2_VIS = False
ITER_LOG = True  # csv, md

#! Hyperparameter =================

# Model Architecture
LAYER_NUM = 31

# Stage 1: Segmentation & Selection
SELECT_THRE = 0.70  # score >= tau * max_score 인 모든 crop select
EARLY_EXIT = False

# Stage 2: Attention Refinement  
AGG_START = 20  # Starting layer for attention aggregation

# Image Resize Ratios
S1_RESIZE_RATIO = 0.25  # Stage 1 crop resize ratio
S2_RESIZE_RATIO = 0.50  # Stage 2 crop resize ratio  
THUMBNAIL_RESIZE_RATIO = 0.10  # Thumbnail resize ratio


# Question
# QUESTION_TEMPLATE="""Where should you tap to {task_prompt}?"""
QUESTION_TEMPLATE="""
You are an assistant trained to navigate the android phone. Given a
task instruction, a screen observation, guess where should you tap.
# Intruction
{task_prompt}"""

#! ==============================================

# Standard Library
import json
import re
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import List
from math import sqrt

# Third-Party Libraries
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
# from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer, set_seed
from transformers import AutoProcessor, AutoTokenizer, set_seed

# Project-Local Modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from qwen_vl_utils import process_vision_info
from iter_logger import init_iter_logger, append_iter_log  # log csv 기록 파일
from gui_actor.modeling_qwen25vl import Qwen2_5_VLForConditionalGenerationWithPointer
from gui_actor.multi_image_inference import inference
from visualize_util import (
    visualize_results, get_highest_attention_patch_bbox, 
    _visualize_early_exit_results, _visualize_stage1_results, _visualize_stage2_results, 
    visualize_crop, visualize_attn_map, visualize_aggregated_attention
)
from crop import crop  #! 어떤 crop 파일 사용?
from thop import profile #! flops

#! ==============================================

class ModelKwargsWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, inputs_dict):
        return self.model(**inputs_dict)

def get_model_inputs(msgs, tokenizer, processor, device): # profiler 한테 모델 입력 주기 위해서
    """Prepares model inputs from a conversation dictionary for FLOPs calculation."""
    user_content = next((item['content'] for item in msgs if item['role'] == 'user'), [])
    images = [content['image'] for content in user_content if content['type'] == 'image']
    text = tokenizer.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(text=[text], images=images, return_tensors='pt')
    # Move all tensor inputs to the correct device
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

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

def check_early_exit_condition(top_point, top_crop_id, crop_list, per_image_outputs, gt_bbox, original_image):
    """Early Exit 조건 확인"""
    if top_point is None or top_crop_id == -1:
        return False, False
    
    ori_w, ori_h = original_image.size
    
    # 썸네일의 최고 attention patch 찾기
    thumb_res = next((res for res in per_image_outputs if res['index'] == 0), None)
    thumb_top_patch_bbox = get_highest_attention_patch_bbox(thumb_res)
    l, t, r, b = thumb_top_patch_bbox
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
    
    return should_exit_early, early_exit_success

def run_selection_pass_with_guiactor(msgs, crop_list, gt_bbox: List, attn_vis_dir: str, original_image, img_path, instruction):
    """Stage 1 inference 및 Early Exit 판단"""
    
    # Inference 수행
    pred = inference(msgs, model, tokenizer, processor, use_placeholder=True, topk=3)
    per_image_outputs = pred["per_image"]
    
    # Attention scores 계산
    compute_attention_scores(crop_list, per_image_outputs)
    
    # Early Exit 체크
    should_exit_early, early_exit_success = False, False
    
    if EARLY_EXIT:
        top_point, top_crop_id = find_top_crop_for_early_exit(crop_list, per_image_outputs)
        should_exit_early, early_exit_success = check_early_exit_condition(
            top_point, top_crop_id, crop_list, per_image_outputs, gt_bbox, original_image
        )    # Early Exit하면 select_crop 스킵
    if should_exit_early:
        top_q_crop_ids = []
        top_q_bboxes = []
    else:
        # Select crop: score >= tau * max_score인 crops 선택
        top_q_crop_ids = select_crop(crop_list, tau=SELECT_THRE)
        top_q_bboxes = [crop["bbox"] for crop in crop_list if crop.get("id") in top_q_crop_ids]
    
    # 시각화 (필요시)
    if STAGE1_VIS and EARLY_EXIT and should_exit_early:
        _visualize_early_exit_results(crop_list, pred, gt_bbox, attn_vis_dir, instruction, img_path)
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
    pred = inference(s2_msgs, model, tokenizer, processor, use_placeholder=True, topk=3)
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

#! ================================================================================================

if __name__ == '__main__':

    set_seed(SEED)

    # Model Import
    model = Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
        MLLM_PATH, torch_dtype="auto", attn_implementation="sdpa",
        device_map={"": "cuda:0"},   # balanced -> 단일 GPU 고정
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MLLM_PATH)
    processor = AutoProcessor.from_pretrained(MLLM_PATH)
    model.eval()

    # save_dir 폴더명이 이미 존재하면 고유한 이름 생성 (save_dir -> save_dir_1 -> save_dir_2)
    save_dir = SAVE_DIR
    suffix = 0
    while os.path.exists(save_dir):
        suffix += 1
        save_dir = f"{SAVE_DIR}_{suffix}"
    os.makedirs(save_dir)

    # 바로바로 log csv, md 저장 어떻게 할지
    init_iter_logger(  
        save_dir=save_dir,
        headers=[  # 순서 그대로 들어감
            "idx", "crop_time", "num_crop", "early_exit", "num_selected_crop", 
            "s1_time", "s1_flops_gflops", "s1_hit", "s2_time", "s2_flops_gflops", "s2_hit", 
            "total_time", "total_flops_gflops", "acc_uptonow", "filename", "instruction"
        ],
        write_md=True,
        use_fsync=True,
        use_lock=False
    )

    # Process
    for task in TASKS:
        task_res = dict()
        dataset = "screenspot_" + task + "_v2.json"
        screenspot_data = json.load(open(os.path.join(SCREENSPOT_JSON, dataset), 'r'))
        screenspot_data = screenspot_data[SAMPLE_RANGE]

        print("Num of sample: " + str(len(screenspot_data)), flush=True)

        # 통계 변수 초기화
        task_res = []
        num_action = 0
        seg_time_sum = s1_time_sum = s2_time_sum = total_flops = 0.0
        early_exit_count = early_exit_success_count = final_success_count = 0

        for j, item in tqdm(enumerate(screenspot_data)):

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
            crop_list = crop(
                image_path=img_path,
                save_visualization=False
            )
            s0_crop_list = resize_crop_list(crop_list=crop_list, ratio=S1_RESIZE_RATIO)
            seg_end = time.time()
            seg_time = seg_end - seg_start

            if STAGE0_VIS and inst_dir:
                all_crops_bboxes = [crop["bbox"] for crop in s0_crop_list]
                visualize_crop(save_dir=inst_dir, gt_bbox=original_bbox, top_q_bboxes=all_crops_bboxes,
                                instruction=instruction, filename="s1_all_crop.png", img_path=img_path, click_point=None)

            #! ==================================================================
            #! Stage 1 | Find Top Q + Inference
            #! ==================================================================

            # Calculate Stage 1 FLOPs
            s1_msgs = create_guiactor_msgs(crop_list=s0_crop_list, instruction=instruction)
            with torch.no_grad():
                s1_inputs = get_model_inputs(s1_msgs, tokenizer, processor, model.device)
                wrapped_model = ModelKwargsWrapper(model)
                s1_flops, _ = profile(wrapped_model, inputs=(s1_inputs,), verbose=False)

            s1_start = time.time()

            s1_top_q_crop_ids, s1_top_q_bboxes, s0_crop_list_out, should_exit_early, early_exit_success = run_selection_pass_with_guiactor(
                msgs=s1_msgs,
                crop_list=s0_crop_list,
                gt_bbox=original_bbox,
                attn_vis_dir=s1_dir or "",
                original_image=original_image,
                img_path=img_path,
                instruction=instruction
            )
            s1_infence_end = time.time()

            if should_exit_early:
               early_exit_count +=1
               if early_exit_success:
                  early_exit_success_count += 1

            s1_time = s1_infence_end - s1_start
            seg_time_sum += seg_time
            s1_time_sum += s1_time

            # GT가 안에 들어가는지 체크
            s1_hit = early_exit_success or (not should_exit_early and check_gt_in_selected_crops(s1_top_q_bboxes, original_bbox))

            # 불필요한 딕셔너리 연산 제거 - 결과 저장용도만
            # res_board_dict는 사실상 미사용
            
            #! ==================================================================
            #! [Stage 2] Attention Refinement Pass
            #! ==================================================================
            
            # Early Exit
            if should_exit_early:
                final_success = early_exit_success
                s2_time = 0.0
                s2_flops = 0.0
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
                with torch.no_grad():
                    s2_inputs = get_model_inputs(s2_msgs, tokenizer, processor, model.device)
                    wrapped_model = ModelKwargsWrapper(model)
                    s2_flops, _ = profile(wrapped_model, inputs=(s2_inputs,), verbose=False)

                s2_inference_start = time.time()

                final_success = run_refinement_pass_with_guiactor(
                    crop_list=s2_input_crops,
                    instruction=instruction,
                    original_image=original_image,
                    save_dir=s2_dir or "",
                    gt_bbox=original_bbox,
                    img_path=img_path
                )
                s2_inference_end = time.time()
                s2_time = s2_inference_end - s2_inference_start
                
            # 성능 로깅
            total_time = seg_time + s1_time + s2_time
            total_flops_this = s1_flops + (s2_flops if not should_exit_early else 0)
            total_flops += total_flops_this

            print(f"✂️  Crops : {len(s0_crop_list_out)-1} | Select Crops : {len(s1_top_q_crop_ids)}")
            print(f"🕖 Times - Seg: {seg_time:.2f}s | S1: {s1_time:.2f}s | S2: {s2_time:.2f}s | Total: {total_time:.2f}s")
            print(f"🔥 FLOPs - S1: {s1_flops/1e9:.2f} | S2: {s2_flops/1e9:.2f} | Total: {total_flops_this/1e9:.2f} GFLOPs")
            print(f"{'✅🚨 Early Exit Success' if should_exit_early and early_exit_success else '❌🚨 Early Exit Fail' if should_exit_early else '✅ Grounding Success' if final_success else '❌ Grounding Fail'}")

            #! ==================================================================
            #! [End]
            #! ==================================================================

            s2_time_sum += s2_time
            final_success_count += final_success

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
                s1_flops_gflops=f"{s1_flops/1e9:.2f}",
                s1_hit="✅" if s1_hit else "❌",
                s2_time=f"{s2_time:.3f}",
                s2_flops_gflops=f"{s2_flops/1e9:.2f}" if not should_exit_early else "0.00",
                s2_hit="✅" if final_success else "❌",
                total_time=f"{total_time:.3f}",
                total_flops_gflops=f"{total_flops_this/1e9:.2f}",
                acc_uptonow=f"{up2now_gt_score:.2f}"
            )

            # JSON 기록 - 핵심 정보만
            item_res = {
                'filename': filename,
                'instruction': instruction,
                'gt_bbox': original_bbox,
                'num_crop': len(s0_crop_list) - 1,  # 썸네일 제외
                'early_exit': should_exit_early,
                'early_exit_success': early_exit_success,
                's1_hit': s1_hit,
                's2_hit': final_success,
                'seg_time': seg_time,
                's1_time': s1_time,
                's2_time': s2_time,
                'total_time': total_time,
                's1_flops': s1_flops,
                's2_flops': s2_flops if not should_exit_early else 0,
                'total_flops': total_flops_this
            }
            task_res.append(item_res)

        #! ==================================================
        # Json 정리
        with open(os.path.join(save_dir, dataset), "w") as f:
            json.dump(task_res, f, indent=4, ensure_ascii=False, cls=NpEncoder)

        # 최종 성능 메트릭 계산
        metrics = {
            "task": task,
            "total_samples": num_action,
            "accuracy": final_success_count / num_action * 100,
            "early_exit_rate": early_exit_count / num_action * 100,
            "early_exit_success_rate": early_exit_success_count / early_exit_count * 100 if early_exit_count > 0 else 0,
            "avg_times": {
                "segmentation": seg_time_sum / num_action,
                "stage1": s1_time_sum / num_action,
                "stage2": s2_time_sum / num_action,
                "total": (seg_time_sum + s1_time_sum + s2_time_sum) / num_action
            },
            "avg_flops_gflops": total_flops / num_action / 1e9,
        }

        with open(os.path.join(save_dir, f"{task}_metrics.json"), "w") as mf:
            json.dump(metrics, mf, ensure_ascii=False, indent=4)

        # 최종 결과 출력
        print("=" * 60)
        print(f"📊 Final Results for {task}:")
        print(f"Total Samples: {num_action}")
        print(f"Accuracy: {metrics['accuracy']:.2f}%")
        print(f"Early Exit Rate: {metrics['early_exit_rate']:.2f}%")
        print(f"Early Exit Success Rate: {metrics['early_exit_success_rate']:.2f}%") 
        print(f"Avg Times: Seg {metrics['avg_times']['segmentation']:.3f}s, S1 {metrics['avg_times']['stage1']:.3f}s, S2 {metrics['avg_times']['stage2']:.3f}s")
        print(f"Avg FLOPs: {metrics['avg_flops_gflops']:.2f} GFLOPs")
        print("=" * 60)