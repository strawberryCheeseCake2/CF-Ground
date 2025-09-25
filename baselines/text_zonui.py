'''
Final version
'''

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('gpu', type=int, default=0, help='GPU number')
parser.add_argument('--r', type=float, default=0.50, help='Stage 1 Resize ratio')
parser.add_argument('--th', type=float, default=0.11, help='Stage 1 Crop threshold')
parser.add_argument('--p', type=int, default=20, help='Stage 1 Crop Padding')
parser.add_argument('--v', action='store_true', help='Whether to save visualization images')
parser.add_argument('--mac', action='store_true', help='Whether to run on Mac (MPS)')
args = parser.parse_args()


args.mac = False
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
STAGE1_ENSEMBLE_RATIO = 0.50                        # Stage1 attention weight
STAGE2_ENSEMBLE_RATIO = 1 - STAGE1_ENSEMBLE_RATIO   # Stage2 crop weight
ENSEMBLE_TOP_PATCHES = 100                          # Stage2에서 앙상블에 사용할 상위 패치 개수 (Qwen2.5VL용)

# 최대 PIXELS 제한
MAX_PIXELS = 3211264  # Process단에서 적용
MIN_PIXELS = 256*28*28  # zonui 논문 세팅 그대로
# MAX_PIXELS = 1003520  # Process단에서 적용

# csv에 기록할 method 이름
method = "text_zonui"

# memo = f"resize{RESIZE_RATIO:.2f}_region_thresh{REGION_THRESHOLD:.2f}_pad{BBOX_PADDING}"
memo = f"text_baseline_10000000000"

#! Argument ==========================================================================================

SEED = 0

# Dataset & Model
MLLM_PATH = "zonghanHZH/ZonUI-3B"
SCREENSPOT_IMGS = "../data/screenspotv2_image"       # input image 경로
SCREENSPOT_JSON = "../data"                          # input image json파일 경로
TASKS = ["mobile", "web", "desktop"]
SAMPLE_RANGE = slice(None)
# SAMPLE_RANGE = slice(0, 10)

# Visualize & Logging
VISUALIZE = False
VIS_ONLY_WRONG = False                                # True면 틀린 것만 시각화, False면 모든 것 시각화
TFOPS_PROFILING = True
MEMORY_VIS = False

STAGE1_VIS = False
STAGE2_VIS = False

# Save Path
SAVE_DIR = f"../attn_output/" + method + "/" + memo

#! ==================================================================================================

# Standard Library
import os
import sys
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
)
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
from scipy.ndimage import gaussian_filter

# Project-Local Modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util.iter_logger import init_iter_logger, append_iter_log  # log csv 기록 파일
# from gui_actor.modeling_qwen25vl import Qwen2_5_VLForConditionalGenerationWithPointer
# from gui_actor.inference import inference
# from gui_actor.multi_image_inference import multi_image_inference
# Qwen2.5-VL base classes (Transformers)
from transformers import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

import matplotlib.pyplot as plt
from util.visualize_util import visualize_stage1_attention_crops, visualize_stage2_multi_attention, visualize_stage3_point_ensemble
from util.visualize_util_qwen25vl import draw_points_on_image
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

_SYSTEM = "Based on the screenshot of the page, I give a text description and you give its corresponding location. The coordinate represents a clickable location [x, y] for an element."


# _SCREENSPOT_SYSTEM = "Based on the screenshot of the page, I give a text description and you give its corresponding location."
# _SYSTEM_point = "The coordinate represents a clickable location [x, y] for an element, which is a relative coordinate on the screenshot, scaled from 0 to 1."
# _SYSTEM_point_int = "The coordinate represents a clickable location [x, y] for an element, which is a relative coordinate on the screenshot, scaled from 1 to 1000."

# _SCREENSPOT_USER = '<|image_1|>{system}{element}'

# def screenspot_to_qwen(element_name, image, xy_int=True):
    
#     transformed_data = []
#     user_content = []

#     if xy_int:
#         system_prompt = _SCREENSPOT_SYSTEM + ' ' + _SYSTEM_point_int
#     else:
#         system_prompt = _SCREENSPOT_SYSTEM + ' ' + _SYSTEM_point

#     '{system}<|image_1|>{element}'
#     user_content.append({"type": "text", "text": system_prompt})
#     user_content.append({"type": "image", "image": image})
    
#     user_content.append({"type": "text",  "text": element_name})

#     # question = _SCREENSPOT_USER.format(system=_SCREENSPOT_SYSTEM, element=element_name)
#     transformed_data.append(
#                 {
#                     "role": "user",
#                     "content": user_content,
#                 },
#             )
#     return transformed_data


def warm_up_model(model, processor, device):
    print("🏋️‍♂️ Warming up the model...")
    dummy_instruction = "Say: ready."
    dummy_image = Image.new("RGB", (640, 640), color=(255, 255, 255))
    msgs = [
        {"role": "user", "content": [
            {"type": "image", "image": dummy_image},
            {"type": "text",  "text": dummy_instruction},
        ]}
    ]
    # Qwen 권장 전처리 흐름
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(msgs)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=8)
    print("🏋️‍♂️ Warm-up complete!")



def _cast(n: str):
    return int(n) if n.isdigit() or (n.startswith(('+', '-')) and n[1:].isdigit()) else float(n)

def parse_coord(text: str):
    """문자열에서 (x, y) 또는 [x, y] 좌표를 모두 추출해 (x, y) 튜플 리스트로 반환."""
    coords = []
    PATTERN = re.compile(
        r"""
        [\(\[]                 # 여는 괄호 '(' 또는 '['
        \s*
        (?P<x>[+-]?\d+(?:\.\d+)?)   # x: 정수 또는 소수
        \s*,\s*
        (?P<y>[+-]?\d+(?:\.\d+)?)   # y: 정수 또는 소수
        \s*
        [\)\]]                 # 닫는 괄호 ')' 또는 ']'
        """,
        re.VERBOSE,
    )
    for m in PATTERN.finditer(text):
        x = _cast(m.group('x'))
        y = _cast(m.group('y'))
        coords.append((x, y))

    print(coords)
    return coords[0]

# def parse_coord(s: str):
#     _NUM = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
#     _PAIR_PAREN = rf"\(\s*({_NUM})\s*,\s*({_NUM})\s*\)"
#     _PAIR_BRACK = rf"\[\s*({_NUM})\s*,\s*({_NUM})\s*\]"
#     _PATTERN_ONE = re.compile(rf"{_PAIR_PAREN}|{_PAIR_BRACK}")
#     # m = re.search(
#     #     # r"\(\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*,\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*\)",
#     #     # s,
#     #     r"[\(\[]\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*,\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*[\)\]]",
#     # )
#     m = _PATTERN_ONE.search(s)
#     if not m:
#         raise ValueError("좌표를 찾지 못했습니다.")
#     x_str, y_str = m.groups()
#     print(x_str)
#     def cast(v: str):
#         # 정수 형태면 int, 아니면 float
#         return int(v) if re.fullmatch(r"[+-]?\d+", v) else float(v)
#     return cast(x_str), cast(y_str)

def denormalize_point(norm_point, orig_w, orig_h):
    
    new_x = norm_point[0] * orig_w
    new_y = norm_point[1] * orig_h
    return [new_x, new_y]

def normalize_point(denorm_point, orig_w, orig_h):
    
    x, y = denorm_point[0], denorm_point[1]

    if 0 <= x <= 1 and 0 <= y <= 1:
        return [x, y]

    new_x = x / orig_w
    new_y = y / orig_h
    return [new_x, new_y]

def create_conversation(image, instruction, resize_ratio=1.0):
    
    conversation = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        # Additional prompt
                        # f"This is a resized screenshot of the whole GUI, scaled by {resize_ratio}. "
                        # previous prompt
                        "You are a GUI agent. Given a screenshot of the current GUI and a human instruction, "
                        "your task is to locate the screen element that corresponds to the instruction. "
                        
                        "You should output a PyAutoGUI action that performs a click on the correct position. "
                        # "To indicate the click location, we will use some special tokens, which is used to refer to a visual patch later. "
                        # "For example, you can output: pyautogui.click(<your_special_token_here>)."
                        "For example, you can output: pyautogui.click(0, 0)."
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
                    "text": f"Where should you tap to {instruction}?",
                    "payload": instruction
                }
            ],
        },
    ]
    return conversation

def inference(converstaion):
    text = processor.apply_chat_template(
        converstaion, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(converstaion)

    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return response


def run_model(image, instruction):
    # conversation = create_conversation(image=image, instruction=instruction)
    
    # conversation = screenspot_to_qwen(element_name=instruction, image=image) # 메세지 생성

    _SYSTEM = "Based on the screenshot of the page, I give a text description and you give its corresponding location. The coordinate represents a clickable location [x, y] for an element."

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": _SYSTEM},
                {"type": "image", "image": image, "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS},
                {"type": "text", "text": instruction}
            ],
        }
    ]




    response = inference(conversation)
    print(response)
    print(type(response))
    if isinstance(response, list):
        response = response[0]
    try:
        coord = parse_coord(response)
    
        print(coord)

        orig_w, orig_h = image.size
        norm_coord = normalize_point(denorm_point=coord, orig_w=orig_w, orig_h=orig_h)
        denorm_coord = denormalize_point(norm_point=norm_coord, orig_w=orig_w, orig_h=orig_h)
    except:
        norm_coord = None
    
    pred = dict()
    pred['resize_ratio'] = 1.0
    pred['original_size'] = image.size
    pred['resized_size'] = image.size
    pred['top_point'] = norm_coord
    print(norm_coord)
    # print(denorm_coord)

    return pred


def point_in_bbox(point, bbox):
    """점이 bbox 안에 있는지 확인"""
    if point is None or bbox is None:
        return False
    return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]

#! ================================================================================================

if __name__ == '__main__':

    set_seed(SEED)


    device_map = "mps" if args.mac else "auto"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MLLM_PATH,
        torch_dtype="auto",
        attn_implementation=ATTN_IMPL,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MLLM_PATH)
    processor = AutoProcessor.from_pretrained(MLLM_PATH, max_pixels=MAX_PIXELS)
    
    if TFOPS_PROFILING:
        prof = FlopsProfiler(model)

    # warm_up_model(model, tokenizer, processor)
    device = "mps" if args.mac else "cuda" if torch.cuda.is_available() else "cpu"
    warm_up_model(model, processor, device)

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
        screenspot_data = json.load(open(os.path.join(SCREENSPOT_JSON, dataset), 'r'))
        screenspot_data = screenspot_data[SAMPLE_RANGE]

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


            if any([STAGE1_VIS, STAGE2_VIS]):
                inst_dir_name = re.sub(r'\W+', '_', instruction).strip('_')
                inst_dir = os.path.join(save_dir, "seg", filename_wo_ext, inst_dir_name)
                s1_dir = os.path.join(inst_dir, "stage1") 
                s2_dir = os.path.join(inst_dir, "stage2")
                os.makedirs(s1_dir, exist_ok=True)
                os.makedirs(s2_dir, exist_ok=True)
            else:
                inst_dir = s1_dir = s2_dir = None
            

            #! ==================================================================
            #! Stage 1 | Attention-based Crop Generation
            #! ==================================================================
            s1_time = s2_time = s3_time = 0
            stage1_success = stage2_success = stage3_success = False
            crop_hit = s1_hit = s2_hit = s3_hit = False
            s3_ensemble_point = s2_corrected_point = (0,0)

            if TFOPS_PROFILING:
                prof.reset_profile()

            s1_start = time.time()
            
            # s1_pred, s1_crop_list, num_crops, resized_image, scaled_gt_bbox 
            # inference

            s1_pred = run_model(image=original_image, instruction=instruction)
            

            s1_end = time.time()
            s1_time = s1_end - s1_start

            if TFOPS_PROFILING:
                s1_tflops = prof.get_total_flops() / 1e12

            # Stage1 Grounding 성공 여부 확인 (실제 예측 결과)
            s1_success = False
            s1_original_point = None
            if s1_pred and "top_point" in s1_pred and s1_pred["top_point"]:
                # s1_predicted_point = s1_pred["topk_points"][0]  # 정규화된 좌표 (0~1)
                s1_predicted_point = s1_pred["top_point"]  # 정규화된 좌표 (0~1)
                # 정규화된 좌표를 원본 이미지 픽셀 좌표로 변환
                s1_original_point = [
                    s1_predicted_point[0] * original_image.size[0],
                    s1_predicted_point[1] * original_image.size[1]
                ]
                print(s1_original_point)
                print(original_bbox)
                s1_success = point_in_bbox(s1_original_point, original_bbox)
            
            s1_hit = "✅" if s1_success else "❌"
            if s1_success:
                stage1_success_count += 1

            # GT bbox와 crop bbox가 겹치는지 확인 (교집합이 있으면 성공)
            crop_success = False
            # for crop in s1_crop_list:
            #     crop_bbox = crop["bbox"]
            #     # crop_bbox: [left, top, right, bottom], original_bbox: [left, top, right, bottom]
            #     left = max(crop_bbox[0], original_bbox[0])
            #     top = max(crop_bbox[1], original_bbox[1])
            #     right = min(crop_bbox[2], original_bbox[2])
            #     bottom = min(crop_bbox[3], original_bbox[3])
            #     if left < right and top < bottom:
            #         crop_success = True
            #         break
            
            # crop_hit = "✅" if crop_success else "❌"
            # if crop_success:
            #     crop_success_count += 1

           

              
                
                

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

            num_attention_crops = 0
            print(f"Task: {task}")
            # print(f"🖼️ Image: {filename} {orig_w}x{orig_h} (Resize Ratio : {s1_pred['resize_ratio']})")
            print(f"✂️  Attention Crops : {num_attention_crops}")
            print(f"🕖 Times - S1: {s1_time:.2f}s | S2: {s2_time:.2f}s | Total: {total_time:.2f}s")
            if TFOPS_PROFILING:
                print(f"🔥 FLOPs - S1: {s1_tflops:.2f} | S2: {s2_tflops:.2f} | Total: {total_tflops_this:.2f} TFLOPs")
            print(f"{'✅ Success' if s1_success else '❌🎯 Fail'}")

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