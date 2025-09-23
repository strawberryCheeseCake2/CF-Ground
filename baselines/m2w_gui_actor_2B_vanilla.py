'''
Final version
'''

# _query 이걸로 모델 추론하면 되는거

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

# Connected Region Based Cropping
REGION_THRESHOLD = args.th              # 연결된 영역 검출을 위한 임계값 (0~1)  # TODO: 0.1 ~ 0.5 중 최적 찾기
MIN_PATCHES = 1                         # 최소 패치 수 (너무 작은 영역 제거)
BBOX_PADDING = args.p                   # bbox 상하좌우로 확장할 픽셀  # TODO: 0 ~ 50 중 최적 찾기

# Ensemble Hyperparameters

# 최대 PIXELS 제한
MAX_PIXELS = 3211264  # Process단에서 적용

# csv에 기록할 method 이름
method = "mind2web_qwen2vl_vanilla_0924"

memo = f"0923_gpt4o"
#! Argument ==========================================================================================

SEED = 0

# Dataset & Model
MLLM_PATH = "microsoft/GUI-Actor-2B-Qwen2-VL"
SCREENSPOT_IMGS = "/data/hj/UGround/offline_evaluation/Multimodal-Mind2Web/data/sample/cross_"       # input image 경로
SCREENSPOT_JSON = "../data"                          # input image json파일 경로
# TASKS = ["domain", "website", "task"]
TASKS = ["domain"]
SAMPLE_RANGE = slice(None)

# Visualize & Logging
VISUALIZE = args.v if args.v else True  # 시각화 여부
VIS_ONLY_WRONG = False                                # True면 틀린 것만 시각화, False면 모든 것 시각화
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
# from gui_actor.modeling_qwen25vl import Qwen2_5_VLForConditionalGenerationWithPointer
from gui_actor.modeling import Qwen2VLForConditionalGenerationWithPointer
from gui_actor.inference import inference, ForceFollowTokensLogitsProcessor
from gui_actor.constants import DEFAULT_POINTER_PAD_TOKEN, DEFAULT_POINTER_END_TOKEN
if TFOPS_PROFILING:
    from deepspeed.profiling.flops_profiler import FlopsProfiler

#! ==============================================================================================


# 

DIR_MAX_LEN = 100

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
    dummy_msgs = create_conversation_stage1(image=dummy_image, instruction=dummy_instruction)
    
    # 예열용 inference 실행
    for _ in range(3):  # 3번 반복
        with torch.no_grad():
            _ = inference(dummy_msgs, model, tokenizer, processor, use_placeholder=True, topk=3)
    print("🏋️‍♂️ Warm-up complete!")

def create_conversation_stage1(image, instruction):
    conversation = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
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



def point_in_bbox(point, bbox):
    """점이 bbox 안에 있는지 확인"""
    if point is None or bbox is None:
        return False
    return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]

#! ================================================================================================

if __name__ == '__main__':
    
    set_seed(SEED)

    # Model Import (NVIDIA CUDA)
    model = Qwen2VLForConditionalGenerationWithPointer.from_pretrained(
        MLLM_PATH, torch_dtype="auto", attn_implementation=ATTN_IMPL,
        device_map="balanced",
        # max_memory=max_memory, 
        low_cpu_mem_usage=True
    )
    # Qwen2VL
    # grounding_system_message = "You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform a series of pyautogui actions to complete the task."
    # Qwen2.5VL
    grounding_system_message = "You are a GUI agent. Given a screenshot of the current GUI and a human instruction, your task is to locate the screen element that corresponds to the instruction. You should output a PyAutoGUI action that performs a click on the correct position. To indicate the click location, we will use some special tokens, which is used to refer to a visual patch later. For example, you can output: pyautogui.click(<your_special_token_here>)."
    # Model Import (Mac)
    # model = Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
    #     MLLM_PATH, torch_dtype="auto", attn_implementation=ATTN_IMPL,
    #     device_map="mps", # Mac
    #     # max_memory=max_memory, 
    #     low_cpu_mem_usage=False
    # )
    tokenizer = AutoTokenizer.from_pretrained(MLLM_PATH)
    processor = AutoProcessor.from_pretrained(MLLM_PATH, max_pixels=MAX_PIXELS)
    logits_processor_pointer = ForceFollowTokensLogitsProcessor(
        token_a_id=tokenizer.encode(DEFAULT_POINTER_PAD_TOKEN)[0],
        forced_sequence=[
            tokenizer.encode(DEFAULT_POINTER_END_TOKEN)[0]
        ]
    )
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
    total_success = 0
    total_s1_time = 0.0
    total_s1_tflops = 0.0

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
    # data_folder = '/data/hj/UGround/offline_evaluation/Multimodal-Mind2Web/data/gpt-4-turbo_results/'
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
        time_sum = 0.0
        tflops_sum = 0.0
        success_count = 0
        
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

            if TFOPS_PROFILING:
                prof.reset_profile()

            s1_start = time.time()
            
            original_bbox = ground_truth_bboxes[0] # 임시로 막아둠.

            conversation = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": grounding_system_message,
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            # "image": example["image"], # PIL.Image.Image or str to path
                            "image": original_image, # PIL.Image.Image or str to path
                            # "image_url": "https://xxxxx.png" or "https://xxxxx.jpg" or "file://xxxxx.png" or "data:image/png;base64,xxxxxxxx", will be split by "base64,"
                        },
                        {
                            "type": "text",
                            "text": item["description"]
                        },
                    ],
                },
            ]
            
            pred = inference(conversation, model, tokenizer, processor, logits_processor=logits_processor_pointer)

            s1_end = time.time()
            s1_time = s1_end - s1_start

            if TFOPS_PROFILING:
                s1_tflops = prof.get_total_flops() / 1e12

            # Stage1 Grounding 성공 여부 확인 (실제 예측 결과)
            success = False
            original_point = None
            if pred and "topk_points" in pred and pred["topk_points"]:
                predicted_point = pred["topk_points"][0]  # 정규화된 좌표 (0~1)
                # 정규화된 좌표를 원본 이미지 픽셀 좌표로 변환
                original_point = [
                    predicted_point[0] * original_image.size[0],
                    predicted_point[1] * original_image.size[1]
                ]
                success = any(point_in_bbox(original_point, bbox) for bbox in ground_truth_bboxes)
            
            
            s1_hit = "✅" if success else "❌"
            if success:
                success_count += 1

            #! ==================================================================
            #! [Visualization - After Time Measurement]
            #! ==================================================================
            

            #! ==================================================================
            #! [Common Processing]
            #! ==================================================================
            
            # 공통 통계 업데이트
            total_time = s1_time
            time_sum += s1_time
            tflops_sum += s1_tflops

            print(f"🕖 Times - S1: {s1_time:.2f}s")
            if TFOPS_PROFILING:
                print(f"🔥 FLOPs - S1: {s1_tflops:.2f} TFLOPs")
            print(f"{'✅ Success' if success else '❌🎯 Fail'}")

            #! ==================================================================
            #! [Statistics & Logging]
            #! ==================================================================

            # data_source별 통계 업데이트
            if data_source not in data_source_stats:
                data_source_stats[data_source] = {
                    'num_action': 0,
                    'time_sum': 0.0,
                    'tflops_sum': 0.0,
                    'success_count': 0
                }
            
            stats = data_source_stats[data_source]
            stats['num_action'] += 1
            stats['time_sum'] += s1_time
            if TFOPS_PROFILING:
                stats['tflops_sum'] += s1_tflops
            if success:
                stats['success_count'] += 1

            up2now_s1_score = stats['success_count'] / stats['num_action'] * 100
            # print(f"Up2Now Crop Accuracy: {up2now_crop_score:.2f}%")
            print(f"Up2Now Accuracy: {up2now_s1_score:.2f}%")

            # Iter log - 개선된 로깅
            append_iter_log(
                idx=j+1,
                orig_w=original_image.size[0],
                orig_h=original_image.size[1],
                s1_time=f"{s1_time:.3f}",
                s1_tflops=f"{s1_tflops:.2f}",
                s1_hit=s1_hit,
                s1_acc_uptonow=f"{up2now_s1_score:.2f}",
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
                'stage1_success': success,
                's1_hit': s1_hit,
                'original_point': original_point,
                's1_time': s1_time,
                'total_time': total_time,
                'total_tflops': s1_tflops
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
            "stage1_accuracy": success_count / num_action * 100,
            "avg_times": {
                "stage1": time_sum / num_action,
            },
            "avg_flops_tflops": {
                "stage1": tflops_sum / num_action,
            },
            "hyperparameters": {
                "region_threshold": REGION_THRESHOLD,
                "bbox_padding": BBOX_PADDING,
                "min_patches": MIN_PATCHES,
                "attn_impl": ATTN_IMPL
            }
        }

        with open(os.path.join(save_dir, f"results_{task}.json"), "w") as mf:
            json.dump(metrics, mf, ensure_ascii=False, indent=4)

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
            round(metrics['stage1_accuracy'], 2),
            round(metrics['avg_times']['stage1'], 4),
            round(metrics['avg_flops_tflops']['stage1'], 2),
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
        total_success += success_count
        total_s1_time += time_sum
        total_s1_tflops += tflops_sum

        # 최종 결과 출력
        print("=" * 60)
        print(f"📊 Final Results for {task}:")
        print(f"Total Samples: {num_action}")
        print(f"Stage1 Accuracy: {metrics['stage1_accuracy']:.2f}%")
        print(f"Avg Times: {metrics['avg_times']['stage1']:.3f}s")
        print(f"Avg FLOPs: {metrics['avg_flops_tflops']['stage1']:.2f} TFLOPs")
        
        print("=" * 60)

    print("\n📊 All Task Done!")

    # 전체 결과 계산 및 저장
    total_success_rate = total_success / total_samples
    
    # 전체 평균 시간
    avg_s1_time = total_s1_time / total_samples
    
    # 전체 평균 TFLOPS
    avg_s1_tflops = total_s1_tflops / total_samples
    
    print(f"Total Sample num: {total_samples}")
    print(f"Total Stage1 Success Rate: {total_success_rate:.4f}")
    print(f"Total avg Stage1 time: {avg_s1_time:.4f}s")
    print(f"Total avg Stage1 TFLOPS: {avg_s1_tflops:.4f}")
    
    # 전체 결과를 CSV로 저장
    cumulative_csv_path = os.path.join("../_results", "results_all.csv")
    
    # 전체 결과 CSV 행 생성
    cumulative_csv_row = [
        method,
        RESIZE_RATIO, REGION_THRESHOLD, BBOX_PADDING,
        total_samples,
        round(total_success_rate * 100, 2),
        round(avg_s1_time, 4),
        round(avg_s1_tflops, 2),
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