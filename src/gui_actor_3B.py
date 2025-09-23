# GUI-Actor-3B + GOLD

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('gpu', type=int, default=0, help='GPU number')
parser.add_argument('--r', type=float, default=0.50, help='Stage 1 Resize ratio')
parser.add_argument('--th', type=float, default=0.12, help='Stage 1 Crop threshold')
parser.add_argument('--p', type=int, default=0, help='Stage 1 Crop Padding')
parser.add_argument('--e', type=float, default=0.7, help='Ensemble ratio for Stage 1 (0~1)')
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
MAX_CROPS = 3  # Maximum number of crops

# Connected Region Based Cropping
REGION_THRESHOLD = args.th              # Threshold for connected region detection (0~1)
MIN_PATCHES = 1                         # Minimum number of patches (remove too small regions)
BBOX_PADDING = args.p                   # Pixels to expand bbox in all directions

# Ensemble Hyperparameters
STAGE1_ENSEMBLE_RATIO = args.e                      # Stage1 attention 가중치
STAGE2_ENSEMBLE_RATIO = 1 - STAGE1_ENSEMBLE_RATIO   # Stage2 crop 가중치

# Maximum PIXELS limit (applied at Process level)
MAX_PIXELS = 3211264
# MAX_PIXELS = 1280*28*28

# Experiment name to record in csv
model_name = "gui-actor-3B+GOLD"
experiment = "visualize"
parameter = f"resize{RESIZE_RATIO:.2f}_maxpixel{MAX_PIXELS}_ensemble{STAGE1_ENSEMBLE_RATIO:.2f}"
SAVE_DIR = f"../attn_output/" + f"{model_name}/" + f"{experiment}/" + parameter

#! Argument ==========================================================================================

SEED = 0

# Dataset & Model
MLLM_PATH = "microsoft/GUI-Actor-3B-Qwen2.5-VL"
SCREENSPOT_IMGS = "../data/screenspotv2_image"
SCREENSPOT_JSON = "../data"
TASKS = ["mobile", "web", "desktop"]
SAMPLE_RANGE = slice(None)

# Visualize & Logging
VISUALIZE = args.v if args.v else False
VIS_ONLY_WRONG = False
TFOPS_PROFILING = True
MEMORY_VIS = False


#! ==================================================================================================

# Standard Library
import os
import sys
import time
import re
import json
from typing import Dict, List
from collections import deque
import logging
logging.disable(logging.CRITICAL)

# Third-Party Libraries
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoProcessor, AutoTokenizer, set_seed

# Project-Local Modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util.iter_logger import init_iter_logger, append_iter_log
from gui_actor.modeling_qwen25vl import Qwen2_5_VLForConditionalGenerationWithPointer
from gui_actor.inference import inference
from gui_actor.multi_image_inference import multi_image_inference
from util.visualize_util import visualize_stage1_attention_crops, visualize_stage2_multi_attention, visualize_stage3_point_ensemble
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
    dummy_image = Image.new("RGB", (1000, 1000), color=(255, 255, 255))
    dummy_msgs = create_conversation_stage1(image=dummy_image, instruction=dummy_instruction, resize_ratio=1.0)
    for _ in range(3):
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

def get_connected_region_bboxes_from_scores(
    image_result: Dict,
    threshold: float,
    min_patches: int
) -> List[Dict]:
    # 1) thresholding
    attn_scores_1d = np.array(image_result["attn_scores"][0], dtype=np.float32)
    n_w = int(image_result["n_width"])
    n_h = int(image_result["n_height"])
    attn = attn_scores_1d.reshape(n_h, n_w)
    
    vmax = float(attn.max()) if attn.size > 0 else 0.0
    thr_val = float(vmax * threshold)
    
    # 2) make mask
    mask = (attn >= thr_val)

    # 3) BFS to find connected regions
    visited = np.zeros_like(mask, dtype=bool)
    regions = []
    neighbors = [(di, dj) for di in (-1,0,1) for dj in (-1,0,1) if not (di==0 and dj==0)]  # 8 directions
    # neighbors = [(-1,0), (1,0), (0,-1), (0,1)]  # TODO: 4 neighbors

    for y in range(n_h):
        for x in range(n_w):
            if not mask[y, x] or visited[y, x]:
                continue

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

    # 4) calculate bbox and scores for each region
    out = []
    for region in regions:
        ys = [p[0] for p in region]
        xs = [p[1] for p in region]
        y_min, y_max = min(ys), max(ys)
        x_min, x_max = min(xs), max(xs)
        
        l = x_min / n_w
        t = y_min / n_h  
        r = (x_max + 1) / n_w
        b = (y_max + 1) / n_h
        
        region_scores = attn[ys, xs]
        score_sum = float(region_scores.sum())
        score_mean = float(region_scores.mean())
        score_max = float(region_scores.max())
        
        out.append({
            "bbox": [l, t, r, b],
            "patch_bbox": [int(x_min), int(y_min), int(x_max), int(y_max)],
            "size": int(len(region)),
            "score_sum": score_sum,
            "score_mean": score_mean,
            "score_max": score_max,
            "score_norm": score_sum / (vmax * len(region) + 1e-9),
        })
    
    # 5) sort by score_sum
    out.sort(key=lambda x: x["score_sum"], reverse=True)

    return out

def run_stage1_attention_inference(original_image, instruction):
    """Stage 1: resize and inference"""

    orig_w, orig_h = original_image.size
    resize_ratio = RESIZE_RATIO
    resized_w, resized_h = int(orig_w * resize_ratio), int(orig_h * resize_ratio)
    resized_image = original_image.resize((resized_w, resized_h))
    
    conversation = create_conversation_stage1(resized_image, instruction, resize_ratio)
    pred = inference(conversation, model, tokenizer, processor, use_placeholder=True, topk=1)
    
    pred['resize_ratio'] = resize_ratio
    pred['original_size'] = (orig_w, orig_h)
    pred['resized_size'] = resized_image.size
    
    return pred, resized_image

def remove_contained_bboxes(regions):
    """Remove completely contained bboxes (keep only larger bboxes)"""
    if len(regions) <= 1:
        return regions

    filtered_regions = []
    
    for i, region in enumerate(regions):
        bbox_i = region['bbox_padded']  # [left, top, right, bottom]
        is_contained = False
        
        # Compare with already filtered regions
        for filtered_region in filtered_regions:
            bbox_f = filtered_region['bbox_padded']
            
            # Check if region is completely contained in filtered_region
            if (bbox_f[0] <= bbox_i[0] and  # left
                bbox_f[1] <= bbox_i[1] and  # top  
                bbox_f[2] >= bbox_i[2] and  # right
                bbox_f[3] >= bbox_i[3]):    # bottom
                is_contained = True
                break
            
            # Check if filtered_region is completely contained in region (found larger region)
            elif (bbox_i[0] <= bbox_f[0] and  # left
                  bbox_i[1] <= bbox_f[1] and  # top
                  bbox_i[2] >= bbox_f[2] and  # right
                  bbox_i[3] >= bbox_f[3]):    # bottom
                # Found larger region, so remove existing one and add new one
                filtered_regions.remove(filtered_region)
                break
        
        # Add only if not contained
        if not is_contained:
            filtered_regions.append(region)
    
    return filtered_regions

def find_connected_regions(pred_result, resized_image, resize_ratio):
    """Find connected regions from attention"""

    regions = get_connected_region_bboxes_from_scores(
        image_result=pred_result,
        threshold=REGION_THRESHOLD,
        min_patches=MIN_PATCHES
    )
    
    resized_w, resized_h = resized_image.size
    orig_w = resized_w / resize_ratio
    orig_h = resized_h / resize_ratio
    
    # Convert each region to original image size and compose information
    connected_regions = []
    for i, region in enumerate(regions):
        # Convert normalized bbox to resized image pixel coordinates
        l, t, r, b = region["bbox"]  # Normalized coordinates (0~1)
        
        # Pixel coordinates in resized image
        resized_left = l * resized_w
        resized_top = t * resized_h
        resized_right = r * resized_w
        resized_bottom = b * resized_h
        
        # Convert to original image size
        orig_left = resized_left / resize_ratio
        orig_top = resized_top / resize_ratio
        orig_right = resized_right / resize_ratio
        orig_bottom = resized_bottom / resize_ratio
        
        # Apply padding to bbox
        padded_left = max(0, int(orig_left - BBOX_PADDING))
        padded_top = max(0, int(orig_top - BBOX_PADDING))
        padded_right = min(orig_w, int(orig_right + BBOX_PADDING))
        padded_bottom = min(orig_h, int(orig_bottom + BBOX_PADDING))
        
        # Calculate region center point (based on bbox before padding)
        center_x = (orig_left + orig_right) / 2
        center_y = (orig_top + orig_bottom) / 2
        
        connected_regions.append({
            'center_x': center_x,
            'center_y': center_y,
            'score_sum': region["score_sum"],  # Sum of scores within region
            'score_mean': region["score_mean"],  # Average score within region
            'score_max': region["score_max"],  # Maximum score within region
            'size': region["size"],  # Number of patches
            'bbox_original': [int(orig_left), int(orig_top), int(orig_right), int(orig_bottom)],  # bbox before padding
            'bbox_padded': [padded_left, padded_top, padded_right, padded_bottom],  # bbox after padding (for actual cropping)
            'region_info': region  # Original region information
        })
    
    # Sort by sum of scores within region (descending)
    connected_regions.sort(key=lambda x: x['score_sum'], reverse=True)

    # Sort by maximum score within region (descending)
    # connected_regions.sort(key=lambda x: x['score_max'], reverse=True)

    # Remove completely contained bboxes (keep only larger bboxes)
    connected_regions = remove_contained_bboxes(connected_regions)
    
    return connected_regions

def create_crops_from_connected_regions(regions, original_image):
    """Crop directly from original image based on connected regions"""
    
    if not regions:
        return []
    
    crops = []
    
    for i, region in enumerate(regions):
        bbox = region['bbox_padded']  # Use bbox with padding applied
        crop_img = original_image.crop(bbox)
        
        crops.append({
            'img': crop_img,
            'bbox': bbox,
            'score': region['score_sum'],
            'id': i + 1,
            'region_info': region  # Include original region information
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
    """Convert multi_image_inference results to original image coordinates"""
    
    # Convert each crop's results to original coordinates
    converted_results = []
    all_candidates = []
    
    for img_idx, img_result in enumerate(multi_pred['per_image']):
        if img_idx >= len(crop_list):
            continue
            
        crop_info = crop_list[img_idx]
        crop_bbox = crop_info['bbox']  # [left, top, right, bottom]
        crop_width = crop_bbox[2] - crop_bbox[0]
        crop_height = crop_bbox[3] - crop_bbox[1]
        
        # Convert topk results of the image to original coordinates
        crop_candidates = []
        for point_idx, (point, score) in enumerate(zip(img_result['topk_points'], img_result['topk_values'])):
            # Convert normalized coordinates to pixel coordinates within crop
            crop_x = point[0] * crop_width
            crop_y = point[1] * crop_height
            
            # Convert crop coordinates to original image coordinates
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
    
    # Sort all candidates by score
    all_candidates.sort(key=lambda x: x['score'], reverse=True)
    
    return all_candidates

def run_stage1_attention_based(original_image, instruction, gt_bbox):
    """New simple Stage 1: Connected region-based crop generation"""
    
    # 1. Resize and inference
    s1_pred, resized_image = run_stage1_attention_inference(original_image, instruction)
    
    # 2. Adjust GT bbox according to resize ratio
    resize_ratio = s1_pred['resize_ratio']
    scaled_gt_bbox = [coord * resize_ratio for coord in gt_bbox]
    
    # 3. Find connected regions
    regions = find_connected_regions(s1_pred, resized_image, resize_ratio)

    regions = regions[:MAX_CROPS]
    
    # 5. Generate crops directly from original image
    crops = create_crops_from_connected_regions(regions, original_image)
    
    num_crops = len(crops)
    
    return s1_pred, crops, num_crops, resized_image, scaled_gt_bbox

def get_stage1_score_at_point(point, s1_attn_scores, s1_n_width, s1_n_height, original_size, resize_ratio):
    """Calculate Stage1 attention score at specific point"""
    
    orig_w, orig_h = original_size
    point_x, point_y = point
    
    # Convert original coordinates to resized coordinates
    resized_x = point_x * resize_ratio
    resized_y = point_y * resize_ratio
    
    # Convert resized coordinates to patch coordinates
    resized_w = orig_w * resize_ratio
    resized_h = orig_h * resize_ratio
    
    patch_x = int((resized_x / resized_w) * s1_n_width)
    patch_y = int((resized_y / resized_h) * s1_n_height)
    
    # Check if patch coordinates are within valid range
    patch_x = max(0, min(patch_x, s1_n_width - 1))
    patch_y = max(0, min(patch_y, s1_n_height - 1))
    
    # Return attention score of the corresponding patch
    patch_idx = patch_y * s1_n_width + patch_x
    if patch_idx < len(s1_attn_scores):
        return float(s1_attn_scores[patch_idx])
    else:
        return 0.0

def point_in_bbox(point, bbox):
    """Check if point is inside bbox"""
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
        # max_memory=max_memory, 
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MLLM_PATH)
    processor = AutoProcessor.from_pretrained(MLLM_PATH, max_pixels=MAX_PIXELS)
    
    if TFOPS_PROFILING:
        prof = FlopsProfiler(model)

    warm_up_model(model, tokenizer, processor)

    if TFOPS_PROFILING:
        prof.start_profile()

    # Generate unique name if save_dir folder already exists (save_dir -> save_dir_1 -> save_dir_2)
    save_dir = SAVE_DIR
    suffix = 0
    while os.path.exists(save_dir):
        suffix += 1
        save_dir = f"{SAVE_DIR}_{suffix}"
    os.makedirs(save_dir)

    # Initialize overall task statistics variables
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

    # Define CSV headers (commonly used across all tasks)
    csv_headers = [
        "model_name", "experiment",
        "resize_ratio", "region_threshold", "bbox_padding","ensemble_ratio",
        "total_samples", "crop_accuracy", "stage1_accuracy", "stage2_accuracy", "stage3_accuracy",
        "avg_stage1_time", "avg_stage2_time", "avg_stage3_time", "avg_total_time",
        "avg_stage1_tflops", "avg_stage2_tflops", "avg_total_tflops",
        "timestamp"
    ]

    # Process
    for task in TASKS:
        # Create separate log file for each task
        init_iter_logger(  
            save_dir=save_dir,
            csv_name=f"iter_log_{task}.csv",
            md_name=f"iter_log_{task}.md",
            headers=[  # Entered in order
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

        # Initialize statistics variables
        task_res = []
        num_action = 0
        s1_time_sum = s2_time_sum = s3_time_sum = 0.0
        s1_tflops_sum = s2_tflops_sum = 0.0
        crop_success_count = stage1_success_count = stage2_success_count = stage3_success_count = 0
        
        # Initialize statistics variables by data_source
        data_source_stats = {}

        if MEMORY_VIS:
            memory_dir = os.path.join(save_dir, "gpu_usage", task)
            os.makedirs(memory_dir, exist_ok=True)

        for j, item in tqdm(enumerate(screenspot_data)):

            s1_tflops = s2_tflops = 0.0
            num_action += 1

            print("\n\n----------------------\n")
            
            # Load file and data
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

            # Extract data_source information (default to "unknown" if not present)
            data_source = item.get("data_source", "unknown")

            #! ==================================================================
            #! Stage 1 | Attention-based Crop Generation
            #! ==================================================================

            if TFOPS_PROFILING:
                prof.reset_profile()

            s1_start = time.time()
            
            s1_pred, s1_crop_list, num_crops, resized_image, scaled_gt_bbox = run_stage1_attention_based(
                original_image=original_image,
                instruction=instruction,
                gt_bbox=original_bbox
            )

            s1_end = time.time()
            s1_time = s1_end - s1_start

            if TFOPS_PROFILING:
                s1_tflops = prof.get_total_flops() / 1e12

            # Check Stage1 Grounding success (actual prediction result)
            s1_success = False
            s1_original_point = None
            if s1_pred and "topk_points" in s1_pred and s1_pred["topk_points"]:
                s1_predicted_point = s1_pred["topk_points"][0]  # Normalized coordinates (0~1)
                # Convert normalized coordinates to original image pixel coordinates
                s1_original_point = [
                    s1_predicted_point[0] * original_image.size[0],
                    s1_predicted_point[1] * original_image.size[1]
                ]
                s1_success = point_in_bbox(s1_original_point, original_bbox)
            
            s1_hit = "✅" if s1_success else "❌"
            if s1_success:
                stage1_success_count += 1

            # Check if GT bbox and crop bbox overlap (success if intersection exists)
            crop_success = False
            for crop in s1_crop_list:
                crop_bbox = crop["bbox"]
                # crop_bbox: [left, top, right, bottom], original_bbox: [left, top, right, bottom]
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
            
            # Multi-image inference
            s2_pred = run_stage2_multi_image_inference(s1_crop_list, instruction)

            # Convert Stage2 multi-image results to original coordinates
            s2_all_candidates = convert_multi_image_results_to_original(s2_pred, s1_crop_list)
            
            # Check Stage2 success
            s2_corrected_point = s2_all_candidates[0]['point']  # Highest point
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
            # Stage1 attention information
            s1_attn_scores = np.array(s1_pred['attn_scores'][0])
            s1_n_width = s1_pred['n_width']
            s1_n_height = s1_pred['n_height']
            s1_resize_ratio = s1_pred['resize_ratio']
            
            # Get Stage1 attention maximum score
            s1_max_score = float(max(s1_attn_scores)) if len(s1_attn_scores) > 0 else 1.0
            
            # Get Stage2 topk candidate maximum score
            s2_topk_scores = [candidate['score'] for candidate in s2_all_candidates]
            s2_max_score = max(s2_topk_scores)

            # Calculate ensemble score for each Stage2 topk point
            ensemble_candidates = []
            
            for i, candidate in enumerate(s2_all_candidates):
                s2_original_point = candidate['point']
                
                # Calculate Stage1 score at that point (normalized value)
                s1_raw_score = get_stage1_score_at_point(
                    s2_original_point, s1_attn_scores, s1_n_width, s1_n_height, 
                    original_image.size, s1_resize_ratio
                )

                # Normalize each score based on maximum
                s1_score = s1_raw_score / s1_max_score
                s2_score = candidate['score'] / s2_max_score
                
                # Calculate ensemble score
                ensemble_score = STAGE1_ENSEMBLE_RATIO * s1_score + STAGE2_ENSEMBLE_RATIO * s2_score
                
                ensemble_candidates.append({
                    'point': s2_original_point,
                    'score': ensemble_score,
                    's1_score': s1_score,
                    's2_score': s2_score,
                    'crop_id': candidate['crop_id'],
                    'rank_in_crop': candidate['rank_in_crop'],
                    's2_rank': i + 1  # Rank within topk
                })
            
            # Select point with highest score
            best_candidate = max(ensemble_candidates, key=lambda x: x['score'])
            s3_ensemble_point = best_candidate['point']

            s3_end = time.time()
            s3_time = s3_end - s3_start
            
            # Save candidates for visualization
            s3_ensemble_candidates = ensemble_candidates
            
            # Check success with ensemble result
            stage3_success = point_in_bbox(s3_ensemble_point, original_bbox)
            
            s3_hit = "✅" if stage3_success else "❌"
            if stage3_success:
                stage3_success_count += 1

            #! ==================================================================
            #! [Visualization - After Time Measurement]
            #! ==================================================================
            
            # if VISUALIZE and (not VIS_ONLY_WRONG or not stage3_success):
            if VISUALIZE and (not VIS_ONLY_WRONG or not stage3_success) and num_crops >= 2:
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
                
                # Stage2 Multi-Image visualization
                if s2_pred and s1_crop_list:  # Visualize only when Stage2 results exist
                    visualize_stage2_multi_attention(
                        s2_pred=s2_pred,
                        crop_list=s1_crop_list,
                        original_image=original_image,
                        save_dir=inst_dir,
                        instruction=instruction,
                        predicted_point=s2_corrected_point
                    )
                
                # Stage3 ensemble visualization
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

            #! ==================================================================
            #! [Common Processing]
            #! ==================================================================
            
            # Update common statistics
            s1_time_sum += s1_time
            s2_time_sum += s2_time
            s3_time_sum += s3_time
            s1_tflops_sum += s1_tflops
            s2_tflops_sum += s2_tflops
                
            # Performance logging
            total_time = s1_time + s2_time
            if TFOPS_PROFILING:
                total_tflops_this = s1_tflops + s2_tflops  # Stage3는 FLOPs 제외

            num_attention_crops = len(s1_crop_list)
            print(f"Task: {task}")
            print(f"🖼️ Image: {filename} {orig_w}x{orig_h} (Resize Ratio : {s1_pred['resize_ratio']})")
            print(f"✂️  Attention Crops : {num_attention_crops}")
            print(f"🕖 Times - S1: {s1_time:.2f}s | S2: {s2_time:.2f}s | Total: {total_time:.2f}s")
            if TFOPS_PROFILING:
                print(f"🔥 FLOPs - S1: {s1_tflops:.2f} | S2: {s2_tflops:.2f} | Total: {total_tflops_this:.2f} TFLOPs")
            print(f"{'✅ Success' if stage3_success else '❌🎯 Fail'}")

            #! ==================================================================
            #! [Statistics & Logging]
            #! ==================================================================

            # Update statistics by data_source
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

            # Iter log - Improved logging
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

            # JSON recording - Core information only
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
        # Organize results Json
        os.makedirs(os.path.join(save_dir, "json"), exist_ok=True)
        with open(os.path.join(save_dir, "json", dataset), "w") as f:
            json.dump(task_res, f, indent=4, ensure_ascii=False, cls=NpEncoder)

        # Calculate final performance metrics
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

        # Save metrics by data_source
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

        # Add one line with overall results to CSV file
        results_csv_path = "../_results"
        os.makedirs(results_csv_path, exist_ok=True)
        csv_file_path = os.path.join(results_csv_path, f"res_{task}.csv")
        
        # Generate CSV data row
        import datetime
        csv_row = [
            model_name, experiment,
            RESIZE_RATIO, REGION_THRESHOLD, BBOX_PADDING,STAGE1_ENSEMBLE_RATIO,
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
        
        # Create with header if CSV file doesn't exist, add only data row if it exists
        import csv
        file_exists = os.path.exists(csv_file_path)
        
        with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Add header if file doesn't exist or is empty
            if not file_exists or os.path.getsize(csv_file_path) == 0:
                writer.writerow(csv_headers)
            
            # Add data row
            writer.writerow(csv_row)
        
        print(f"📝 Results saved to CSV: {csv_file_path}")

        # Accumulate to overall task statistics
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

        # Output final results
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

    # Calculate and save overall results
    total_crop_success_rate = total_crop_success / total_samples
    total_stage1_success_rate = total_stage1_success / total_samples
    total_stage2_success_rate = total_stage2_success / total_samples
    total_stage3_success_rate = total_stage3_success / total_samples
    
    # Overall average time
    avg_s1_time = total_s1_time / total_samples
    avg_s2_time = total_s2_time / total_samples
    avg_s3_time = total_s3_time / total_samples
    avg_total_time = (total_s1_time + total_s2_time + total_s3_time) / total_samples
    
    # Overall average TFLOPS
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
    
    # Save overall results to CSV
    cumulative_csv_path = os.path.join("../_results", "res_all.csv")
    
    # Generate overall results CSV row
    cumulative_csv_row = [
        model_name, experiment,
        RESIZE_RATIO, REGION_THRESHOLD, BBOX_PADDING,STAGE1_ENSEMBLE_RATIO,
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
        
        # Add header if file doesn't exist or is empty
        if not file_exists or os.path.getsize(cumulative_csv_path) == 0:
            writer.writerow(csv_headers)
        
        # Add overall results row
        writer.writerow(cumulative_csv_row)

    print(f"📝 Total Results : {cumulative_csv_path}")