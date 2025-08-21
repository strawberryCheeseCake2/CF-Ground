# Standard Library
import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Third-Party Libraries
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer, set_seed, logging

# Project-Local Modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from qwen_vl_utils import process_vision_info
from crop_clean import run_segmentation_recursive

# Suppress harmless warnings
logging.set_verbosity_error()

#! Argument =======================
SEED = 0
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "3"

# Dataset & Model
MLLM_PATH = "zonghanHZH/ZonUI-3B"
SCREENSPOT_IMGS = "./data/screenspotv2_imgs"
SCREENSPOT_TEST = "./data"
SAVE_DIR = "./output/" + "0822_v5_highlighted_segments_prompt" #! 저장 경로 변경

# Data Processing
TASKS = ["mobile"]
SAMPLE_RANGE = slice(0, 50)

#! Hyperparameter =================
VAR_THRESH = 120
WINDOW_SIZE = 120
TOP_Q = 0.5
S1_RESIZE_RATIO = 0.30
S2_RESIZE_RATIO = 0.50
THUMBNAIL_RESIZE_RATIO = 0.05
LAYER_NUM = 31

# This global variable will be updated in the main loop for each image.
img_path = ""

#! Helper Functions =================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mllm_path', type=str, default=MLLM_PATH)
    parser.add_argument('--screenspot_imgs', type=str, default=SCREENSPOT_IMGS)
    parser.add_argument('--screenspot_test', type=str, default=SCREENSPOT_TEST)
    parser.add_argument('--save-dir', type=str, default=SAVE_DIR)
    return parser.parse_args()

def point_in_bbox(pt, bbox):
    if not pt or not bbox: return False
    x, y = pt
    left, top, right, bottom = bbox
    return left <= x <= right and top <= y <= bottom

def calc_acc(score_list):
    return sum(score_list) / len(score_list) if score_list else 0.0

def parse_coordinates(text: str) -> Tuple[int, int] | None:
    match = re.search(r'\[\s*(\d+)\s*,\s*(\d+)\s*\]', text)
    return (int(match.group(1)), int(match.group(2))) if match else None

def create_msgs(crop_list, question):
    msgs = [{"role": "user", "content": []}]
    for crop in crop_list:
        msgs[0]["content"].append({"type": "image", "image": crop["resized_img"]})
    msgs[0]["content"].append({"type": "text", "text": question})
    return msgs

def create_stage_crop_list(crop_list: List, resize_dict: Dict, use_thumbnail: bool = True):
    stage_crop_list = []
    for crop in crop_list:
        crop_level = crop.get("level")
        if not use_thumbnail and crop_level == 0: continue
        ratio = resize_dict.get(crop_level, resize_dict.get(1))
        if ratio is None: continue
        new_crop = crop.copy()
        crop_img = new_crop.get("img")
        if not isinstance(crop_img, Image.Image): continue
        crop_width, crop_height = crop_img.size
        new_crop["resized_img"] = crop_img.resize((int(crop_width * ratio), int(crop_height * ratio)))
        stage_crop_list.append(new_crop)
    return stage_crop_list

def find_vision_spans(input_ids, vs_id, ve_id):
    ids = input_ids.tolist()
    spans = []
    i = 0
    while True:
        try:
            s = ids.index(vs_id, i) + 1
            e = ids.index(ve_id, s)
            spans.append((s, e))
            i = e + 1
        except ValueError:
            break
    return spans

def run_single_forward(model, processor, inputs, crop_list: List):
    vision_start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
    spans = find_vision_spans(inputs["input_ids"][0], vision_start_id, vision_end_id)
    for i, span in enumerate(spans):
      if i < len(crop_list):
        crop_list[i]["token_span"] = span
    with torch.no_grad():
      output = model(**inputs, output_attentions=True)
    return output, crop_list

# Functions for Stage 1
def run_selection_pass(model, processor, msgs, crop_list, top_q):
    import pandas as pd
    inputs = processor(
        text=processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True),
        images=[c['resized_img'] for c in crop_list],
        padding=True, return_tensors="pt"
    ).to(model.device)
    
    attn_output, s1_crop_list_with_span = run_single_forward(model, processor, inputs, crop_list)
    
    _data, index = [], []
    for crop in s1_crop_list_with_span:
        if "token_span" not in crop: continue
        (st, end) = crop["token_span"]
        attn_sum = sum(
            attn_output.attentions[li][0, :, -1, st:end].mean().item()
            for li in range(20, LAYER_NUM)
        )
        _data.append(attn_sum)
        index.append(crop["id"])
    
    if not _data: return [], []
    
    attn_scores = pd.Series(_data, index=index)
    thr = attn_scores.quantile(1 - top_q)
    top_q_crop_ids = attn_scores.index[attn_scores >= thr].tolist()
    top_q_crop_ids = top_q_crop_ids if top_q_crop_ids else [attn_scores.idxmax()]
    
    print(f"Stage 1 Selected crops: {top_q_crop_ids}")
    top_q_bboxes = [c["bbox"] for c in crop_list if c.get("id") in top_q_crop_ids]
    return top_q_crop_ids, top_q_bboxes

def run_scaled_prompt_generation(model, processor, crop_list: List, question_template: str, instruction: str, original_dims: Tuple[int, int], gt_bbox: list) -> Tuple[bool, Tuple[int, int] | None]:
    print("Running Stage 2: Scaled Prompt Generation...")
    
    s2_resized_crop_list = create_stage_crop_list(
        crop_list=crop_list,
        resize_dict={0: THUMBNAIL_RESIZE_RATIO, 1: S2_RESIZE_RATIO},
        use_thumbnail=True
    )
    
    original_width, original_height = original_dims
    scale_factor = round(1 / THUMBNAIL_RESIZE_RATIO)

    question = question_template.format(
        original_width=original_width,
        original_height=original_height,
        scale_factor=scale_factor,
        task_prompt=instruction
    )
    
    msgs = create_msgs(s2_resized_crop_list, question)
    inputs = processor(
        text=processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True),
        images=[c['resized_img'] for c in s2_resized_crop_list],
        padding=True, return_tensors="pt"
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=30, do_sample=False, pad_token_id=processor.tokenizer.eos_token_id)
        
    response_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    generated_text = response_text.split("assistant\n")[-1].strip()
    print(f"Model generated response: '{generated_text}'")
    
    predicted_point = parse_coordinates(generated_text)
    is_success = point_in_bbox(predicted_point, gt_bbox)
    
    if predicted_point:
        result_msg = "✅ Success" if is_success else "❌ Fail"
        print(f"{result_msg}: Predicted {predicted_point} | GT Bbox: {gt_bbox}")
    else:
        print("❌ Parsing failed: Could not find coordinates in the response.")
        
    return is_success, predicted_point

def visualize_s1_result(save_dir, gt_bbox, top_q_bboxes) -> str:
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)
    draw.rectangle(gt_bbox, outline="lime", width=3)
    for bbox in top_q_bboxes:
        draw.rectangle(bbox, outline="red", width=2)
    os.makedirs(save_dir, exist_ok=True)
    result_path = os.path.join(save_dir, "stage1_selection.png")
    img.save(result_path)
    return result_path

def visualize_s2_result(s1_path, save_dir, predicted_point, is_success):
    if not predicted_point or not os.path.exists(s1_path): return
    img = Image.open(s1_path)
    draw = ImageDraw.Draw(img)
    px, py = predicted_point
    radius = 15
    color = "yellow" if is_success else "magenta"
    draw.line([(px, py - radius), (px, py + radius)], fill=color, width=4)
    draw.line([(px - radius, py), (px + radius, py)], fill=color, width=4)
    draw.line([(px - radius*0.7, py - radius*0.7), (px + radius*0.7, py + radius*0.7)], fill=color, width=4)
    draw.line([(px - radius*0.7, py + radius*0.7), (px + radius*0.7, py - radius*0.7)], fill=color, width=4)
    os.makedirs(save_dir, exist_ok=True)
    status = "SUCCESS" if is_success else "FAIL"
    final_path = os.path.join(save_dir, f"result_final_{status}.png")
    img.save(final_path)
    print(f"Final visualization saved to: {final_path}")

#! Main Execution ===================
if __name__ == '__main__':
    set_seed(SEED)
    args = parse_args()
    
    print("Loading model and processor...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.mllm_path, torch_dtype="auto", attn_implementation="eager", device_map="auto")
    processor = AutoProcessor.from_pretrained(args.mllm_path)
    model.eval()
    print("Model loaded successfully.")

    s1_question_template = "Given a task instruction, a screen observation, guess where should you tap.\n# Instruction\n{task_prompt}"
    
    #
    # ## 💡 여기가 바로 V5를 위해 수정된 Stage 2 프롬프트 템플릿입니다.
    #
    s2_question_template = """You are an expert AI assistant for mobile UI navigation. You will be provided with several images:
1. A small **thumbnail** of the entire screen.
2. One or more **zoomed-in segments**. These are areas from the original screen with a high probability of containing the target for the instruction.

**Image Context:**
- The original screen resolution was {original_width}x{original_height}.
- The thumbnail is a {scale_factor}x scaled-down version of the original screen.

Your task is to analyze all provided images and identify the precise (x, y) coordinate on the **original {original_width}x{original_height} screen**. Your answer must be only in the format [x, y].

# Instruction
{task_prompt}
"""

    for task in TASKS:
        dataset_path = os.path.join(args.screenspot_test, f"screenspot_{task}_v2.json")
        if not os.path.exists(dataset_path): continue
        with open(dataset_path, 'r') as f: screenspot_data = json.load(f)[SAMPLE_RANGE]
        
        print(f"\n--- Starting task: {task} ({len(screenspot_data)} samples) ---")
        s2_scores = []
        
        for item in tqdm(screenspot_data, desc=f"Processing {task}"):
            filename = item["img_filename"]
            img_path = os.path.join(args.screenspot_imgs, filename) # Set global img_path for visualizers
            if not os.path.exists(img_path): continue

            instruction = item["instruction"]
            bbox = item["bbox"]
            original_bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            
            original_image = Image.open(img_path).convert("RGB")
            original_dims = original_image.size
            
            inst_dir_name = re.sub(r'\W+', '_', instruction).strip('_')[:50]
            base_save_dir = Path(args.save_dir) / os.path.splitext(filename)[0] / inst_dir_name
            
            # --- Stage 1: Selection ---
            segmentation_dir = base_save_dir / "segmentation"
            segmentation_dir.mkdir(parents=True, exist_ok=True)
            
            crop_list = run_segmentation_recursive(
                image_path=img_path,
                max_depth=1,
                var_thresh=VAR_THRESH,
                window_size=WINDOW_SIZE,
                output_json_path=str(segmentation_dir / "output.json"),
                output_image_path=str(segmentation_dir),
                start_id=0
            )
            
            if not crop_list: 
                crop_list = [{'id': 0, 'level': 0, 'bbox': [0, 0, *original_dims], 'img': original_image}]

            s1_crop_list = create_stage_crop_list(crop_list, {0: THUMBNAIL_RESIZE_RATIO, 1: S1_RESIZE_RATIO})
            s1_msgs = create_msgs(s1_crop_list, s1_question_template.format(task_prompt=instruction))
            s1_ids, s1_bboxes = run_selection_pass(model, processor, s1_msgs, s1_crop_list, TOP_Q)
            
            s1_save_dir = base_save_dir / "stage1"
            s1_result_path = visualize_s1_result(s1_save_dir, original_bbox, s1_bboxes)

            # --- Stage 2: Scaled Prompt Generation ---
            original_crop_map = {c['id']: c for c in crop_list}
            s2_input_crops = [original_crop_map[cid] for cid in sorted({0} | set(s1_ids)) if cid in original_crop_map]
            
            s2_success, predicted_point = run_scaled_prompt_generation(model, processor, s2_input_crops, s2_question_template, instruction, original_dims, original_bbox)
            s2_scores.append(1 if s2_success else 0)

            s2_save_dir = base_save_dir / "stage2"
            visualize_s2_result(s1_result_path, s2_save_dir, predicted_point, s2_success)

            # --- Rename folder with final status ---
            status_symbol = '✅' if s2_success else '❌'
            final_dir = Path(args.save_dir) / os.path.splitext(filename)[0] / f"{inst_dir_name} {status_symbol}"
            if base_save_dir.exists() and str(base_save_dir) != str(final_dir):
                if final_dir.exists(): shutil.rmtree(final_dir)
                base_save_dir.rename(final_dir)
            
            print(f"Current Accuracy: {calc_acc(s2_scores):.4f}")

        # Final Results
        final_accuracy = calc_acc(s2_scores)
        print(f"\n{'='*60}\nTask '{task}' Final Results:\n  - Total Samples: {len(s2_scores)}\n  - Final Accuracy: {final_accuracy:.4f}\n{'='*60}")
        
        metrics = {"task": task, "total_samples": len(s2_scores), "accuracy": final_accuracy, "approach": "V5 - Highlighted Segments Prompt"}
        Path(args.save_dir).mkdir(exist_ok=True)
        with open(Path(args.save_dir) / f"{task}_metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)