# Standard Library
import argparse
import json
import os
import re
import shutil
import sys
import time
from collections import deque
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

# Third-Party Libraries
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer, set_seed

from crop_clean import run_segmentation_recursive  # ! crop

# Project-Local Modules
# (Local module paths might need adjustment depending on your project structure)
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from qwen_vl_utils import process_vision_info
from crop_clean import run_segmentation_recursive

# Mock functions for local modules if they are not available
# def run_segmentation_recursive(**kwargs):
#     print("Warning: Using mock 'run_segmentation_recursive'. Please provide the actual implementation.")
#     return []

# def process_vision_info(msgs):
#     print("Warning: Using mock 'process_vision_info'. Please provide the actual implementation.")
#     image_inputs = [content['image'] for content in msgs[0]['content'] if content['type'] == 'image']
#     return image_inputs, None


#! Argument =======================

SEED = 0

# Enviroment
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "3"
max_memory = {
    0: "75GiB",
    "cpu": "120GiB",
}

# Dataset & Model
MLLM_PATH = "zonghanHZH/ZonUI-3B"
SCREENSPOT_IMGS = "./data/screenspotv2_imgs"
SCREENSPOT_TEST = "./data"
SAVE_DIR = "./attn_output/" + "0821_direct_coordinate_stage2"

# Data Processing
SAMPLING = False
TASKS = ["mobile"]
SAMPLE_RANGE = slice(5,55)

#! Hyperparameter =================

# Model Architecture
LAYER_NUM_LIST = [3, 7, 11, 15, 31]
LAYER_NUM = 31

# Stage 1: Segmentation & Selection
VAR_THRESH = 120
TOP_Q = 0.5

# Image Resize Ratios
S1_RESIZE_RATIO = 0.30
S2_RESIZE_RATIO = 0.50
THUMBNAIL_RESIZE_RATIO = 0.05

#! ================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mllm_path', type=str, default=MLLM_PATH)
    parser.add_argument("--sampling", action='store_true', default=SAMPLING, help="do sampling for mllm")
    parser.add_argument('--screenspot_imgs', type=str, default=SCREENSPOT_IMGS)
    parser.add_argument('--screenspot_test', type=str, default=SCREENSPOT_TEST)
    parser.add_argument('--save-dir', type=str, default=SAVE_DIR)
    parser.add_argument("--vis_flag", action='store_true', help="visualize mid-results")
    # A default for 'img_path' which is used globally later
    parser.add_argument('--img_path', type=str, default="", help="Path to the current image being processed")
    args = parser.parse_args()
    return args

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

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

def point_in_bbox(pt, bbox):
    x, y = pt
    left, top, right, bottom = bbox
    return left <= x <= right and top <= y <= bottom

def calc_acc(score_list):
    return sum(score_list) / len(score_list) if len(score_list) != 0 else 0

def process_inputs(msgs, crop_list):
    text = processor.apply_chat_template(
      msgs, tokenize=False, add_generation_prompt=True
    )

    image_inputs = [c['resized_img'] for c in crop_list]

    inputs = processor(
        text=text,
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to(model.device)
    return inputs, crop_list

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
        if ratio is None:
            print(f"Warning: Skipping crop ID {crop.get('id')} due to missing resize ratio.")
            continue
        new_crop = deepcopy(crop)
        crop_img = new_crop["img"]
        crop_width, crop_height = crop_img.size
        crop_img = crop_img.resize((int(crop_width * ratio), int(crop_height * ratio)))
        new_crop["resized_img"] = crop_img
        stage_crop_list.append(new_crop)
    return stage_crop_list

def run_single_forward(inputs, crop_list: List):
    vision_start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
    spans = find_vision_spans(inputs["input_ids"][0], vision_start_id, vision_end_id)
    for i, span in enumerate(spans):
      if i < len(crop_list):
        crop_list[i]["token_span"] = span
    with torch.no_grad():
      output = model(**inputs, output_attentions=True)
    return output, crop_list

def attn2df(attn_output, crop_list: List):
    _data = []
    for i, crop in enumerate(crop_list):
        if "token_span" not in crop: continue
        (st, end) = crop["token_span"]
        layer_avgs = []
        for li in range(LAYER_NUM):
            att_vector = attn_output.attentions[li][0, :, -1, st:end].mean(dim=0).to(torch.float32).cpu().numpy()
            layer_avgs.append(att_vector.mean())
        _data.append(layer_avgs)
    df = pd.DataFrame(
            _data,
            columns=[f"layer{layer_idx+1}" for layer_idx in range(LAYER_NUM)],
            index=[crop["id"] for crop in crop_list if "token_span" in crop]
            )
    return df

def get_top_q_crop_ids(top_q, attn_df):
    if attn_df.empty: return []
    df = deepcopy(attn_df)
    cols_to_sum = [f"layer{i}" for i in range(20, LAYER_NUM + 1)]
    df[f"sum_until_{LAYER_NUM}"] = df[cols_to_sum].sum(axis=1)
    thr = df[f"sum_until_{LAYER_NUM}"].quantile(1 - top_q)
    top_q_crop_ids = df.index[df[f"sum_until_{LAYER_NUM}"] >= thr].tolist()
    if not top_q_crop_ids:
        return [df[f"sum_until_{LAYER_NUM}"].idxmax()]
    return top_q_crop_ids

def check_gt_in_top_q_crops(top_q_bboxes: List, gt_bbox: List):
    gt_center_x = (gt_bbox[0] + gt_bbox[2]) / 2.0
    gt_center_y = (gt_bbox[1] + gt_bbox[3]) / 2.0
    gt_point = (gt_center_x, gt_center_y)
    return any(point_in_bbox(gt_point, bbox) for bbox in top_q_bboxes)

def run_selection_pass(msgs, crop_list, top_q, drop_indices: List, attn_vis_dir: str):
    s1_inputs, s1_crop_list = process_inputs(msgs=msgs, crop_list=crop_list)
    attn_output, s1_crop_list_with_span = run_single_forward(inputs=s1_inputs, crop_list=s1_crop_list)
    df = attn2df(attn_output=attn_output, crop_list=s1_crop_list_with_span)
    
    # Pass the updated crop list with spans to the visualizer
    visualize_attn_map(attn_output=attn_output, msgs=msgs, crop_list=s1_crop_list_with_span, attn_vis_dir=attn_vis_dir)

    raw_attn_res = df.to_dict(orient="index")
    df.drop(index=[i for i in drop_indices if i in df.index], errors='ignore', inplace=True)
    
    top_q_crop_ids = get_top_q_crop_ids(top_q=top_q, attn_df=df)
    print(f"Selected crops: {top_q_crop_ids}")
    
    top_q_bboxes = [crop["bbox"] for crop in crop_list if crop.get("id") in top_q_crop_ids]
    return top_q_crop_ids, top_q_bboxes, raw_attn_res, s1_crop_list_with_span

def parse_coordinates(text: str) -> Tuple[int, int] | None:
    match = re.search(r'\[\s*(\d+)\s*,\s*(\d+)\s*\]', text)
    return (int(match.group(1)), int(match.group(2))) if match else None

def run_coordinate_generation_pass(model, processor, crop_list: List, question_template: str, instruction: str, gt_bbox: List) -> Tuple[bool, Tuple[float, float] | None]:
    print("Running Stage 2: Direct Coordinate Generation...")
    s2_resized_crop_list = create_stage_crop_list(
        crop_list=crop_list,
        resize_dict={0: THUMBNAIL_RESIZE_RATIO, 1: S2_RESIZE_RATIO},
        use_thumbnail=True
    )
    thumbnail_crop = next((c for c in s2_resized_crop_list if c.get("level") == 0), None)
    if not thumbnail_crop:
        print("Error: Thumbnail not found. Skipping.")
        return False, None
    width, height = thumbnail_crop["resized_img"].size
    question = question_template.format(task_prompt=instruction, width=width, height=height)
    msgs = create_msgs(crop_list=s2_resized_crop_list, question=question)
    inputs, _ = process_inputs(msgs=msgs, crop_list=s2_resized_crop_list)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=30, do_sample=False, pad_token_id=processor.tokenizer.eos_token_id)
    response_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    prompt_len = len(processor.batch_decode(inputs['input_ids'], skip_special_tokens=True)[0])
    generated_text = response_text[prompt_len:].strip()
    print(f"Model generated response: '{generated_text}'")
    
    coords = parse_coordinates(generated_text)
    if not coords:
        print("❌ Coordinate parsing failed.")
        return False, None
        
    px, py = coords
    L, T, R, B = thumbnail_crop['bbox']
    orig_w, orig_h = R - L, B - T
    resized_w, resized_h = thumbnail_crop["resized_img"].size
    if resized_w == 0 or resized_h == 0: return False, None
    
    global_x = L + (px / resized_w) * orig_w
    global_y = T + (py / resized_h) * orig_h
    point = (global_x, global_y)
    is_success = point_in_bbox(point, gt_bbox)
    print(f"✅ Grounding Success: {point} in {gt_bbox}" if is_success else f"❌ Grounding Fail: {point} not in {gt_bbox}")
    return is_success, point

def visualize_result(save_dir, gt_bbox, top_q_bboxes) -> str:
    result_img = Image.open(img_path)
    draw = ImageDraw.Draw(result_img)
    draw.rectangle(gt_bbox, outline="lime", width=3)
    for bbox in top_q_bboxes:
        draw.rectangle(bbox, outline="red", width=2)
    os.makedirs(save_dir, exist_ok=True)
    result_path = os.path.join(save_dir, "result_stage1_selection.png")
    print(f"Saving Stage 1 visualization to: {result_path}")
    result_img.save(result_path)
    return result_path

def visualize_stage2_result(s1_path: str, save_dir: str, point: Tuple[float, float], is_success: bool):
    if not os.path.exists(s1_path): return
    result_img = Image.open(s1_path)
    draw = ImageDraw.Draw(result_img)
    px, py = point
    radius = 15
    color = "yellow" if is_success else "magenta"
    draw.line([(px, py - radius), (px, py + radius)], fill=color, width=4)
    draw.line([(px - radius, py), (px + radius, py)], fill=color, width=4)
    draw.line([(px - radius*0.7, py - radius*0.7), (px + radius*0.7, py + radius*0.7)], fill=color, width=4)
    draw.line([(px - radius*0.7, py + radius*0.7), (px + radius*0.7, py - radius*0.7)], fill=color, width=4)
    os.makedirs(save_dir, exist_ok=True)
    final_path = os.path.join(save_dir, f"result_final_{'SUCCESS' if is_success else 'FAIL'}.png")
    result_img.save(final_path)
    print(f"Final visualization saved to: {final_path}")

def visualize_attn_map(attn_output, msgs, crop_list, attn_vis_dir):
    image_inputs, _ = process_vision_info(msgs)
    img_proc_out = processor.image_processor(images=image_inputs, return_tensors="pt")
    grid = img_proc_out.get("image_grid_thw")
    if grid is None: return
    if grid.ndim == 3: grid = grid[0]
    final_shapes = [(t.item(), (h // 2).item(), (w // 2).item()) for t, h, w in grid]
    
    num_imgs = len(crop_list)
    if num_imgs == 0: return
    fig, axes = plt.subplots(1, num_imgs, figsize=(5*num_imgs, 5))
    if num_imgs == 1: axes = [axes]

    for i, crop in enumerate(crop_list):
        if "token_span" not in crop or i >= len(final_shapes): continue
        (st, end) = crop["token_span"]
        t, h2, w2 = final_shapes[i]
        att_maps = []
        for li in range(20, LAYER_NUM):
            att = attn_output.attentions[li][0, :, -1, st:end].mean(dim=0).to(torch.float32).cpu().numpy()
            
            # --- START of ERROR FIX ---
            expected_size = t * h2 * w2
            actual_size = att.shape[0]
            if actual_size != expected_size:
                print(f"  [Warning] Mismatch in visualize_attn_map for crop{crop.get('id', 'N/A')}: "
                      f"Token span size is {actual_size}, but expected grid size is {expected_size} ({t}x{h2}x{w2}). "
                      f"Truncating/padding attention vector to fit.")
                if actual_size > expected_size:
                    att = att[:expected_size]
                else:
                    att = np.pad(att, (0, expected_size - actual_size), 'constant')
            # --- END of ERROR FIX ---

            att_map = att.reshape(t, h2, w2).mean(axis=0)
            att_maps.append(att_map)
        
        if not att_maps: continue
        att_avg = np.mean(att_maps, axis=0)
        ax = axes[i]
        ax.imshow(att_avg, cmap="viridis", interpolation="nearest")
        ax.set_title(f"crop{crop.get('id', 'N/A')}")
        ax.axis("off")

    plt.tight_layout()
    out_dir = Path(attn_vis_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_path = out_dir / "attn_map.png"
    fig.savefig(_save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

if __name__ == '__main__':
    set_seed(SEED)
    args = parse_args()
    seg_save_base_dir = f"{args.save_dir}/seg"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.mllm_path, torch_dtype="auto", attn_implementation="eager", device_map="balanced", max_memory=max_memory, low_cpu_mem_usage=True)
    processor = AutoProcessor.from_pretrained(args.mllm_path)
    model.eval()

    question_template = "Given a task instruction, a screen observation, guess where should you tap.\n# Instruction\n{task_prompt}"

    for task in TASKS:
        dataset = f"screenspot_{task}_v2.json"
        try:
            with open(os.path.join(args.screenspot_test, dataset), 'r') as f:
                screenspot_data = json.load(f)[SAMPLE_RANGE]
        except FileNotFoundError:
            print(f"Dataset not found: {dataset}. Skipping.")
            continue
        
        print(f"Num of samples for task '{task}': {len(screenspot_data)}")
        task_res, s1_scores, s2_scores = [], [], []
        
        for item in tqdm(screenspot_data):
            filename = item["img_filename"]
            img_path = os.path.join(args.screenspot_imgs, filename) # Set global img_path for visualizer
            if not os.path.exists(img_path): continue

            instruction = item["instruction"]
            bbox = item["bbox"]
            original_bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            
            inst_dir_name = re.sub(r'\W+', '_', instruction).strip('_')[:50] # Limit length
            dir_path = f"{seg_save_base_dir}/{os.path.splitext(filename)[0]}/{inst_dir_name}"
            s1_dir = os.path.join(dir_path, "s1")
            os.makedirs(s1_dir, exist_ok=True)
            
            crop_list = run_segmentation_recursive(
                image_path=img_path, max_depth=1, window_size=120,
                output_json_path=f"{s1_dir}/output.json",
                output_image_path=s1_dir, start_id=0, var_thresh=VAR_THRESH
            )
            if not crop_list: 
                # Add original image as a single crop if segmentation fails
                original_image = Image.open(img_path).convert("RGB")
                w, h = original_image.size
                crop_list = [{'id': 0, 'level': 0, 'bbox': [0, 0, w, h], 'img': original_image}]

            # Stage 1
            s1_crop_list = create_stage_crop_list(crop_list, {0: THUMBNAIL_RESIZE_RATIO, 1: S1_RESIZE_RATIO})
            msgs = create_msgs(s1_crop_list, question_template.format(task_prompt=instruction))
            s1_ids, s1_bboxes, _, _ = run_selection_pass(msgs, s1_crop_list, TOP_Q, [0], os.path.join(s1_dir, "attn_map"))
            s1_success = check_gt_in_top_q_crops(s1_bboxes, original_bbox)
            s1_scores.append(1 if s1_success else 0)
            s1_result_path = visualize_result(s1_dir, original_bbox, s1_bboxes)
            print(f"Stage 1 GT Contained: {s1_success}")

            # Stage 2
            question_template_s2 = "Provide the (x, y) coordinate to tap for instruction: '{task_prompt}'. Image is {width}x{height}. Answer format: [x, y]"
            original_crop_map = {c['id']: c for c in crop_list}
            s2_ids = {0} | set(s1_ids)
            s2_crops = [original_crop_map[cid] for cid in sorted(list(s2_ids)) if cid in original_crop_map]
            
            s2_success = False
            if len(s2_crops) > 1:
                s2_success, point = run_coordinate_generation_pass(model, processor, s2_crops, question_template_s2, instruction, original_bbox)
                if point:
                    s2_dir = os.path.join(dir_path, "s2_direct_coord")
                    visualize_stage2_result(s1_result_path, s2_dir, point, s2_success)
            s2_scores.append(1 if s2_success else 0)
            
            # Rename folder based on final success
            new_dir_path = f"{os.path.splitext(dir_path)[0]} {'✅' if s2_success else '❌'}"
            if os.path.exists(dir_path):
                if os.path.exists(new_dir_path): shutil.rmtree(new_dir_path)
                os.rename(dir_path, new_dir_path)
                
            print(f"Up to now S2 Accuracy: {calc_acc(s2_scores):.4f}\n---")
            task_res.append({'filename': filename, 'instruction': instruction, 's1_success': s1_success, 's2_success': s2_success})

        with open(os.path.join(args.save_dir, f"{task}_results.json"), "w") as f:
            json.dump(task_res, f, indent=4)
        
        print("\n==================================================")
        print(f"Task: {task}, Total: {len(screenspot_data)}")
        print(f"Stage 1 (Selection) Accuracy: {calc_acc(s1_scores):.4f}")
        print(f"Stage 2 (Coordination) Accuracy: {calc_acc(s2_scores):.4f}")