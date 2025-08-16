
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict
from copy import deepcopy
import math
import os
import re
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
import copy
import json
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
#Qwen 2.5
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer, set_seed

from qwen_vl_utils import process_vision_info

import re
import json

from crop import run_segmentation
import cv2 # for bilinear interpolation

set_seed(0)

torch.cuda.empty_cache()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "3"  # Set the GPUs 2 and 3 to use

max_memory = {
    0: "75GiB",
    # 1: "75GiB",
    # 2: "75GiB",
    "cpu": "120GiB",  # 남는 건 CPU 오프로딩xs
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mllm_path', type=str, required=False)
    parser.add_argument("--sampling", action='store_true', default=False, help="do sampling for mllm")
    parser.add_argument('--screenspot_imgs', type=str, required=False)
    parser.add_argument('--screenspot_test', type=str, required=False)
    parser.add_argument('--save-dir', type=str, default=None)
    parser.add_argument("--vis_flag", action='store_true', help="visualize mid-results")
    args = parser.parse_args()
    return args

args = parse_args()

# args.mllm_path = "Qwen/Qwen2.5-VL-3B-Instruct"
# args.mllm_path = "Qwen/Qwen2.5-VL-3B-Instruct"
# args.mllm_path = "ByteDance-Seed/UI-TARS-1.5-7B"
args.mllm_path = "zonghanHZH/ZonUI-3B"
# args.mllm_path = "xlangai/Jedi-3B-1080p"

# args.screenspot_imgs = "./data/screenspot_imgs"
args.screenspot_imgs = "./data/screenspotv2_imgs"
args.screenspot_test = "./data"

# args.save_dir = "./output"
# args.save_dir = "./attn_output/sevenb/eval_res"
# args.save_dir = "./attn_output/ui_tar/eval_res"
# args.save_dir = "./attn_output/zonui/eval_res"
# args.save_dir = "./attn_output/zonui/eval_res"
# args.save_dir = "./attn_output/qwen/eval_res" # resize=0.05
# args.save_dir = "./attn_output/jedi_k65/eval_res" # resize=0.05
# args.save_dir = "./attn_output/jedi_k70_resize10/eval_res" # resize
# args.save_dir = "./attn_output/jedi_k70_resize05/eval_res" # 0805
# args.save_dir = "./attn_output/jedi_k50_resize20/eval_res"
# args.save_dir = "./attn_output/jedi_k50_resize10_thumnail/eval_res"

# args.save_dir = "./attn_output/zonui_k70_resize30/eval_res" # resize
# args.save_dir = "./attn_output/zonui_q50_resize20_thumb/eval_res" # resize
# args.save_dir = "./attn_output/zonui_q50_resize20_thumb_s2/eval_res" # resize
# args.save_dir = "./attn_output/ensemble_q50_r20_thumb_s2_agg20_try2/eval_res" # resize
# args.save_dir = "./attn_output/zonui_q50_resize20/eval_res" # resize
# args.save_dir = "./attn_output/zonui_k50_resize30/eval_res" # resize
# args.save_dir = "./attn_output/qwen_k50_resize05/eval_res" # resize
args.save_dir = "./attn_output/course_repeat/eval_res" # resize


args.sampling = False

seg_save_base_dir = f"{args.save_dir}/seg"
# layer_num_list = [4,8,12,16,32] #12
layer_num_list = [3,7,11,15,31] #12
layer_num = layer_num_list[-1]
# layer_num_list = [4,8,12,16,27] #12

# top_k_ratio = 0.7
top_q = 0.5

s1_resize_ratio = 0.20
deep_resize_ratio = 0.35  # level >= 2 기본 리사이즈 비율
thumnail_resize_ratio = 0.05

# num_stage = 2

# tasks = ["mobile", "desktop", "web"]
tasks = ["mobile"]



# 현재 best : resize 5% layer 12

#프롬프트 full, 프롬프트 group

# Qwen 2.5
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    args.mllm_path,
    torch_dtype="auto",
    attn_implementation="eager",
    # device_map="cuda:0",
    device_map="balanced",
    max_memory=max_memory,
    low_cpu_mem_usage=True
)


tokenizer = AutoTokenizer.from_pretrained(args.mllm_path)
processor = AutoProcessor.from_pretrained(args.mllm_path)
model.eval()

def contains_bbox(target, gt):
    gt_left, gt_top, gt_right, gt_bottom = gt
    target_left, target_top, target_right, target_bottom = target

    
    return (
      gt_left >= target_left
      and gt_top >= target_top
      and gt_right <= target_right
      and gt_bottom <= target_bottom
  )

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

def visualize_result(save_dir, gt_bbox, top_q_bboxes, instruction, click_point=None, base_image=None):
    # Visualize ground truth and selected crop on the image
    if base_image is not None:
        result_img = base_image.copy()
    else:
        # fallback: 기존 코드 호환 (가능하면 base_image를 넘겨 주세요)
        result_img = Image.open(img_path)

    draw = ImageDraw.Draw(result_img)
    # Draw ground truth bbox in green
    draw.rectangle(gt_bbox, outline="green", width=2)

    font = None
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
    except IOError:
        font = ImageFont.load_default()
    

    for bbox in top_q_bboxes:
        draw.rectangle(bbox, outline="red", width=2)
        text_to_draw = f"{instruction}"
        crop_left, crop_top, crop_right, crop_bottom = bbox
        inst_position = (crop_left, crop_top)
        draw.text(inst_position, text_to_draw, fill="red", font=font)


    # Draw click point as an orange circle
    if click_point is not None:
        click_x, click_y = click_point[0], click_point[1]
        radius = 5
        draw.ellipse((click_x - radius, click_y - radius, click_x + radius, click_y + radius), outline="orange", width=3)

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    # Save the result image
    result_path = os.path.join(save_dir ,f"result.png")
    print(result_path)
    result_img.save(result_path)

def point_in_bbox(pt, bbox):
    x, y = pt
    left, top, right, bottom = bbox
    return left <= x <= right and top <= y <= bottom


def calc_acc(score_list):
    return sum(score_list) / len(score_list) if len(score_list) != 0 else 0


def process_inputs(msgs):
    text = processor.apply_chat_template(
      msgs, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(msgs)

    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to(model.device)

    return inputs

def create_msgs(crop_list, question):
     # msg에 추가
    msgs = [{
        "role": "user",
        "content": [],
    }]

    for crop in crop_list:
        img = crop["resized_img"]
        msgs[0]["content"].append({"type": "image", "image": img})
    msgs[0]["content"].append({"type": "text", "text": question})
    return msgs


def create_stage_crop_list(crop_list: List,
                           resize_dict: Dict, 
                           use_thumbnail: bool = True,
                           default_resize: float | None = None):
    stage_crop_list = []

    for crop in crop_list:
        crop_level = crop.get("level")
        # 리사이즈 비율 결정: 사전에 없으면 default_resize 사용
        ratio = resize_dict.get(crop_level, default_resize)
        if ratio is None:
            continue
        if not use_thumbnail and crop_level == 0: continue

        new_crop = deepcopy(crop)
        crop_img = deepcopy(crop["img"])

        # Resize image to 50% size
        if ratio is None:
            raise Exception(f"No resize dict entry for crop level {crop_level}")

        crop_width, crop_height = crop_img.size
        crop_img = crop_img.resize((int(crop_width * ratio), int(crop_height * ratio)))

        new_crop["resized_img"] = crop_img
        
        stage_crop_list.append(new_crop)

    return stage_crop_list


def run_single_forward(msgs: List, crop_list: List):
    # Process Input
    inputs = process_inputs(msgs=msgs)

    # Add vision span to crop list from inputs
    vision_start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")

    spans = find_vision_spans(inputs["input_ids"][0], vision_start_id, vision_end_id)


    # crop이랑 span 같이 관리
    for i, span in enumerate(spans):
      crop_list[i]["token_span"] = span
    

    ##### inference
    with torch.no_grad():
      output = model(**inputs, output_attentions=True)
    

    return output, crop_list

def attn2df(attn_output, crop_list: List):
    _data = []
        
    # crop마다 받은 어텐션 뽑아내기
    

    for i, crop in enumerate(crop_list):
        (st, end) = crop["token_span"] # crop의 토큰 시작, 끝 index 뽑기

        layer_avgs = [] # e.g. [0.0(레이어 0에서 이 crop이 받은 어텐션), 0.1 (레이어 1에서 crop이 받은 어텐션), 0.2, ...]
        for li in range(layer_num):
            # 레이어마다 해당 crop이 받은 어텐션 추출 
            att = (
                attn_output.attentions[li][0, :, -1, st:end] # starting answer token(텍스트 토큰 마지막)
                .mean(dim=0)
                .to(torch.float32)
                .cpu()
                .numpy()
            )
            layer_avgs.append(att.mean()) 

        _data.append(layer_avgs)

    
    
    df = pd.DataFrame(
            _data,
            columns=[f"layer{layer_idx+1}" for layer_idx in range(layer_num_list[-1])],
            index=[crop["id"] for crop in crop_list]
            )

    return df



def get_top_q_crop_ids(top_q, attn_df):

    df = deepcopy(attn_df)
     # layer num까지의 어텐션 합 구하기
    cols_to_sum = [f"layer{i}" for i in range(20, layer_num + 1)] #HERE!!!! #20?1?attn_map 20으로 한결과
    df[f"sum_until_{layer_num}"] = df[cols_to_sum].sum(axis=1)


    # 1.1. get top-q crop index 

    
    # quantile 임계치로 선택
    keep_frac = top_q  # 예: 0.2 → 상위 20% 유지
    thr = df[f"sum_until_{layer_num}"].quantile(1 - keep_frac)


    top_q_crop_ids = df.index[df[f"sum_until_{layer_num}"] >= thr].tolist()

    # # guardrail: 최소/최대 개수 보정(선택)

    if len(top_q_crop_ids) < 1:
        raise Exception("no crop selected")
    return top_q_crop_ids


def visualize_attn_map(attn_output, msgs, crop_list, attn_vis_dir):

    image_inputs, _ = process_vision_info(msgs)

    # 3) grid 크기 뽑아두기
    img_proc_out = processor.image_processor(images=image_inputs)
    grid = img_proc_out["image_grid_thw"]
    # (batch, num_imgs, 3) 혹은 (num_imgs, 3) 형태일 수 있으니 안전하게 뽑기
    if grid.ndim == 3:
        grid = grid[0]   # (num_imgs, 3)
    # grid = grid.cpu().numpy().astype(int)  # 이제 (4, 3)

    # 최종 token-map 차원: t × (h//2) × (w//2)
    final_shapes = [
        (t, h//2, w//2)
        for t, h, w in grid
    ]

    num_imgs = len(crop_list)
    fig, axes = plt.subplots(1, num_imgs, figsize=(5*num_imgs, 5))


    for i, crop in enumerate(crop_list):
        (st, end) = crop["token_span"] # crop의 토큰 시작, 끝 index 뽑기
        t, h2, w2 = final_shapes[i]
        att_maps = []
        
        # for li in range(L):
        for li in range(20, 32):
            att = (
                attn_output.attentions[li]         # (batch, heads, seq_q, seq_k)
                [0, :, -1, st:end]              # batch=0, 마지막 query 토큰, vision span
                .mean(dim=0)                 # head 평균
                .to(torch.float32)           # bfloat16 → float32
                .cpu()
                .numpy()
            )
            att_map = att.reshape(t, h2, w2).mean(axis=0)  # 시간축 평균
            att_maps.append(att_map)

        att_avg = np.mean(att_maps, axis=0)  # 32개 레이어 평균

        ax = axes[i] if num_imgs > 1 else axes
        im = ax.imshow(att_avg, cmap="viridis", interpolation="nearest")
        ax.set_title(f"crop{crop['id']}")
        ax.axis("off")
    
    plt.tight_layout()

    out_dir = Path(attn_vis_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _save_path = os.path.join(out_dir, "attn_map.png")

    fig.savefig(_save_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"attn_map saved at: {_save_path}")



def run_stage(msgs, crop_list, top_q, drop_indices: List, attn_vis_dir: str):

    attn_output, crop_list = run_single_forward(msgs=msgs, crop_list=crop_list)
    df = attn2df(attn_output=attn_output, crop_list=crop_list)
    visualize_attn_map(attn_output=attn_output, msgs=msgs, crop_list=crop_list, attn_vis_dir=attn_vis_dir)
    raw_attn_res = df.to_dict(orient="index")

    # exclude index 0 if present before selecting top-k

    for i in drop_indices:
        df = df.drop(index=i, errors='ignore')  


    # Step 1. thre 넘은 crop(top_k crop) 가져오기
    top_q_crop_ids = get_top_q_crop_ids(top_q=top_q, attn_df=df)
        
    print(f"instruction: {instruction}, selected crops: {top_q_crop_ids}")

    # 1.2. get top-k bboxes
    top_q_bboxes = []

    for crop in crop_list:
        if crop.get("id") not in drop_indices and crop.get("id") in top_q_crop_ids:
            top_q_bboxes.append(crop.get("bbox"))

    return top_q_crop_ids, top_q_bboxes, raw_attn_res, crop_list


def check_gt_in_top_q_crops(top_q_bboxes: List, gt_bbox: List):
    gt_center_x = (gt_bbox[0] + gt_bbox[2]) / 2.0
    gt_center_y = (gt_bbox[1] + gt_bbox[3]) / 2.0
    gt_center_point = (gt_center_x, gt_center_y)
    gt_point_in_top_q_crops = any(point_in_bbox(gt_center_point, bbox) for bbox in top_q_bboxes)

    return gt_point_in_top_q_crops

# ----------------------------------------------------
# 추가한 함수

def visualize_aggregated_attention(attn_output, crop_list, original_image, processor, save_path, individual_maps_dir=None):
    """
    각 crop의 어텐션 맵과 이를 모두 합산한 최종 맵을 시각화합니다.
    [수정됨] 최종 합산은 level 1 crop만 대상으로 합니다.
    """
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    
    if individual_maps_dir:
        os.makedirs(individual_maps_dir, exist_ok=True)

    original_width, original_height = original_image.size
    aggregated_attention_map = np.zeros((original_height, original_width), dtype=np.float32)

    image_inputs, _ = process_vision_info(create_msgs(crop_list, ""))
    img_proc_out = processor.image_processor(images=image_inputs)
    grid = img_proc_out["image_grid_thw"]
    if grid.ndim == 3:
        grid = grid[0]
    final_shapes = [(t, h//2, w//2) for t, h, w in grid]

    for i, crop in enumerate(crop_list):
        (st, end) = crop["token_span"]
        t, h2, w2 = final_shapes[i]
        
        att_maps = []
        for li in range(layer_num):
            if li >= len(attn_output.attentions): continue
            att = (
                attn_output.attentions[li][0, :, -1, st:end]
                .mean(dim=0).to(torch.float32).cpu().numpy()
            )
            att_map = att.reshape(t, h2, w2).mean(axis=0)
            att_maps.append(att_map)

        if not att_maps: continue
        att_avg_low_res = np.mean(att_maps, axis=0)

        crop_bbox = crop.get("bbox")
        crop_width = int(crop_bbox[2] - crop_bbox[0])
        crop_height = int(crop_bbox[3] - crop_bbox[1])
        if crop_width <= 0 or crop_height <= 0: continue
        att_avg_upscaled = cv2.resize(att_avg_low_res, (crop_width, crop_height), interpolation=cv2.INTER_LINEAR)

        # 개별 맵 시각화는 모든 레벨에 대해 수행
        if individual_maps_dir:
            individual_map = np.zeros((original_height, original_width), dtype=np.float32)
            left, top, right, bottom = map(int, crop_bbox)
            individual_map[top:bottom, left:right] = att_avg_upscaled
            
            plt.figure(figsize=(10, 10 * original_height / original_width))
            plt.imshow(original_image, extent=(0, original_width, original_height, 0))
            plt.imshow(individual_map, cmap='viridis', alpha=0.6, extent=(0, original_width, original_height, 0))
            plt.axis('off')
            plt.title(f"Attention for Crop ID: {crop.get('id')} (Level: {crop.get('level')})")
            individual_save_path = os.path.join(individual_maps_dir, f"individual_attn_crop_{crop.get('id')}.png")
            plt.savefig(individual_save_path, dpi=150, bbox_inches='tight', pad_inches=0)
            plt.close()

        # ▼▼▼▼▼▼▼▼▼▼▼▼ [수정된 부분] ▼▼▼▼▼▼▼▼▼▼▼▼
        # level이 1인 crop의 어텐션만 최종 맵에 누적합니다.
        if crop.get("level") == 1:
            left, top, right, bottom = map(int, crop_bbox)
            aggregated_attention_map[top:bottom, left:right] += att_avg_upscaled
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    # 최종 합산된 맵 시각화
    plt.figure(figsize=(10, 10 * original_height / original_width))
    plt.imshow(original_image, extent=(0, original_width, original_height, 0))
    plt.imshow(aggregated_attention_map, cmap='viridis', alpha=0.6, extent=(0, original_width, original_height, 0))
    plt.axis('off')
    plt.title("Aggregated Attention Map (Level 1 Only)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"Aggregated attention map saved at: {save_path}")
    if individual_maps_dir:
        print(f"Individual attention maps saved in: {individual_maps_dir}")


# pilot_stage_ensemble.py 파일 내 run_stage 함수 안
def run_stage2(msgs, crop_list, top_q, drop_indices: List, attn_vis_dir: str, original_image):

    attn_output, crop_list = run_single_forward(msgs=msgs, crop_list=crop_list)
    df = attn2df(attn_output=attn_output, crop_list=crop_list)
    
    # 기존 시각화 함수 호출
    visualize_attn_map(attn_output=attn_output, msgs=msgs, crop_list=crop_list, attn_vis_dir=attn_vis_dir)
    
    # ▼▼▼▼▼▼▼▼▼▼▼▼ 여기에 새로운 함수 호출 추가 ▼▼▼▼▼▼▼▼▼▼▼▼
    # 1. 합쳐진 최종 맵 저장 경로
    agg_save_path = os.path.join(attn_vis_dir, "aggregated_attention_map.png")
    # 2. 합치기 전 개별 맵들을 저장할 폴더 경로
    individual_maps_save_dir = os.path.join(attn_vis_dir, "individual_maps")

    visualize_aggregated_attention(
        attn_output=attn_output,
        crop_list=crop_list,
        original_image=original_image,
        processor=processor,
        save_path=agg_save_path,
        individual_maps_dir=individual_maps_save_dir # 이 인자를 추가합니다.
    )
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
    
    raw_attn_res = df.to_dict(orient="index")

    # exclude index 0 if present before selecting top-k

    for i in drop_indices:
        df = df.drop(index=i, errors='ignore')  


    # Step 1. thre 넘은 crop(top_k crop) 가져오기
    top_q_crop_ids = get_top_q_crop_ids(top_q=top_q, attn_df=df)
        
    print(f"instruction: {instruction}, selected crops: {top_q_crop_ids}")

    # 1.2. get top-k bboxes
    top_q_bboxes = []

    for crop in crop_list:
        if crop.get("id") not in drop_indices and crop.get("id") in top_q_crop_ids:
            top_q_bboxes.append(crop.get("bbox"))

    return top_q_crop_ids, top_q_bboxes, raw_attn_res, crop_list


# -----------------------------------------------------

question_template="""
You are an assistant trained to navigate the android phone. Given a
task instruction, a screen observation, guess where should you tap.
# Intruction
{task_prompt}"""


#! Hyper Parameter
AREA_THRESHOLD_RATIO = 0.20   # 20% 이상이면 크롭
MAX_RECURSION_ITERS = 6       # 무한 루프 방지를 위한 최대 재귀(반복) 횟수
S1_WINDOW_SIZE = 120          # stage1 최초 세분화 크기 (기존 120)
REFINE_WINDOW_SIZE = 75       # 재세분화 크기 (기존 stage2에서 사용하던 75)


def _area_of(bbox: List[int | float]) -> float:
    left, top, right, bottom = bbox
    return max(0.0, float(right - left)) * max(0.0, float(bottom - top))


def refine_crops_until_area_threshold(
    original_image: Image.Image,
    base_dir: str,
    filename_wo_ext: str,
    inst_dir_name: str,
    initial_crops: List[Dict],
    area_threshold_ratio: float,
    refine_window_size: int,
    start_id: int = 0,
) -> List[Dict]:
    """
    초기 세분화 결과(initial_crops)에서 level>=1 crop 중 면적이 임계 비율보다 큰 경우,
    해당 crop을 잘라서 재세분화(run_segmentation)하고, 자식들을 원본 좌표계로 보정하여 대체한다.
    - level은 부모+1로 증가시킴
    - 세분화 실패 시(유효한 자식 없음) 부모를 유지하고 다시 시도하지 않도록 lock
    - 반복은 모든 crop이 임계 이하가 되거나 최대 반복 수에 도달하면 종료
    반환: level>=1인 최종 crop 리스트(썸네일 제외)
    """
    W, H = original_image.size
    area_threshold_px = W * H * float(area_threshold_ratio)

    # 썸네일(level 0)은 유지하되 재세분화 대상에서 제외
    thumbnail = next((c for c in initial_crops if c.get("level") == 0), None)
    working = [deepcopy(c) for c in initial_crops if c.get("level", 1) >= 1]

    # next id 계산
    next_id = (max([c.get("id", -1) for c in initial_crops], default=-1) + 1)

    iter_idx = 0
    while iter_idx < MAX_RECURSION_ITERS:
        iter_idx += 1
        changed = False
        new_working = []
        for crop in working:
            if crop.get("locked"):
                new_working.append(crop)
                continue

            area = _area_of(crop.get("bbox"))
            if area <= area_threshold_px:
                new_working.append(crop)
                continue

            # 재세분화 시도
            parent_level = int(crop.get("level", 1))
            left, top, right, bottom = map(int, crop.get("bbox"))

            # 저장 경로 준비 및 이미지 저장
            refine_dir = os.path.join(base_dir, filename_wo_ext, inst_dir_name, "s1", "refine", f"crop{crop.get('id')}", f"iter{iter_idx}")
            os.makedirs(refine_dir, exist_ok=True)
            refine_img_path = os.path.join(refine_dir, "selected.png")
            crop_img: Image.Image = crop["img"]
            crop_img.save(refine_img_path)

            output_json_path = os.path.join(refine_dir, "output.json")
            output_image_path = refine_dir  # 내부 저장용

            try:
                detail_crops = run_segmentation(
                    image_path=refine_img_path,
                    max_depth=1,
                    window_size=refine_window_size,
                    output_json_path=output_json_path,
                    output_image_path=output_image_path,
                    start_id=next_id,
                )
            except Exception as e:
                # 실패 시 이 crop은 더 이상 시도하지 않음
                crop["locked"] = True
                new_working.append(crop)
                continue

            # 유효한 자식(level 1)만 사용
            children = [dc for dc in detail_crops if int(dc.get("level", 1)) == 1]
            if not children:
                crop["locked"] = True
                new_working.append(crop)
                continue

            # 자식 좌표 보정 및 level, id 재할당
            adjusted_children = []
            for ch in children:
                ch = deepcopy(ch)
                ch_bbox = ch.get("bbox", [0, 0, 0, 0])
                ch_left, ch_top, ch_right, ch_bottom = ch_bbox
                ch["bbox"] = [
                    ch_left + left,
                    ch_top + top,
                    ch_right + left,
                    ch_bottom + top,
                ]
                ch["level"] = parent_level + 1
                ch["id"] = next_id
                next_id += 1
                adjusted_children.append(ch)

            # 변경 반영
            new_working.extend(adjusted_children)
            changed = True

        working = new_working
        if not changed:
            break

    # 최종 필터: 임계 이하만 남김
    final_crops = [c for c in working if _area_of(c.get("bbox")) <= area_threshold_px]

    # 썸네일은 외부에서 필요 시 추가
    if thumbnail is not None:
        return [thumbnail] + final_crops
    return final_crops


for task in tasks:
    task_res = dict()
    dataset = "screenspot_" + task + "_v2.json"
    screenspot_data = json.load(open(os.path.join(args.screenspot_test, dataset), 'r'))
    # vanilla_qwen_data = json.load(open(os.path.join(args.screenspot_test, "vanilla_qwen_screenspot_mobile.json"), 'r'))
    # screenspot_data = screenspot_data[:150]
    screenspot_data = screenspot_data[:30]
    # screenspot_data = screenspot_data[5:10]  #! 데이터 몇개 쓸건지

    print("Num of sample: " + str(len(screenspot_data)), flush=True)
    
    task_res = list()
    num_action = 0
    corr_action = 0

    # gt_included = []

    res_board_dict = dict()

    for stage_num in range(1, 2+1):
      if stage_num == 1:  
        res_board_dict[stage_num] = {
          "is_coarse": True,
          "gt_score_list": [],
          "top_q_crop_ids": [],
          "is_gt_in_top_q": False,
          "raw_attn_dict": None,
        }
      elif stage_num == 2:
        res_board_dict[stage_num] = {
          "is_coarse": True,
          "gt_score_list": [],
          "sub_res": [],
        }

    # sub_res  {"parent_crop_id": 1, "top_q_crop_ids": [], "raw_attn_dict": {}}
    # res_board_dict[stage_num] = {
    #   "is_coarse": True,
    #   "top_q_crop_ids": [],
    #   "gt_score_list": [],
    #   "raw_attn_dict": None,
    #   "next_stage_res": {
    #       "is_coarse": False,
    #       "gt_score_list": [],
    #       "sub_res": []
    #   } 
    # }

    # stage 1: mode, gt_score_list, top crop ids, raw_attn_dict, is_gt
    # stage 2: mode, gt_score_list, is_gt, stage 1 selected crop 별로 top_crop_ids, raw_attn_dict, 
    
    num_wrong_format = 0
    num_segmentation_failed = 0


    for j, item in tqdm(enumerate(screenspot_data)):
        item_res = dict()
        num_action += 1
        filename = item["img_filename"]
        filename_wo_ext, ext = os.path.splitext(filename)
        img_path = os.path.join(args.screenspot_imgs, filename)
        if not os.path.exists(img_path):
            print(f"{img_path} not found", flush=True)
            input()

        original_image = Image.open(img_path).convert("RGB")
        # original_img_size = original_image.size

        image = original_image

        instruction = item["instruction"]
        original_bbox = item["bbox"]
        original_bbox = [original_bbox[0], original_bbox[1], original_bbox[0] + original_bbox[2], original_bbox[1] + original_bbox[3]]
       
        bbox = original_bbox
        # item_res['original_bbox'] = copy.copy(original_bbox)

        # item_res['bbox'] = copy.copy(original_bbox)

        question = question_template.format(task_prompt=instruction)        
        
        ################
        ####### crop 추가
        inst_dir_name = re.sub(r'\W+', '_', instruction).strip('_')
        os.makedirs(f"{seg_save_base_dir}/{filename_wo_ext}/{inst_dir_name}/s1", exist_ok=True)
        os.makedirs(f"{seg_save_base_dir}/{filename_wo_ext}/{inst_dir_name}/s1_processed", exist_ok=True)

        
        crop_list = run_segmentation(
            image_path=img_path,
            max_depth=1,
            window_size=S1_WINDOW_SIZE,
            output_json_path=f"{seg_save_base_dir}/{filename_wo_ext}/{inst_dir_name}/s1/output.json",
            output_image_path=f"{seg_save_base_dir}/{filename_wo_ext}/{inst_dir_name}/s1"
        )
        # print(crop_list)


        # 무조건 전체샷은 id=0
        for crop in crop_list:
            if crop.get("level") == 0:
                crop["id"] = 0

        # segmentation fault 검사 (초기)
        only_lv_1 = [crop for crop in crop_list if crop.get("level") == 1]
        if len(only_lv_1) == 0:
            print("segmenation failed")
            num_segmentation_failed += 1
            continue

        # [변경] Stage1 내에서 재귀 세분화: 임계 면적 이하가 될 때까지 반복 세분화
        refined_crops = refine_crops_until_area_threshold(
            original_image=original_image,
            base_dir=seg_save_base_dir,
            filename_wo_ext=filename_wo_ext,
            inst_dir_name=inst_dir_name,
            initial_crops=crop_list,
            area_threshold_ratio=AREA_THRESHOLD_RATIO,
            refine_window_size=REFINE_WINDOW_SIZE,
            start_id=len(crop_list),
        )

        # stage에 맞는 crop만 뽑아내기 (썸네일 포함, 나머지는 임계 이하)
        stage_crop_list = create_stage_crop_list(
            crop_list=refined_crops,
            resize_dict={0: thumnail_resize_ratio, 1: s1_resize_ratio},
            use_thumbnail=True,
            default_resize=deep_resize_ratio,
        )
        msgs = create_msgs(crop_list=stage_crop_list, question=question)

        for _stage_crop in stage_crop_list:
            _stage_crop['resized_img'].save(f"{seg_save_base_dir}/{filename_wo_ext}/{inst_dir_name}/s1_processed/{_stage_crop['id']}.png")
        #########

# -----------------------------------------------------------------------------------------------
        # [Stage 1]
        # 1. Run stage
        # top_q_crop_ids, top_q_bboxes, raw_res_dict, stage1_crop_list = run_stage(msgs=msgs, crop_list=stage_crop_list, top_q=top_q)
        # is_gt_in_top_q = check_gt_in_top_q_crops(top_q_bboxes=top_q_bboxes, gt_bbox=original_bbox)
        attn_vis_dir = os.path.join(seg_save_base_dir, filename_wo_ext, inst_dir_name , "s1/attn_map")
        s1_crop_list = stage_crop_list
        s1_top_q_crop_ids, s1_top_q_bboxes, s1_raw_res_dict, s1_crop_list = run_stage( # s1_raw_res_dict 가 attention?
            msgs=msgs, crop_list=stage_crop_list, top_q=top_q, drop_indices=[0], 
            attn_vis_dir=attn_vis_dir
        )
        s1_is_gt_in_top_q = check_gt_in_top_q_crops(top_q_bboxes=s1_top_q_bboxes, gt_bbox=original_bbox)


        gt_vis_dir = os.path.join(seg_save_base_dir, filename_wo_ext, inst_dir_name , f"s1")
        # 2. Visualize Result
        visualize_result(
            save_dir=gt_vis_dir,
            gt_bbox=original_bbox,
            top_q_bboxes=s1_top_q_bboxes,
            instruction=instruction,
            click_point=None,
            base_image=original_image
        )

        # Update Score
        res_board_dict[1]["gt_score_list"].append(1 if s1_is_gt_in_top_q else 0)
        print(f"s1 gt_contained: {s1_is_gt_in_top_q}")

        res_board_dict[1]["top_q_crop_ids"] = s1_top_q_crop_ids
        res_board_dict[1]["raw_attn_dict"] = s1_raw_res_dict
        res_board_dict[1]["is_gt_in_top_q"] = s1_is_gt_in_top_q

        for c in s1_crop_list:
            c.pop("img", None)
            c.pop("resized_img", None)
            c.pop("token_span", None)
        res_board_dict[1]["crop_list"] = s1_crop_list

        print(f"Selected Crops from Stage 1: {s1_top_q_crop_ids}")

        # -------------------------------
        # [Stage 2] - cropping removed, reuse Stage 1 crops
        s2_is_gt_in_top_q_list = []
        res_board_dict[2]["sub_res"] = []

        # thumbnail reference
        thumbnail_crop = next((c for c in refined_crops if c.get('level') == 0), None)

        for crop_id in s1_top_q_crop_ids:
            print(f"[Stage 2 - reuse stage1 crop {crop_id} without new cropping]")
            s1_selected_crop = next((c for c in refined_crops if c.get('id') == crop_id), None)
            if s1_selected_crop is None:
                continue

            # Prepare stage2 crop list: thumbnail + the selected crop only
            s2_crop_list_base = [thumbnail_crop, s1_selected_crop] if thumbnail_crop is not None else [s1_selected_crop]

            # Resize for model input
            s2_crop_list = create_stage_crop_list(
                crop_list=s2_crop_list_base,
                resize_dict={0: thumnail_resize_ratio, 1: s1_resize_ratio},
                use_thumbnail=True,
                default_resize=deep_resize_ratio,
            )

            # Save resized inputs
            os.makedirs(f"{seg_save_base_dir}/{filename_wo_ext}/{inst_dir_name}/s2_processed/crop{crop_id}", exist_ok=True)
            for _stage_crop in s2_crop_list:
                _stage_crop['resized_img'].save(f"{seg_save_base_dir}/{filename_wo_ext}/{inst_dir_name}/s2_processed/crop{crop_id}/{_stage_crop['id']}.png")

            # Run Stage 2 evaluation/visualization without new crops
            s2_attn_vis_dir = os.path.join(seg_save_base_dir, filename_wo_ext, inst_dir_name , f"s2/crop{crop_id}/attn_map")
            s2_msgs = create_msgs(crop_list=s2_crop_list, question=question)

            # drop only thumbnail (keep selected crop)
            s2_top_q_crop_ids, s2_top_q_bboxes, s2_raw_res_dict, s2_crop_list = run_stage2(
                msgs=s2_msgs, crop_list=s2_crop_list, top_q=0.3, drop_indices=[0], 
                attn_vis_dir=s2_attn_vis_dir,
                original_image=original_image
            )

            s2_is_gt_in_top_q = check_gt_in_top_q_crops(top_q_bboxes=s2_top_q_bboxes, gt_bbox=original_bbox)
            s2_is_gt_in_top_q_list.append(s2_is_gt_in_top_q)

            # Visualize result overlay
            viz_dir = os.path.join(seg_save_base_dir, filename_wo_ext, inst_dir_name , f"s2", f"crop{crop_id}")
            visualize_result(
                save_dir=viz_dir,
                gt_bbox=original_bbox,
                top_q_bboxes=s2_top_q_bboxes,
                instruction=instruction,
                click_point=None,
                base_image=original_image
            )

            # strip large fields before storing
            for c in s2_crop_list:
                c.pop("img", None)
                c.pop("resized_img", None)
                c.pop("token_span", None)

            sub_res_dict = {
                "parent_crop_id": crop_id,
                "is_gt_in_top_q": s2_is_gt_in_top_q,
                "top_q_crop_ids": s2_top_q_crop_ids,
                "raw_attn_dict": s2_raw_res_dict,
                "crop_list": s2_crop_list
            }
            res_board_dict[2]["sub_res"].append(sub_res_dict)

        # Aggregate Stage 2 metrics per sample
        s2_is_gt_in_top_q_tot = any(s2_is_gt_in_top_q_list) if s2_is_gt_in_top_q_list else False
        res_board_dict[2]["gt_score_list"].append(1 if s2_is_gt_in_top_q_tot else 0)
        res_board_dict[2]["is_gt_in_top_q"] = s2_is_gt_in_top_q_tot
        print(f"s2 gt_contained: {s2_is_gt_in_top_q_tot}")

        #########

    # 단일 스테이지 기준 성능 집계
    _gt_score_list = res_board_dict[1]["gt_score_list"]
    up2now_gt_score = calc_acc(_gt_score_list)
    print(f"Up2Now gt_score:{up2now_gt_score}")
    print("------")
    print()
        


    item_res['filename'] = filename
    item_res['data_type'] = item["data_type"]
    item_res['data_source'] = item["data_source"]
    item_res['instruction'] = instruction
    item_res['stage1_res'] = res_board_dict[1]
    item_res['stage2_res'] = res_board_dict[2]
    item_res['gt_bbox'] = original_bbox

        # item_res['attn_result'] = attn_result # json dict 형태로 # stage 별로
        # item_res['gt_included'] = s2_is_gt_in_top_q_tot
        # item_res['top_q_crop_ids'] = top_q_crop_ids

    item_res['num_crop'] = len(stage_crop_list)
    # item_res['crop_bbox'] = [crop.get("bbox") for crop in stage_crop_list] #고치기
    task_res.append(item_res)


    with open(os.path.join(args.save_dir, dataset), "w") as f:
        json.dump(task_res, f, indent=4, ensure_ascii=False)

    print(task + ": Total num: " + str(num_action))
    print(task + ": Wrong format num: " + str(num_wrong_format))
    final_acc = calc_acc(res_board_dict[1]["gt_score_list"]) if res_board_dict and 1 in res_board_dict else 0
    final_acc_s2 = calc_acc(res_board_dict[2]["gt_score_list"]) if res_board_dict and 2 in res_board_dict else 0
    print(task + f": Ground truth included Acc (S1): " + str(final_acc))
    print(task + f": Ground truth included Acc (S2): " + str(final_acc_s2))


    metrics = {
        "task": task,
        "total_num": num_action,
        "segmentation_failed": num_segmentation_failed,
        "num_segmentation_failed": num_segmentation_failed,
    "acc": final_acc,
    "acc_s2": final_acc_s2
    }

    with open(os.path.join(args.save_dir, f"{task}_metrics.json"), "w") as mf:
        json.dump(metrics, mf, ensure_ascii=False, indent=4)