import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "2,3"  # 몇번 GPU 사용할지 ("0,1", "2" 등)
max_memory = {
    0: "75GiB",
    1: "75GiB",
    # 2: "75GiB",
    "cpu": "120GiB",  # 남는 건 CPU 오프로딩xs
}


import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from copy import deepcopy

import re
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
import json
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer, set_seed
from qwen_vl_utils import process_vision_info
from crop import run_segmentation_recursive
import torch.nn.functional as F
from collections import deque
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import time

#! Argument =======================

SEED = 0

# Enviroment


# Dataset & Model
MLLM_PATH = "zonghanHZH/ZonUI-3B"
SCREENSPOT_IMGS = "./data/screenspotv2_imgs"  # input image 경로
SCREENSPOT_TEST = "./data"  # json파일 경로
SAVE_DIR = "./attn_output/" + "0819_crop"  #! 결과 저장 경로 (방법을 바꾼다면 바꿔서 기록하기)

# Data Processing
SAMPLING = False  # data 섞을지
TASKS = ["mobile"]
SAMPLE_RANGE = slice(None)  # 샘플 범위 지정 (5~9번 샘플이면 5,10 / 전체 사용이면 None)
# SAMPLE_RANGE = slice()

#! Hyperparameter =================

# Model Architecture
LAYER_NUM_LIST = [3, 7, 11, 15, 31]
LAYER_NUM = 31

# Stage 1: Segmentation & Selection
VAR_THRESH = 120  # Segmentation variance threshold
TOP_Q = 0.5  # Top quantile for crop selection

# Stage 2: Attention Refinement  
AGG_START = 20  # Starting layer for attention aggregation

# Image Resize Ratios
S1_RESIZE_RATIO = 0.30  # Stage 1 crop resize ratio
S2_RESIZE_RATIO = 0.50  # Stage 2 crop resize ratio  
THUMBNAIL_RESIZE_RATIO = 0.05  # Thumbnail resize ratio


TOPK_SINKS  = 20             # 공통 sink 좌표로 잡을 개수
PER_MAP_TOPN_FRAC = 0.05     # 각 맵에서 상위 몇 %를 "상위값"으로 간주할지 (빈도수 집계용)
RENORMALIZE = False           # 0으로 만든 후 vision span 내에서 재정규화할지 여부
SKIP_INDICES = {}

#! ================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mllm_path', type=str, default=MLLM_PATH)
    parser.add_argument("--sampling", action='store_true', default=SAMPLING, help="do sampling for mllm")
    parser.add_argument('--screenspot_imgs', type=str, default=SCREENSPOT_IMGS)
    parser.add_argument('--screenspot_test', type=str, default=SCREENSPOT_TEST)
    parser.add_argument('--save-dir', type=str, default=SAVE_DIR)
    parser.add_argument("--vis_flag", action='store_true', help="visualize mid-results")
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

    image_inputs, video_inputs = process_vision_info(msgs)

    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to(model.device)


    # # 3) grid 크기 뽑아두기
    img_proc_out = processor.image_processor(images=image_inputs)
    grid = img_proc_out["image_grid_thw"]
    # # (batch, num_imgs, 3) 혹은 (num_imgs, 3) 형태일 수 있으니 안전하게 뽑기
    if grid.ndim == 3:
        grid = grid[0]   # (num_imgs, 3)

    # # 최종 token-map 차원: t × (h//2) × (w//2)
    # final_shapes = [
    #     (t, h//2, w//2)
    #     for t, h, w in grid
    # ]

    for i, (t,h,w) in enumerate(grid):
        crop_list[i]['patch_shape'] = (t, h//2, w//2)
        

    return inputs, crop_list

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

def create_stage_crop_list(crop_list: List, resize_dict: Dict, use_thumbnail: bool = True):
    stage_crop_list = []

    for crop in crop_list:
        crop_level = crop.get("level")
        if not use_thumbnail and crop_level == 0: continue

        # 현재 crop의 level에 맞는 resize 비율을 가져옵니다.
        ratio = resize_dict.get(crop_level)

        # 만약 level이 resize_dict에 없다면(e.g., None, 2), crop을 누락시키지 않습니다.
        # 대신, level 1의 비율을 기본값으로 사용합니다.
        if ratio is None:
            ratio = resize_dict.get(1) # level 1 비율로 대체
            if ratio is None: # level 1 비율조차 없다면 경고 후 스킵
                print(f"Warning: Skipping crop ID {crop.get('id')} due to missing resize ratio for level {crop_level} and no fallback.")
                continue

        new_crop = deepcopy(crop)
        crop_img = deepcopy(crop["img"])

        # 이미지 리사이즈
        crop_width, crop_height = crop_img.size
        crop_img = crop_img.resize((int(crop_width * ratio), int(crop_height * ratio)))

        new_crop["resized_img"] = crop_img

        stage_crop_list.append(new_crop)

    return stage_crop_list

def run_single_forward(inputs, crop_list: List):
    # Process Input
    # inputs = process_inputs(msgs=msgs)

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
        for li in range(LAYER_NUM):
            # 레이어마다 해당 crop이 받은 어텐션 벡터 추출
            att_vector = (
                attn_output.attentions[li][0, :, -1, st:end] # starting answer token(텍스트 토큰 마지막)
                .mean(dim=0)
                .to(torch.float32)
                .cpu()
                .numpy()
            )
            # 어텐션 싱크를 제외하고 평균 계산
            mean_att_without_sink = att_vector.mean()  # calculate_mean_without_sinks 함수가 없어서 일단 평균으로 대체
            layer_avgs.append(mean_att_without_sink)

        _data.append(layer_avgs)

    df = pd.DataFrame(
            _data,
            columns=[f"layer{layer_idx+1}" for layer_idx in range(LAYER_NUM_LIST[-1])],
            index=[crop["id"] for crop in crop_list]
            )

    return df

def get_top_q_crop_ids(top_q, attn_df):
    df = deepcopy(attn_df)
     # LAYER_NUM까지의 어텐션 합 구하기
    cols_to_sum = [f"layer{i}" for i in range(20, LAYER_NUM + 1)] #HERE!!!! #20?1?attn_map 20으로 한결과
    df[f"sum_until_{LAYER_NUM}"] = df[cols_to_sum].sum(axis=1)

    # 1.1. get top-q crop index
    # quantile 임계치로 선택
    keep_frac = top_q  # 예: 0.2 → 상위 20% 유지
    thr = df[f"sum_until_{LAYER_NUM}"].quantile(1 - keep_frac)

    top_q_crop_ids = df.index[df[f"sum_until_{LAYER_NUM}"] >= thr].tolist()

    # # guardrail: 최소/최대 개수 보정(선택)
    if len(top_q_crop_ids) < 1:
        raise Exception("no crop selected")
    return top_q_crop_ids

def check_gt_in_top_q_crops(top_q_bboxes: List, gt_bbox: List):
    gt_center_x = (gt_bbox[0] + gt_bbox[2]) / 2.0
    gt_center_y = (gt_bbox[1] + gt_bbox[3]) / 2.0
    gt_center_point = (gt_center_x, gt_center_y)
    gt_point_in_top_q_crops = any(point_in_bbox(gt_center_point, bbox) for bbox in top_q_bboxes)
    return gt_point_in_top_q_crops

def calculate_entropy(attention_map: np.ndarray) -> float:
    """
    어텐션 맵(2D numpy array)의 엔트로피를 계산합니다.
    """
    epsilon = 1e-9
    prob_map = attention_map / (attention_map.sum() + epsilon)
    non_zero_probs = prob_map[prob_map > 0]
    entropy = -np.sum(non_zero_probs * np.log2(non_zero_probs))
    return entropy

def run_selection_pass(msgs, crop_list, top_q, drop_indices: List, attn_vis_dir: str):
    """
    [Stage 1용] Attention 스코어를 계산하고, threshold(top_q)를 넘는 상위 crop들을 선택(selection)합니다.
    """
    s1_inputs, crop_list = process_inputs(msgs=msgs, crop_list=crop_list)
    attn_output, crop_list = run_single_forward(inputs=s1_inputs, crop_list=crop_list)
    df = attn2df(attn_output=attn_output, crop_list=crop_list)

    # Stage 1 진단용 시각화 (개별 crop의 low-res attention)
    visualize_attn_map(attn_output=attn_output, msgs=msgs, crop_list=crop_list, attn_vis_dir=attn_vis_dir)

    raw_attn_res = df.to_dict(orient="index")

    for i in drop_indices:
        df = df.drop(index=i, errors='ignore')

    top_q_crop_ids = get_top_q_crop_ids(top_q=top_q, attn_df=df)
    print(f"instruction: {instruction}, selected crops: {top_q_crop_ids}")

    top_q_bboxes = []
    for crop in crop_list:
        if crop.get("id") not in drop_indices and crop.get("id") in top_q_crop_ids:
            top_q_bboxes.append(crop.get("bbox"))

    return top_q_crop_ids, top_q_bboxes, raw_attn_res, crop_list

def run_refinement_pass(crop_list: List, question: str, original_image: Image, save_dir: str, gt_bbox: List):
    """
    [Stage 2용] 선택된 crop 리스트 전체를 사용하여 최종 Attention Map을 생성하고,
    Grounding Accuracy를 계산하여 성공 여부를 반환합니다.
    """
    print("Running second forward pass for refinement...")
    os.makedirs(save_dir, exist_ok=True)

    # 1. Stage 2에 맞는 크기로 이미지들을 리사이즈하고 모델 입력을 준비합니다.
    # for crop in crop_list:
    #     print(f"id: {crop['id']}, level{crop['level']}")

    s2_resized_crop_list = create_stage_crop_list(
        crop_list=crop_list,
        resize_dict={0: THUMBNAIL_RESIZE_RATIO, 1: S2_RESIZE_RATIO},
        use_thumbnail=True
    )

    os.makedirs(f"{seg_save_base_dir}/{filename_wo_ext}/{inst_dir_name}/s2_processed", exist_ok=True)
    for _crop in s2_resized_crop_list:
        # os.makedirs(f"{seg_save_base_dir}/{filename_wo_ext}/{inst_dir_name}/s2_processed/crop{_crop['id']}", exist_ok=True)
        # _crop['resized_img'].save(f"{seg_save_base_dir}/{filename_wo_ext}/{inst_dir_name}/s2_processed/crop{_crop['id']}/{_crop['id']}.png")
        _crop['resized_img'].save(f"{seg_save_base_dir}/{filename_wo_ext}/{inst_dir_name}/s2_processed/crop{_crop['id']}.png")


    msgs_s2 = create_msgs(crop_list=s2_resized_crop_list, question=question)

    # 2. 모델을 실행하여 attention 결과만 얻습니다.
    s2_inputs, s2_crop_list_w_shape = process_inputs(msgs=msgs_s2, crop_list=s2_resized_crop_list)
    s2_attn_output, s2_final_crop_list = run_single_forward(
        inputs=s2_inputs, 
        crop_list=s2_crop_list_w_shape)

    # 3. 모든 crop의 attention을 종합하여 최종 맵을 생성하고 Grounding 성공 여부를 확인합니다.
    print("Generating final aggregated attention map and checking grounding accuracy...")
    
    # 저장 파일명을 "result.png"로 변경
    s2_agg_save_path = os.path.join(save_dir, "result.png")
    s2_individual_maps_dir = os.path.join(save_dir, "individual_maps_refined")


    # (0) 사용할 query text token 정하기
    input_ids_1d = s2_inputs["input_ids"][0]
    last_vision_end = get_last_vision_end_index(input_ids_1d, processor)
    seq_len = input_ids_1d.shape[0]
    if last_vision_end < 0 or last_vision_end >= seq_len - 1:
        query_indices = [seq_len - 1]
    else:
        query_indices = list(range(last_vision_end + 1, seq_len))


    exclude = set()

    # (a) 마지막 텍스트 토큰(여기서는 편의상 마지막 토큰) 제외
    if seq_len > 0:
        exclude.add(seq_len - 1)

    # (b) 마지막 <|im_start|> 제외
    im_start_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_pos = (input_ids_1d == im_start_id).nonzero(as_tuple=False).flatten().tolist()
    if len(im_pos) > 0:
        exclude.add(int(im_pos[-1]))

    # (c) '?' 토큰 제외 (토큰 문자열 정규화 후 비교)
    cand_ids = input_ids_1d[query_indices].tolist()
    cand_tokens = processor.tokenizer.convert_ids_to_tokens(cand_ids)

    def is_qmark_token(tok: str) -> bool:
        # GPT2류 'Ġ', sentencepiece류 '▁' 제거 + trim
        t = tok.replace("Ġ", "").replace("▁", "").strip()
        return t in {"?", "？"}

    for idx, tok in zip(query_indices, cand_tokens):
        if is_qmark_token(tok):
            exclude.add(int(idx))

    # 최종 필터링
    query_indices = [i for i in query_indices if i not in exclude]


    # (1) 모든 레이어×모든 텍스트 질의 토큰에 대한 맵 수집 & 빈도 누적
    

    s2_final_crop_list = collect_maps_and_sink_counts(
        s2_attn_output, 
        s2_inputs, 
        s2_final_crop_list,
        query_indices=query_indices,
        # grounding_query=exclude,
        layer_range=range(AGG_START, LAYER_NUM), 
        processor=processor, 
        per_map_topn_frac=PER_MAP_TOPN_FRAC,

    )

    # (2) 이미지별 공통 sink 좌표 Top-K 추출
    for i, crop in enumerate(s2_final_crop_list):
        t, h2, w2 = crop['patch_shape']
        sink_freqs = crop['sink_freqs']
        # sink_freqs = per_img_sink_freqs[img_idx]

        sinks = pick_common_sinks(sink_freqs, h2, w2, k=TOPK_SINKS)
        crop['sink_coords'] = sinks
        # sink_coords_per_img.append(sinks)


    # (3) 마스킹 후 맵 재계산
    

    s2_final_crop_list = zero_out_sinks_and_remap(
        model_output=s2_attn_output,
        crop_list=s2_final_crop_list,
        layer_range=range(AGG_START, LAYER_NUM),
        query_indices=exclude,
        renormalize=RENORMALIZE
    )

    for i, crop in enumerate(s2_final_crop_list):
    # 원본 평균맵
        # if len(per_img_maps[img_idx]) > 0:
        if len(crop["att_maps"]) > 0:
            att_avg_orig = np.mean(crop["att_maps"], axis=0)
            crop["att_avg_orig"] = att_avg_orig
        else:
            # t, h2, w2 = final_shapes[img_idx]
            t, h2, w2 = crop["patch_shape"]
            att_avg_orig = np.zeros((h2, w2), dtype=np.float32)
            crop["att_avg_orig"] = att_avg_orig

        # 마스킹 평균맵
        # if len(per_img_maps_masked[img_idx]) > 0:
        if len(crop["masked_att_maps"]) > 0:
            att_avg_masked = np.mean(crop["masked_att_maps"], axis=0)
            crop["att_avg_masked"] = att_avg_masked
        else:
            t, h2, w2 = crop["patch_shape"]
            att_avg_masked = np.zeros((h2, w2), dtype=np.float32)
            crop["att_avg_masked"] = att_avg_masked




    is_success = visualize_aggregated_attention(
        # attn_output=s2_attn_output, 
        crop_list=s2_final_crop_list,
        original_image=original_image, processor=processor,
        save_path=s2_agg_save_path,
        gt_bbox=gt_bbox,
        individual_maps_dir=s2_individual_maps_dir
    )
    print("Stage 2 refinement complete.")
    if is_success:
        print("✅ Grounding Success")
    else:
        print("❌ Grounding Fail")

    return is_success

#! ================================================================================================
# Accumulate time logs
TIME_LOGS = []
STAGE_TIMES = {}

def measure_and_log_time(stage_name, start_time, end_time):
    """
    Logs the elapsed time for a stage to a global list and groups by stage.

    Parameters:
    - stage_name: Name of the stage being measured.
    - start_time: Start time of the stage.
    - end_time: End time of the stage.

    Returns:
    - elapsed_time: The time taken for the stage in seconds.
    """
    elapsed_time = end_time - start_time
    TIME_LOGS.append(f"{stage_name}: {elapsed_time:.2f} seconds")
    if stage_name not in STAGE_TIMES:
        STAGE_TIMES[stage_name] = 0
    STAGE_TIMES[stage_name] += elapsed_time
    return elapsed_time

def save_time_logs(log_file_path):
    """
    Save all accumulated time logs to a file, including grouped stage times and total time.

    Parameters:
    - log_file_path: Path to the log file where the times will be recorded.
    """
    with open(log_file_path, "w") as log_file:
        log_file.write("\n".join(TIME_LOGS) + "\n\n")
        log_file.write("Stage-wise Summary:\n")
        for stage, total_time in STAGE_TIMES.items():
            log_file.write(f"{stage}: {total_time:.2f} seconds\n")
        log_file.write("\nTotal Time: {:.2f} seconds\n".format(sum(STAGE_TIMES.values())))

#! ================================================================================================
# Visualize

def visualize_result(save_dir, gt_bbox, top_q_bboxes, instruction, click_point=None):
    # Visualize ground truth and selected crop on the image
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




def visualize_attn_map(attn_output, msgs, crop_list, attn_vis_dir):
    image_inputs, _ = process_vision_info(msgs)

    # 3) grid 크기 뽑아두기
    img_proc_out = processor.image_processor(images=image_inputs)
    grid = img_proc_out["image_grid_thw"]
    # (batch, num_imgs, 3) 혹은 (num_imgs, 3) 형태일 수 있으니 안전하게 뽑기
    if grid.ndim == 3:
        grid = grid[0]   # (num_imgs, 3)

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
    # print(f"attn_map saved at: {_save_path}")


def get_mean_att_map(model_output, span, img_idx, layer_range, img_patch_shapes):

    # if grid.ndim == 3:
    #     grid = grid[0]
    # final_shapes = [(t, h//2, w//2) for t, h, w in grid]
    
    # 어텐션 뽑기
    (st, end) = span
    t, h2, w2 = img_patch_shapes[img_idx]

    per_layer_att_maps = []
    for li in layer_range:
        if li >= len(model_output.attentions): continue
        att = (
            model_output.attentions[li][0, :, -1, st:end]
            .mean(dim=0).to(torch.float32).cpu().numpy()
        )
        att_map = att.reshape(t, h2, w2).mean(axis=0)
        per_layer_att_maps.append(att_map)


    att_avg = np.mean(per_layer_att_maps, axis=0)
    return att_avg


def upsample_att_map(att_map, size: Tuple):
    """
    param: att_map, size (height, width)
    """

    att_map_to_upsample = torch.from_numpy(att_map).to(torch.float32).unsqueeze(0).unsqueeze(0)
    upsampled_map = F.interpolate(att_map_to_upsample, size=size, mode='bilinear', align_corners=False)
    upsampled_map = upsampled_map.squeeze(0).squeeze(0).cpu().numpy()

    return upsampled_map




def get_last_vision_end_index(input_ids_1d, processor):
    """시퀀스에서 마지막 <|vision_end|> 토큰 인덱스 반환 (없으면 -1)."""
    vision_end_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
    ve_positions = (input_ids_1d == vision_end_id).nonzero(as_tuple=False).flatten()
    return int(ve_positions[-1].item()) if len(ve_positions) > 0 else -1

def top_indices(flat_values, topn):
    """1D 배열에서 상위 topn 인덱스(정렬된 내림차순) 반환."""
    topn = int(min(topn, flat_values.size))
    if topn <= 0:
        return np.array([], dtype=np.int64)
    # argpartition으로 빠르게 상위 집합 뽑고, 값 기준 정렬
    part = np.argpartition(flat_values, -topn)[-topn:]
    return part[np.argsort(flat_values[part])[::-1]]


def pick_common_sinks(counts, h2, w2, k=10):
    """
    빈도 누적(counts: shape=(h2*w2,))에서 공통적으로 높은 좌표 상위 k개 선택.
    반환: [(r,c), ...], 길이<=k (counts가 적으면 더 적을 수 있음)
    """
    hw = h2 * w2 # 이미지 패치 개수
    k = min(k, hw)
    if k <= 0:
        return []
    top_flat = top_indices(counts.astype(np.float32), k)
    coords = [(int(idx // w2), int(idx % w2)) for idx in top_flat]
    return coords


def zero_sink_in_att_vec(att_vec, t, h2, w2, sink_coords, renormalize=False, eps=1e-12):
    """
    vision span 내 어텐션 벡터(att_vec: shape=(t*h2*w2,))에서
    sink 좌표들의 모든 시간 위치를 0으로 설정. (선택적 재정규화)
    """
    vec = att_vec.copy()
    hw = h2 * w2 # 이미지 패치 개수
    base_idxs = [row * w2 + col for (row, col) in sink_coords]
    if len(base_idxs) == 0:
        return vec
    # 시간 축으로 stride=hw
    for base in base_idxs:
        vec[base: hw * t: hw] = 0.0
    # if renormalize:
    #     s = vec.sum()
    #     if s > eps:
    #         vec = vec / s
    return vec


def zero_out_sinks_and_remap(
        model_output, 
        # spans, 
        # final_shapes, 
        # sink_coords_per_img,
        crop_list, 
        layer_range, 
        query_indices, 
        renormalize=False
    ):
    """
    정해진 sink 좌표들을 0으로 만든 뒤, 다시 (h2,w2) 맵들을 계산하여 반환.
    Returns:
        per_img_maps_masked: 이미지별로 (마스킹 후) (h2,w2) 맵 리스트
    """
    # per_img_maps_masked = []
    # for img_idx, (st, end) in enumerate(spans):
    for i, crop in enumerate(crop_list):
        # t, h2, w2 = final_shapes[img_idx]
        t, h2, w2 = crop['patch_shape']
        # sink_coords = sink_coords_per_img[img_idx]
        sink_coords = crop['sink_coords']
        crop_span = crop['token_span']
        st, end = crop_span

        img_maps = []

        for li in layer_range:
            if li < 0 or li >= len(model_output.attentions):
                continue
            layer_att = model_output.attentions[li]  # (B,H,Q,K)
            for q_idx in query_indices:
                att_vec = (
                    layer_att[0, :, q_idx, st:end]
                    .mean(dim=0)
                    .to(torch.float32)
                    .detach()
                    .cpu()
                    .numpy()
                )  # (t*h2*w2,)
                att_vec_masked = zero_sink_in_att_vec(att_vec, t, h2, w2, sink_coords, renormalize=renormalize)
                att_map_masked = att_vec_masked.reshape(t, h2, w2).mean(axis=0)
                img_maps.append(att_map_masked)
        crop['masked_att_maps'] = img_maps
        # per_img_maps_masked.append(img_maps)
    return crop_list

def collect_maps_and_sink_counts(
        model_output,
        inputs, 
        # spans, 
        crop_list,
        # final_shapes, 
        layer_range, 
        # vision_end_id,
        query_indices,
        # grounding_query,
        processor, 
        per_map_topn_frac=0.01,
    ):
    """
    모든 이미지 span, 모든 레이어, 모든 (마지막 VE 이후) 텍스트 토큰 질의에 대해
    (h2, w2) 어텐션 맵을 수집하고, 각 맵에서 상위값 빈도수를 누적(counts)합니다.
    
    Returns:
        per_img_maps:  리스트 길이=num_imgs, 각 원소는 (맵 리스트). 각 맵 shape=(h2, w2)
        per_img_counts: 패치마다 어텐션 스코어 top n에 든 빈도수; 리스트 길이=num_imgs, 각 원소는 counts 배열 shape=(h2*w2,) 
    """
    input_ids_1d = inputs["input_ids"][0]
    seq_len = input_ids_1d.shape[0]
    vision_end_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
    last_vision_end = get_last_vision_end_index(input_ids_1d, processor)

    # 마지막 <|vision_end|> 이후의 모든 텍스트 토큰 인덱스 (없으면 마지막 토큰만 사용)
    # if last_vision_end < 0 or last_vision_end >= seq_len - 1:
    #     query_indices = [seq_len - 1]
    # else:
    #     # 너무 길면 메모리 폭발 방지: 필요하면 슬라이스/샘플링 가능
    #     query_indices = list(range(last_vision_end + 1, seq_len))

    per_img_maps   = []  # 각 이미지별로 [(h2,w2)맵, ...]
    per_img_sink_freqs = []  # 각 이미지별로 (h2*w2,) 카운트

    for i, crop in enumerate(crop_list):
        crop_span = crop['token_span']
        st, end = crop_span
        # crop에서 shape도 뽑기
        crop_patch_shape = crop['patch_shape']

        t, h2, w2 = crop_patch_shape
        hw = h2 * w2
        span_len = (end - st)
        assert span_len == t * hw, f"Span/token shape mismatch: span={span_len}, t*h*w={t*hw}"

        all_maps = []
        sink_freqs = np.zeros(hw, dtype=np.int64) # 이미지 당 이미지 패치 개수

        # per-map 상위 집계 기준 개수 (최소 5개는 집계)
        per_map_topn = max(5, int(per_map_topn_frac * hw))

        for li in layer_range:
            # 안전장치: 레이어 인덱스 범위 확인
            if li < 0 or li >= len(model_output.attentions):
                continue
            layer_att = model_output.attentions[li]  # (Batch, Head, Query, Key)

            for q_idx in query_indices:
            # for q_idx in grounding_query:
                # H 평균, batch=0, 질의 q_idx, key는 vision span s:e
                att_vec = (
                    layer_att[0, :, q_idx, st:end]
                    .mean(dim=0)
                    .to(torch.float32)
                    .detach()
                    .cpu()
                    .numpy()
                )  
                
                # shape: (t*h*w,)
                # 시간 평균하여 (h2,w2) 맵
                att_map = att_vec.reshape(t, h2, w2).mean(axis=0)  # (h2,w2)
                all_maps.append(att_map)

                # 상위값 빈도 누적
                flat = att_map.reshape(-1)
                idxs = top_indices(flat, per_map_topn)
                sink_freqs[idxs] += 1

        crop['att_maps'] = all_maps
        crop['sink_freqs'] = sink_freqs
        # per_img_maps.append(all_maps)
        # per_img_sink_freqs.append(sink_freqs)
    # return per_img_maps, per_img_sink_freqs
    return crop_list


def visualize_aggregated_attention(
        # attn_output, 
        crop_list,
        original_image, processor, save_path, gt_bbox, individual_maps_dir=None):
    """
    가장 높은 어텐션 포인트(Grounding Point)를 기준으로 성공 여부를 판단하고,
    시각화 시 해당 포인트와 예측 BBox, Ground Truth BBox를 모두 표시합니다.
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
    img_patch_shapes = [(t, h//2, w//2) for t, h, w in grid]


    for i, crop in enumerate(crop_list):
        

        crop_span = crop["token_span"]
        # get average attention
        # att_avg_low_res = get_mean_att_map(
        #     model_output=attn_output,
        #     span=crop_span,
        #     img_idx=i,
        #     layer_range=range(AGG_START, LAYER_NUM),
        #     grid=grid
        # )

        att_avg_low_res = crop["att_avg_masked"]

        
        


        # Calculate Entropy
        # entropy = calculate_entropy(att_avg_low_res)
        # weight = max(0.0, 1/entropy)


        # Upsample Attention Map
        crop_bbox = crop.get("bbox")
        crop_width = int(crop_bbox[2] - crop_bbox[0])
        crop_height = int(crop_bbox[3] - crop_bbox[1])
        if crop_width <= 0 or crop_height <= 0: continue

        att_avg_upscaled = upsample_att_map(att_avg_low_res, size=(crop_height, crop_width))



        if individual_maps_dir:
            individual_map = np.zeros((original_height, original_width), dtype=np.float32)
            left, top, right, bottom = map(int, crop_bbox)
            individual_map[top:bottom, left:right] = att_avg_upscaled

            plt.figure(figsize=(10, 10 * original_height / original_width))
            plt.imshow(original_image, extent=(0, original_width, original_height, 0))
            plt.imshow(individual_map, cmap='viridis', alpha=0.6, extent=(0, original_width, original_height, 0))
            plt.axis('off')
            # plt.title(f"Crop ID: {crop.get('id')}, Level: {crop.get('level')}, Weight: {weight:.2f}")
            plt.title(f"Crop ID: {crop.get('id')}, Level: {crop.get('level')}")
            individual_save_path = os.path.join(individual_maps_dir, f"individual_attn_crop_{crop.get('id')}.png")
            plt.savefig(individual_save_path, dpi=150, bbox_inches='tight', pad_inches=0)
            plt.close()

        if crop.get("level") == 1 or crop.get("level") == 0:
            left, top, right, bottom = map(int, crop_bbox)
            # aggregated_attention_map[top:bottom, left:right] += weight * att_avg_upscaled
            aggregated_attention_map[top:bottom, left:right] += att_avg_upscaled

    # --- Grounding Accuracy 측정 로직 ---
    is_grounding_success = False
    highest_point = None
    if aggregated_attention_map.max() > 0:
        max_yx = np.unravel_index(np.argmax(aggregated_attention_map, axis=None), aggregated_attention_map.shape)
        highest_point = (max_yx[1], max_yx[0])
        is_grounding_success = point_in_bbox(highest_point, gt_bbox)
        print(f"Highest attention point (Grounding Point): {highest_point}, GT bbox: {gt_bbox}, Success: {is_grounding_success}")
    else:
        print("Aggregated attention map is all zeros. Grounding failed.")

    # --- 시각화 ---
    fig, ax = plt.subplots(figsize=(10, 10 * original_height / original_width))
    ax.imshow(original_image, extent=(0, original_width, original_height, 0))
    ax.imshow(aggregated_attention_map, cmap='viridis', alpha=0.6, extent=(0, original_width, original_height, 0))

    if aggregated_attention_map.max() > 0:
        normalized_map = aggregated_attention_map / aggregated_attention_map.max()
    else:
        normalized_map = aggregated_attention_map

    if np.any(normalized_map > 0):
        threshold = np.percentile(normalized_map[normalized_map > 0], 99)
    else:
        threshold = 1.0
    
    # 어텐션이 높은 여러 영역(Predicted BBox)을 빨간 박스로 표시
    bounding_boxes = find_bounding_boxes_from_heatmap(normalized_map, threshold=threshold)
    print(f"Found {len(bounding_boxes)} high-attention regions.")

    highest_bbox = deepcopy(bounding_boxes[0])

    for bbox_info in bounding_boxes:
        coords, confidence = bbox_info
        left, top, right, bottom = coords
        rect = patches.Rectangle((left, top), right - left, bottom - top, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        score_text = f'{confidence:.2f}'
        ax.text(left, top - 5, score_text, color='white', fontsize=8, fontweight='bold',
                bbox=dict(facecolor='red', alpha=0.6, edgecolor='none', pad=1))
        
        if confidence > highest_bbox[1]:
            highest_bbox = bbox_info

    # Ground Truth BBox를 초록색 박스로 표시
    gt_left, gt_top, gt_right, gt_bottom = gt_bbox
    gt_width = gt_right - gt_left
    gt_height = gt_bottom - gt_top
    gt_rect = patches.Rectangle((gt_left, gt_top), gt_width, gt_height, linewidth=3, edgecolor='lime', facecolor='none')
    ax.add_patch(gt_rect)

    bbox_left, bbox_top, bbox_right, bbox_bottom = highest_bbox[0]
    highest_point = [(bbox_left + bbox_right) / 2, (bbox_top + bbox_bottom) / 2]
    
    # --- 수정된 코드 시작 ---
    # 가장 높은 어텐션 포인트(Grounding Point)를 노란 별표로 표시
    if highest_point is not None:
        ax.plot(highest_point[0], highest_point[1], 'y*', markersize=15, markeredgecolor='black')

    # 범례(Legend) 수정
    red_patch = patches.Patch(color='red', label='Predicted BBox (High Attention Area)')
    green_patch = patches.Patch(color='lime', label='Ground Truth BBox')
    handles = [red_patch, green_patch]
    if highest_point is not None:
        yellow_star = Line2D([0], [0], marker='*', color='w', label='Grounding Point (Max Attention)',
                              markerfacecolor='yellow', markeredgecolor='black', markersize=15)
        handles.append(yellow_star)
    ax.legend(handles=handles, loc='best')
    # --- 수정된 코드 끝 ---

    ax.axis('off')
    ax.set_title("Final Result: Attention Map, Grounding, and Ground Truth")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"Final result visualization saved at: {save_path}")
    if individual_maps_dir:
        print(f"Individual attention maps saved in: {individual_maps_dir}")

    return is_grounding_success


def find_bounding_boxes_from_heatmap(attention_map, threshold):
    """
    주어진 어텐션 맵(히트맵)에서 임계값(threshold)을 초과하는
    연결된 영역들을 찾고, 각 영역의 바운딩 박스를 반환합니다.
    """
    if not isinstance(attention_map, np.ndarray):
        attention_map = np.array(attention_map)

    height, width = attention_map.shape
    visited = np.zeros_like(attention_map, dtype=bool)
    bounding_boxes = []

    def bfs(start_i, start_j):
        queue = deque([(start_i, start_j)])
        region_pixels = []

        if visited[start_i][start_j] or attention_map[start_i][start_j] <= threshold:
            return None

        visited[start_i][start_j] = True
        queue = deque([(start_i, start_j)])
        region_pixels.append((start_i, start_j))

        while queue:
            i, j = queue.popleft()
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0: continue
                    ni, nj = i + di, j + dj
                    if (0 <= ni < height and 0 <= nj < width and
                        not visited[ni][nj] and attention_map[ni][nj] > threshold):
                        visited[ni][nj] = True
                        queue.append((ni, nj))
                        region_pixels.append((ni, nj))
        return region_pixels

    for i in range(height):
        for j in range(width):
            if attention_map[i][j] > threshold and not visited[i][j]:
                region = bfs(i, j)
                if region:
                    min_i = min(p[0] for p in region)
                    max_i = max(p[0] for p in region)
                    min_j = min(p[1] for p in region)
                    max_j = max(p[1] for p in region)
                    bbox_confidence = np.mean(attention_map[min_i:max_i+1, min_j:max_j+1])
                    bounding_boxes.append([[min_j, min_i, max_j, max_i], bbox_confidence])
    return bounding_boxes


#! ================================================================================================

if __name__ == '__main__':
    
    # SEED
    set_seed(SEED)

    # Argument Parsing
    args = parse_args()
    seg_save_base_dir = f"{args.save_dir}/seg"

    # Model Import
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.mllm_path, torch_dtype="auto", attn_implementation="eager",
        device_map="balanced", max_memory=max_memory, low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.mllm_path)
    processor = AutoProcessor.from_pretrained(args.mllm_path)
    model.eval()

    # Question
    # question_template="""Where should you tap to {task_prompt}?"""
    question_template="""
You are an assistant trained to navigate the android phone. Given a
task instruction, a screen observation, guess where should you tap.
# Intruction
{task_prompt}"""

    # Process
    for task in TASKS:
        task_res = dict()
        dataset = "screenspot_" + task + "_v2.json"
        screenspot_data = json.load(open(os.path.join(args.screenspot_test, dataset), 'r'))
        screenspot_data = screenspot_data[SAMPLE_RANGE]

        print("Num of sample: " + str(len(screenspot_data)), flush=True)

        task_res = list()
        num_action = 0
        corr_action = 0
        res_board_dict = dict()

        for stage_num in range(1, 2+1):
          if stage_num == 1:
            res_board_dict[stage_num] = {
              "is_coarse": True, "gt_score_list": [], "top_q_crop_ids": [],
              "is_gt_in_top_q": False, "raw_attn_dict": None,
            }
          elif stage_num == 2:
            res_board_dict[stage_num] = {
              "is_coarse": True, "gt_score_list": [], "sub_res": [],
            }

        num_wrong_format = 0
        num_segmentation_failed = 0

        for j, item in tqdm(enumerate(screenspot_data)):
            item_res = dict()
            num_action += 1
            filename = item["img_filename"]
            filename_wo_ext, ext = os.path.splitext(filename)
            img_path = os.path.join(args.screenspot_imgs, filename)
            if not os.path.exists(img_path):
                continue

            original_image = Image.open(img_path).convert("RGB")
            instruction = item["instruction"]
            original_bbox = item["bbox"]
            original_bbox = [original_bbox[0], original_bbox[1], original_bbox[0] + original_bbox[2], original_bbox[1] + original_bbox[3]]
            question = question_template.format(task_prompt=instruction)

            inst_dir_name = re.sub(r'\W+', '_', instruction).strip('_')
            s1_dir = f"{seg_save_base_dir}/{filename_wo_ext}/{inst_dir_name}/s1"
            s1_processed_dir = f"{seg_save_base_dir}/{filename_wo_ext}/{inst_dir_name}/s1_processed"
            os.makedirs(s1_dir, exist_ok=True)
            os.makedirs(s1_processed_dir, exist_ok=True)

            dir = f"{seg_save_base_dir}/{filename_wo_ext}/{inst_dir_name}"
            log_file_path = os.path.join(dir, "time_log.txt")

            # Stage 1: Segmentation
            stage1_start = time.time()
            crop_list = run_segmentation_recursive(
                image_path=img_path, max_depth=1, window_size=120,
                output_json_path=f"{s1_dir}/output.json",
                output_image_path=s1_dir,
                start_id=0,
                var_thresh=VAR_THRESH
            )
            stage1_end = time.time()
            measure_and_log_time("Stage 1 Segmentation", stage1_start, stage1_end)

            if crop_list is None:
                print(f"Segmentation failed for {img_path}. Skipping this image.")
                num_segmentation_failed += 1
                continue

            # Stage 1 Processed
            stage1_processed_start = time.time()
            only_lv_1 = [crop for crop in crop_list if crop.get("level") == 1]
            if len(only_lv_1) == 0:
                print(f"No level 1 crops found for {img_path}. Skipping this image.")
                num_segmentation_failed += 1
                continue

            stage_crop_list = create_stage_crop_list(
                crop_list=crop_list,
                resize_dict={0: THUMBNAIL_RESIZE_RATIO, 1: S1_RESIZE_RATIO},
                use_thumbnail=True
            )
            stage1_processed_end = time.time()
            measure_and_log_time("Stage 1 Processed", stage1_processed_start, stage1_processed_end)

            # Stage 2: Attention Refinement
            stage2_start = time.time()
            msgs = create_msgs(crop_list=stage_crop_list, question=question)
            attn_vis_dir = os.path.join(s1_dir, "attn_map")

            print()

            for _stage_crop in stage_crop_list:
                _stage_crop['resized_img'].save(f"{seg_save_base_dir}/{filename_wo_ext}/{inst_dir_name}/s1_processed/{_stage_crop['id']}.png")
            s1_top_q_crop_ids, s1_top_q_bboxes, s1_raw_res_dict, s1_crop_list_out = run_selection_pass(
                msgs=msgs, crop_list=stage_crop_list, top_q=TOP_Q, drop_indices=[0], attn_vis_dir=attn_vis_dir
            )
            s1_is_gt_in_top_q = check_gt_in_top_q_crops(top_q_bboxes=s1_top_q_bboxes, gt_bbox=original_bbox)

            gt_vis_dir = os.path.join(s1_dir, "gt_vis")
            visualize_result(
                save_dir=gt_vis_dir, gt_bbox=original_bbox, top_q_bboxes=s1_top_q_bboxes,
                instruction=instruction, click_point=None
            )

            stage2_end = time.time()
            measure_and_log_time("Stage 2 Attention Refinement", stage2_start, stage2_end)

            # Total Time
            total_time = (stage1_end - stage1_start) + (stage1_processed_end - stage1_processed_start) + (stage2_end - stage2_start)
            with open(log_file_path, "a") as log_file:
                log_file.write(f"Total Time: {total_time:.2f} seconds\n")
            print(f"🕖 Total Time: {total_time:.2f} seconds")

            # [Stage 1]
            attn_vis_dir = os.path.join(seg_save_base_dir, filename_wo_ext, inst_dir_name , "s1/attn_map")
            s1_top_q_crop_ids, s1_top_q_bboxes, s1_raw_res_dict, s1_crop_list_out = run_selection_pass(
                msgs=msgs, crop_list=stage_crop_list, top_q=TOP_Q, drop_indices=[0], attn_vis_dir=attn_vis_dir
            )
            s1_is_gt_in_top_q = check_gt_in_top_q_crops(top_q_bboxes=s1_top_q_bboxes, gt_bbox=original_bbox)

            gt_vis_dir = os.path.join(seg_save_base_dir, filename_wo_ext, inst_dir_name , f"s1")
            visualize_result(
                save_dir=gt_vis_dir, gt_bbox=original_bbox, top_q_bboxes=s1_top_q_bboxes,
                instruction=instruction, click_point=None
            )

            res_board_dict[1]["gt_score_list"].append(1 if s1_is_gt_in_top_q else 0)
            print(f"s1 gt_contained: {s1_is_gt_in_top_q}")

            res_board_dict[1]["top_q_crop_ids"] = s1_top_q_crop_ids
            res_board_dict[1]["raw_attn_dict"] = s1_raw_res_dict
            res_board_dict[1]["is_gt_in_top_q"] = bool(s1_is_gt_in_top_q)

            for c in s1_crop_list_out:
                if "img" in c: del c["img"]
                if "resized_img" in c: del c["resized_img"]
                if "token_span" in c: del c["token_span"]
            res_board_dict[1]["crop_list"] = s1_crop_list_out

            print(f"Selected Crops from Stage 1: {s1_top_q_crop_ids}")


            # [Stage 2] Attention Refinement Pass-------------------------------------
            print("\n[Stage 2] Starting Attention Refinement Pass...")

            original_crop_map = {c['id']: c for c in crop_list}
            s2_input_crop_ids = set()
            if 0 in original_crop_map:
                s2_input_crop_ids.add(0)
            for crop_id in s1_top_q_crop_ids:
                s2_input_crop_ids.add(crop_id)
            s2_input_crops = [original_crop_map[cid] for cid in s2_input_crop_ids if cid in original_crop_map]

            if len(s2_input_crops) <= 1:
                print("No crops from Stage 1 to refine. Skipping Stage 2.")
                final_success = False # Stage 2를 건너뛰면 실패로 간주
                res_board_dict[2]["is_gt_in_top_q"] = bool(final_success)
                res_board_dict[2]["gt_score_list"].append(1 if final_success else 0)
            else:
                print(f"Stage 2 will use crops: {[c['id'] for c in s2_input_crops]}")
                s2_save_dir = os.path.join(seg_save_base_dir, filename_wo_ext, inst_dir_name, "s2_refined")
                
                # run_refinement_pass 호출 시 gt_bbox를 넘겨주고, 성공 여부를 final_success에 저장
                final_success = run_refinement_pass(
                    crop_list=s2_input_crops,
                    question=question,
                    original_image=original_image,
                    save_dir=s2_save_dir,
                    gt_bbox=original_bbox # gt_bbox 전달
                )

                # res_board_dict[2]["is_gt_in_top_q"] 변수명을 is_grounding_success 등으로 바꿔도 좋지만,
                # 일단 기존 변수명을 재활용하여 점수 리스트에 추가합니다.
                res_board_dict[2]["is_gt_in_top_q"] = bool(final_success)
                res_board_dict[2]["gt_score_list"].append(1 if final_success else 0)

            _gt_score_list = res_board_dict[2]["gt_score_list"]
            up2now_gt_score = calc_acc(_gt_score_list)
            print(f"Up2Now S2 Grounding Accuracy:{up2now_gt_score}") # 출력 메시지 수정
            print("------")
            print()

            item_res['filename'] = filename
            item_res['data_type'] = item["data_type"]
            item_res['data_source'] = item["data_source"]
            item_res['instruction'] = instruction
            item_res['stage1_res'] = res_board_dict[1]
            item_res['stage2_res'] = res_board_dict[2]
            item_res['gt_bbox'] = original_bbox
            item_res['num_crop'] = len(stage_crop_list)
            task_res.append(item_res)


        # 마지막 결과 모음 정리
        # print(task_res[0])
        with open(os.path.join(args.save_dir, dataset), "w") as f:
            json.dump(task_res, f, indent=4, ensure_ascii=False, cls=NpEncoder)

        print("==================================================")
        print(task + ": Total num: " + str(num_action))
        print(task + ": Wrong format num: " + str(num_wrong_format))
        print(task + f": Ground truth included Acc: " + str(calc_acc(_gt_score_list)))

        metrics = {
            "task": task,
            "total_num": num_action,
            "segmentation_failed": num_segmentation_failed,
            "num_segmentation_failed": num_segmentation_failed,
            "acc": calc_acc(_gt_score_list)
        }

        with open(os.path.join(args.save_dir, f"{task}_metrics.json"), "w") as mf:
            json.dump(metrics, mf, ensure_ascii=False, indent=4)

    # Save all time logs at the end
    save_time_logs(log_file_path)