import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict
from copy import deepcopy
import os
import re
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
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


#! ============================================================================================================================
# Centralized user settings

SEED: int = 0

# GPU visibility (e.g., "0", "0,1", "2,3", "1,2,3"...)
# nvidia-smi 하고 사용할 GPU 넣기
CUDA_VISIBLE: str = "2,3"

# HF device map / memory hints
attn_implementation: str = "eager"
device_map: str = "balanced"
max_memory = {
    0: "75GiB",
    1: "75GiB",
    "cpu": "120GiB",
}

# Model/paths
mllm_path: str = "zonghanHZH/ZonUI-3B"
screenspot_imgs: str = "./data/screenspotv2_imgs"
screenspot_test: str = "./data"
save_dir: str = "./attn_output/q50_r20_thumb_s2_agg20_try2/eval_res"

TEST_SAMPLE_LIMIT: int = 12  # test data 개수 (None이면 제한 없이 전체 데이터를 사용)

# Flags
sampling: bool = False
vis_flag: bool = False

# Attention/cropping hyperparameters
layer_num_list = [3,7,11,15,31]
layer_num = layer_num_list[-1]
top_q: float = 0.5           # Stage 1 selection ratio
stage2_top_q: float = 0.3    # Stage 2 selection ratio


#! 현재 best : resize 5% layer 12
s1_resize_ratio: float = 0.20
s2_resize_ratio: float = 0.35
thumnail_resize_ratio: float = 0.05

# Tasks to run
tasks = ["mobile"]

#! ============================================================================================================================

set_seed(SEED)

torch.cuda.empty_cache()


# Check CUDA availability and print status
if torch.cuda.is_available():
    print(f"CUDA is available. Using device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("CUDA is not available. Running on CPU.")


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= CUDA_VISIBLE  # Set the GPUs to use

seg_save_base_dir = f"{save_dir}/seg"

# Qwen 2.5
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    mllm_path,
    torch_dtype="auto",
    attn_implementation=attn_implementation,
    # device_map="cuda:0",
    device_map=device_map,
    max_memory=max_memory,
    low_cpu_mem_usage=True
)

tokenizer = AutoTokenizer.from_pretrained(mllm_path)
processor = AutoProcessor.from_pretrained(mllm_path)
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
                           use_thumbnail: bool = True):
    stage_crop_list = []

    for crop in crop_list:
        crop_level = crop.get("level")
        if crop_level not in resize_dict.keys(): continue
        if not use_thumbnail and crop_level == 0: continue

        new_crop = deepcopy(crop)
        crop_img = deepcopy(crop["img"])

        # Resize image to 50% size

        ratio = resize_dict.get(crop_level)
        if ratio is None: raise Exception(f"No resize dict entry for crop level {crop_level}")

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

def get_top_q_crop_ids(top_q, attn_df):  #! 어텐션 기반 상위 crop 선별 유틸

    df = deepcopy(attn_df)
    # layer num까지의 어텐션 합 구하기
    cols_to_sum = [f"layer{i}" for i in range(20, layer_num + 1)] #? 20? 1? attn_map 20으로 한결과

    # 중요도 스코어 열 생성
    df[f"sum_until_{layer_num}"] = df[cols_to_sum].sum(axis=1)

    # 상위 top_q 비율을 quantile로 커트
    keep_frac = top_q  # 예: 0.2 → 상위 20%
    thr = df[f"sum_until_{layer_num}"].quantile(1 - keep_frac)
    top_q_crop_ids = df.index[df[f"sum_until_{layer_num}"] >= thr].tolist()

    # guardrail: 최소/최대 개수 보정(선택)
    if len(top_q_crop_ids) < 1:
        raise Exception("no crop selected")
    
    # TODO: layer 범위는 하이퍼파라미터로 승격 (cli args)
    # TODO: 합이 아니라 평균/가중합(예: 레이어별 가중)도 옵션화
    # TODO: 정규화(각 crop별 토큰 수/해상도 차이 보정) 옵션 추가
    
    return top_q_crop_ids

def visualize_attn_map(attn_output, msgs, crop_list, attn_vis_dir):  #! 어텐션 맵 시각화

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

    # TODO: 레이어 범위/집계 방식(합/평균/가중)을 argparse로 노출
    # TODO: 'relative attention'(일반 설명 프롬프트로 정규화) 실험 옵션 추가
    # TODO: gradient-weighting(pure-grad/grad-att) 확장 포인트 추가
    
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


    # Step 1. thre 넘은 crop(top_q crop) 가져오기
    top_q_crop_ids = get_top_q_crop_ids(top_q=top_q, attn_df=df)
        
    # 전역 의존 최소화: 메시지만 출력
    print(f"selected crops: {top_q_crop_ids}")

    # Get top-q bboxes
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


# TODO: 어텐션 맵 뽑는 거 만들기
# gpu setting


#! Question Template
question_template="""
You are an assistant trained to navigate the android phone. Given a
task instruction, a screen observation, guess where should you tap.
# Intruction
{task_prompt}"""


for task in tasks:
    task_res = dict()
    dataset = "screenspot_" + task + "_v2.json"
    screenspot_data = json.load(open(os.path.join(screenspot_test, dataset), 'r'))
    # vanilla_qwen_data = json.load(open(os.path.join(args.screenspot_test, "vanilla_qwen_screenspot_mobile.json"), 'r'))
    # 테스트 데이터 개수 제한 적용 (None이면 전체 데이터 사용)
    if TEST_SAMPLE_LIMIT is not None:
        screenspot_data = screenspot_data[:TEST_SAMPLE_LIMIT]

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
        img_path = os.path.join(screenspot_imgs, filename)
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
        item_res['original_bbox'] = original_bbox.copy()
        item_res['bbox'] = original_bbox.copy()

        question = question_template.format(task_prompt=instruction)        
        

        # crop 추가
        inst_dir_name = re.sub(r'\W+', '_', instruction).strip('_')
        os.makedirs(f"{seg_save_base_dir}/{filename_wo_ext}/{inst_dir_name}/s1", exist_ok=True)
        os.makedirs(f"{seg_save_base_dir}/{filename_wo_ext}/{inst_dir_name}/s1_processed", exist_ok=True)

        
        crop_list = run_segmentation(
            image_path=img_path,
            max_depth=1,
            window_size=120,
            output_json_path=f"{seg_save_base_dir}/{filename_wo_ext}/{inst_dir_name}/s1/output.json",
            output_image_path=f"{seg_save_base_dir}/{filename_wo_ext}/{inst_dir_name}/s1"
        )
        # print(crop_list)


        # 무조건 전체샷은 id=0
        for crop in crop_list:
            if crop.get("level") == 0:
                crop["id"] = 0

        # segmentation fault 검사
        only_lv_1 = [crop for crop in crop_list if crop.get("level") == 1]
        if len(only_lv_1) == 0:
            print("segmenation failed")
            num_segmentation_failed += 1
            continue
        
        
        # stage에 맞는 crop만 뽑아내기

        stage_crop_list = create_stage_crop_list(crop_list=crop_list,
            resize_dict={0: thumnail_resize_ratio, 1: s1_resize_ratio}, 
            use_thumbnail=True)
        msgs = create_msgs(crop_list=stage_crop_list, question=question)

        for _stage_crop in stage_crop_list:
            _stage_crop['resized_img'].save(f"{seg_save_base_dir}/{filename_wo_ext}/{inst_dir_name}/s1_processed/{_stage_crop['id']}.png")

        #! ---------------------------------------------
        #! Stage 1
        #! - 입력: 썸네일+레벨1 crop 묶음(msgs, crop_list)
        #! - 출력: 어텐션 top-q crop들의 bbox, 선택결과 시각화, GT 포함 여부 기록
        #! - 주의: 썸네일(id=0)은 드롭(drop_indices=[0])해서 분산 낮추기
        #! ---------------------------------------------
        
        #! 1. Run stage
        # top_q_crop_ids, top_q_bboxes, raw_res_dict, stage1_crop_list = run_stage(msgs=msgs, crop_list=stage_crop_list, top_q=top_q)
        # is_gt_in_top_q = check_gt_in_top_q_crops(top_q_bboxes=top_q_bboxes, gt_bbox=original_bbox)
        attn_vis_dir = os.path.join(seg_save_base_dir, filename_wo_ext, inst_dir_name , "s1/attn_map")
        s1_crop_list = stage_crop_list

        s1_top_q_crop_ids, s1_top_q_bboxes, s1_raw_res_dict, s1_crop_list = run_stage(
            msgs=msgs,
            crop_list=stage_crop_list,
            top_q=top_q,
            drop_indices=[0],        # 썸네일 제외
            attn_vis_dir=attn_vis_dir
        )
        s1_is_gt_in_top_q = check_gt_in_top_q_crops(top_q_bboxes=s1_top_q_bboxes, gt_bbox=original_bbox)


        
        #! 2. Visualize Result
        gt_vis_dir = os.path.join(seg_save_base_dir, filename_wo_ext, inst_dir_name , f"s1")
        visualize_result(
            save_dir=gt_vis_dir,
            gt_bbox=original_bbox,
            top_q_bboxes=s1_top_q_bboxes,
            instruction=instruction,
            click_point=None
        )

        # Update Score
        res_board_dict[1]["gt_score_list"].append(1 if s1_is_gt_in_top_q else 0)
        print(f"s1 gt_contained: {s1_is_gt_in_top_q}")

        res_board_dict[1]["top_q_crop_ids"] = s1_top_q_crop_ids
        res_board_dict[1]["raw_attn_dict"] = s1_raw_res_dict
        res_board_dict[1]["is_gt_in_top_q"] = s1_is_gt_in_top_q

        # 큰 이미지 / 토큰 span 제거 (메모리 절약)
        for c in s1_crop_list:
              del c["img"]
              del c["resized_img"]
              del c["token_span"]
        res_board_dict[1]["crop_list"] = s1_crop_list

        print(f"Selected Crops from Stage 1: {s1_top_q_crop_ids}")




        #! ---------------------------------------------
        #! [Stage 2] stage 1에서 뽑은 각 crop에 대해서 stage_run
        #! - 입력: Stage 1에서 뽑은 각 top crop
        #! - 절차:
        #!   1) 해당 crop 이미지를 기준으로 재세그멘테이션(세부 레벨)
        #!   2) 새로 생긴 세부 crop들의 bbox를 '원본 이미지 좌표계'로 복원
        #!   3) 썸네일 + 선택된 상위 crop + 세부 crop 묶어서 다시 전방패스
        #! - 실패 케이스: 세부 crop이 2개 미만이면 Stage1 crop만으로 GT 포함 여부 체크
        #! ---------------------------------------------

        # [Stage 2] stage 1에서 뽑은 각 crop에 대해서 stage_run

        # for crop in s1_top_q_crop_ids
        # crop_list에서 cropid로 crop 찾아서 변수에 img 저장
        # create new messages, crop imgs


        s2_is_gt_in_top_q_list = []
        start_id = len(s1_crop_list) # id 충돌 방지용

        for crop_id in s1_top_q_crop_ids:
            print(f"[Stage 2 - crop{crop_id} from stage 1]")
            s1_selected_crop = next((c for c in crop_list if c.get('id') == crop_id), None)
            # if crop is None:
            if s1_selected_crop is None:
                print(f"crop_id {crop_id} not found; skip")
                continue  
            
            thumbnail_crop = next((c for c in crop_list if c.get('level') == 0), None)

            s2_crop_list = [thumbnail_crop, s1_selected_crop]
            
            # 선택 crop를 파일로 저장해 stage2 segmentation 입력으로 활용
            os.makedirs(f"{seg_save_base_dir}/{filename_wo_ext}/{inst_dir_name}/s2/crop{crop_id}", exist_ok=True)
            os.makedirs(f"{seg_save_base_dir}/{filename_wo_ext}/{inst_dir_name}/s2_processed", exist_ok=True)
            s1_selected_crop["img"].save(
                f"{seg_save_base_dir}/{filename_wo_ext}/{inst_dir_name}/s2/crop{crop_id}/selected_crop_from_s1.png"
            )

            # 원점 기준 재세그멘테이션 (세부 crop 뽑기)
            s2_detail_crops = run_segmentation(
                image_path=f"{seg_save_base_dir}/{filename_wo_ext}/{inst_dir_name}/s2/crop{crop_id}/selected_crop_from_s1.png",
                max_depth=1,
                window_size=75,
                output_json_path=f"{seg_save_base_dir}/{filename_wo_ext}/{inst_dir_name}/s2/output.json",
                output_image_path=f"{seg_save_base_dir}/{filename_wo_ext}/{inst_dir_name}/s2/crop{crop_id}",
                start_id=start_id
            )

            #? 실패 처리: 세부 crop이 너무 적으면 Stage1 선택결과로 대체
            if len(s2_detail_crops) < 2 :  
                _is_gt_in_curr_crop = check_gt_in_top_q_crops(top_q_bboxes=[s1_selected_crop['bbox']], gt_bbox=original_bbox)
                s2_is_gt_in_top_q_list.append(_is_gt_in_curr_crop)
                print(f"cropping failed at stage 2 crop {crop_id}")
                continue
                # raise Exception("Cropping fault")


            start_id += len(s2_detail_crops) #id 안겹치게
            
            # 레벨 정리 및 '원점 복원'
            s2_detail_crops = [c for c in s2_detail_crops if c.get("level") == 1]
            s1_left, s1_top, _, _ = s1_selected_crop['bbox']
            for detail_crop in s2_detail_crops:
                if detail_crop.get("level") == 1: # 레벨1을 레벨 2로 강등
                    detail_crop["level"] = 2
                # 원점 복원: stage1 crop의 (left, top)을 더해 전체 좌표계로 환원
                detail_crop["bbox"][0] += s1_left # left에 s1_left(원점 x) 더하기
                detail_crop["bbox"][2] += s1_left # right에 s1_left 더하기
                detail_crop["bbox"][1] += s1_top # top에 s1_top(원점 y) 더하기
                detail_crop["bbox"][3] += s1_top # bottom에 s1_top(원점 y) 더하기
                s2_crop_list.append(detail_crop)  #forward 돌릴 crop list에 추가
        
            
            # Stage2 crop 묶음 만들기 + 메시지 생성
            s2_crop_list = create_stage_crop_list(crop_list=s2_crop_list,
                resize_dict={0: thumnail_resize_ratio, 1: s1_resize_ratio, 2: s2_resize_ratio}, 
                # resize_dict={0: thumnail_resize_ratio, 1: 0.4, 2: 0.2}, 
                use_thumbnail=True)

            msgs = create_msgs(crop_list=s2_crop_list, question=question)

            #* 시각화용 저장
            os.makedirs(f"{seg_save_base_dir}/{filename_wo_ext}/{inst_dir_name}/s2_processed/crop{crop_id}", exist_ok=True)
            for _stage_crop in s2_crop_list:
                _stage_crop['resized_img'].save(f"{seg_save_base_dir}/{filename_wo_ext}/{inst_dir_name}/s2_processed/crop{crop_id}/{_stage_crop['id']}.png")
            

            # TODO: run_stage(...) 다시 호출하여 Stage2 평가 + GT 포함 여부 업데이트
            #       시각화/로그/보드 업데이트는 Stage1과 동일 패턴으로 처리


            # 1. Run stage 2
            s2_attn_vis_dir = os.path.join(seg_save_base_dir, filename_wo_ext, inst_dir_name , f"s2/crop{crop_id}/attn_map")
            s2_top_q_crop_ids, s2_top_q_bboxes, s2_raw_res_dict, s2_crop_list = run_stage(
                msgs=msgs, crop_list=s2_crop_list, top_q=stage2_top_q, drop_indices=[0, crop_id], 
                attn_vis_dir=s2_attn_vis_dir)
            s2_is_gt_in_top_q = check_gt_in_top_q_crops(top_q_bboxes=s2_top_q_bboxes, gt_bbox=original_bbox)
            s2_is_gt_in_top_q_list.append(s2_is_gt_in_top_q)



            # 2. Visualize Result
            viz_dir = os.path.join(seg_save_base_dir, filename_wo_ext, inst_dir_name , f"s2", f"crop{crop_id}")

            visualize_result(
                save_dir=viz_dir,
                gt_bbox=original_bbox,
                top_q_bboxes=s2_top_q_bboxes,
                instruction=instruction,
                click_point=None
            )

            # Update Result

            # s2_copy = deepcopy(s2_crop_list)
            # for c in s2_copy:
                
            #     c.pop("img")
            #     c.pop("resized_img")

            for c in s2_crop_list:
              del c["img"]
              del c["resized_img"]
              del c["token_span"]

            sub_res_dict = {
                "parent_crop_id": crop_id,
                "is_gt_in_top_q": s2_is_gt_in_top_q,
                "top_q_crop_ids": s2_top_q_crop_ids,
                "raw_attn_dict": s2_raw_res_dict,
                "crop_list": s2_crop_list # PIL되는지 체크
            }

            res_board_dict[2]["sub_res"].append(sub_res_dict)


        
        # 2. Update Score
        print(s2_is_gt_in_top_q_list)
        s2_is_gt_in_top_q_tot = any(s2_is_gt_in_top_q_list)  
        res_board_dict[2]["gt_score_list"].append(1 if s2_is_gt_in_top_q_tot else 0)
        res_board_dict[2]["is_gt_in_top_q"] = s2_is_gt_in_top_q_tot

        print(f"s2 gt_contained: {s2_is_gt_in_top_q_tot}")

        
      

        #! =============================================================================

        _gt_score_list = res_board_dict[2]["gt_score_list"]
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



    #! ---------------------------------------------
    #! 작업 결과 저장 + 메트릭 요약
    #! - task_res: 각 샘플 결과 집합 (파일명/지시문/Stage1,2 요약/GT bbox 등)
    #! - metrics: 전체 카운트 및 정확도 요약 (분석 스크립트가 읽기 쉬운 형태)
    #! ---------------------------------------------
    with open(os.path.join(save_dir, dataset), "w") as f:
        json.dump(task_res, f, indent=4, ensure_ascii=False)

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

    with open(os.path.join(save_dir, f"{task}_metrics.json"), "w") as mf:
        json.dump(metrics, mf, ensure_ascii=False, indent=4)
