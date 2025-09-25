import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('gpu', type=int, default=0, help='GPU number')
parser.add_argument('--r', type=float, default=1.00, help='Resize ratio')
parser.add_argument('--max_pixels', type=int, default=3211264, help='Max pixels for the image encoder')
parser.add_argument('--2b', action='store_true', help='Use 2B model if set, otherwise use 3B model')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import torch

import json
import argparse
import time
import csv
import datetime
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoProcessor, set_seed
from PIL import Image
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gui_actor.constants import chat_template
from gui_actor.modeling import Qwen2VLForConditionalGenerationWithPointer
from gui_actor.modeling_qwen25vl import Qwen2_5_VLForConditionalGenerationWithPointer
from gui_actor.inference import inference, ForceFollowTokensLogitsProcessor
from gui_actor.utils import do_boxes_overlap
from gui_actor.constants import DEFAULT_POINTER_PAD_TOKEN, DEFAULT_POINTER_END_TOKEN
from copy import deepcopy

from deepspeed.profiling.flops_profiler import FlopsProfiler


IMAGE_PATCH_SIZE =14

RESIZE_RATIO=args.r

model_type = "qwen25vl" if not args.__dict__['2b'] else "qwen2vl"
model_name = "gui-actor-3B" if not args.__dict__['2b'] else "gui-actor-2B"
MAX_PIXELS=args.max_pixels
set_seed(0)
SAVE_DIR = "../attn_output/gui-actor-2B" if model_type == "qwen2vl" else "../attn_output/gui-actor-3B"

PRED_PATH = "{save_dir}/{task}_{resize_ratio}_preds.json"
METRIC_PATH = "{save_dir}/{task}_{resize_ratio}_metrics.txt"

TASKS = ["mobile", "web", "desktop"]

#!=================================================================================

def calc_acc(score_list):
    return sum(score_list) / len(score_list) if len(score_list) != 0 else 0

def normalize_bbox(bbox_x1y1x2y2, img_width, img_height):
    # if bbox_x1y1x2y2 is not normalized to [0, 1], normalize it
    x1, y1, x2, y2 = bbox_x1y1x2y2
    if (0 <= x1 <= 1) and (0 <= y1 <= 1) and (0 <= x2 <= 1) and (0 <= y2 <= 1):
        return bbox_x1y1x2y2
    else:
        x1 = x1 / img_width
        y1 = y1 / img_height
        x2 = x2 / img_width
        y2 = y2 / img_height
        return x1, y1, x2, y2

def evaluate(model_name_or_path, use_placeholder, topk, task, prof):
    # initialize model
    data_processor = AutoProcessor.from_pretrained(model_name_or_path, max_pixels=MAX_PIXELS)
    tokenizer = data_processor.tokenizer
    for k, v in tokenizer.added_tokens_encoder.items():
        print(v, k)

    logits_processor_pointer = ForceFollowTokensLogitsProcessor(
        token_a_id=tokenizer.encode(DEFAULT_POINTER_PAD_TOKEN)[0],
        forced_sequence=[
            tokenizer.encode(DEFAULT_POINTER_END_TOKEN)[0]
        ]
    )

    dataset = json.load(open(os.path.join("../data", f"screenspot_{task}_v2.json"), 'r'))

    results = []
    score_list = []

    inference_time_sum = 0.0
    iter = 0
    correct_count = 0
    total_flops = 0.0

    for j, example in tqdm(enumerate(dataset)):
        iter += 1
        
        filename = example["img_filename"]
        img_path = os.path.join("../data/screenspotv2_image", filename)
        if not os.path.exists(img_path):
            print("img not found", flush=True)
            input()

        original_image = Image.open(img_path).convert("RGB")
        w_orig, h_orig = original_image.size

        # 리사이즈 처리
        if RESIZE_RATIO < 1.00:
            print(f"Resized image from ({w_orig}, {h_orig})", end=" -> ")
            w_resized, h_resized = int(w_orig * RESIZE_RATIO), int(h_orig * RESIZE_RATIO)
            resized_image = original_image.resize((w_resized, h_resized))
            print(f"({w_resized}, {h_resized}) (ratio: {RESIZE_RATIO})")
            img = deepcopy(resized_image)
        else:
            img = deepcopy(original_image)
            w_resized, h_resized = w_orig, h_orig
        
        # GT bbox를 리사이즈 비율에 맞춰 조정
        gt_bbox = example["bbox"]
        gt_bbox = [gt_bbox[0], gt_bbox[1], gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]]
        
        # GT bbox를 리사이즈된 좌표계로 변환
        if RESIZE_RATIO < 1.00:
            gt_bbox_resized = [coord * RESIZE_RATIO for coord in gt_bbox]
        else:
            gt_bbox_resized = gt_bbox.copy()

        ele = {
            "file_name": example["img_filename"],
            "data_type": example["data_type"],
            "instruction": example["instruction"],
            "img_size": [w_resized, h_resized],  # 리사이즈된 이미지 크기 사용
            "bbox_x1y1x2y2": example["bbox"],
            "hit_top1": 0,
            "overlap_top1": 0,
            "hit_topk": 0,
            "overlap_topk": 0,
        }

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
                        "image": img, # PIL.Image.Image or str to path
                        # "image_url": "https://xxxxx.png" or "https://xxxxx.jpg" or "file://xxxxx.png" or "data:image/png;base64,xxxxxxxx", will be split by "base64,"
                    },
                    {
                        "type": "text",
                        "text": example["instruction"]
                    },
                ],
            },
        ]

        #! =======================================================
        prof.reset_profile()
        #! =======================================================

        inference_start = time.time()
        pred = inference(conversation, model, tokenizer, data_processor, logits_processor=logits_processor_pointer, use_placeholder=use_placeholder, topk=topk)
        inference_end = time.time() 

        #! =======================================================
        tflops = prof.get_total_flops()
        tflops /= 1e12
        print()
        print(f"TFLOPs: {tflops}")
        total_flops += tflops
        #! =======================================================

        topk_points = pred["topk_points"]
        
        # compute the metrics
        px, py = topk_points[0]
        
        # 예측 좌표를 원본 좌표계로 되돌리기
        if RESIZE_RATIO < 1.00:
            px_orig = px * w_resized / RESIZE_RATIO  # 리사이즈된 좌표를 원본 좌표로 변환
            py_orig = py * h_resized / RESIZE_RATIO
        else:
            px_orig = px * w_orig
            py_orig = py * h_orig

        print(f"Predicted (resized): ({px * w_resized}, {py * h_resized})")
        print(f"Predicted (original): ({px_orig}, {py_orig})")
        print(f"GT bbox (original): {gt_bbox}")
        
        x1, y1, x2, y2 = gt_bbox  # 원본 좌표계의 GT bbox 사용

        if (x1 <= px_orig <= x2) and (y1 <= py_orig <= y2):
            ele["hit_top1"] = 1
            ele["hit_topk"] = 1
            score_list.append(1)
            print("correct")
            correct_count += 1
        else: 
            print("wrong")
            score_list.append(0)
        print(f"up2now: {calc_acc(score_list)}")

        # overlap 계산 (원본 좌표계 사용)
        pred_bbox = [px_orig - IMAGE_PATCH_SIZE, py_orig - IMAGE_PATCH_SIZE, 
                    px_orig + IMAGE_PATCH_SIZE, py_orig + IMAGE_PATCH_SIZE]
        if do_boxes_overlap(pred_bbox, gt_bbox):
            ele["overlap_top1"] = 1
            ele["overlap_topk"] = 1

        # topk 평가 (원본 좌표계로 변환)
        for px, py in topk_points[1:]:
            if RESIZE_RATIO < 1.00:
                px_orig_k = px * w_resized / RESIZE_RATIO
                py_orig_k = py * h_resized / RESIZE_RATIO
            else:
                px_orig_k = px * w_orig
                py_orig_k = py * h_orig
                
            if (x1 <= px_orig_k <= x2) and (y1 <= py_orig_k <= y2):
                ele["hit_topk"] = 1
            pred_bbox_k = [px_orig_k - IMAGE_PATCH_SIZE, py_orig_k - IMAGE_PATCH_SIZE, 
                          px_orig_k + IMAGE_PATCH_SIZE, py_orig_k + IMAGE_PATCH_SIZE]
            if do_boxes_overlap(pred_bbox_k, gt_bbox):
                ele["overlap_topk"] = 1
        elapsed = inference_end - inference_start

        inference_time_sum += elapsed
        print(f"elapsed: {elapsed}")

        print("----------------------------\n")

        results.append(ele)
    
    print()
    print(f"mean tflops: {total_flops / iter}")
    print(f"total time: {inference_time_sum}")
    print(f"mean time: {inference_time_sum / iter}")
    print("iter:", iter)
    print("correct_count:", correct_count)

    return results, (inference_time_sum / iter), (total_flops / iter)


def get_metric(list_of_examples, time_mean, tflops_mean,
               domains=["mobile", "desktop", "web"],
               data_types=["text", "icon"]):
    """
    Computes metrics over a list of examples and prints/plots a table.
    
    Each element in list_of_examples is a dict containing:
        - "domain": Domain name (e.g., "web", "mobile", "desktop")
        - "data_type": Data type (e.g., "text", "icon")
        - "hit_top1", "overlap_top1", "hit_topk", "overlap_topk": binary (0 or 1)
    
    The final table has columns for each domain broken down by UI type (plus a domain-average)
    and overall columns ("All-text", "All-icon", "All-average").
    
    The rows of the table are:
        - hit_top1
        - overlap_top1
        - hit_topk
        - overlap_topk
    """
    
    # List of metric keys to compute.
    metrics = ["hit_top1", "overlap_top1", "hit_topk", "overlap_topk"]

    # Helper function to compute the mean of a given key from a list of examples.
    def compute_mean(examples, key):
        if not examples:
            return None
        return sum(example.get(key, 0) for example in examples) / len(examples)

    # Prepare results dictionary: structure {metric: {column_name: value}}.
    results = {metric: {} for metric in metrics}
    
    # Compute metrics for each group broken down by UI type.
    for domain in domains:
        # Filter examples for the current group.
        domain_examples = [ex for ex in list_of_examples if ex.get("domain") == domain]
        for data_type in data_types:
            # Filter further for the specific UI type.
            domain_data_type_examples = [ex for ex in domain_examples if ex.get("data_type") == data_type]
            col_name = f"{domain}-{data_type}"
            for metric in metrics:
                results[metric][col_name] = compute_mean(domain_data_type_examples, metric)
        
        # Compute domain-average (all UI types for this domain).
        col_name_avg = f"{domain}-avg"
        for metric in metrics:
            results[metric][col_name_avg] = compute_mean(domain_examples, metric)

    # Compute overall metrics for each UI type across all domains.
    for data_type in data_types:
        data_type_examples = [ex for ex in list_of_examples if ex.get("data_type") == data_type]
        col_name = f"All-{data_type}"
        for metric in metrics:
            results[metric][col_name] = compute_mean(data_type_examples, metric)
    
    # Compute overall average across all examples.
    overall_key = "All-avg"
    for metric in metrics:
        results[metric][overall_key] = compute_mean(list_of_examples, metric)
    
    # Define the order of columns.
    columns_order = []
    for domain in domains:
        for data_type in data_types:
            columns_order.append(f"{domain}-{data_type}")
        columns_order.append(f"{domain}-avg")
    for data_type in data_types:
        columns_order.append(f"All-{data_type}")
    columns_order.append("All-avg")
    
    # ------------- Print Table to Console -------------
    # Prepare header row.
    header = [""] + columns_order
    # Calculate column widths for console printing.
    col_widths = [max(len(col), 12) for col in header]
    
    def format_cell(cell):
        if isinstance(cell, float):
            return f"{cell*100:.2f}"
        elif cell is None:
            return "N/A"
        return str(cell)
    
    # Print header.
    header_line = " | ".join(word.ljust(width) for word, width in zip(header, col_widths))
    separator_line = "-+-".join("-" * width for width in col_widths)
    print(header_line)
    print(separator_line)
    
    for metric in metrics:
        row = [metric]
        for col in columns_order:
            val = results[metric].get(col)
            row.append(format_cell(val))
        row_line = " | ".join(word.ljust(width) for word, width in zip(row, col_widths))
        print(row_line)
    
    # ------------- Print Tab-delimited Version (for Excel Copy-Paste) -------------
    metric_info = "Tab-delimited Table for Excel:\n"
    # Header row.
    header_tab = "\t".join([""] + columns_order)
    metric_info += (header_tab + "\n")
    # Each row.
    for metric in metrics:
        row = [metric] + [format_cell(results[metric].get(col)) for col in columns_order]
        metric_info += ("\t".join(row) + "\n")
    print(metric_info)
    print(time_mean)
    
    # 간단한 정확도 계산 추가
    total_samples = len(list_of_examples)
    correct_samples = sum(1 for ex in list_of_examples if ex.get("hit_top1", 0) == 1)
    accuracy = (correct_samples / total_samples * 100) if total_samples > 0 else 0
    
    metric_info += f"\nAccuracy: {correct_samples}/{total_samples} = {accuracy:.4f}%"
    metric_info += f"\nTime: {time_mean}"
    metric_info += f"\nTFLOPS: {tflops_mean}"

    print(tflops_mean)
    return metric_info



"""
# cd to project root directory
python eval/screenSpot_v2.py --save_path <path_to_save_results>
"""
if __name__ == "__main__":

    if model_type == "qwen2vl":
        model_name_or_path = "microsoft/GUI-Actor-2B-Qwen2-VL"
        print(f"Loading model with Qwen2-VL backbone from {model_name_or_path}")
        model = Qwen2VLForConditionalGenerationWithPointer.from_pretrained(
            model_name_or_path,
            # torch_dtype=torch.bfloat16,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation="eager",
            do_sample=False
        ).eval()
        grounding_system_message = "You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform a series of pyautogui actions to complete the task."
    elif model_type == "qwen25vl":
        model_name_or_path = "microsoft/GUI-Actor-3B-Qwen2.5-VL"
        print(f"Loading model with Qwen2.5-VL backbone from {model_name_or_path}")
        model = Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
            model_name_or_path,
            # torch_dtype=torch.bfloat16,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation="eager",
            do_sample=False
        ).eval()
        grounding_system_message = "You are a GUI agent. Given a screenshot of the current GUI and a human instruction, your task is to locate the screen element that corresponds to the instruction. You should output a PyAutoGUI action that performs a click on the correct position. To indicate the click location, we will use some special tokens, which is used to refer to a visual patch later. For example, you can output: pyautogui.click(<your_special_token_here>)."
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    print(f"Loaded model from {model_name_or_path}")

    prof = FlopsProfiler(model)
    prof.start_profile()

    # 전체 task 통계 변수 초기화
    total_samples = 0
    total_correct = 0
    total_time = 0.0
    total_tflops = 0.0

    print(TASKS)
    for task in TASKS:
        save_path = SAVE_DIR
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        # pred_path = PRED_PATH.format(save_dir=SAVE_DIR, task=task, max_pixels=MAX_PIXELS)
        metric_path = METRIC_PATH.format(save_dir=SAVE_DIR, task=task, resize_ratio=RESIZE_RATIO)
   
    
        print(f"Evaluating {model_name_or_path}...")
        results, time_mean, tflops_mean = evaluate(
            model_name_or_path, 
            use_placeholder=False, 
            topk=3,
            task=task,
            prof=prof
        )
        # with open(pred_path, "w") as f:
        #     json.dump(results, f)
        # print(f"Saved {len(results)} predictions to {pred_path}")

        if not os.path.exists(metric_path):
            metric_info = get_metric(list_of_examples=results, time_mean=time_mean, tflops_mean=tflops_mean)
            with open(metric_path, "w") as f:
                f.write(metric_info)
            print(f"Saved metric to {metric_path}")
        
        # 전체 task 통계에 누적
        task_samples = len(results)
        task_correct = sum(1 for ex in results if ex.get("hit_top1", 0) == 1)
        
        total_samples += task_samples
        total_correct += task_correct
        total_time += time_mean * task_samples  # 총 시간으로 계산
        total_tflops += tflops_mean * task_samples  # 총 TFLOPs로 계산
        
        print(f"Task {task} completed: {task_correct}/{task_samples} accuracy = {task_correct/task_samples*100:.2f}%")
    
    # 모든 task 완료 후 전체 결과 계산 및 CSV 저장
    print("\n📊 All Tasks Completed!")
    
    # 전체 평균 계산
    avg_accuracy = (total_correct / total_samples * 100) if total_samples > 0 else 0
    avg_time = (total_time / total_samples) if total_samples > 0 else 0
    avg_tflops = (total_tflops / total_samples) if total_samples > 0 else 0
    
    print(f"Total Results:")
    print(f"Total Samples: {total_samples}")
    print(f"Total Correct: {total_correct}")
    print(f"Average Accuracy: {avg_accuracy:.2f}%")
    print(f"Average Time: {avg_time:.4f}s")
    print(f"Average TFLOPs: {avg_tflops:.2f}")
    
    # CSV로 저장
    results_csv_path = "../_results"
    os.makedirs(results_csv_path, exist_ok=True)
    csv_file_path = os.path.join(results_csv_path, "result_vanilla.csv")
    
    # CSV 헤더 정의
    csv_headers = [
        "model", "resize_ratio", "avg_accuracy", "avg_time", "avg_tflops", "total_samples", "timestamp"
    ]
    
    # CSV 데이터 행 생성
    csv_row = [
        model_name,
        RESIZE_RATIO,
        round(avg_accuracy, 2),
        round(avg_time, 4),
        round(avg_tflops, 2),
        total_samples,
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ]
    
    # CSV 파일이 없으면 헤더와 함께 생성, 있으면 데이터 행만 추가
    file_exists = os.path.exists(csv_file_path)
    
    with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # 파일이 없거나 비어있으면 헤더 추가
        if not file_exists or os.path.getsize(csv_file_path) == 0:
            writer.writerow(csv_headers)
        
        # 결과 행 추가
        writer.writerow(csv_row)
    
    print(f"📝 Results saved to CSV: {csv_file_path}")
        
