import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "0"  # 몇번 GPU 사용할지 ("0,1", "2" 등)
device_map = "balanced"
max_memory = {
    0: "75GiB",
    # 1: "75GiB",
    # 2: "75GiB",
    "cpu": "120GiB",  # 남는 건 CPU 오프로딩xs
}

import torch
from copy import deepcopy
import json
import argparse
import time
import sys

from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoProcessor, set_seed, GenerationConfig

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gui_actor.constants import chat_template
from gui_actor.modeling import Qwen2VLForConditionalGenerationWithPointer
# from gui_actor.modeling_qwen25vl import Qwen2_5_VLForConditionalGenerationWithPointer
from gui_actor.modeling_fastv import Qwen2_5_VLForConditionalGenerationWithPointer
# from gui_actor.inference import inference, ForceFollowTokensLogitsProcessor
from gui_actor.fastv_inference import inference, ForceFollowTokensLogitsProcessor
from gui_actor.utils import do_boxes_overlap
from gui_actor.constants import DEFAULT_POINTER_PAD_TOKEN, DEFAULT_POINTER_END_TOKEN
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLAttention, Qwen2_5_VLSdpaAttention
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration


from deepspeed.profiling.flops_profiler import get_model_profile
from deepspeed.profiling.flops_profiler import FlopsProfiler


#!=================================================================================
# from thop import profile  # FLOPs 측정
import numpy as np
#!=================================================================================

IMAGE_PATCH_SIZE =14
RATIO = 1.0

set_seed(0)
# MAX_PIXELS = 3200 * 1800
# MAX_PIXELS = 1920 * 1080
MAX_PIXELS = 3211264
# MAX_PIXELS = 1280 * 28 * 28

SAVE_DIR = "../output/fastv"
CALC_FLOPS = True

# 버리는 비율
FASTV_R = 0.5
# FASTV_R = 0.7

PRED_PATH = "{save_dir}/{FASTV_R}/{task}_{max_pixels}_preds.json"
METRIC_PATH = "{save_dir}/{FASTV_R}/{task}_{max_pixels}_metrics.txt"

TASKS = ["mobile", "web", "desktop"]

#!=================================================================================


def resize_image(image, resize_to_pixels=MAX_PIXELS):
    image_width, image_height = image.size
    if (resize_to_pixels is not None) and ((image_width * image_height) != resize_to_pixels):
        resize_ratio = (resize_to_pixels / (image_width * image_height)) ** 0.5
        image_width_resized, image_height_resized = int(image_width * resize_ratio), int(image_height * resize_ratio)
        image = image.resize((image_width_resized, image_height_resized))
    return image, image_width_resized, image_height_resized 

#!=================================================================================

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

def calc_acc(score_list):
    return sum(score_list) / len(score_list) if len(score_list) != 0 else 0

def evaluate(model_name_or_path, model_type, use_placeholder, topk, task, device_map, max_memory=max_memory):
    # initialize model
    data_processor = AutoProcessor.from_pretrained(model_name_or_path, max_pixels=MAX_PIXELS)
    tokenizer = data_processor.tokenizer
    for k, v in tokenizer.added_tokens_encoder.items():
        print(v, k)

    if model_type == "qwen2vl":
        print(f"Loading model with Qwen2-VL backbone from {model_name_or_path}")
        model = Qwen2VLForConditionalGenerationWithPointer.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            max_memory=max_memory,
            attn_implementation="flash_attention_2",
            do_sample=False
        ).eval()
        grounding_system_message = "You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform a series of pyautogui actions to complete the task."
    elif model_type == "qwen25vl":
        print(f"Loading model with Qwen2.5-VL backbone from {model_name_or_path}")
        model = Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            # device_map=device_map,
            device_map="auto",
            max_memory=max_memory,
            attn_implementation="eager",  #!
            do_sample=False
        ).eval()
        grounding_system_message = "You are a GUI agent. Given a screenshot of the current GUI and a human instruction, your task is to locate the screen element that corresponds to the instruction. You should output a PyAutoGUI action that performs a click on the correct position. To indicate the click location, we will use some special tokens, which is used to refer to a visual patch later. For example, you can output: pyautogui.click(<your_special_token_here>)."
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    print(f"Loaded model from {model_name_or_path}")

    logits_processor_pointer = ForceFollowTokensLogitsProcessor(
        token_a_id=tokenizer.encode(DEFAULT_POINTER_PAD_TOKEN)[0],
        forced_sequence=[
            tokenizer.encode(DEFAULT_POINTER_END_TOKEN)[0]
        ]
    )
    
    
    
    dataset = json.load(open(os.path.join("../data", f"screenspot_{task}_v2.json"), 'r'))

    results = []

    inference_time_sum = 0.0
    mobile_num = 0

    score_list = []
    correct_predictions = 0  # 정확한 예측 횟수
    total_predictions = 0    # 전체 예측 횟수
    
    if CALC_FLOPS:
        prof = FlopsProfiler(model)
        prof.start_profile()
    tflops_sum = 0.0

    for j, example in tqdm(enumerate(dataset)):

        mobile_num += 1

        filename = example["img_filename"]
        img_path = os.path.join("../data/screenspotv2_image", filename)
        if not os.path.exists(img_path):
            print("img not found", flush=True)
            input()


        print(img_path)

        original_image = Image.open(img_path).convert("RGB")
        w_orig, h_orig = original_image.size 

        img = deepcopy(original_image)
        w_resized, h_resized = w_orig, h_orig 
        # if w_orig * h_orig > MAX_PIXELS:
        #   img, w_resized, h_resized = resize_image(original_image)
        
        gt_bbox = example["bbox"]
        gt_bbox = [gt_bbox[0], gt_bbox[1], gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]]
        
        ele = {
            "file_name": example["img_filename"],
            "data_type": example["data_type"],
            # "domain": domain_dict[example["data_source"]],
            "instruction": example["instruction"],
            "img_size": [w_orig, h_orig],
            "bbox_x1y1x2y2": normalize_bbox(example["bbox"], w_orig, h_orig),
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




        if CALC_FLOPS:
            prof.reset_profile()
        
        inference_start = time.time()

        pred = inference(conversation, model, tokenizer, 
            data_processor, logits_processor=logits_processor_pointer, 
            use_placeholder=use_placeholder, topk=3, fastv_r=FASTV_R)
        inference_end = time.time() 
        
        if CALC_FLOPS:
            flops = prof.get_total_flops()
            tflops = flops/1e12
            print(f"Total: {tflops:.2f} TFLOPs")
            tflops_sum += tflops

        topk_points = pred["topk_points"]
        
        # gt_bbox = ele["bbox_x1y1x2y2"]
        
        # compute the metrics
        px, py = topk_points[0]
        px, py = px * w_orig, py * h_orig
        x1, y1, x2, y2 = gt_bbox

        total_predictions += 1  # 전체 예측 횟수 증가
        
        if (x1 <= px <= x2) and (y1 <= py <= y2):
            ele["hit_top1"] = 1
            ele["hit_topk"] = 1
            score_list.append(1)
            correct_predictions += 1  # 정확한 예측 횟수 증가
        else: 
            score_list.append(0)
        
        # 현재까지의 정확도 출력
        current_accuracy = correct_predictions / total_predictions * 100
        print(f"Sample {j+1}/{len(dataset)} | Correct: {correct_predictions}/{total_predictions} | Accuracy: {current_accuracy:.2f}%")

        # w, h = example["image"].size
        pred_bbox = [px - IMAGE_PATCH_SIZE / w_orig, py - IMAGE_PATCH_SIZE / h_orig, px + IMAGE_PATCH_SIZE / w_orig, py + IMAGE_PATCH_SIZE / h_orig]
        if do_boxes_overlap(pred_bbox, gt_bbox):
            ele["overlap_top1"] = 1
            ele["overlap_topk"] = 1

        for px, py in topk_points[1:]:
            if (x1 <= px <= x2) and (y1 <= py <= y2):
                ele["hit_topk"] = 1
            pred_bbox = [px - IMAGE_PATCH_SIZE / w_orig, py - IMAGE_PATCH_SIZE / h_orig, px + IMAGE_PATCH_SIZE / w_orig, py + IMAGE_PATCH_SIZE / h_orig]
            if do_boxes_overlap(pred_bbox, gt_bbox):
                ele["overlap_topk"] = 1
        elapsed = inference_end - inference_start

        inference_time_sum += elapsed
        print(f"elapsed: {elapsed}")

        results.append(ele)
    
    # 태스크별 최종 통계 계산
    final_accuracy = correct_predictions / total_predictions * 100 if total_predictions > 0 else 0
    mean_inference_time = inference_time_sum / mobile_num if mobile_num > 0 else 0
    mean_tflops = tflops_sum / mobile_num if mobile_num > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"Task '{task}' Results:")
    print(f"{'='*60}")
    print(f"Total samples: {total_predictions}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {final_accuracy:.2f}%")
    print(f"Mean inference time: {mean_inference_time:.4f}s")
    print(f"Mean TFLOPs: {mean_tflops:.4f}")
    print(f"{'='*60}")

    #! =======================================================
    # 평균 GFLOPs 콘솔 출력
    # if mobile_num > 0 and total_flops > 0:
    #     avg_gflops = (total_flops / mobile_num) / 1e9
    #     print(f"Avg FLOPs (GFLOPs): {avg_gflops:.2f}")
    #! =======================================================


    return results, mean_inference_time, mean_tflops, final_accuracy, correct_predictions, total_predictions


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
    print(tflops_mean)
    metric_info += "\nTime: " + str(time_mean) + "\nTFLOPS: " + str(tflops_mean)
    return metric_info


"""
# cd to project root directory
python eval/screenSpot_v2.py --save_path <path_to_save_results>
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="qwen2vl", choices=["qwen2vl", "qwen25vl"])
    parser.add_argument("--model_name_or_path", type=str, default="microsoft/GUI-Actor-2B-Qwen2-VL")
    parser.add_argument("--save_path", type=str, default="../")
    parser.add_argument('--topk', type=int, default=3, help='Topk')
    parser.add_argument('--no-placeholder', dest='use_placeholder', action='store_false', help='Disable the placeholder')
    parser.set_defaults(use_placeholder=True)

    args = parser.parse_args()
    args.model_name_or_path = "microsoft/GUI-Actor-3B-Qwen2.5-VL"
    args.model_type = "qwen25vl"

    # 전체 태스크 결과를 저장할 리스트
    all_task_results = []
    total_correct = 0
    total_samples = 0
    total_time_sum = 0.0
    total_tflops_sum = 0.0

    for task in TASKS:
        save_path = SAVE_DIR
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        pred_path = PRED_PATH.format(save_dir=SAVE_DIR, FASTV_R=FASTV_R, task=task, max_pixels=MAX_PIXELS)
        metric_path = METRIC_PATH.format(save_dir=SAVE_DIR, FASTV_R=FASTV_R, task=task, max_pixels=MAX_PIXELS)

        print(f"\n🔄 Evaluating {args.model_name_or_path} on {task} task...")
        results, time_mean, tflops_mean, accuracy, correct_count, total_count = evaluate(
            args.model_name_or_path, args.model_type, args.use_placeholder, 
            args.topk, task=task, device_map=device_map
        )
        
        # 태스크별 결과 저장
        task_result = {
            'task': task,
            'accuracy': accuracy,
            'correct_predictions': correct_count,
            'total_samples': total_count,
            'mean_inference_time': time_mean,
            'mean_tflops': tflops_mean
        }
        all_task_results.append(task_result)
        
        # 전체 통계에 누적
        total_correct += correct_count
        total_samples += total_count
        total_time_sum += time_mean
        total_tflops_sum += tflops_mean
        
        # 예측 결과 저장
        with open(pred_path, "w") as f:
            json.dump(results, f)
        print(f"✅ Saved {len(results)} predictions to {pred_path}")

        # 태스크별 성능 요약 저장
        task_summary_path = pred_path.replace('_preds.json', '_task_summary.txt')
        with open(task_summary_path, 'w') as f:
            f.write(f"Task: {task}\n")
            f.write(f"FastV Ratio: {FASTV_R}\n")
            f.write(f"Max Pixels: {MAX_PIXELS}\n")
            f.write(f"Accuracy: {accuracy:.2f}%\n")
            f.write(f"Correct Predictions: {correct_count}/{total_count}\n")
            f.write(f"Mean Inference Time: {time_mean:.4f}s\n")
            f.write(f"Mean TFLOPs: {tflops_mean:.4f}\n")
        print(f"✅ Saved task summary to {task_summary_path}")

        # 기존 상세 메트릭 저장
        if not os.path.exists(metric_path):
            metric_info = get_metric(list_of_examples=results, time_mean=time_mean, tflops_mean=tflops_mean)
            with open(metric_path, "w") as f:
                f.write(metric_info)
            print(f"✅ Saved detailed metrics to {metric_path}")

    # 전체 요약 결과 출력
    overall_accuracy = total_correct / total_samples * 100 if total_samples > 0 else 0
    avg_time = total_time_sum / len(TASKS) if len(TASKS) > 0 else 0
    avg_tflops = total_tflops_sum / len(TASKS) if len(TASKS) > 0 else 0
    
    print(f"\n{'='*80}")
    print(f"🎉 OVERALL RESULTS (FastV Ratio: {FASTV_R})")
    print(f"{'='*80}")
    print(f"Total Samples: {total_samples}")
    print(f"Total Correct: {total_correct}")
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")
    print(f"Average Inference Time: {avg_time:.4f}s")
    print(f"Average TFLOPs: {avg_tflops:.4f}")
    print(f"{'='*80}")
    
    print(f"📊 Task-wise Results:")
    for task_result in all_task_results:
        print(f"  {task_result['task']}: {task_result['accuracy']:.2f}% "
              f"({task_result['correct_predictions']}/{task_result['total_samples']}) | "
              f"Time: {task_result['mean_inference_time']:.4f}s | "
              f"TFLOPs: {task_result['mean_tflops']:.4f}")
    
    # 전체 요약을 파일에 저장
    overall_summary_path = os.path.join(SAVE_DIR, f"overall_summary_fastv_{FASTV_R}.txt")
    with open(overall_summary_path, 'w') as f:
        f.write(f"FastV Performance Summary (Ratio: {FASTV_R})\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Overall Results:\n")
        f.write(f"Total Samples: {total_samples}\n")
        f.write(f"Total Correct: {total_correct}\n")
        f.write(f"Overall Accuracy: {overall_accuracy:.2f}%\n")
        f.write(f"Average Inference Time: {avg_time:.4f}s\n")
        f.write(f"Average TFLOPs: {avg_tflops:.4f}\n\n")
        
        f.write(f"Task-wise Results:\n")
        for task_result in all_task_results:
            f.write(f"Task: {task_result['task']}\n")
            f.write(f"  Accuracy: {task_result['accuracy']:.2f}%\n")
            f.write(f"  Correct/Total: {task_result['correct_predictions']}/{task_result['total_samples']}\n")
            f.write(f"  Mean Inference Time: {task_result['mean_inference_time']:.4f}s\n")
            f.write(f"  Mean TFLOPs: {task_result['mean_tflops']:.4f}\n\n")
    
    print(f"\n💾 Overall summary saved to: {overall_summary_path}")
    print(f"{'='*80}")
