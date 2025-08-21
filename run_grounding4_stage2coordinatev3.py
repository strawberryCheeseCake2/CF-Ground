# Standard Library
import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Tuple

# Third-Party Libraries
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer, set_seed

# ==============================================================================
# ## 💡 코드가 매우 간결해졌습니다.
# ## 불필요한 함수들을 모두 제거하고 핵심 로직만 남겼습니다.
# ==============================================================================

#! Argument =======================
SEED = 0
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "3"

# Dataset & Model
MLLM_PATH = "zonghanHZH/ZonUI-3B"
SCREENSPOT_IMGS = "./data/screenspotv2_imgs"
SCREENSPOT_TEST = "./data"
SAVE_DIR = "./output/" + "0822_direct_original_resolution" #! 저장 경로 변경

# Data Processing
TASKS = ["mobile"]
SAMPLE_RANGE = slice(0, 50) #! 테스트할 샘플 범위

#! Helper Functions =================

def parse_args():
    parser = argparse.ArgumentParser(description="Run VLM grounding task with original resolution.")
    parser.add_argument('--mllm_path', type=str, default=MLLM_PATH)
    parser.add_argument('--screenspot_imgs', type=str, default=SCREENSPOT_IMGS)
    parser.add_argument('--screenspot_test', type=str, default=SCREENSPOT_TEST)
    parser.add_argument('--save-dir', type=str, default=SAVE_DIR)
    return parser.parse_args()

def point_in_bbox(pt: Tuple[int, int], bbox: list) -> bool:
    """좌표가 바운딩 박스 내에 있는지 확인합니다."""
    x, y = pt
    left, top, right, bottom = bbox
    return left <= x <= right and top <= y <= bottom

def calc_acc(score_list: list) -> float:
    """정확도를 계산합니다."""
    return sum(score_list) / len(score_list) if score_list else 0.0

def parse_coordinates(text: str) -> Tuple[int, int] | None:
    """모델이 생성한 텍스트에서 '[x, y]' 형식의 좌표를 파싱합니다."""
    match = re.search(r'\[\s*(\d+)\s*,\s*(\d+)\s*\]', text)
    return (int(match.group(1)), int(match.group(2))) if match else None

def visualize_final_result(image_path, save_dir, gt_bbox, predicted_point, is_success):
    """최종 예측 결과를 원본 이미지 위에 시각화합니다."""
    if not predicted_point: return
    
    os.makedirs(save_dir, exist_ok=True)
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    # Ground Truth BBox (초록색)
    draw.rectangle(gt_bbox, outline="lime", width=4)

    # Predicted Point (성공: 노란색 별, 실패: 자홍색 별)
    px, py = predicted_point
    radius = 15
    color = "yellow" if is_success else "magenta"
    
    draw.line([(px, py - radius), (px, py + radius)], fill=color, width=4)
    draw.line([(px - radius, py), (px + radius, py)], fill=color, width=4)
    draw.line([(px - radius*0.7, py - radius*0.7), (px + radius*0.7, py + radius*0.7)], fill=color, width=4)
    draw.line([(px - radius*0.7, py + radius*0.7), (px + radius*0.7, py - radius*0.7)], fill=color, width=4)
    
    status = "SUCCESS" if is_success else "FAIL"
    final_path = os.path.join(save_dir, f"result_{status}.png")
    img.save(final_path)
    print(f"Final visualization saved to: {final_path}")


#! Main Execution ===================

if __name__ == '__main__':
    set_seed(SEED)
    args = parse_args()
    
    print("Loading model and processor...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.mllm_path,
        torch_dtype="auto",
        attn_implementation="eager",
        device_map="auto" # 간단한 모델이므로 auto로 설정
    )
    processor = AutoProcessor.from_pretrained(args.mllm_path)
    model.eval()
    print("Model loaded successfully.")

    # 모델에게 원본 해상도를 알려주고 좌표를 직접 요청하는 새로운 프롬프트
    question_template = """You are an expert AI assistant for mobile UI navigation. Based on the full-screen image provided, your task is to identify the precise (x, y) coordinate to tap for the given instruction.
The screen resolution is {width}x{height}.
Your answer must contain only the coordinates in the format [x, y].

# Instruction
{task_prompt}
"""

    for task in TASKS:
        dataset_path = os.path.join(args.screenspot_test, f"screenspot_{task}_v2.json")
        if not os.path.exists(dataset_path):
            print(f"Dataset not found: {dataset_path}")
            continue
        
        with open(dataset_path, 'r') as f:
            screenspot_data = json.load(f)[SAMPLE_RANGE]
        
        print(f"\n--- Starting task: {task} ({len(screenspot_data)} samples) ---")
        scores = []
        
        for item in tqdm(screenspot_data, desc=f"Processing {task}"):
            filename = item["img_filename"]
            img_path = os.path.join(args.screenspot_imgs, filename)
            if not os.path.exists(img_path): continue

            instruction = item["instruction"]
            bbox = item["bbox"]
            original_bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            
            original_image = Image.open(img_path).convert("RGB")
            width, height = original_image.size
            
            # 1. 프롬프트 및 입력 데이터 생성
            question = question_template.format(width=width, height=height, task_prompt=instruction)
            msgs = [{"role": "user", "content": [
                {"type": "image", "image": original_image},
                {"type": "text", "text": question}
            ]}]
            
            inputs = processor(
                text=processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True),
                images=[original_image],
                padding=True,
                return_tensors="pt",
            ).to(model.device)

            # 2. 모델 추론
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=30, do_sample=False, pad_token_id=processor.tokenizer.eos_token_id)
            
            response_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
            prompt_len = len(processor.batch_decode(inputs['input_ids'], skip_special_tokens=True)[0])
            generated_text = response_text[prompt_len:].strip()
            print(f"\nInstruction: {instruction}")
            print(f"Model generated response: '{generated_text}'")
            
            # 3. 결과 파싱 및 평가
            predicted_point = parse_coordinates(generated_text)
            is_success = False
            if predicted_point:
                is_success = point_in_bbox(predicted_point, original_bbox)
                result_msg = "✅ Success" if is_success else "❌ Fail"
                print(f"{result_msg}: Predicted {predicted_point} | GT Bbox: {original_bbox}")
            else:
                print("❌ Parsing failed: Could not find coordinates in the response.")
            
            scores.append(1 if is_success else 0)
            
            # 4. 시각화 및 결과 폴더 정리
            inst_dir_name = re.sub(r'\W+', '_', instruction).strip('_')[:50]
            status_symbol = '✅' if is_success else '❌'
            save_dir_name = f"{os.path.splitext(filename)[0]}/{inst_dir_name} {status_symbol}"
            final_save_dir = os.path.join(args.save_dir, save_dir_name)
            
            visualize_final_result(
                image_path=img_path,
                save_dir=final_save_dir,
                gt_bbox=original_bbox,
                predicted_point=predicted_point,
                is_success=is_success
            )
            print(f"Up-to-date Accuracy: {calc_acc(scores):.4f}")

        # 최종 결과 요약
        final_accuracy = calc_acc(scores)
        print("\n==================================================")
        print(f"Task '{task}' Final Results:")
        print(f"  - Total Samples: {len(scores)}")
        print(f"  - Final Accuracy: {final_accuracy:.4f}")
        print("==================================================")
        
        # 메트릭 파일 저장
        metrics = {
            "task": task,
            "total_samples": len(scores),
            "accuracy": final_accuracy,
            "approach": "Direct Prediction on Original Resolution"
        }
        with open(os.path.join(args.save_dir, f"{task}_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)