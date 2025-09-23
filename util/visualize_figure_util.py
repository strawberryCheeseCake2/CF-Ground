import os
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import torch
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from copy import deepcopy



@torch.inference_mode()
def get_attn_map(image: Image.Image, attn_scores: list, n_width: int, n_height: int) -> Image.Image:
    """
    주어진 이미지 위에 어텐션 스코어로 히트맵을 생성하여 겹칩니다.

    Args:
        image (Image.Image): 원본 PIL 이미지.
        attn_scores (list): 1차원으로 펼쳐진 어텐션 스코어 리스트.
        n_width (int): 어텐션 맵의 너비 (패치 개수).
        n_height (int): 어텐션 맵의 높이 (패치 개수).

    Returns:
        Image.Image: 히트맵이 겹쳐진 PIL 이미지.
    """
    w, h = image.size
    # 1차원 스코어를 2차원 배열로 변환
    scores = np.array(attn_scores).reshape(n_height, n_width)

    # 스코어를 0~1 사이로 정규화
    min_val, max_val = scores.min(), scores.max()
    denom = (max_val - min_val) if (max_val - min_val) != 0 else 1.0
    scores_norm = (scores - min_val) / denom

    # 정규화된 스코어를 원본 이미지 크기에 맞게 리사이즈하여 흑백 맵 생성
    score_map = Image.fromarray((scores_norm * 255).astype(np.uint8)).resize(
        (w, h), resample=Image.NEAREST
    )
    
    # Matplotlib의 'jet' 컬러맵을 사용하여 흑백 맵을 컬러 히트맵으로 변환
    colormap = plt.get_cmap('jet')
    colored_score_map_array = colormap(np.array(score_map) / 255.0)[:, :, :3]
    colored_overlay = Image.fromarray((colored_score_map_array * 255).astype(np.uint8))
    
    # 원본 이미지와 히트맵을 투명도(alpha)를 조절하여 합성
    blended_image = Image.blend(image.convert('RGB'), colored_overlay, alpha=0.4)
    
    return blended_image

# -----------------------------------------
# 메인 시각화 함수
# -----------------------------------------


def vis_gt():
    pass

def visualize_stage1_attention_crops(s1_pred, resized_image, crop_list, original_image, save_dir, instruction, gt_bbox=None, s1_predicted_point=None):
    """Stage 1 attention 맵과 생성된 crop들을 시각화"""
    
    print(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    resized = deepcopy(resized_image)

    original_image.save(f"{save_dir}/origianl.png")
    resized.save(f"{save_dir}/resized.png")
    gt_vis_img = deepcopy(resized_image)

    # 1. Attention 맵 생성
    if 'attn_scores' in s1_pred and s1_pred['attn_scores']:
        attn_scores = np.array(s1_pred['attn_scores'][0])
        n_width = s1_pred['n_width']
        n_height = s1_pred['n_height']
        
        # 리사이즈된 이미지에 attention 맵 오버레이
        blended_img = get_attn_map(
            image=resized,
            attn_scores=attn_scores,
            n_width=n_width,
            n_height=n_height
        )
        
        # 리사이즈 비율 계산
        resize_ratio = s1_pred.get('resize_ratio', 1.0)
        resized_w, resized_h = resized.size
        orig_w, orig_h = original_image.size
        
        draw = ImageDraw.Draw(blended_img)
        draw_gt = ImageDraw.Draw(gt_vis_img)
        
        # 2. GT 박스를 초록색으로 그리기 (리사이즈된 좌표계로 변환)
        if gt_bbox is not None:
            gt_left, gt_top, gt_right, gt_bottom = gt_bbox
            # 원본 좌표를 리사이즈된 좌표로 변환
            scaled_gt_left = int(gt_left * resize_ratio)
            scaled_gt_top = int(gt_top * resize_ratio)
            scaled_gt_right = int(gt_right * resize_ratio)
            scaled_gt_bottom = int(gt_bottom * resize_ratio)
            
            # 이미지 범위 내로 클리핑
            scaled_gt_left = max(0, min(scaled_gt_left, resized_w))
            scaled_gt_top = max(0, min(scaled_gt_top, resized_h))
            scaled_gt_right = max(0, min(scaled_gt_right, resized_w))
            scaled_gt_bottom = max(0, min(scaled_gt_bottom, resized_h))
            
            draw_gt.rectangle([scaled_gt_left, scaled_gt_top, scaled_gt_right, scaled_gt_bottom], 
                         outline="green", width=4)
            gt_vis_img.save(f"{save_dir}/gt_vis.png")
        
        # S1 예측점 그리기 (파란색 원)
        if s1_predicted_point is not None:
            # 원본 좌표를 리사이즈된 좌표로 변환
            s1_x = int(s1_predicted_point[0] * resize_ratio)
            s1_y = int(s1_predicted_point[1] * resize_ratio)
            
            # 이미지 범위 내로 클리핑
            s1_x = max(0, min(s1_x, resized_w))
            s1_y = max(0, min(s1_y, resized_h))
            
            # 예측점을 파란색 원으로 그리기
            radius = 8
            # draw.ellipse([s1_x - radius, s1_y - radius, s1_x + radius, s1_y + radius], 
            #             fill="blue", outline="white", width=2)
        
        for crop in crop_list:
            bbox = crop["bbox"]
            crop_id = crop["id"]

            cropped = original_image.crop(bbox)
            cropped.save(f"{save_dir}/cropped_{crop_id}.png")

        save_path = os.path.join(save_dir, "s1_attention.png")
        blended_img.save(save_path)

        # 3. Crop 박스들을 빨간색으로 그리기 (원본 좌표를 리사이즈된 좌표로 변환)
        for crop in crop_list:
            # 원본 좌표를 리사이즈된 좌표로 변환
            bbox = crop["bbox"]
            left, top, right, bottom = bbox
            
            scaled_left = int(left * resize_ratio)
            scaled_top = int(top * resize_ratio)
            scaled_right = int(right * resize_ratio)
            scaled_bottom = int(bottom * resize_ratio)
            
            # 이미지 범위 내로 클리핑
            scaled_left = max(0, min(scaled_left, resized_w))
            scaled_top = max(0, min(scaled_top, resized_h))
            scaled_right = max(0, min(scaled_right, resized_w))
            scaled_bottom = max(0, min(scaled_bottom, resized_h))
            
            # 유효한 크기인지 확인
            if scaled_right > scaled_left and scaled_bottom > scaled_top:
                draw.rectangle([scaled_left, scaled_top, scaled_right, scaled_bottom], 
                             outline="red", width=3)
        
        # 4. 결과 저장
        # os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "s1_attention_crops.png")
        blended_img.save(save_path)
        # print(f"🌄 Stage 1 visualization : {save_path}")
    
    else:
        print("⚠️ No attention scores found for visualization")


def visualize_stage2_multi_attention(s2_pred, crop_list, original_image, save_dir, instruction, predicted_point=None):
    """Stage 2 multi-image inference 결과 시각화 - 각 crop별 attention과 원본 이미지에 합성된 attention 맵"""
    
    if not s2_pred or not s2_pred.get('per_image') or not crop_list:
        print("⚠️ No Stage2 multi-image results found for visualization")
        return
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 개별 crop들의 attention 시각화
    num_crops = len(crop_list)
    per_image_results = s2_pred['per_image']
    
    # 개별 crop 시각화를 위한 서브플롯 설정
    fig_width = min(20, num_crops * 6)  # 최대 20, crop당 6인치
    fig_height = 8
    
    plt.figure(figsize=(fig_width, fig_height))
    
    for img_idx, img_result in enumerate(per_image_results):
        if img_idx >= len(crop_list):
            continue
            
        crop_info = crop_list[img_idx]
        crop_img = crop_info['img']
        crop_bbox = crop_info['bbox']
        
        # attention 정보 추출
        attn_scores = np.array(img_result['attn_scores'][0])
        n_width = img_result['n_width']
        n_height = img_result['n_height']
        topk_points = img_result['topk_points']
        topk_values = img_result['topk_values']
        
        # 서브플롯에 crop 이미지와 attention 표시
        plt.subplot(2, num_crops, img_idx + 1)
        
        # attention 맵이 겹쳐진 이미지 생성
        blended_crop = get_attn_map(
            image=crop_img,
            attn_scores=attn_scores,
            n_width=n_width,
            n_height=n_height
        )

        blended_crop.save(f"{save_dir}/s2_crop_att_{crop_info['id']}.png")
        
        plt.imshow(blended_crop)
        
        # 예측 점들 표시 (crop 내 좌표)
        crop_w, crop_h = crop_img.size
        for i, (point, score) in enumerate(zip(topk_points[:3], topk_values[:3])):
            # 정규화된 좌표를 픽셀 좌표로 변환
            pixel_x = point[0] * crop_w
            pixel_y = point[1] * crop_h
            
            # 점수에 따라 색상과 크기 조정
            color = 'red' if i == 0 else 'orange' if i == 1 else 'yellow'
            size = 100 if i == 0 else 80 if i == 1 else 60
            
            plt.scatter(pixel_x, pixel_y, c=color, s=size, marker='*', 
                       edgecolors='white', linewidth=2, alpha=0.9)
        
        plt.axis('off')
        
        # 아래쪽에 원본 이미지에서의 위치 표시
        plt.subplot(2, num_crops, num_crops + img_idx + 1)
        
        # 원본 이미지에 해당 crop 영역 표시
        orig_img_copy = original_image.copy()
        draw = ImageDraw.Draw(orig_img_copy)
        
        # crop 박스 그리기
        draw.rectangle(crop_bbox, outline='red', width=3)
        
        plt.imshow(orig_img_copy)
        plt.axis('off')
    
    plt.tight_layout()
    
    # 개별 crop 시각화 저장
    individual_save_path = os.path.join(save_dir, "s2_multi_individual_crops.png")
    plt.savefig(individual_save_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    # print(f"🌄 Stage2 multi-image visualization saved:")
    # print(f"  - Individual crops: {individual_save_path}")


def visualize_stage3_point_ensemble(s3_ensemble_candidates, original_image, crop_list, original_bbox, 
                                    s3_ensemble_point, s2_corrected_point, s1_original_point,
                                    stage1_ratio, stage2_ratio, save_dir, vis_only_wrong=False, 
                                    stage3_success=True):
    """Stage 3 포인트 기반 앙상블 시각화 - 후보 포인트들과 최종 선택된 포인트 표시"""
    
    if s3_ensemble_point is None or not save_dir:
        return
        
    if vis_only_wrong and stage3_success:
        return
    
    # 원본 이미지 크기 계산
    orig_w, orig_h = original_image.size
    
    # 여백 크기 계산 (이미지 높이의 15%)
    margin_height = int(orig_h * 0.15)
    legend_height = int(orig_h * 0.12)
    
    # 확장된 캔버스 크기
    canvas_w = orig_w
    canvas_h = orig_h + margin_height + legend_height
    
    # matplotlib 설정
    fig_width = canvas_w / 100  # DPI 100 기준
    fig_height = canvas_h / 100
    
    plt.figure(figsize=(fig_width, fig_height), dpi=100)
    
    # 이미지 영역과 여백 영역 분리
    ax_main = plt.axes([0, legend_height/canvas_h, 1, orig_h/canvas_h])
    ax_legend = plt.axes([0, 0, 1, legend_height/canvas_h])
    
    # 메인 이미지 영역
    ax_main.imshow(original_image, alpha=1.0)
    
    # GT 박스 표시 (초록색)
    if original_bbox is not None:
        from matplotlib.patches import Rectangle
        gt_rect = Rectangle((original_bbox[0], original_bbox[1]), 
                           original_bbox[2] - original_bbox[0], 
                           original_bbox[3] - original_bbox[1],
                           fill=False, edgecolor='green', linewidth=4, alpha=0.9)
        ax_main.add_patch(gt_rect)
    
    # 크롭 박스들 표시
    for i, crop in enumerate(crop_list):
        bbox = crop['bbox']
        from matplotlib.patches import Rectangle
        rect = Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], 
                        fill=False, edgecolor='yellow', linewidth=2, alpha=0.8)
        ax_main.add_patch(rect)
    
    # 주요 포인트들 표시 (기본 크기)
    base_point_size = 600  # 기본 포인트 크기 (2배로 크게)
    
    # Stage1 결과 (연두색 동그라미)
    if s1_original_point:
        ax_main.scatter(s1_original_point[0], s1_original_point[1], 
                       c='limegreen', s=base_point_size, marker='o', alpha=0.9, 
                       edgecolors='darkgreen', linewidth=3, label='Stage1')
    
    # Stage2 결과 (파란색 동그라미)
    if s2_corrected_point:
        ax_main.scatter(s2_corrected_point[0], s2_corrected_point[1], 
                       c='blue', s=base_point_size, marker='o', alpha=0.9, 
                       edgecolors='darkblue', linewidth=3, label='Stage2')
    
    # Stage3 앙상블 후보 포인트들 표시
    if s3_ensemble_candidates:
        # 점수 순으로 정렬된 후보들 표시
        sorted_candidates = sorted(s3_ensemble_candidates, key=lambda x: x['score'], reverse=True)
        
        # Stage3 1등 (빨간색 별, 같은 크기) - 배경 추가하여 잘 보이게
        if len(sorted_candidates) > 0:
            x, y = sorted_candidates[0]['point']
            score = sorted_candidates[0]['score']
            ax_main.scatter(x, y, c='red', s=base_point_size*1.5, marker='*', alpha=1.0, 
                           edgecolors='darkred', linewidth=3, label='Stage3 #1')
        
        # Stage3 나머지 후보들 (빨간색 동그라미, 점점 작아지게)
        for i, candidate in enumerate(sorted_candidates[1:5]):  # 2등~5등만 표시
            x, y = candidate['point']
            score = candidate['score']
            # 크기 점점 작아지게 (2등: 80%, 3등: 60%, 4등: 40%, 5등: 20%)
            size_ratio = max(0.2, 1.0 - (i + 1) * 0.2)
            point_size = int(base_point_size * size_ratio)
            # 빨간색 동그라미로 통일
            color = 'red'
            ax_main.scatter(x, y, c=color, s=point_size, marker='o', alpha=0.8, 
                           edgecolors='darkred', linewidth=2)
    
    # 축 설정
    ax_main.set_xlim(0, orig_w)
    ax_main.set_ylim(orig_h, 0)  # y축 뒤집기 (이미지 좌표계)
    ax_main.set_aspect('equal')
    ax_main.axis('off')
    
    # 이미지와 범례 사이에 구분선 추가
    fig = plt.gcf()
    line_y = legend_height / canvas_h  # 범례 영역 상단
    fig.add_artist(plt.Line2D([0, 1], [line_y, line_y], color='gray', linewidth=2, alpha=0.8))
    
    # 범례 영역 설정
    ax_legend.axis('off')
    
    # 저장
    success_str = "success" if stage3_success else "failure"
    filename = f"s3_point_ensemble_{success_str}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=100, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    
    # print(f"💾 Stage3 point ensemble visualization saved: {filepath}")
