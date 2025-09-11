import os
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import torch
import matplotlib.pyplot as plt

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

@torch.inference_mode()
def draw_top_patch_attentions(image: Image.Image, attn_scores: np.ndarray, n_width: int, n_height: int, top_k: int = 3) -> Image.Image:
    """
    어텐션 스코어가 가장 높은 Top-K 패치의 중심에 해당 스코어 값을 텍스트로 그립니다.
    """
    img_w, img_h = image.size
    patch_pixel_w = img_w / n_width
    patch_pixel_h = img_h / n_height

    # 어텐션 스코어가 높은 순서대로 인덱스를 정렬
    # np.argsort()는 오름차순이므로 뒤집어줌 `[::-1]`
    top_indices = np.argsort(attn_scores)[::-1][:top_k]

    draw = ImageDraw.Draw(image)
    try:
        # 폰트 로드 (없을 경우 기본 폰트 사용)
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    for idx in top_indices:
        # 1차원 인덱스를 2차원 패치 좌표로 변환
        patch_y = idx // n_width
        patch_x = idx % n_width
        
        # 패치의 중심 픽셀 좌표 계산
        center_x = (patch_x + 0.5) * patch_pixel_w
        center_y = (patch_y + 0.5) * patch_pixel_h
        
        # 어텐션 값 텍스트 준비
        score_val = attn_scores[idx]
        text = f"{score_val:.3f}"
        
        # 텍스트 배경 및 위치 계산
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        text_bg_rect = [
            (center_x - text_w / 2 - 2, center_y - text_h / 2 - 2), # 좌상단
            (center_x + text_w / 2 + 2, center_y + text_h / 2 + 2)  # 우하단
        ]
        
        # 텍스트 배경(흰색 사각형)과 텍스트(검은색) 그리기
        draw.rectangle(text_bg_rect, fill="white")
        draw.text(
            (center_x - text_w / 2, center_y - text_h / 2),
            text,
            fill="black",
            font=font
        )
        
    return image

def visualize_stage1_attention_crops(s1_pred, resized_image, crop_list, original_image, save_dir, instruction, gt_bbox=None, s1_predicted_point=None):
    """Stage 1 attention 맵과 생성된 crop들을 시각화"""
    
    # 1. Attention 맵 생성
    if 'attn_scores' in s1_pred and s1_pred['attn_scores']:
        attn_scores = np.array(s1_pred['attn_scores'][0])
        n_width = s1_pred['n_width']
        n_height = s1_pred['n_height']
        
        # 리사이즈된 이미지에 attention 맵 오버레이
        blended_img = get_attn_map(
            image=resized_image,
            attn_scores=attn_scores,
            n_width=n_width,
            n_height=n_height
        )
        
        # 리사이즈 비율 계산
        resize_ratio = s1_pred.get('resize_ratio', 1.0)
        resized_w, resized_h = resized_image.size
        orig_w, orig_h = original_image.size
        
        draw = ImageDraw.Draw(blended_img)
        
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
            
            draw.rectangle([scaled_gt_left, scaled_gt_top, scaled_gt_right, scaled_gt_bottom], 
                         outline="green", width=4)
            
            # GT 라벨 추가
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except IOError:
                font = ImageFont.load_default()
            
            draw.text((scaled_gt_left + 5, scaled_gt_top - 20), "GT", fill="green", font=font)
        
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
            draw.ellipse([s1_x - radius, s1_y - radius, s1_x + radius, s1_y + radius], 
                        fill="blue", outline="white", width=2)
            
            # S1 라벨 추가
            try:
                font = ImageFont.truetype("arial.ttf", 14)
            except IOError:
                font = ImageFont.load_default()
            
            draw.text((s1_x + 12, s1_y - 10), "S1", fill="blue", font=font)
        
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
                
                # crop 번호 표시
                crop_id = crop.get("id", "?")
                try:
                    font = ImageFont.truetype("arial.ttf", 16)
                except IOError:
                    font = ImageFont.load_default()
                
                text = f"C{crop_id}"
                text_x, text_y = scaled_left + 5, scaled_top + 5
                draw.text((text_x, text_y), text, fill="red", font=font)
        
        # 4. 결과 저장
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "s1_attention_crops.png")
        blended_img.save(save_path)
        # print(f"🌄 Stage 1 visualization : {save_path}")
    
    else:
        print("⚠️ No attention scores found for visualization")


def visualize_stage2_merged_attention(s2_pred, merged_img, save_dir, instruction, predicted_point=None):
    """Stage 2 합쳐진 이미지에 attention 맵과 예측 점을 시각화"""
    
    # 1. Attention 맵 생성
    if 'attn_scores' in s2_pred and s2_pred['attn_scores']:
        attn_scores = np.array(s2_pred['attn_scores'][0])
        n_width = s2_pred['n_width']
        n_height = s2_pred['n_height']
        
        # 합쳐진 이미지에 attention 맵 오버레이
        blended_img = get_attn_map(
            image=merged_img,
            attn_scores=attn_scores,
            n_width=n_width,
            n_height=n_height
        )
        
        # 2. 예측 점에 별 표시
        if predicted_point and s2_pred.get("topk_points"):
            # 정규화된 좌표를 픽셀 좌표로 변환
            top_point_normalized = s2_pred["topk_points"][0]
            
            draw = ImageDraw.Draw(blended_img)
            
            # 별 그리기
            img_w, img_h = merged_img.size
            star_x = int(top_point_normalized[0] * img_w)
            star_y = int(top_point_normalized[1] * img_h)
            
            # 별 모양 그리기 (간단한 X 모양)
            star_size = 20
            draw.line([star_x - star_size, star_y - star_size, star_x + star_size, star_y + star_size], 
                     fill="yellow", width=5)
            draw.line([star_x - star_size, star_y + star_size, star_x + star_size, star_y - star_size], 
                     fill="yellow", width=5)
            draw.line([star_x, star_y - star_size, star_x, star_y + star_size], 
                     fill="yellow", width=5)
            draw.line([star_x - star_size, star_y, star_x + star_size, star_y], 
                     fill="yellow", width=5)
            
            # 중심점 표시
            draw.ellipse([star_x - 5, star_y - 5, star_x + 5, star_y + 5], 
                        fill="red", outline="black", width=2)
        
        # 4. Top-5 attention 점수 표시한 결과 저장
        os.makedirs(save_dir, exist_ok=True)
        attn_scores_np = np.array(s2_pred['attn_scores'][0])
        img_with_attns = draw_top_patch_attentions(
            blended_img.copy(),
            attn_scores_np,
            n_width,
            n_height,
            top_k=5
        )
        
        save_path = os.path.join(save_dir, "s2_merged_attention.png")
        img_with_attns.save(save_path)
        # print(f"🌄 Stage 2 visualization : {save_path}")
    
    else:
        print("⚠️ No attention scores found for Stage 2 visualization")


def visualize_stage3_ensemble_attention(ensemble_map, original_image, crop_list, original_bbox, 
                                       s3_ensemble_point, s2_corrected_point, s1_original_point,
                                       stage1_ratio, stage2_ratio, save_dir, vis_only_wrong=False, 
                                       stage3_success=True):
    """Stage 3 앙상블 어텐션 시각화 - 설명 박스가 이미지를 가리지 않도록 여백 추가"""
    
    if s3_ensemble_point is None or not save_dir:
        return
        
    if vis_only_wrong and stage3_success:
        return
    
    # 원본 이미지 크기 계산
    orig_w, orig_h = original_image.size

    # 이미지 크기에 비례한 폰트 크기 계산 (최소 8, 최대 24)
    base_font_size = max(8, min(40, int(orig_w / 40)))  # 40픽셀당 1pt
    title_font_size = base_font_size
    label_font_size = base_font_size//3*2
    
    # 여백 크기 계산 (이미지 높이의 15%)
    margin_height = int(orig_h * 0.15)
    legend_height = int(orig_h * 0.12)
    colorbar_width = int(orig_w * 0.12)  # 컬러바를 위한 우측 여백
    
    # 확장된 캔버스 크기 (우측에 컬러바 여백 추가)
    canvas_w = orig_w + colorbar_width
    canvas_h = orig_h + margin_height + legend_height
    
    # matplotlib 설정
    fig_width = canvas_w / 100  # DPI 100 기준
    fig_height = canvas_h / 100
    
    plt.figure(figsize=(fig_width, fig_height), dpi=100)
    
    # 이미지 영역과 여백 영역 분리 (우측 여백 고려)
    image_width_ratio = orig_w / canvas_w
    ax_main = plt.axes([0, legend_height/canvas_h, image_width_ratio, orig_h/canvas_h])
    ax_legend = plt.axes([0, 0, image_width_ratio, legend_height/canvas_h])
    
    # 메인 이미지 영역
    ax_main.imshow(original_image, alpha=0.8)
    ax_main.imshow(ensemble_map, cmap='hot', alpha=0.6, interpolation='bilinear')
    
    # 크롭 박스들 표시
    for i, crop in enumerate(crop_list):
        bbox = crop['bbox']
        from matplotlib.patches import Rectangle
        rect = Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], 
                        fill=False, edgecolor='yellow', linewidth=2, alpha=0.8)
        ax_main.add_patch(rect)
        
        # 크롭 라벨을 박스 내부 좌상단에 배치 (폰트 크기 조정)
        label_x = bbox[0] + 5
        label_y = bbox[1] + 15
        ax_main.text(label_x, label_y, f'C{i+1}', color='yellow', fontsize=label_font_size, 
                    weight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
    
    # GT 박스 표시
    gt_rect = Rectangle((original_bbox[0], original_bbox[1]), 
                       original_bbox[2] - original_bbox[0], 
                       original_bbox[3] - original_bbox[1],
                       linewidth=3, edgecolor='lime', facecolor='none')
    ax_main.add_patch(gt_rect)
    
    # GT 라벨 (폰트 크기 조정)
    gt_label_x = original_bbox[0] + 5
    gt_label_y = original_bbox[1] + 15
    ax_main.text(gt_label_x, gt_label_y, 'GT', color='lime', fontsize=label_font_size, 
                weight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
    
    # 예측점들 표시 (마커 크기도 이미지 크기에 비례)
    marker_size = max(8, min(16, int(orig_w / 120)))  # 120픽셀당 1pt
    ax_main.plot(s3_ensemble_point[0], s3_ensemble_point[1], 'bo', markersize=marker_size + 2, 
                markeredgecolor='white', markeredgewidth=3, alpha=0.9)
    ax_main.plot(s2_corrected_point[0], s2_corrected_point[1], 'ro', markersize=marker_size, 
                markeredgecolor='white', markeredgewidth=2, alpha=0.9)
    
    if s1_original_point is not None:
        ax_main.plot(s1_original_point[0], s1_original_point[1], 'go', markersize=marker_size, 
                    markeredgecolor='white', markeredgewidth=2, alpha=0.9)
    
    ax_main.set_xlim(0, orig_w)
    ax_main.set_ylim(orig_h, 0)  # Y축 뒤집기
    ax_main.axis('off')
    
    # 범례 영역 (하단 여백) - 폰트 크기 조정
    ax_legend.axis('off')
    
    # 제목 (폰트 크기 조정)
    title_text = f'Stage3 Ensemble Attention (S1:{stage1_ratio:.2f}, S2:{stage2_ratio:.2f})'
    ax_legend.text(0.5, 0.85, title_text, ha='center', va='top', fontsize=title_font_size, weight='bold',
                  transform=ax_legend.transAxes)
    
    # 범례 정보
    legend_items = [
        ('●', 'green', 'S1 Original Prediction'),
        ('●', 'red', 'S2 Original Prediction'),
        ('●', 'blue', 'Ensemble Prediction'),
        ('■', 'yellow', 'Generated Crops'),
        ('■', 'lime', 'Ground Truth')
    ]
    
    # 범례를 2열로 배치
    items_per_row = 3
    for i, (symbol, color, label) in enumerate(legend_items):
        row = i // items_per_row
        col = i % items_per_row
        
        x_pos = 0.1 + col * 0.3
        y_pos = 0.5 - row * 0.25
        
        ax_legend.text(x_pos, y_pos, symbol, color=color, fontsize=base_font_size/3*2, weight='bold',
                      transform=ax_legend.transAxes, ha='left', va='center')
        ax_legend.text(x_pos + 0.03, y_pos, label, fontsize=base_font_size//3*2,
                      transform=ax_legend.transAxes, ha='left', va='center')
    
    # 컬러바를 우측에 추가
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    
    norm = Normalize(vmin=ensemble_map.min(), vmax=ensemble_map.max())
    sm = ScalarMappable(norm=norm, cmap='hot')
    sm.set_array([])
    
    # 컬러바 위치 조정 (메인 이미지 우측)
    cbar_ax = plt.axes([0.92, legend_height/canvas_h + 0.1, 0.02, (orig_h/canvas_h) * 0.8])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Ensemble Attention Score', fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    # 저장
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 's3_ensemble_attention.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    # print(f"🌄 Stage 3 ensemble visualization : {save_path}")