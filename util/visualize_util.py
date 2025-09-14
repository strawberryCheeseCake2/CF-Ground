import os
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import torch
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# matplotlib 폰트 설정 (이모지 및 한글 지원)
try:
    # 시스템에서 사용 가능한 폰트 찾기
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 한글 및 이모지 지원 폰트 우선순위
    preferred_fonts = ['Noto Color Emoji', 'Apple Color Emoji', 'Segoe UI Emoji', 
                      'DejaVu Sans', 'Arial Unicode MS', 'Noto Sans CJK JP', 'Malgun Gothic']
    
    selected_font = None
    for font in preferred_fonts:
        if font in available_fonts:
            selected_font = font
            break
    
    if selected_font:
        plt.rcParams['font.family'] = selected_font
    
    # 마이너스 기호 깨짐 방지
    plt.rcParams['axes.unicode_minus'] = False
    
except:
    # 폰트 설정 실패 시 기본 설정 유지
    pass

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
        
        # 2. Top-K 예측점들을 모두 시각화
        if s2_pred.get("topk_points"):
            topk_points = s2_pred["topk_points"]
            img_w, img_h = merged_img.size
            
            draw = ImageDraw.Draw(blended_img)
            
            # 색상 설정 (상위 점수순으로 색상 진하게)
            colors = [
                "yellow",    # 1등 - 가장 밝은 노란색
                "gold",      # 2등
                "orange",    # 3등
                "red",       # 4등
                "magenta",   # 5등
                "purple",    # 6등
                "blue",      # 7등
                "cyan",      # 8등
                "lime",      # 9등
                "white"      # 10등
            ]
            
            # 각 예측점을 순위에 따라 다른 색상과 크기로 표시
            for i, point_normalized in enumerate(topk_points[:10]):  # 최대 10개
                star_x = int(point_normalized[0] * img_w)
                star_y = int(point_normalized[1] * img_h)
                
                # 순위에 따른 크기 조정 (1등이 가장 크게)
                star_size = max(8, 50 - i * 3)  # 1등: 50px, 10등: 20px
                line_width = max(2, 5 - i // 2)  # 선 두께도 순위에 따라 조정
                
                color = colors[i] if i < len(colors) else "gray"
                
                # 별 모양 그리기
                draw.line([star_x - star_size, star_y - star_size, star_x + star_size, star_y + star_size], 
                         fill=color, width=line_width)
                draw.line([star_x - star_size, star_y + star_size, star_x + star_size, star_y - star_size], 
                         fill=color, width=line_width)
                draw.line([star_x, star_y - star_size, star_x, star_y + star_size], 
                         fill=color, width=line_width)
                draw.line([star_x - star_size, star_y, star_x + star_size, star_y], 
                         fill=color, width=line_width)
                
                # 중심점 표시 (순위에 따라 크기 조정)
                center_radius = max(2, 5 - i // 3)
                draw.ellipse([star_x - center_radius, star_y - center_radius, 
                            star_x + center_radius, star_y + center_radius], 
                            fill="black", outline="white", width=1)
                
                # 순위 번호 표시 (1~3등만)
                if i < 3:
                    try:
                        font = ImageFont.truetype("arial.ttf", 6)
                    except IOError:
                        font = ImageFont.load_default()
                    
                    rank_text = str(i + 1)
                    # 텍스트 배경 박스
                    text_bbox = draw.textbbox((0, 0), rank_text, font=font)
                    text_w = text_bbox[2] - text_bbox[0]
                    text_h = text_bbox[3] - text_bbox[1]
                    
                    text_x = star_x + star_size + 5
                    text_y = star_y - text_h // 2
                    
                    # 배경 박스
                    draw.rectangle([text_x - 1, text_y - 1, text_x + text_w + 1, text_y + text_h + 1], 
                                 fill="black", outline="white", width=1)
                    # 텍스트
                    draw.text((text_x, text_y), rank_text, fill=color, font=font)
        
        # 3. 범례 추가 (이미지 하단에)
        if s2_pred.get("topk_points"):
            try:
                legend_font = ImageFont.truetype("arial.ttf", 6)
            except IOError:
                legend_font = ImageFont.load_default()
            
            # 범례 텍스트
            legend_text = "★ Top-10 Predictions (1st=Yellow, 2nd=Gold, 3rd=Orange...)"
            legend_bbox = draw.textbbox((0, 0), legend_text, font=legend_font)
            legend_w = legend_bbox[2] - legend_bbox[0]
            legend_h = legend_bbox[3] - legend_bbox[1]
            
            # 이미지 하단 중앙에 범례 배치
            legend_x = (img_w - legend_w) // 2
            legend_y = img_h - legend_h - 10
            
            # 배경 박스
            draw.rectangle([legend_x - 5, legend_y - 3, legend_x + legend_w + 5, legend_y + legend_h + 3], 
                         fill="black", outline="white", width=1)
            # 범례 텍스트
            draw.text((legend_x, legend_y), legend_text, fill="white", font=legend_font)
        
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


def visualize_stage2_individual_crops(all_crop_results, save_dir, instruction):
    """개별 crop inference 결과들을 시각화 (각 crop별로 attention map과 예측점 표시)"""
    
    if not all_crop_results or not save_dir:
        return
    
    # 가장 높은 점수를 가진 crop 찾기
    best_result = max(all_crop_results, key=lambda x: x['max_attn_score'])
    
    for i, result in enumerate(all_crop_results):
        crop_img = result['crop_img']
        s2_pred = result['pred']
        crop_point = result['crop_point']
        is_best = (result == best_result)
        
        # Attention 맵 생성
        if 'attn_scores' in s2_pred and s2_pred['attn_scores']:
            attn_scores = np.array(s2_pred['attn_scores'][0])
            n_width = s2_pred['n_width']
            n_height = s2_pred['n_height']
            
            # crop 이미지에 attention 맵 오버레이
            blended_img = get_attn_map(
                image=crop_img,
                attn_scores=attn_scores,
                n_width=n_width,
                n_height=n_height
            )
            
            # 예측점 표시
            if crop_point:
                draw = ImageDraw.Draw(blended_img)
                
                star_x = int(crop_point[0])
                star_y = int(crop_point[1])
                
                # 별 모양 그리기
                star_size = 15
                color = "gold" if is_best else "yellow"
                draw.line([star_x - star_size, star_y - star_size, star_x + star_size, star_y + star_size], 
                         fill=color, width=4)
                draw.line([star_x - star_size, star_y + star_size, star_x + star_size, star_y - star_size], 
                         fill=color, width=4)
                draw.line([star_x, star_y - star_size, star_x, star_y + star_size], 
                         fill=color, width=4)
                draw.line([star_x - star_size, star_y, star_x + star_size, star_y], 
                         fill=color, width=4)
                
                # 중심점 표시
                draw.ellipse([star_x - 4, star_y - 4, star_x + 4, star_y + 4], 
                            fill="red", outline="black", width=2)
                
                # Best crop 표시
                if is_best:
                    try:
                        font = ImageFont.truetype("arial.ttf", 16)
                    except IOError:
                        font = ImageFont.load_default()
                    
                    draw.text((10, 10), "BEST", fill="gold", font=font,
                             stroke_width=2, stroke_fill="black")
                
                # Attention score 표시
                try:
                    font = ImageFont.truetype("arial.ttf", 12)
                except IOError:
                    font = ImageFont.load_default()
                
                score_text = f"Score: {result['max_attn_score']:.3f}"
                draw.text((10, crop_img.height - 25), score_text, fill="white", font=font,
                         stroke_width=1, stroke_fill="black")
            
            # 저장
            os.makedirs(save_dir, exist_ok=True)
            filename = f"s2_individual_crop_{i}_{'best' if is_best else 'other'}.png"
            save_path = os.path.join(save_dir, filename)
            blended_img.save(save_path)
    
    # print(f"🌄 Stage 2 individual crops visualization saved")


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

    # 이미지 크기에 비례한 폰트 크기 계산 (최소 8, 최대 24)
    base_font_size = max(8, min(30, int(orig_w / 50)))  # 50픽셀당 1pt
    title_font_size = base_font_size
    label_font_size = base_font_size
    
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
        
        # GT 라벨 추가
        gt_label_x = original_bbox[0] + 5
        gt_label_y = original_bbox[1] - 20
        ax_main.text(gt_label_x, gt_label_y, 'GT', color='green', fontsize=label_font_size, 
                    weight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
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
            # 1등 점수 표시 (배경 추가하여 잘 보이게)
            ax_main.text(x+30, y-16, f'{score:.3f}', fontsize=label_font_size//2, 
                        color='red', weight='bold',
                        bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7))
        
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
            # 순위와 점수 함께 표시 (좌표도 2배로)
            ax_main.text(x+16, y-10, f'{i+2}({score:.3f})', fontsize=label_font_size//3, 
                        color='black', weight='bold', 
                        bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7))
    
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
    
    # 제목 및 정보 텍스트
    title_text = f"Stage 3 Point Ensemble (Success: {stage3_success})"
    ax_legend.text(0.5, 0.85, title_text, fontsize=title_font_size, weight='bold', 
                  ha='center', va='center', transform=ax_legend.transAxes)
    
    # 앙상블 정보
    info_text = f"Ensemble Ratio - Stage1: {stage1_ratio:.2f}, Stage2: {stage2_ratio:.2f}"
    ax_legend.text(0.5, 0.65, info_text, fontsize=label_font_size, 
                  ha='center', va='center', transform=ax_legend.transAxes)
    
    # 색상 및 범례 정보 (시각적 마커와 함께 표시)
    # 범례 영역에 실제 마커들을 그려서 표시
    legend_items = [
        ('S1', 'limegreen', 'o'),
        ('S2', 'blue', 'o'), 
        ('S3', 'red', '*'),
        ('S3-others', 'red', 'o'), 
        ('GT', 'green', 's'),  # GT 박스 추가
        ('Crops', 'yellow', 's')
    ]
    
    # 범례 마커들을 가로로 배치
    x_start = 0.1
    x_spacing = 0.15
    y_pos = 0.3
    
    for i, (label, color, marker) in enumerate(legend_items):
        x_pos = x_start + i * x_spacing
        
        # 마커 그리기
        ax_legend.scatter(x_pos, y_pos, c=color, s=300, marker=marker, 
                         alpha=0.9, edgecolors='black', linewidth=1,
                         transform=ax_legend.transAxes)
        
        # 라벨 텍스트
        ax_legend.text(x_pos + 0.03, y_pos, label, fontsize=label_font_size-4,
                      ha='left', va='center', transform=ax_legend.transAxes)
    
    # 저장
    success_str = "success" if stage3_success else "failure"
    filename = f"s3_point_ensemble_{success_str}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=100, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    
    # print(f"💾 Stage3 point ensemble visualization saved: {filepath}")


def visualize_stage3_ensemble_attention(ensemble_map, original_image, crop_list, original_bbox, 
                                       s3_ensemble_point, s2_corrected_point, s1_original_point,
                                       stage1_ratio, stage2_ratio, save_dir, vis_only_wrong=False, 
                                       stage3_success=True):
    """Stage 3 앙상블 어텐션 시각화 - 설명 박스가 이미지를 가리지 않도록 여백 추가 (기존 방식용)"""
    
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