import numpy as np
from PIL import Image, ImageFont
import torch
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw
import numpy as np
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


@torch.inference_mode()
def draw_point(image: Image.Image, point: tuple, radius=10, color=(255, 0, 0)) -> Image.Image:
    """
    이미지 위의 특정 좌표에 원을 그립니다.

    Args:
        image (Image.Image): PIL 이미지.
        point (tuple): (x, y) 좌표 (0~1 사이의 정규화된 값).
        radius (int): 원의 반지름.
        color (tuple): 원의 색상 (R, G, B).

    Returns:
        Image.Image: 원이 그려진 PIL 이미지.
    """
    w, h = image.size
    # 정규화된 좌표를 실제 픽셀 좌표로 변환
    abs_x, abs_y = int(point[0] * w), int(point[1] * h)

    draw = ImageDraw.Draw(image)
    # 원의 외곽선 그리기
    draw.ellipse(
        [(abs_x - radius, abs_y - radius), (abs_x + radius, abs_y + radius)],
        outline=color,
        width=3  # 선 두께
    )
    return image

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


def visualize_results(crop_list: list, prediction_results: dict, instruction: str, save_path: str):
    """
    추론 결과를 받아 어텐션 맵, Top-3 어텐션 값, Top-1 예측 지점, 그리고
    각 이미지의 어텐션 총합을 시각화합니다.
    """
    per_image_outputs = prediction_results.get('per_image', [])
    
    if not per_image_outputs:
        print("시각화할 결과가 없습니다.")
        return

    num_images = len(per_image_outputs)
    fig, axes = plt.subplots(1, num_images, figsize=(8 * num_images, 8))
    if num_images == 1:
        axes = [axes]
        
    fig.suptitle(f"Instruction: {instruction}", fontsize=16)

    for result in per_image_outputs:
        idx = result['index']
        # original_image = original_images[idx]
        original_image = crop_list[idx]['resized_img']
        
        # 1. 어텐션 맵 생성
        blended_img = get_attn_map(
            image=original_image,
            attn_scores=result['attn_scores'][0],
            n_width=result['n_width'],
            n_height=result['n_height']
        )
        
        # 2. Top-3 어텐션 값 텍스트로 그리기
        attn_scores_np = np.array(result['attn_scores'][0])
        img_with_attns = draw_top_patch_attentions(
            blended_img,
            attn_scores_np,
            result['n_width'],
            result['n_height'],
            top_k=3
        )
        
        # 3. Top-1 예측 지점 그리기
        if result['topk_points']:
            top_point = result['topk_points'][0]
            final_img = draw_point(img_with_attns, top_point, color=(0, 255, 0)) # 초록색 원
        else:
            final_img = img_with_attns

        # ⭐ 핵심 수정 부분: 어텐션 스코어 총합 계산 및 제목에 추가
        total_attention_score = np.sum(attn_scores_np)
            
        # 4. 결과 이미지 표시
        ax = axes[idx]
        ax.imshow(final_img)
        # 제목에 이미지 번호와 어텐션 총합을 함께 표시
        ax.set_title(f"Image {idx+1}\nTotal Attention: {total_attention_score:.4f}", fontsize=12)
        ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.show() # 화면에 표시
    plt.savefig(save_path) # 파일로 저장
    # print(f"결과가 '{save_path}'에 저장되었습니다.")


def get_highest_attention_patch_bbox(image_result: dict) -> list:
    """
    per_image 결과에서 어텐션 스코어가 가장 높은 패치를 찾아 
    해당 패치의 정규화된 바운딩 박스 좌표를 반환합니다.

    Args:
        image_result (dict): prediction_results['per_image'] 리스트의 단일 아이템.

    Returns:
        list: [left, top, right, bottom] 형태의 정규화된 좌표 리스트.
    """
    # 1. 입력 데이터 추출
    attn_scores = np.array(image_result['attn_scores'][0])
    n_width = image_result['n_width']
    n_height = image_result['n_height']

    # 2. 어텐션 스코어가 가장 높은 패치의 1차원 인덱스 찾기
    highest_score_idx = np.argmax(attn_scores)

    # 3. 1차원 인덱스를 2차원 패치 그리드 좌표 (patch_x, patch_y)로 변환
    # (patch_x는 가로 인덱스, patch_y는 세로 인덱스)
    patch_y = highest_score_idx // n_width
    patch_x = highest_score_idx % n_width

    # 4. 패치 그리드 좌표를 정규화된 바운딩 박스 좌표로 계산
    # 각 패치의 정규화된 너비와 높이
    patch_norm_width = 1.0 / n_width
    patch_norm_height = 1.0 / n_height
    
    # 바운딩 박스 계산
    left = patch_x * patch_norm_width
    top = patch_y * patch_norm_height
    right = (patch_x + 1) * patch_norm_width
    bottom = (patch_y + 1) * patch_norm_height
    
    return [left, top, right, bottom]