import os
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from pathlib import Path
from qwen_vl_utils import process_vision_info


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

        # 어텐션 스코어 총합 계산
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


def visualize_crop(save_dir, gt_bbox, top_q_bboxes, instruction, filename, img_path, click_point=None):
    """Visualize ground truth and selected crop on the image"""
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
        radius = 13
        draw.ellipse((click_x - radius, click_y - radius, click_x + radius, click_y + radius), outline="purple", width=3)

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    # Save the result image
    result_path = os.path.join(save_dir, filename)
    result_img.save(result_path)


def visualize_attn_map(attn_output, msgs, crop_list, attn_vis_dir, processor, agg_start=20, layer_num=31):
    """Visualize attention maps for crops"""
    image_inputs, _ = process_vision_info(msgs)

    # grid 크기 뽑아두기
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
        for li in range(agg_start, layer_num):
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
    plt.close(fig)


def upsample_att_map(att_map_low_res: np.ndarray, size):
    """
    Pillow를 이용한 bilinear 업샘플 (size=(H, W))
    입력/출력 모두 float32 유지
    """
    h, w = size
    # 안전장치: 음수/NaN 제거
    m = np.nan_to_num(att_map_low_res.astype(np.float32), nan=0.0, neginf=0.0, posinf=0.0)
    if m.size == 0:
        return np.zeros((h, w), dtype=np.float32)
    # 값 범위를 일단 0 이상으로 클램프
    m[m < 0] = 0.0
    im = Image.fromarray(m)
    im = im.resize((w, h), resample=Image.BILINEAR)
    out = np.array(im).astype(np.float32)
    # scale에 따라 값이 약간 변할 수 있어 0 이상으로 재클램프
    out[out < 0] = 0.0
    return out


def boxfilter_sum(arr: np.ndarray, r: int):
    """
    끝부분 보정 없음(neighbor-sum).
    (2r+1)x(2r+1) 창과 실제 겹치는 부분의 '합'만 계산.
    평균 아님. 가장자리는 창이 덜 겹치므로 손해보게 됨.
    """
    if r <= 0:
        return arr.astype(np.float32, copy=True)

    a = arr.astype(np.float32, copy=False)
    H, W = a.shape

    # 적분영상: 상단/좌측 0 패딩을 한 칸 추가해서 벡터화 계산 용이하게 구성
    ii = np.pad(a, ((1, 0), (1, 0)), mode='constant').cumsum(axis=0).cumsum(axis=1)

    ys = np.arange(H)[:, None]   # Hx1
    xs = np.arange(W)[None, :]   # 1xW

    y0 = np.clip(ys - r, 0, H)
    y1 = np.clip(ys + r + 1, 0, H)
    x0 = np.clip(xs - r, 0, W)
    x1 = np.clip(xs + r + 1, 0, W)

    # 적분영상 인덱스는 +1 패딩 고려해서 그대로 사용 가능
    S = ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]
    return S


def visualize_aggregated_attention(
        crop_list,
        original_image, inst_dir, gt_bbox, individual_maps_dir=None,
        neigh_radius = 20,         #! 이웃합(box filter) 반경 r → (2r+1)^2 창
        topk_points = 5,           # 상위 점 개수 (보여주기용)
        min_dist_pix = 200,        # 상위 점 사이 최소 간격 (픽셀)
        star_marker_size= 8,      # 별 크기 (1등)
        dot_marker_size = 5,      # 점 크기 (2~5등)
        text_fontsize= 7          # 점수 텍스트 폰트 크기
    ):
    """
    이웃합 기반 최대점 탐색:
    - 합성 맵 정규화 후 boxfilter_sum(neigh_radius)로 이웃합 계산
    - greedy 비최대 억제(NMS)로 상위 topk_points 좌표 선택(간격 min_dist_pix)
    - 시각화:
        • s2_result_only: 원본+합성맵+GT 박스만
        • s2_result_star: top-1은 별(*), top-k 모두는 이웃합 점수 텍스트로 표시
    - 성공 판정은 top-1 점이 GT 박스 안이면 True
    """

    os.makedirs(os.path.dirname(inst_dir + "/stage2"), exist_ok=True)
    if individual_maps_dir:
        os.makedirs(individual_maps_dir, exist_ok=True)

    # 캔버스 및 합성 맵 준비
    W, H = original_image.size
    aggregated_attention_map = np.zeros((H, W), dtype=np.float32)

    # 각 crop의 맵을 원본 좌표계로 업샘플하여 합성
    for crop in crop_list:
        if 'bbox' not in crop or 'att_avg_masked' not in crop:
            continue

        left, top, right, bottom = map(int, crop['bbox'])
        cw = max(0, right - left)
        ch = max(0, bottom - top)
        if cw == 0 or ch == 0:
            continue

        att_low = crop['att_avg_masked']
        att_up = upsample_att_map(att_low, size=(ch, cw))

        # 개별 맵 저장(옵션)
        if individual_maps_dir:
            indiv = np.zeros((H, W), dtype=np.float32)
            indiv[top:bottom, left:right] = att_up
            plt.figure(figsize=(10, 10 * H / W))
            plt.imshow(original_image, extent=(0, W, H, 0))
            plt.imshow(indiv, cmap='viridis', alpha=0.6, extent=(0, W, H, 0))
            plt.axis('off')
            ttl = f"Crop ID: {crop.get('id','?')}"
            plt.title(ttl)
            path = os.path.join(individual_maps_dir, f"individual_attn_crop_{crop.get('id','unk')}.png")
            plt.savefig(path, dpi=150, bbox_inches='tight', pad_inches=0)
            plt.close()

        aggregated_attention_map[top:bottom, left:right] += att_up

    # 이웃합 기반 상위 점 선정
    top_points = []
    scores = []  # boxfilter_sum으로 얻은 이웃합 값

    if aggregated_attention_map.max() > 0:
        normalized = aggregated_attention_map / (aggregated_attention_map.max() + 1e-8)
        smoothed = boxfilter_sum(normalized, neigh_radius)

        # greedy NMS로 상위 K개 점 선택
        sm = smoothed.copy()
        Hh, Ww = sm.shape

        for _ in range(int(topk_points)):
            idx = int(np.argmax(sm))
            vy, vx = divmod(idx, Ww)
            best_val = sm[vy, vx]
            if not np.isfinite(best_val) or best_val <= 0:
                break
            # 점 기록
            top_points.append((int(vx), int(vy)))
            scores.append(float(best_val))
            # 정사각형 억제
            y1 = max(0, vy - min_dist_pix); y2 = min(Hh - 1, vy + min_dist_pix)
            x1 = max(0, vx - min_dist_pix); x2 = min(Ww - 1, vx + min_dist_pix)
            sm[y1:y2+1, x1:x2+1] = -np.inf

    # 성공 판정: top-1 기준
    is_grounding_success = False
    if len(top_points) > 0:
        cx, cy = top_points[0]
        gl, gt, gr, gb = gt_bbox
        is_grounding_success = (gl <= cx <= gr) and (gt <= cy <= gb)
        print(f"🎯 Our Grounding: {(cx, cy)} , GT: {gt_bbox}, Neigh_sum: {scores[0]:.2f}")
    else:
        print("Aggregated attention map empty 또는 peak 없음")

    # 시각화: 공통 바탕
    fig, ax = plt.subplots(figsize=(10, 10 * H / W))
    ax.imshow(original_image, extent=(0, W, H, 0))
    ax.imshow(aggregated_attention_map, cmap='viridis', alpha=0.6, extent=(0, W, H, 0))

    # 그냥 Attention 상태만 저장 -> 가리는거 없이 보이도록.
    plt.savefig(inst_dir + "/s2_result_only.png", dpi=300, bbox_inches="tight", pad_inches=0)

    # GT 박스(초록)
    gl, gt, gr, gb = gt_bbox
    gt_rect = patches.Rectangle((gl, gt), gr - gl, gb - gt, linewidth=3, edgecolor='lime', facecolor='none')
    ax.add_patch(gt_rect)

    # 범례
    green_patch = patches.Patch(color='lime', label='Ground Truth BBox')
    star_legend = Line2D([0], [0], marker='*', color='w', label='NeighSum Top-1', 
                         markerfacecolor='yellow', markeredgecolor='black', markersize=star_marker_size)
    ax.legend([green_patch, star_legend], ['Ground Truth BBox', 'NeighSum Top-1'], loc='best')

    ax.axis('off')
    ax.set_title("Attention (aggregated) + NeighSum Peaks")
    plt.tight_layout()

    # 시각화: top-1 별표, top-2~5 검정 점, top-k 텍스트 라벨
    if len(top_points) > 0:
        # top-1 별표
        ax.plot(top_points[0][0], top_points[0][1], 'y*',
                markersize=star_marker_size, markeredgecolor='black')

        # top-2~5 검정 점
        for i in range(1, min(len(top_points), topk_points)):
            px, py = top_points[i]
            ax.plot(px, py, 'o', 
                    markersize=dot_marker_size, markerfacecolor='black', markeredgecolor='white', markeredgewidth=0.9)

        # top-k 텍스트(모두 표기: 점수만)
        for (px, py), sc in zip(top_points, scores):
            label = f"{sc:.3f}"
            ax.text(px + 10, py - 10, label,
                    fontsize=text_fontsize, color='white', ha='left', va='top',
                    path_effects=[pe.withStroke(linewidth=2, foreground='black')])

    plt.savefig(inst_dir + "/s2_result_star.png", dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    return bool(is_grounding_success)

def _visualize_early_exit_results(crop_list, pred, top_point, gt_bbox, attn_vis_dir, instruction, img_path):
    """Early Exit 시각화"""
    s1_att_vis_path = attn_vis_dir + "/output.png"
    visualize_results(crop_list, pred, instruction=instruction, save_path=s1_att_vis_path)
    
    # 임시로 빈 리스트로 처리 (Early Exit이므로 crop selection 없음)
    visualize_crop(save_dir=attn_vis_dir, gt_bbox=gt_bbox, 
                   top_q_bboxes=[], instruction=instruction, filename="ee_gt_vis.png", img_path=img_path,
                    click_point=top_point)


def _visualize_stage1_results(crop_list, pred, attn_vis_dir, instruction):
    """일반 Stage1 시각화"""
    s1_att_vis_path = attn_vis_dir + "/output.png"
    visualize_results(crop_list, pred, instruction=instruction, save_path=s1_att_vis_path)


def _visualize_stage2_results(save_dir, crop_list, pred, gt_bbox, click_point, instruction, img_path):
    """Stage 2 결과 시각화"""
    s2_att_vis_path = save_dir + "/output.png"
    visualize_results(crop_list, pred, instruction=instruction, save_path=s2_att_vis_path)
    visualize_crop(save_dir=save_dir, gt_bbox=gt_bbox, top_q_bboxes=[], 
                   instruction=instruction, filename="gt_vis.png", click_point=click_point, img_path=img_path)