import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ! Hyper Parameter ===================================================================================

MIN_RESIZE = 0.3
MAX_RESIZE = 0.7

# ! Argument ==========================================================================================

SCREENSPOT_IMGS = "../data/screenspotv2_image"  # input image 경로
SCREENSPOT_JSON = "../data"  # input image json파일 경로
TASKS = ["mobile", "web", "desktop"]
SAVE_DIR = f"../_sharpness_test/"
# SAMPLE_RANGE = slice(None)
SAMPLE_RANGE = slice(0,20)

#! ================================================================================================

def compute_fft_sharpness(image, size=100):
    np_img = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    # print(np_img.shape)  # 디버그 출력 제거
    (h, w, _) = np_img.shape
    (cx, cy) = (int(w / 2.0), int(h / 2.0))
    fft = np.fft.fft2(gray)
    fftShift = np.fft.fftshift(fft)


    fftShift[cy - size : cy + size, cx - size : cx + size] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)
    
    magnitude = 20 * np.log(np.abs(recon))
    mean = np.mean(magnitude)

    return mean

def sigmoid(x, alpha=0.20, x0=12): #alpha=0.05 #alpha=0.20?
    return 1 / (1 + np.exp(-alpha * (x - x0) ))

def minmax_normalize(x, new_min=MIN_RESIZE, new_max=MAX_RESIZE):
    """0~1 범위의 값을 new_min~new_max 범위로 정규화하는 함수"""
    return x * (new_max - new_min) + new_min

def get_fft_blur_score(image, min_resize=MIN_RESIZE, max_resize=MAX_RESIZE):
    sharpness_score = compute_fft_sharpness(image)

    normed_sharpness_score = sigmoid(sharpness_score)

    blur_score = minmax_normalize(1 - normed_sharpness_score, new_min=min_resize, new_max=max_resize)
    return blur_score

# ! ================================================================================================

def draw_sharpness_info(image, sharpness_score, normed_sharpness_score, blur_score):
    """
    이미지에 선명도 정보를 그려서 반환하는 함수
    """
    # 이미지 복사
    result_img = image.copy()
    draw = ImageDraw.Draw(result_img)
    
    # 폰트 크기를 이미지 크기에 비례하도록 설정
    img_width, img_height = result_img.size
    font_size = max(20, min(img_width, img_height) // 30)  # 최소 20, 이미지 크기에 따라 조정
    
    try:
        # 시스템 폰트 사용 시도 (더 큰 폰트)
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
    
    # 텍스트 내용 (중요한 정보만)
    text_lines = [
        f"FFT Sharpness: {sharpness_score:.2f}",
        f"Normalized: {normed_sharpness_score:.3f}", 
        f"Blur Score: {blur_score:.3f}",
        f"Resize: {img_width}x{img_height}"
    ]
    
    # 텍스트 배경 크기 계산
    max_width = 0
    total_height = 0
    line_heights = []
    
    for line in text_lines:
        bbox = font.getbbox(line)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        max_width = max(max_width, width)
        line_heights.append(height)
        total_height += height
    
    # 여백 추가
    padding = 15
    bg_width = max_width + 2 * padding
    bg_height = total_height + len(text_lines) * 5 + 2 * padding
    
    # 중앙 하단 위치 계산
    bg_x = (img_width - bg_width) // 2
    bg_y = img_height - bg_height - 20  # 하단에서 20픽셀 위
    
    # 반투명 배경 그리기
    overlay = Image.new('RGBA', result_img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle([bg_x, bg_y, bg_x + bg_width, bg_y + bg_height], 
                          fill=(0, 0, 0, 180))  # 반투명 검은색
    
    # 원본 이미지와 오버레이 합성
    result_img = Image.alpha_composite(result_img.convert('RGBA'), overlay).convert('RGB')
    draw = ImageDraw.Draw(result_img)
    
    # 텍스트 그리기
    current_y = bg_y + padding
    for i, line in enumerate(text_lines):
        # 색상 구분
        if "FFT Sharpness" in line:
            color = "cyan"
        elif "Normalized" in line:
            color = "yellow"
        elif "Blur Score" in line:
            color = "orange"
        else:
            color = "white"
            
        draw.text((bg_x + padding, current_y), line, fill=color, font=font)
        current_y += line_heights[i] + 5
    
    return result_img

def create_side_by_side_comparison(orig_img, resize_ratio):
    """
    원본 이미지와 리사이즈된 이미지를 나란히 비교하는 이미지 생성
    """
    # 리사이즈된 이미지 생성
    new_width = int(orig_img.width * resize_ratio)
    new_height = int(orig_img.height * resize_ratio)
    resized_img = orig_img.resize((new_width, new_height), Image.LANCZOS)
    
    # 각각의 선명도 계산
    orig_sharpness = compute_fft_sharpness(orig_img)
    orig_normed = sigmoid(orig_sharpness)
    orig_blur = minmax_normalize(1 - orig_normed)
    
    resized_sharpness = compute_fft_sharpness(resized_img)
    resized_normed = sigmoid(resized_sharpness)
    resized_blur = minmax_normalize(1 - resized_normed)
    
    # 시각화된 이미지들 생성
    orig_vis = draw_sharpness_info(orig_img, orig_sharpness, orig_normed, orig_blur)
    resized_vis = draw_sharpness_info(resized_img, resized_sharpness, resized_normed, resized_blur)
    
    # 제목 추가
    draw_orig = ImageDraw.Draw(orig_vis)
    draw_resized = ImageDraw.Draw(resized_vis)
    
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
    except:
        title_font = ImageFont.load_default()
    
    # 원본에 "Original" 텍스트 추가
    draw_orig.text((10, 10), "Original (1.0)", fill="lime", font=title_font)
    
    # 리사이즈에 비율 정보 추가
    draw_resized.text((10, 10), f"Resized ({resize_ratio:.2f})", fill="lime", font=title_font)
    
    # 두 이미지를 나란히 배치
    max_height = max(orig_vis.height, resized_vis.height)
    total_width = orig_vis.width + resized_vis.width + 20  # 20픽셀 간격
    
    combined_img = Image.new('RGB', (total_width, max_height), 'black')
    
    # 원본 이미지 배치 (왼쪽)
    y_offset_orig = (max_height - orig_vis.height) // 2
    combined_img.paste(orig_vis, (0, y_offset_orig))
    
    # 리사이즈된 이미지 배치 (오른쪽)
    y_offset_resized = (max_height - resized_vis.height) // 2
    combined_img.paste(resized_vis, (orig_vis.width + 20, y_offset_resized))
    
    return combined_img, {
        'orig_sharpness': orig_sharpness,
        'orig_blur': orig_blur,
        'resized_sharpness': resized_sharpness,
        'resized_blur': resized_blur,
        'resize_ratio': resize_ratio
    }

#! ================================================================================================

if __name__ == '__main__':

    import os
    import json
    from tqdm import tqdm

    save_dir = SAVE_DIR
    
    # 처리된 이미지 파일명을 추적하는 세트
    processed_images = set()

    # Process
    for task in TASKS:
        dataset = "screenspot_" + task + "_v2.json"
        screenspot_data = json.load(open(os.path.join(SCREENSPOT_JSON, dataset), 'r'))
        screenspot_data = screenspot_data[SAMPLE_RANGE]
        
        os.makedirs(os.path.join(save_dir, task), exist_ok=True)

        print(f"Processing {task} - Num of samples: {len(screenspot_data)}", flush=True)
        
        task_processed_count = 0

        for j, item in tqdm(enumerate(screenspot_data)):
            fname = item["img_filename"]
            
            # 이미 처리된 이미지인지 확인
            if fname in processed_images:
                continue
                
            image_path = os.path.join(SCREENSPOT_IMGS, fname)
            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                continue

            orig_img = Image.open(image_path).convert("RGB")
            
            # 원본 이미지 선명도 계산
            sharpness_score = compute_fft_sharpness(orig_img)
            normed_sharpness_score = sigmoid(sharpness_score)
            blur_score = minmax_normalize(1 - normed_sharpness_score)

            # 원본과 리사이즈된 이미지 나란히 비교
            comparison_img, comparison_results = create_side_by_side_comparison(orig_img, blur_score)
            
            # 비교 이미지 저장
            save_path = os.path.join(save_dir, task, f"{os.path.splitext(fname)[0]}_comparison.png")
            comparison_img.save(save_path)
            
            # 처리된 이미지로 기록
            processed_images.add(fname)
            task_processed_count += 1
            
            # 결과 로그 출력 (처음 5개만)
            if task_processed_count <= 5:
                print(f"\n{fname}:")
                print(f"  Original - Sharpness: {comparison_results['orig_sharpness']:.2f}, Blur: {comparison_results['orig_blur']:.3f}")
                print(f"  Resized ({comparison_results['resize_ratio']:.2f}) - Sharpness: {comparison_results['resized_sharpness']:.2f}, Blur: {comparison_results['resized_blur']:.3f}")

        print(f"Completed {task} processing - Processed: {task_processed_count}")
    
    print(f"\nAll processing completed!")
    print(f"Total unique images processed: {len(processed_images)}")
    print(f"Results saved in: {save_dir}")