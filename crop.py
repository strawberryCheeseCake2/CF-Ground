import os
import json
from PIL import Image

from dcgen.utils import *

def crop_and_save(image_path: str, json_path: str, output_dir: str, start_id: int):
    """
    image_path: 원본 이미지 파일 경로
    json_path: bbox 정보가 담긴 JSON 파일 경로
    output_dir: 잘라낸 이미지를 저장할 디렉터리
    """
    # 출력 디렉터리 생성
    os.makedirs(output_dir, exist_ok=True)
    # 자른 이미지와 메타데이터를 담을 리스트
    results = []

    # JSON 로드
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 원본 이미지 열기
    img = Image.open(image_path)

    # 각 bbox별로 크롭 및 저장
    for idx, item in enumerate(data):
        left, top, right, bottom = item["bbox"]
        level = item.get("level", None)

        # 영역 자르기
        
        cropped = img.crop((left, top, right, bottom))
        # if level != 0:
        #     # reduce the cropped image size by ~40% (scale to 60% of original)
        #     width, height = cropped.size
        #     new_size = (int(width * 0.6), int(height * 0.6))
        #     cropped = cropped.resize(new_size, Image.ANTIALIAS)

        # 파일명 지정 (예: crop_0_level0.png)
        if level is not None:
            filename = f"crop_{start_id + idx}_level{level}.png"
        else:
            filename = f"crop_{start_id + idx}.png"

        # 저장
        save_path = os.path.join(output_dir, filename)
        cropped.save(save_path)
        # print(f"Saved: {save_path}")
        results.append({
            "img": cropped,
            "id": start_id + idx,
            "bbox": item["bbox"],
            "level": level
        })

    return results

def run_segmentation(image_path: str, max_depth: int, window_size: int, 
                     output_json_path: str, output_image_path: str, start_id: int = 0, var_thresh: int = 120):
    """
    ImgSegmentation을 실행하여 crop을 생성합니다. 만약 crop이 하나도 생성되지 않으면
    var_thresh를 두 배씩 올려가며 재시도합니다.

    하이퍼파라미터:
    - var_thresh: 시작 threshold
    - MAX_VAR_THRESH: 재시도 시 상한
    """
    #! Segmentation 재시도 하이퍼파라미터
    MAX_VAR_THRESH = 5000  #! var_thresh 상한 (요청: 최대 3500)

    attempt = 0
    # 요청: 항상 var_thresh으로 시작, 크롭이 없을 때만 두 배로 증가
    cur_thresh = var_thresh
    last_error = None
    crop_list = None

    tried_values = []
    while True:
        tried_values.append(cur_thresh)
        try:
            img_seg = ImgSegmentation(
                img=image_path,
                max_depth=max_depth,
                var_thresh=cur_thresh,
                diff_thresh=45,
                diff_portion=0.9,
                window_size=window_size
            )

            # 동일 경로에 덮어쓰기 (최근 시도 기준)
            img_seg.to_json(path=output_json_path)
            crop_list = crop_and_save(image_path, output_json_path, output_image_path, start_id)
            
            # level 0 원본 제외 하나라도 나왔는지 확인
            if crop_list and len(crop_list) > 1:
                # print(f"Segmentation succeeded with var_thresh={cur_thresh} (attempt {attempt+1}, tried={tried_values})")
                return crop_list
            else:
                print(f"🚨 Segmentation produced no crops with var_thresh={cur_thresh}")
                # return crop_list  #! segmentation 그냥 일단 취소 (threshold 올리면서 반복할거면 주석해제 / 근데 오래걸림.)
        except Exception as e:
            last_error = e
            print(f"Segmentation error with var_thresh={cur_thresh}: {e}")

        # 다음 시도 준비 (두 배 증가)
        attempt += 1
        next_thresh = cur_thresh * 5
        if next_thresh <= MAX_VAR_THRESH:
            cur_thresh = next_thresh
        else:
            # 상한을 마지막으로 한 번 더 시도 (ex: 120→...→2048→3500)
            if cur_thresh != MAX_VAR_THRESH:
                cur_thresh = MAX_VAR_THRESH
            else:
                break

    print(f"❌ segmentation failed: No crops generated for {image_path} after tries {tried_values}")
    if last_error:
        print(f"Last error: {last_error}")
    return None

def run_segmentation_recursive(image_path: str, max_depth: int, window_size: int, 
                               output_json_path: str, output_image_path: str, start_id: int = 0, var_thresh: int = 120,
                               max_area_ratio: float = 0.20, max_recursion: int = 3):
    """
    재귀적으로 segmentation을 수행하여 crop 크기를 제한합니다.
    기존 run_segmentation 방식을 반복 사용하여 큰 crop들을 더 작은 단위로 분할합니다.

    Parameters:
    - image_path: 원본 이미지 파일 경로
    - max_depth: ImgSegmentation의 최대 깊이
    - window_size: ImgSegmentation의 윈도우 크기
    - output_json_path: bbox 정보를 저장할 JSON 파일 경로
    - output_image_path: 잘라낸 이미지를 저장할 디렉터리
    - start_id: crop ID 시작 번호
    - max_area_ratio: 원본 면적 대비 최대 허용 비율 (기본값: 0.20 = 20%)
    - max_recursion: 최대 재귀 깊이 (기본값: 3)
    """
    #! 재귀적 crop 분할을 위한 하이퍼파라미터
    MAX_AREA_RATIO = max_area_ratio        # 원본 면적 대비 최대 허용 비율
    MAX_RECURSION_DEPTH = max_recursion    # 최대 재귀 깊이 (무한 반복 방지)
    
    # 원본 이미지 정보
    original_img = Image.open(image_path)
    original_area = original_img.width * original_img.height
    
    # 초기 segmentation 실행 (내부에서 var_thresh 재시도)
    # print(f"Starting initial segmentation for {image_path}")
    
    crop_list = run_segmentation(image_path, max_depth, window_size, output_json_path, output_image_path, start_id, var_thresh)


    #! Segmentation 결과가 없을 경우 바로 리턴
    if not crop_list:
        print(f"Segmentation failed: No crops generated for {image_path}")
        return None

    # 재귀적으로 큰 crop들을 처리
    recursion_count = 0
    next_crop_id = start_id + len(crop_list)

    while recursion_count < MAX_RECURSION_DEPTH:
        # 면적이 기준을 초과하는 큰 crop들 찾기 (level==0 제외)
        large_crops = []
        for crop in crop_list:
            if crop.get("level") != 0:
                left, top, right, bottom = crop["bbox"]
                crop_area = (right - left) * (bottom - top)
                if crop_area > original_area * MAX_AREA_RATIO:
                    large_crops.append(crop)
        
        if not large_crops:
            print(f"All crops meet the size criteria. Stopping recursion at depth {recursion_count}")
            break
            
        print(f"Recursion depth {recursion_count + 1}: Found {len(large_crops)} crops exceeding {MAX_AREA_RATIO*100:.1f}% of original area")
        
        # 큰 crop들에 대해 추가 segmentation 수행
        new_crops = []
        for large_crop in large_crops:
            # 큰 crop을 임시 이미지로 저장
            left, top, right, bottom = large_crop["bbox"]
            temp_crop_img = original_img.crop((left, top, right, bottom))
            
            # 임시 파일 경로 생성
            temp_dir = os.path.join(output_image_path, f"temp_recursion_{recursion_count}")
            os.makedirs(temp_dir, exist_ok=True)
            temp_image_path = os.path.join(temp_dir, f"temp_crop_{large_crop['id']}.png")
            temp_crop_img.save(temp_image_path)
            
            # 임시 crop에 대해 segmentation 수행
            temp_json_path = os.path.join(temp_dir, f"temp_crop_{large_crop['id']}.json")
            temp_output_dir = os.path.join(temp_dir, f"sub_crops_{large_crop['id']}")
            
            # try:
            sub_crops = run_segmentation(
                temp_image_path, max_depth, window_size, 
                temp_json_path, temp_output_dir, next_crop_id
            )

            if sub_crops is not None:
            
              # 상대 좌표를 절대 좌표로 변환 + 최종 crop 생성
              for sub_crop in sub_crops:
                  if sub_crop.get("level") != 0:
                      s_left, s_top, s_right, s_bottom = sub_crop["bbox"]
                      absolute_bbox = [left + s_left, top + s_top, left + s_right, top + s_bottom]
                      sub_crop["bbox"] = absolute_bbox
                      sub_crop["parent_id"] = large_crop["id"]
                      sub_crop["recursion_depth"] = recursion_count + 1
                      
                      # 원본 이미지에서 다시 crop
                      final_cropped = original_img.crop(tuple(absolute_bbox))
                      sub_crop["img"] = final_cropped
                      
                      new_crops.append(sub_crop)
                      next_crop_id += 2
                   
                
                # 임시 파일 정리
                # try:
                #     os.remove(temp_image_path)
                # except Exception:
                #     pass
                # try:
                #     if os.path.exists(temp_json_path):
                #         os.remove(temp_json_path)
                # except Exception:
                #     pass
                    
            # except Exception as e:
            #     print(f"Warning: Failed to process large crop {large_crop['id']}: {e}")
            #     continue
        
        # 큰 crop들을 새로운 작은 crop들로 교체
        crop_list = [crop for crop in crop_list if crop not in large_crops]
        crop_list.extend(new_crops)
        
        
        recursion_count += 1
        print(f"Recursion depth {recursion_count} completed. Total crops: {len(crop_list)}")
    
    if recursion_count >= MAX_RECURSION_DEPTH:
        remaining_large = []
        for crop in crop_list:
            if crop.get("level") != 0:
                l, t, r, b = crop["bbox"]
                crop_area = (r - l) * (b - t)
                if crop_area > original_area * MAX_AREA_RATIO:
                    remaining_large.append(crop["id"])
        if remaining_large:
            print(f"Warning: Maximum recursion depth reached. {len(remaining_large)} crops still exceed size limit: {remaining_large}")
    
    print(f"Recursive segmentation completed. Final crop count: {len(crop_list)}")

    # 최종 crop_list 개수 출력
    print(f"[seg] final_count={len(crop_list)}")

    return crop_list


# if __name__ == "__main__":
#     # 예시 사용법
#     IMAGE_PATH = "input.jpg"        # 원본 이미지 파일
#     JSON_PATH  = "bboxes.json"      # bbox 정보 JSON
#     OUTPUT_DIR = "crops"           # 잘라낸 이미지를 저장할 폴더

#     crop_and_save(IMAGE_PATH, JSON_PATH, OUTPUT_DIR)