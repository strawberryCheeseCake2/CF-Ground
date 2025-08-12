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
        print(f"Saved: {save_path}")
        results.append({
            "img": cropped,
            "id": start_id + idx,
            "bbox": item["bbox"],
            "level": level
        })

    return results

def run_segmentation(image_path: str, max_depth: int, window_size: int, 
                     output_json_path: str, output_image_path: str, start_id: int = 0):
    
    img_seg = ImgSegmentation(
      img=image_path,
      max_depth=max_depth,
      var_thresh=120,
      diff_thresh=45,
      diff_portion=0.9,
      # window_size=75
      window_size=window_size
    )

    # img_seg.display_tree(save_path=output_image_path)
    img_seg.to_json(path=output_json_path)


    crop_list = crop_and_save(image_path, output_json_path, output_image_path, start_id)
    return crop_list

# if __name__ == "__main__":
#     # 예시 사용법
#     IMAGE_PATH = "input.jpg"        # 원본 이미지 파일
#     JSON_PATH  = "bboxes.json"      # bbox 정보 JSON
#     OUTPUT_DIR = "crops"           # 잘라낸 이미지를 저장할 폴더

#     crop_and_save(IMAGE_PATH, JSON_PATH, OUTPUT_DIR)