'''
crop2.py는 crop 1개짜리가 많음
-> 오히려 그냥 1개짜리 crop이 top-q로 잃는게 없고, GUI Actor의 정확도가 워낙 높아서 좋아진거일 수 있음
그리고 crop이 실패하면 나중에 Peak Memory가 증가함.
그래서 오래걸리더라도 crop 실패가 적도록 반복문을 바꿈
'''

from utils_dcgen import ImgSegmentation
from PIL import Image, ImageDraw

import os
from time import time 
import json

# TODO: pip uninstall pillow → pip install pillow-simd로 교체 설치시 resize 빨라진다고 함.

#! Hyper Parameter
# 수직 최소 분할 비율: 폭이 너무 좁은 조각(=좌우로 얇음)은 이웃과 병합
Y_MIN_RATIO = 0.20   # 전체 너비의 20% 미만이면 병합

# 수평 최소 분할 비율: 높이가 너무 낮은 조각(=상하로 얇음)은 이웃과 병합
X_MIN_RATIO = 0.1   # 전체 높이의 10% 미만이면 병합  # TODO: crop이 잘게 10개 넘게씩이 좋을까 아니면 그냥 뭉치게가 좋을까 0.05 / 0.1

# 리사이즈 비율
RESIZE_RATIO_1 = 0.1   # 1차 coarse segmentation


#! 1차 분할 파라미터만 사용
FIRST_PASS = dict(
    max_depth=1,      # 트리 분할 최대 깊이 (값 ↑ → 더 많이 잘라서 세부적으로 분할)
    var_thresh=150,   # 픽셀 분산 기준 (값 ↑ → 단색/균일 구간도 "내용 있음"으로 인식 → 분할 줄어듦)
    diff_thresh=20,   # 행간 픽셀 차이 허용치 (값 ↑ → 작은 경계 무시, 오직 큰 차이만 분리 → 분할 줄어듦)
    diff_portion=0.7, # 차이가 일정 비율 이상일 때만 경계 인정 (값 ↑ → 더 강한 변화 필요 → 분할 줄어듦)
    window_size=50    # 슬라이딩 윈도우 높이 (값 ↑ → 큰 구간 단위로 평균내서 안정적 경계 탐지, 작은 변화 무시됨)
)


#! ---------------------------- Util ----------------------------

def bbox_area(b):
    l, t, r, btm = b
    return max(0, r - l) * max(0, btm - t)

def bbox_w(b):
    return max(0, b[2] - b[0])

def bbox_h(b):
    return max(0, b[3] - b[1])

def overlap_1d(a1, a2, b1, b2):
    """ [a1,a2]와 [b1,b2] 구간의 겹침 길이 """
    return max(0, min(a2, b2) - max(a1, b1))

def vertical_overlap(a, b):
    """ 두 bbox의 세로 방향 겹침 비율 (작은 쪽 기준) """
    ov = overlap_1d(a[1], a[3], b[1], b[3])
    denom = max(1, min(bbox_h(a), bbox_h(b)))
    return ov / denom

def horizontal_overlap(a, b):
    """ 두 bbox의 가로 방향 겹침 비율 (작은 쪽 기준) """
    ov = overlap_1d(a[0], a[2], b[0], b[2])
    denom = max(1, min(bbox_w(a), bbox_w(b)))
    return ov / denom

def merge_two(a, b):
    """ 두 bbox의 외접 사각형으로 병합 """
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))

def collect_leaves_from_tree(tree_dict, base_level=0):
    """
    to_json_tree() 결과(dict)에서 리프 노드들만 (bbox, level)로 수집.
    level은 base_level 기준으로 +1씩 내려감.
    """
    out = []
    def _rec(node, level):
        ch = node.get("children", [])
        if not ch:
            out.append((tuple(node["bbox"]), level))
            return
        for c in ch:
            _rec(c, level + 1)
    _rec(tree_dict, base_level)
    return out

def merge_until_stable(leaves, parent_size, min_w_ratio, min_h_ratio):
    """
    인접(pairwise) 병합만 반복 수행.
    - 바로 좌/우(또는 상/하) '이웃' 중에서만 합침
    - 우선순위: merge_count 적은 쪽 → 동률이면 더 짧은 쪽(폭/높이)
    - 임계(min_w_ratio, min_h_ratio) 이상만 남을 때까지 반복
    """
    W, H = parent_size
    cur = [(tuple(b), lvl, 0) for (b, lvl) in leaves]  # (bbox, level, merge_count)

    def _center(b):
        return ((b[0]+b[2])*0.5, (b[1]+b[3])*0.5)

    def _adjacent_pass(cur, span, min_ratio, axis):  # axis: 'x' or 'y'
        if span <= 0 or min_ratio <= 0 or len(cur) <= 1:
            return cur, False

        # 정렬: x축이면 x중심, y축이면 y중심
        order = sorted(range(len(cur)), key=lambda i: _center(cur[i][0])[0 if axis=='x' else 1])
        used  = [False]*len(order)
        out   = []
        changed = False

        for p, i in enumerate(order):
            if used[p]:
                continue
            b, lvl, cnt = cur[i]
            w, h = bbox_w(b), bbox_h(b)
            size = w if axis=='x' else h
            need_merge = (size / span) < min_ratio

            if need_merge:
                # 바로 왼쪽/오른쪽(또는 위/아래) '사용 안 된' 이웃만 후보
                left_p = p-1
                while left_p >= 0 and used[left_p]:
                    left_p -= 1
                right_p = p+1
                while right_p < len(order) and used[right_p]:
                    right_p += 1

                candidates = []
                if left_p >= 0:
                    j = order[left_p]
                    b2, lvl2, cnt2 = cur[j]
                    # 동률이면 '더 짧은 쪽' 우선: x축 병합이면 폭, y축 병합이면 높이
                    short_metric = bbox_w(b2) if axis=='x' else bbox_h(b2)
                    candidates.append((cnt2, short_metric, left_p, j, b2, lvl2))
                if right_p < len(order):
                    j = order[right_p]
                    b2, lvl2, cnt2 = cur[j]
                    short_metric = bbox_w(b2) if axis=='x' else bbox_h(b2)
                    candidates.append((cnt2, short_metric, right_p, j, b2, lvl2))

                if candidates:
                    # merge_count 적은 쪽 → 동률이면 더 짧은 쪽
                    candidates.sort(key=lambda x: (x[0], x[1]))
                    _, _, picked_p, j, b2, lvl2 = candidates[0]
                    new_bbox = merge_two(b, b2)
                    new_cnt  = max(cnt, cur[j][2]) + 1
                    out.append((new_bbox, max(lvl, lvl2), new_cnt))
                    used[p] = True
                    used[picked_p] = True
                    changed = True
                    continue  # 이 i는 처리 끝

            # 병합 불필요 또는 이웃 없음 → 그대로 유지
            used[p] = True
            out.append((b, lvl, cnt))

        return out, changed

        # 끝 _adjacent_pass

    # 임계 만족할 때까지 x→y 순으로 반복
    while True:
        changed_any = False
        cur, ch1 = _adjacent_pass(cur, W, min_w_ratio, axis='x')
        changed_any |= ch1
        cur, ch2 = _adjacent_pass(cur, H, min_h_ratio, axis='y')
        changed_any |= ch2

        # 더 이상 합칠 게 없으면 종료
        if not changed_any:
            break

        # 혹시 아직도 임계 미달이 남았는지 확인해서 남아있으면 다음 루프로
        small_exists = False
        for b, _, _ in cur:
            if (W > 0 and (bbox_w(b) / W) < min_w_ratio) or (H > 0 and (bbox_h(b) / H) < min_h_ratio):
                small_exists = True
                break
        if not small_exists:
            break

    # (bbox, level)로 반환
    return [(b, lvl) for (b, lvl, _) in cur]


#! ================================================================================================


def crop(image_path,
         output_json_path,
         output_image_path,
         save_visualization=True):
    """
    crop5.run_segmentation_recursive 와 동일한 return 포맷을 반환.
    - 썸네일(dict): {"img": PIL.Image, "id":0, "bbox":[0,0,W,H]}
    - 크롭(dict): {"img": PIL.Image, "id":k, "bbox":[l,t,r,b], "recursion_depth":0, "fail":False, "filename": None}

    출력(파일 저장/시각화)은 기존 동작을 그대로 유지.
    """

    start = time()

    # 0) 원본/작업 이미지 로드 및 리사이즈
    orig_img_full = Image.open(image_path).convert("RGB")
    orig_w, orig_h = orig_img_full.size
    resized_w1, resized_h1 = int(orig_w * RESIZE_RATIO_1), int(orig_h * RESIZE_RATIO_1)
    resized_w1 = max(1, resized_w1)
    resized_h1 = max(1, resized_h1)
    work_img = orig_img_full.resize((resized_w1, resized_h1))

    abs_W1 = orig_w * RESIZE_RATIO_1
    abs_H1 = orig_h * RESIZE_RATIO_1

    time0 = time()
    print(f"[Crop] [0] {time0 - start:.3f}s", end = " | ")

    # 1-2) 1차 분할
    img_seg = ImgSegmentation(
        img=work_img,
        max_depth=FIRST_PASS["max_depth"],
        var_thresh=FIRST_PASS["var_thresh"],
        diff_thresh=FIRST_PASS["diff_thresh"],
        diff_portion=FIRST_PASS["diff_portion"],
        window_size=FIRST_PASS["window_size"]
    )

    # 트리에서 리프만 수집(level 기준: 루트 0, 리프는 1)
    tree = img_seg.to_json_tree()
    leaves_lvl1 = collect_leaves_from_tree(tree, base_level=0)

    time1 = time()
    print(f"[1] {time1 - time0:.3f}s", end = " | ")


    # 2) 제로-드롭 병합 보정(너무 얇은/낮은 조각은 이웃과 병합)
    leaves_lvl1_merged = merge_until_stable(
        leaves=leaves_lvl1,
        parent_size=(abs_W1, abs_H1),  # ← 부모 bbox 말고, 원본×리사이즈 상수
        min_w_ratio=Y_MIN_RATIO,
        min_h_ratio=X_MIN_RATIO
    )

    time2 = time()
    print(f"[2] {time2 - time1:.3f}s", end = " | ")

    # 3) 결과 JSON 저장(리스트 평면 구조: {"bbox":[l,t,r,b], "level":k})
    final_items = [(b_work, max(lvl, 1)) for (b_work, lvl) in leaves_lvl1_merged]
    json_out = [{"bbox": [int(b[0]), int(b[1]), int(b[2]), int(b[3])], "level": int(lvl)} for (b, lvl) in final_items]

    with open(output_json_path, "w") as f:
        json.dump(json_out, f, indent=2)

    # === 반환 리스트(grounding 호환 포맷) 구성 ===
    results_for_grounding = []
    # 0번 썸네일
    results_for_grounding.append({
        "img": orig_img_full.copy(),
        "id": 0,
        "bbox": [0, 0, orig_w, orig_h]
    })

    # work_img → original 스케일팩터
    sx = orig_w / float(resized_w1)
    sy = orig_h / float(resized_h1)

    # k번 크롭들 (원본 좌표계 bbox + 이미지를 메모리에서 잘라서 포함)
    k = 1
    for item in json_out:
        bbox = item.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        l, t, r, b = bbox
        # work_img → 원본 좌표로 스케일백
        L = int(round(l * sx)); T = int(round(t * sy))
        R = int(round(r * sx)); B = int(round(b * sy))
        # 유효성 체크
        L = max(0, min(orig_w, L)); R = max(0, min(orig_w, R))
        T = max(0, min(orig_h, T)); B = max(0, min(orig_h, B))
        if R <= L or B <= T:
            continue
        crop_img = orig_img_full.crop((L, T, R, B))
        results_for_grounding.append({
            "img": crop_img,
            "id": k,
            "bbox": [L, T, R, B],
            "recursion_depth": 0,
            "fail": False,
            "filename": None
        })
        k += 1

    end = time()

    print(f"[3] {end - time2:.3f}s", end = " | ")
    
    print(f"🕖 Total Time : {end - start:.3f}s", end = " | ")

    print(f"✂️ crops : {len(final_items)}", end = " | ")

    if save_visualization==False:
        print()
        return results_for_grounding
    
    #! ---------------------------- 시각화(원본 크기) ----------------------------

    # 시각화는 기존 방식 유지
    if json_out:
        if save_visualization:
            orig_img = orig_img_full.copy()
            draw = ImageDraw.Draw(orig_img)

            palette = {
                0: (255, 0, 0),
                1: (0, 255, 0),
                2: (0, 0, 255),
                3: (255, 165, 0),
                4: (255, 0, 255),
                5: (0, 255, 255),
            }
            line_w = max(2, int(min(orig_w, orig_h) * 0.003))

            for item in json_out:
                bbox = item.get("bbox")
                level = item.get("level", 0)
                if not bbox or len(bbox) != 4:
                    continue
                l, t, r, b = bbox
                L = int(round(l * sx)); T = int(round(t * sy))
                R = int(round(r * sx)); B = int(round(b * sy))
                color = palette.get(level % len(palette), (255, 0, 0))
                draw.rectangle([L, T, R, B], outline=color, width=line_w)

            save_path = output_image_path + f"result.png"
            orig_img.save(save_path)
            print(f"[SAVE] {save_path}")
    else:
        print("[INFO] No bbox results to visualize.")

    return results_for_grounding


#! ================================================================================================

if __name__ == '__main__':
    # 테스트용 main: 데이터/출력 경로는 main 전역으로 유지
    data_path = "./data/screenspotv2_imgs/"

    jsonlist = json.load(open("./data/screenspot_mobile_v2.json"))
    target_imgs = sorted(set(item["img_filename"] for item in jsonlist))

    for fname in target_imgs:
        os.makedirs(f"./crop_test/{fname}", exist_ok=True)
        # 테스트 실행: 저장 경로는 main에서 지정한 output_path 사용
        crop(image_path = data_path + fname,
                output_json_path = f"./crop_test/{fname}/json.json",
                output_image_path = f"./crop_test/{fname}/",
                save_visualization = True)
