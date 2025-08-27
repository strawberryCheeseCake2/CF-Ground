'''
# 2-Stage Segmentation:
#   Stage 1) 좌우 분할만 수행 → 좌우 병합 기준으로 정리
#   Stage 2) 각 좌우 세그먼트 내부에서 상하 분할 → 상하 병합 기준으로 정리
# 정책:
#   - 버림 금지(드랍 없음) 기본
#   - 너무 얇은 조각은 인접 이웃과 병합
#   - 병합 편향 방지(좌/우, 상/하 병합 횟수 균형)
#   - 최종적으로 모델 최소 변 길이(28px) 원본 좌표계에서 보장
'''

from dcgen_segmentation import ImgSegmentation
from PIL import Image, ImageDraw
import math
import os
from time import time
import json

#! Hyper Parameter
# 1단계 좌우 분할(수직 컷) 후 병합 기준
LR_MIN_W_RATIO = 0.20     # 부모 폭 대비 최소 너비 비율 기준
LR_MIN_W_PX    = 28       # 절대 최소 너비 픽셀 기준

# 2단계 상하 분할(수평 컷) 후 병합 기준
TB_MIN_H_RATIO = 0.1      # 부모 높이 대비 최소 높이 비율 기준
TB_MIN_H_PX    = 28       # 절대 최소 높이 픽셀 기준

# 겹침 임계
V_OVERLAP_THR = 0.30      # 세로 겹침(좌우 병합 시) 임계
H_OVERLAP_THR = 0.30      # 가로 겹침(상하 병합 시) 임계

# 리사이즈 비율
RESIZE_RATIO_1 = 0.10     # 작업 이미지 비율

# 세그먼트 생성 파라미터(조금 더 보수적으로 경계만 잡도록 설정 권장)
FIRST_PASS_LR = dict(     # 좌우 분할 전용 세그 파라미터
    max_depth=1,      # 트리 분할 최대 깊이 (값 ↑ → 더 많이 잘라서 세부적으로 분할)
    var_thresh=150,   # 픽셀 분산 기준 (값 ↑ → 단색/균일 구간도 "내용 있음"으로 인식 → 분할 줄어듦)
    diff_thresh=20,   # 행간 픽셀 차이 허용치 (값 ↑ → 작은 경계 무시, 오직 큰 차이만 분리 → 분할 줄어듦)
    diff_portion=0.7, # 차이가 일정 비율 이상일 때만 경계 인정 (값 ↑ → 더 강한 변화 필요 → 분할 줄어듦)
    window_size=50    # 슬라이딩 윈도우 높이 (값 ↑ → 큰 구간 단위로 평균내서 안정적 경계 탐지, 작은 변화 무시됨)
)
FIRST_PASS_TB = dict(     # 상하 분할 전용(기존과 유사)
    max_depth=1,
    var_thresh=150,
    diff_thresh=20,
    diff_portion=0.70,
    window_size=50
)

# 모델 전처리 제약
MODEL_MIN_SIDE = 28       # Qwen2-VL smart_resize 최소 변
PAD_COLOR = (255, 255, 255)  # 패딩 색

#! ---------------------------- Util ----------------------------

def bbox_area(b):
    l, t, r, btm = b
    return max(0, r - l) * max(0, btm - t)

def bbox_w(b):
    return max(0, b[2] - b[0])

def bbox_h(b):
    return max(0, b[3] - b[1])

def overlap_1d(a1, a2, b1, b2):
    return max(0, min(a2, b2) - max(a1, b1))

def vertical_overlap(a, b):
    ov = overlap_1d(a[1], a[3], b[1], b[3])
    denom = max(1, min(bbox_h(a), bbox_h(b)))
    return ov / denom

def horizontal_overlap(a, b):
    ov = overlap_1d(a[0], a[2], b[0], b[2])
    denom = max(1, min(bbox_w(a), bbox_w(b)))
    return ov / denom

def merge_two(a, b):
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))

def by_cx(b):
    return (b[0] + b[2]) * 0.5

def by_cy(b):
    return (b[1] + b[3]) * 0.5

def grow_bbox_to_min(b, img_w, img_h, min_w=MODEL_MIN_SIDE, min_h=MODEL_MIN_SIDE):
    L, T, R, B = b
    w, h = R - L, B - T

    if w < min_w:
        lack = min_w - w
        left = lack // 2
        right = lack - left
        L = max(0, L - left)
        R = min(img_w, R + right)
        if (R - L) < min_w:
            miss = min_w - (R - L)
            if L == 0 and R + miss <= img_w:
                R += miss
            elif R == img_w and L - miss >= 0:
                L -= miss

    if h < min_h:
        lack = min_h - h
        up = lack // 2
        down = lack - up
        T = max(0, T - up)
        B = min(img_h, B + down)
        if (B - T) < min_h:
            miss = min_h - (B - T)
            if T == 0 and B + miss <= img_h:
                B += miss
            elif B == img_h and T - miss >= 0:
                T -= miss

    L, T = max(0, L), max(0, T)
    R, B = min(img_w, R), min(img_h, B)
    return (L, T, R, B)


# [ADDED] 좌우 병합(폭 기준) - Repeated merging with directional counters
def merge_lr(bboxes, parent_w, v_overlap_thr=V_OVERLAP_THR,
             min_w_ratio=LR_MIN_W_RATIO, min_w_px=LR_MIN_W_PX):
    threshold = max(int(math.ceil(parent_w * min_w_ratio)), min_w_px)
    # Initialize each crop with directional merge counts
    items = [{'box': b, 'L': 0, 'R': 0} for b in sorted(bboxes, key=by_cx)]
    changed = True
    while changed:
        changed = False
        i = 0
        new_items = []
        skip = False
        while i < len(items):
            if skip:
                skip = False
                i += 1
                continue
            item = items[i]
            if bbox_w(item['box']) < threshold and len(items) > 1:
                left_available = (i > 0)
                right_available = (i < len(items) - 1)
                if left_available and right_available:
                    # Decide direction based on merge counter
                    cost_left = items[i-1]['R']
                    cost_right = items[i+1]['L']
                    if cost_left <= cost_right:
                        new_box = merge_two(items[i-1]['box'], item['box'])
                        items[i-1]['box'] = new_box
                        items[i-1]['R'] += 1
                        changed = True
                    else:
                        new_box = merge_two(item['box'], items[i+1]['box'])
                        items[i+1]['box'] = new_box
                        items[i+1]['L'] += 1
                        changed = True
                        skip = True
                elif left_available:
                    new_box = merge_two(items[i-1]['box'], item['box'])
                    items[i-1]['box'] = new_box
                    items[i-1]['R'] += 1
                    changed = True
                elif right_available:
                    new_box = merge_two(item['box'], items[i+1]['box'])
                    items[i+1]['box'] = new_box
                    items[i+1]['L'] += 1
                    changed = True
                    skip = True
                else:
                    new_items.append(item)
            else:
                new_items.append(item)
            i += 1
        items = sorted(new_items, key=lambda x: by_cx(x['box']))
    return [item['box'] for item in items]


# [ADDED] 상하 병합(높이 기준) - Repeated merging with directional counters
def merge_tb(bboxes, parent_h, h_overlap_thr=H_OVERLAP_THR,
             min_h_ratio=TB_MIN_H_RATIO, min_h_px=TB_MIN_H_PX):
    threshold = max(int(math.ceil(parent_h * min_h_ratio)), min_h_px)
    items = [{'box': b, 'U': 0, 'D': 0} for b in sorted(bboxes, key=by_cy)]
    changed = True
    while changed:
        changed = False
        i = 0
        new_items = []
        skip = False
        while i < len(items):
            if skip:
                skip = False
                i += 1
                continue
            item = items[i]
            if bbox_h(item['box']) < threshold and len(items) > 1:
                up_available = (i > 0)
                down_available = (i < len(items) - 1)
                if up_available and down_available:
                    cost_up = items[i-1]['D']
                    cost_down = items[i+1]['U']
                    if cost_up <= cost_down:
                        new_box = merge_two(items[i-1]['box'], item['box'])
                        items[i-1]['box'] = new_box
                        items[i-1]['D'] += 1
                        changed = True
                    else:
                        new_box = merge_two(item['box'], items[i+1]['box'])
                        items[i+1]['box'] = new_box
                        items[i+1]['U'] += 1
                        changed = True
                        skip = True
                elif up_available:
                    new_box = merge_two(items[i-1]['box'], item['box'])
                    items[i-1]['box'] = new_box
                    items[i-1]['D'] += 1
                    changed = True
                elif down_available:
                    new_box = merge_two(item['box'], items[i+1]['box'])
                    items[i+1]['box'] = new_box
                    items[i+1]['U'] += 1
                    changed = True
                    skip = True
                else:
                    new_items.append(item)
            else:
                new_items.append(item)
            i += 1
        items = sorted(new_items, key=lambda x: by_cy(x['box']))
    return [item['box'] for item in items]
# [ADDED] 한 방향 강제 세그 생성 유틸
def segment_once(segger, img, bbox, line_direct):
    # ImgSegmentation 내부의 cut_img_bbox를 직접 호출해서 한 방향만 분할
    cuts = segger.cut_img_bbox(img, bbox, line_direct=line_direct, verbose=False, save_cut=False)
    return cuts if cuts else []

#! ================================================================================================

def crop_img(image_path, output_image_path=None, save_visualization=False, print_latency=False, skip_vertical_split=False):
    start = time()

    # 0) 원본/작업 이미지
    orig_img = Image.open(image_path).convert("RGB")
    W, H = orig_img.size
    w1, h1 = max(1, int(W * RESIZE_RATIO_1)), max(1, int(H * RESIZE_RATIO_1))

    #! resize
    work_img = orig_img.resize((w1, h1))

    sx, sy = W / float(w1), H / float(h1)

    # 1) Stage 1 - 좌우 분할만
    if skip_vertical_split:
        lr_merged = [(0, 0, W, H)]
    else:
        seg_lr = ImgSegmentation(
            img=work_img,
            max_depth=FIRST_PASS_LR["max_depth"],
            var_thresh=FIRST_PASS_LR["var_thresh"],
            diff_thresh=FIRST_PASS_LR["diff_thresh"],
            diff_portion=FIRST_PASS_LR["diff_portion"],
            window_size=FIRST_PASS_LR["window_size"]
        )
        root_bbox_work = (0, 0, w1, h1)
        lr_work = segment_once(seg_lr, work_img, root_bbox_work, line_direct="y")  # 수직 컷

        # 분할이 안 되면 전체 1개 처리
        if not lr_work:
            lr_work = [root_bbox_work]

        # work→orig 스케일백
        lr_orig = []
        for l,t,r,b in lr_work:
            L = int(math.floor(l * sx)); T = int(math.floor(t * sy))
            R = int(math.ceil (r * sx)); B = int(math.ceil (b * sy))
            L, T = max(0, min(W, L)), max(0, min(H, T))
            R, B = max(0, min(W, R)), max(0, min(H, B))
            if R > L and B > T:
                lr_orig.append((L,T,R,B))

        # 좌우 병합
        lr_merged = merge_lr(lr_orig, parent_w=W)

    # 2) Stage 2 - 각 좌우 세그 내부에서 상하 분할
    final_boxes = []
    for seg in lr_merged:
        L0,T0,R0,B0 = seg
        # 해당 세그 영역을 작업 좌표계로 변환
        l0 = int(math.floor(L0 / sx)); t0 = int(math.floor(T0 / sy))
        r0 = int(math.ceil (R0 / sx)); b0 = int(math.ceil (B0 / sy))
        l0, t0 = max(0, min(w1, l0)), max(0, min(h1, t0))
        r0, b0 = max(0, min(w1, r0)), max(0, min(h1, b0))
        sub_bbox_work = (l0, t0, r0, b0)

        seg_tb = ImgSegmentation(
            img=work_img,
            max_depth=FIRST_PASS_TB["max_depth"],
            var_thresh=FIRST_PASS_TB["var_thresh"],
            diff_thresh=FIRST_PASS_TB["diff_thresh"],
            diff_portion=FIRST_PASS_TB["diff_portion"],
            window_size=FIRST_PASS_TB["window_size"]
        )
        tb_work = segment_once(seg_tb, work_img, sub_bbox_work, line_direct="x")  # 수평 컷
        if not tb_work:
            tb_work = [sub_bbox_work]

        # work→orig 스케일백
        tb_orig = []
        for l,t,r,b in tb_work:
            L = int(math.floor(l * sx)); T = int(math.floor(t * sy))
            R = int(math.ceil (r * sx)); B = int(math.ceil (b * sy))
            L, T = max(L0, L), max(T0, T)
            R, B = min(R0, R), min(B0, B)
            if R > L and B > T:
                tb_orig.append((L,T,R,B))

        # 상하 병합
        tb_merged = merge_tb(tb_orig, parent_h=(B0 - T0))
        final_boxes.extend(tb_merged)

    # 4) 결과 JSON 구성
    json_out = [{"bbox": [int(b[0]), int(b[1]), int(b[2]), int(b[3])], "level": 1} for b in final_boxes]

    # 5) 결과 리스트(grounding 포맷)
    results_for_grounding = []
    results_for_grounding.append({"img": orig_img.copy(), "id": 0, "bbox": [0,0,W,H]})
    k = 1
    for b in final_boxes:
        crop_img = orig_img.crop(b)
        # 모델 전처리 안전을 위해 최종적으로도 최소 변 확보
        if min(crop_img.size) < MODEL_MIN_SIDE:
            # 여유 패딩으로 보장
            from PIL import ImageOps
            need_w = max(0, MODEL_MIN_SIDE - crop_img.size[0])
            need_h = max(0, MODEL_MIN_SIDE - crop_img.size[1])
            pad = (need_w//2, need_h//2, need_w - need_w//2, need_h - need_h//2)
            crop_img = ImageOps.expand(crop_img, border=pad, fill=PAD_COLOR)
        results_for_grounding.append({
            "img": crop_img,
            "id": k,
            "bbox": [int(b[0]), int(b[1]), int(b[2]), int(b[3])],
            "recursion_depth": 0,
            "fail": False,
            "filename": None
        })
        k += 1

    end = time()
    if print_latency:
        print(f"🕖 Time: {end - start:.3f}s", end=" | ")
        print(f"✂️ Crops: {len(final_boxes)}", end=" | ")

    # 6) 시각화
    if save_visualization and output_image_path:
        vis = orig_img.copy()
        draw = ImageDraw.Draw(vis)
        line_w = max(2, int(min(W,H) * 0.003))
        palette = [(255,0,0),(0,255,0),(0,0,255),(255,165,0),(255,0,255),(0,255,255)]
        for idx, b in enumerate(final_boxes):
            color = palette[idx % len(palette)]
            draw.rectangle(b, outline=color, width=line_w)
        save_path = os.path.join(output_image_path)
        vis.save(save_path)
        if print_latency:
            print(f"[SAVE] {save_path}")

    return results_for_grounding

#! ================================================================================================

if __name__ == '__main__':
    data_path = "./data/screenspotv2_imgs/"
    jsonlist = json.load(open("./data/screenspot_mobile_v2.json"))
    target_imgs = sorted(set(item["img_filename"] for item in jsonlist))
    os.makedirs(f"./crop_test/", exist_ok=True)
    for fname in target_imgs:
        crop_img(
            image_path = os.path.join(data_path, fname),
            output_image_path = f"./crop_test/{fname}",
            save_visualization = True,
            print_latency = True,
            skip_vertical_split = False
        )
