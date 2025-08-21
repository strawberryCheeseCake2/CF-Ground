# crop5.py
"""
요구사항 요약:
    - s1 폴더에 all.png 저장
    - 20% 넘는 큰 그림은 최종적으로 반환하지 않으면 oversize/에 저장
    - 20% 이하(또는 실패로 강제 포함)는 s1 폴더 바로 아래에
        crop{번호}_rec{깊이}[ _fail].png 로 저장
    * 번호는 '최종적으로 사용되는 crop'에 대해 1부터 부여
    * level==0(썸네일)은 id=0 고정, 파일 저장은 all.png 만
    - 좌표는 절대 좌표 유지, 중복 저장 금지
    - MAX_RECURSION_DEPTH 번 반복해도 실패하면 fail로 강제 포함
반환:
    List[dict]: 0번 썸네일 + level1 crop들(dict는 run_grounding.py가 쓰는 스펙 유지)
"""

import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from PIL import Image

from dcgen.utils2 import ImgSegmentation  #* dcgen util 사용

# ------------------------------
# 설정 상수
# ------------------------------
DIFF_THRESH    = 45            # dcgen ImgSegmentation 인자 유지
DIFF_PORTION   = 0.9           # dcgen ImgSegmentation 인자 유지

MAX_AREA_RATIO = 0.40          #! 40% 보다 크면 재귀
MAX_RECURSION_DEPTH = 3        # 최대 재귀 횟수
RETRY_GROWTH   = 5             # var_thresh 증가 배율

@dataclass
class CropItem:
    id: int
    level: int
    bbox: List[int]                  # [left, top, right, bottom] (절대 좌표)
    recursion_depth: int = 0
    fail: bool = False               # 세그 실패로 강제 채택 표시
    parent_id: Optional[int] = None
    filename: Optional[str] = None   # 디스크 저장 파일명

# ------------------------------
# 유틸
# ------------------------------

def _area(b):
    return max(0, b[2]-b[0]) * max(0, b[3]-b[1])

def _ratio(b, W, H):
    A = _area(b)
    return A / float(max(1, W*H))

def _bbox_to_tuple(b):
    return (int(b[0]), int(b[1]), int(b[2]), int(b[3]))

def _dedup_key(b):
    # 박스 중복 제거 키 (좌표 스냅)
    return (int(b[0]), int(b[1]), int(b[2]), int(b[3]))

def _safe_remove_tree(p: Path):
    try:
        if p.exists():
            shutil.rmtree(p)
    except Exception:
        pass

# ------------------------------
# 세그멘테이션 1회 실행 + JSON 읽기
# ------------------------------

def _run_imgseg_once(image_path: str, max_depth: int, window_size: int, var_thresh: int, json_out: Path) -> Optional[List[Dict]]:
    """
    ImgSegmentation 1회 실행하고 JSON을 json_out에 저장한 뒤 파싱해서 반환
    각 항목은 {"bbox":[l,t,r,b], "level":int} 형태라고 가정
    """
    try:
        seg = ImgSegmentation(
            img=image_path,
            max_depth=max_depth,
            var_thresh=var_thresh,
            diff_thresh=DIFF_THRESH,
            diff_portion=DIFF_PORTION,
            window_size=window_size
        )
        seg.to_json(path=str(json_out))
        with open(json_out, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return None
        return data
    except Exception as e:
        print(f"[seg] error var_thresh={var_thresh}: {e}")
        return None

def _run_imgseg_with_retries(image_path: str, max_depth: int, window_size: int, var_thresh: int, json_out: Path) -> Optional[List[Dict]]:
    """
    var_thresh를 점증시키며 시도
    MAX_RECURSION_DEPTH 만큼 
    """
    tried = []
    cur = var_thresh
    last = None

    for _ in range(MAX_RECURSION_DEPTH):
        tried.append(cur)
        data = _run_imgseg_once(image_path, max_depth, window_size, cur, json_out)
        if data is not None and len(data) > 0:
            # 최소 level=1이 존재하는지 확인
            if any((d.get("level", 1) != 0) for d in data):
                return data
        last = data
        cur *= RETRY_GROWTH

    # 최종 실패
    print(f"[seg] no result after tries {tried}")
    return last

# ------------------------------
# 하위(부분) 이미지에 대해 재귀 분할
# ------------------------------

def _recursive_split(
    original_img: Image.Image,
    abs_bbox: List[int],
    depth: int,
    max_depth_limit: int,
    max_area_ratio: float,
    max_depth: int, window_size: int, var_thresh: int,
    work_root: Path
) -> Tuple[List[Tuple[List[int], int, bool]], List[List[int]]]:
    """
    반환:
      kept: List of (abs_bbox, recursion_depth, fail_flag)
      oversize_dump: 최종적으로 반환하지 않는(oversize) 절대 박스들
    """
    W, H = original_img.size

    # 1) 큰 영역을 잘라 임시 파일로 세그 시도
    l, t, r, b = _bbox_to_tuple(abs_bbox)
    sub = original_img.crop((l, t, r, b))

    tmp_dir = work_root / f"tmp_rec_{depth}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_img = tmp_dir / "img.png"
    tmp_json = tmp_dir / "seg.json"
    sub.save(tmp_img)

    data = _run_imgseg_with_retries(str(tmp_img), max_depth=max_depth, window_size=window_size, var_thresh=var_thresh, json_out=tmp_json)

    kept: List[Tuple[List[int], int, bool]] = []
    oversize_dump: List[List[int]] = []

    if data is None:
        # 세그 자체가 실패 → 원래 abs_bbox를 fail로 포함
        kept.append((abs_bbox, depth, True))
        return kept, oversize_dump

    # data 안에는 level 0(썸네일) + level 1들이 함께 있을 수 있음
    # level 1만 처리
    subs = [d for d in data if d.get("level", 1) != 0 and "bbox" in d]
    if len(subs) == 0:
        # 분할 결과 없음 → abs_bbox fail 포함
        kept.append((abs_bbox, depth, True))
        return kept, oversize_dump

    for d in subs:
        rel = d["bbox"]  # 부분 이미지 기준 좌표
        sL, sT, sR, sB = int(rel[0]), int(rel[1]), int(rel[2]), int(rel[3])
        # 절대 좌표 변환
        abs_child = [l + sL, t + sT, l + sR, t + sB]
        ratio = _ratio(abs_child, W, H)

        if ratio <= max_area_ratio:
            kept.append((abs_child, depth, False))
        else:
            # 더 쪼갤 수 있는지 확인
            if depth + 1 < max_depth_limit:
                child_kept, child_oversize = _recursive_split(
                    original_img, abs_child, depth + 1, max_depth_limit, max_area_ratio,
                    max_depth, window_size, var_thresh, work_root
                )
                kept.extend(child_kept)
                oversize_dump.extend(child_oversize)
            else:
                # 더 못 쪼갬 → oversize로 보관 (반환 안 함)
                oversize_dump.append(abs_child)

    return kept, oversize_dump

# ------------------------------
# 메인 엔트리
# ------------------------------

def run_segmentation_recursive( image_path: str,
                                max_depth: int,
                                window_size: int,
                                output_json_path: str,
                                output_image_path: str,
                                start_id: int = 0,
                                var_thresh: int = 120,
                                max_area_ratio: float = MAX_AREA_RATIO,
                                max_recursion: int = 3):
    
    s1_dir = Path(output_image_path)
    s1_dir.mkdir(parents=True, exist_ok=True)
    oversize_dir = s1_dir / "oversize"
    oversize_dir.mkdir(exist_ok=True)

    # 0) 원본 저장
    try:
        original_img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[seg] cannot open image: {e}")
        return None
    W, H = original_img.size
    (s1_dir / "all.png").write_bytes(Path(image_path).read_bytes())

    # 1) 초기 세그
    output_json_path = Path(output_json_path)
    data = _run_imgseg_with_retries(image_path, max_depth=max_depth, window_size=window_size, var_thresh=var_thresh, json_out=output_json_path)

    if data is None or len(data) == 0:
        print("[seg] initial segmentation failed → 썸네일만 반환")
        return [{
            "img": original_img,
            "id": 0,
            "bbox": [0, 0, W, H],
            "level": 0
        }]

    # 2) 수집 단계
    final_kept: List[Tuple[List[int], int, bool]] = []   # (abs_bbox, recursion_depth, fail_flag)
    final_dump: List[List[int]] = []                    # oversize만 모아 저장용
    seen = set()

    # 초기 level==1 후보
    init_candidates = [d for d in data if d.get("level", 1) != 0 and "bbox" in d]

    # 초기 후보 각각 처리
    for d in init_candidates:
        b = [int(x) for x in d["bbox"]]
        key = _dedup_key(b)
        if key in seen:
            continue
        seen.add(key)

        ratio = _ratio(b, W, H)
        if ratio <= max_area_ratio:
            final_kept.append((b, 0, False))
        else:
            # 재귀 분할
            work_root = s1_dir / "_tmp_work"
            work_root.mkdir(exist_ok=True)
            kept, dump = _recursive_split(
                original_img, b, 1, max_recursion, max_area_ratio,
                max_depth, window_size, var_thresh,
                work_root=work_root
            )
            final_kept.extend(kept)
            final_dump.extend(dump)
            _safe_remove_tree(work_root)

    # 3) oversize 저장 (반환하지 않는 것만)
    dump_seen = set()
    for b in final_dump:
        k = _dedup_key(b)
        if k in dump_seen:
            continue
        dump_seen.add(k)
        try:
            crop = original_img.crop(_bbox_to_tuple(b))
            # 이름: over_{l}_{t}_{r}_{b}.png (고유)
            fname = f"over_{b[0]}_{b[1]}_{b[2]}_{b[3]}.png"
            crop.save(oversize_dir / fname)
        except Exception as e:
            print(f"[seg] oversize save fail: {e}")

    # 4) 최종 반환 목록(레벨/ID 정리)
    #    - id 0: 썸네일
    #    - id 1..N: 채택된 crop (fail 포함)
    #    - 번호 부여는 '최종적으로 사용되는 crop' 기준으로 1부터
    # 좌표 중복 제거
    kept_unique = []
    kept_seen = set()
    for b, rec, fail in final_kept:
        k = _dedup_key(b)
        if k in kept_seen:
            continue
        kept_seen.add(k)
        kept_unique.append((b, rec, fail))

    # 정렬 기준: 상단-좌측 우선
    kept_unique.sort(key=lambda x: (x[0][1], x[0][0]))

    results: List[Dict] = []
    # 썸네일 먼저
    results.append({
        "img": original_img,
        "id": 0,
        "bbox": [0, 0, W, H],
        "level": 0
    })

    # 파일 저장 + id 부여
    running_idx = 1
    for b, rec, fail in kept_unique:
        try:
            crop_img = original_img.crop(_bbox_to_tuple(b))
            # 파일명 규칙
            fname = f"crop{running_idx}_rec{rec}{'_fail' if fail else ''}.png"
            crop_img.save(s1_dir / fname)

            results.append({
                "img": crop_img,
                "id": running_idx,
                "bbox": b,
                "level": 1,
                "recursion_depth": rec,
                "fail": bool(fail),
                "filename": fname
            })
            running_idx += 1
        except Exception as e:
            print(f"[seg] save error: {e}")

    # 디버그 출력
    print(f"✂️  Total crop: {running_idx-1}")

    return results
