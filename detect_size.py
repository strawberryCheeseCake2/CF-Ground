'''
폰트 크기(읽힘) 임계 기반 이분탐색 리사이즈 & 시각화 도구
- Laplacian 분산 + Tenengrad(소벨 에너지)로 읽힘 점수 측정
- 3회 이분탐색으로 최소 필요 배율(best_scale) 탐색
- 탐색 스냅샷/최종 리사이즈 이미지 저장 + CSV 로그
'''

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
from time import time
import json
import csv

# ===================== Hyper Parameters =====================
# 이분탐색 구간과 반복 횟수
BISECT_LOW = 0.25
BISECT_HIGH = 0.75
BISECT_STEPS = 3            # 3~5 권장

# 읽힘 점수 기준(단변 기준 512에서의 임계)
BASE_SHORT_SIDE = 512       # 480~640 권장
BASE_THRESH = 0.20          # 0.16~0.24에서 데이터로 튠

# 점수 내부 파라미터
SOBEL_KEEP_FRAC = 0.06      # 상위 몇 %의 소벨 에너지를 남길지
BOXFILTER_K = 3             # 3x3 박스 스무딩

# 프리뷰 저장 품질
JPEG_QUALITY = 92
# ============================================================


# ---------------------- 유틸: CSV 로깅 ----------------------
def log_to_csv(device_type, data_rows):
    out_dir = "./cs_test"
    os.makedirs(out_dir, exist_ok=True)
    csv_file = os.path.join(out_dir, f"log_{device_type}.csv")
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "Relative Path", "Height", "Width",
            "Best Scale", "Final Score", "Final Threshold",
            "Trace (scale:score:thr:ok; ...)", "Processing Time (s)"
        ])
        w.writerows(data_rows)


# ---------------------- 읽힘 점수 계산 ----------------------
def _boxfilter3(img_f32):
    # 매우 가벼운 3x3 박스 필터 (np.cumsum 이용)
    k = BOXFILTER_K
    pad = k // 2
    cs = np.pad(img_f32, ((pad, pad), (pad, pad)), mode="edge").cumsum(0).cumsum(1)
    return (cs[k:, k:] - cs[:-k, k:] - cs[k:, :-k] + cs[:-k, :-k]) / (k * k)

def _clarity_score(img_pil, sobel_thresh_frac=SOBEL_KEEP_FRAC):
    """
    Laplacian variance + Tenengrad 결합 점수 (높을수록 더 또렷)
    입력은 이미 리사이즈된 작은 이미지
    """
    g = np.asarray(img_pil.convert("L"), dtype=np.float32)

    # 가벼운 스무딩(노이즈 억제)
    sm = _boxfilter3(g)

    # 1차 미분(간단 Sobel 대체): abs(diff)
    gx = np.abs(np.pad(np.diff(sm, axis=1), ((0, 0), (1, 0)), mode="edge"))
    gy = np.abs(np.pad(np.diff(sm, axis=0), ((1, 0), (0, 0)), mode="edge"))
    g2 = gx * gx + gy * gy

    # 상위 일부만 남기는 적응 임계
    thr_sobel = np.quantile(g2, 1.0 - sobel_thresh_frac)
    tenengrad = np.sqrt(np.maximum(g2 - thr_sobel, 0.0)).mean()

    # ✅ shape-safe Laplacian (입력과 동일한 크기 보장)
    # ∇²I ≈ I_up + I_down + I_left + I_right - 4*I
    sm_up    = np.pad(sm[:-1, :], ((1, 0), (0, 0)), mode="edge")
    sm_down  = np.pad(sm[1:,  :], ((0, 1), (0, 0)), mode="edge")
    sm_left  = np.pad(sm[:, :-1], ((0, 0), (1, 0)), mode="edge")
    sm_right = np.pad(sm[:, 1: ], ((0, 0), (0, 1)), mode="edge")
    lap = (sm_up + sm_down + sm_left + sm_right) - 4.0 * sm

    lap_var = lap.var()

    # 크기 정규화
    H, W = sm.shape
    area_norm = np.sqrt(max(1.0, float(H * W)))
    score = (0.6 * lap_var + 0.4 * tenengrad) / area_norm
    return float(score)



def _is_readable_by_scale(orig_img_pil, scale, base_short_side=BASE_SHORT_SIDE, base_thresh=BASE_THRESH):
    """
    지정 배율에서 읽힘 여부 판단
    반환: (ok, score, thresh, img_scaled)
    """
    W0, H0 = orig_img_pil.size
    new_w, new_h = max(2, int(W0 * scale)), max(2, int(H0 * scale))
    img_scaled = orig_img_pil.resize((new_w, new_h), Image.BILINEAR)

    score = _clarity_score(img_scaled)

    short_side = min(new_w, new_h)
    # 단변 길이에 따른 임계 보정(작을수록 관대, 클수록 엄격)
    scale_factor = np.clip(short_side / float(base_short_side), 0.85, 1.10)
    thresh = base_thresh * scale_factor

    return (score >= thresh), float(score), float(thresh), img_scaled


def find_min_readable_scale_bisect(orig_img_pil,
                                   low=BISECT_LOW, high=BISECT_HIGH, max_steps=BISECT_STEPS,
                                   base_short_side=BASE_SHORT_SIDE, base_thresh=BASE_THRESH,
                                   prefer_smaller=True):
    """
    이분탐색으로 읽힘 임계를 만족하는 최소 scale 찾기
    반환: (best_scale, final_score, final_thresh, trace, snapshots)
      - trace: [(scale, score, thresh, ok), ...]
      - snapshots: {scale_str: PIL.Image}  # 탐색 중간 이미지들
    """
    trace = []
    snapshots = {}
    best_scale = None
    best_score = None
    best_thresh = None

    lo, hi = low, high
    for _ in range(max_steps):
        mid = (lo + hi) / 2.0
        ok, sc, th, img_mid = _is_readable_by_scale(orig_img_pil, mid, base_short_side, base_thresh)
        mid_key = f"{mid:.4f}"
        snapshots[mid_key] = img_mid
        trace.append((round(mid, 4), round(sc, 4), round(th, 4), bool(ok)))

        if ok:
            best_scale, best_score, best_thresh = mid, sc, th
            hi = mid     # 더 작은 쪽 탐색
        else:
            lo = mid     # 더 큰 쪽 탐색

    # 마지막 hi가 보통 최소 OK 스케일 후보
    if best_scale is None:
        # 끝까지도 OK가 안 나오면 hi로 세팅(가장 큰 배율로라도)
        ok, sc, th, img_hi = _is_readable_by_scale(orig_img_pil, hi, base_short_side, base_thresh)
        best_scale, best_score, best_thresh = hi, sc, th
        snapshots[f"{hi:.4f}"] = img_hi
        trace.append((round(hi, 4), round(sc, 4), round(th, 4), bool(ok)))

    if prefer_smaller and best_scale is not None:
        # 최소 배율을 더 깎을 수 있나 한 번 더 확인(10% 축소)
        cand = max(low, best_scale * 0.9)
        ok2, sc2, th2, img_cand = _is_readable_by_scale(orig_img_pil, cand, base_short_side, base_thresh)
        snapshots[f"{cand:.4f}"] = img_cand
        trace.append((round(cand, 4), round(sc2, 4), round(th2, 4), bool(ok2)))
        if ok2:
            best_scale, best_score, best_thresh = cand, sc2, th2

    return float(best_scale), float(best_score), float(best_thresh), trace, snapshots


# ---------------------- 시각화/저장 ----------------------
def _try_load_font(size):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        try:
            return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Unicode.ttf", size)
        except Exception:
            try:
                return ImageFont.load_default()
            except Exception:
                return None

def save_scaled_preview(img_pil, scale, score, thresh, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    W, H = img_pil.size
    new_w, new_h = max(2, int(W * scale)), max(2, int(H * scale))
    img_scaled = img_pil.resize((new_w, new_h), Image.BILINEAR)

    # 상단 라벨 그리기
    draw = ImageDraw.Draw(img_scaled)
    font = _try_load_font(max(14, new_w // 40))
    label = f"scale={scale:.4f}  score={score:.4f}  thr={thresh:.4f}"
    if font:
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    else:
        tw, th = len(label) * 6, 11
    pad = 6
    box = (5, 5, 5 + tw + 2 * pad, 5 + th + 2 * pad)
    draw.rectangle(box, fill="white", outline="black", width=1)
    draw.text((5 + pad, 5 + pad), label, fill="red", font=font if font else None)

    img_scaled.save(out_path, quality=JPEG_QUALITY, subsampling=0)
    return out_path

def save_composite(original, snapshots, best_key, out_path, cols=3, cell_max_w=540):
    """
    원본 + 스냅샷들을 그리드로 붙인 콤포지트 생성
    snapshots: { "scale": PIL.Image(resized_already) }
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # 보여줄 목록 구성: 원본 1장 + 스냅샷 일부 + best (중복 제거)
    keys_sorted = sorted(snapshots.keys(), key=lambda k: float(k))
    show_keys = []
    for k in keys_sorted:
        if k not in show_keys:
            show_keys.append(k)
    if best_key not in show_keys:
        show_keys.append(best_key)

    # 썸네일 만들기
    thumbs = []
    labels = ["original"] + show_keys
    images = [original] + [snapshots[k] for k in show_keys]
    for img in images:
        w, h = img.size
        scale = min(1.0, cell_max_w / float(w))
        if scale < 1.0:
            img_t = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
        else:
            img_t = img.copy()
        thumbs.append(img_t)

    # 그리드 배치
    rows = int(np.ceil(len(thumbs) / float(cols)))
    cell_w = max(im.size[0] for im in thumbs) + 20
    cell_h = max(im.size[1] for im in thumbs) + 40

    comp = Image.new("RGB", (cell_w * cols, cell_h * rows), (245, 245, 245))
    draw = ImageDraw.Draw(comp)
    font = _try_load_font(16)

    for idx, (im, lab) in enumerate(zip(thumbs, labels)):
        r, c = divmod(idx, cols)
        x = c * cell_w + 10
        y = r * cell_h + 10
        comp.paste(im, (x, y))
        tag = lab if lab == "original" else f"scale={float(lab):.4f}"
        if font:
            draw.text((x, y + im.size[1] + 6), tag, fill="black", font=font)
        else:
            draw.text((x, y + im.size[1] + 6), tag, fill="black")

    comp.save(out_path, quality=JPEG_QUALITY, subsampling=0)
    return out_path


# ============================== MAIN ==============================
if __name__ == '__main__':
    # 테스트용 main
    data_path = "./data/screenspotv2_image/"
    device_types = ["mobile", "web", "desktop"]

    for device_type in device_types:
        jsonlist = json.load(open(f"./data/screenspot_{device_type}_v2.json"))
        target_imgs = sorted(set(item["img_filename"] for item in jsonlist if "img_filename" in item))

        log_data = []
        for fname in target_imgs:
            image_path = os.path.join(data_path, fname)
            orig_img = Image.open(image_path).convert("RGB")
            orig_w, orig_h = orig_img.size

            start_time = time()

            # 이분탐색으로 최소 읽힘 배율 탐색
            best_scale, final_score, final_thresh, trace, snapshots = find_min_readable_scale_bisect(
                orig_img,
                low=BISECT_LOW, high=BISECT_HIGH, max_steps=BISECT_STEPS,
                base_short_side=BASE_SHORT_SIDE, base_thresh=BASE_THRESH,
                prefer_smaller=True
            )

            processing_time = time() - start_time

            # 저장 경로 구성(입력 폴더 구조 반영)
            rel = os.path.relpath(image_path, start=data_path)  # data_path 기준 상대경로
            rel_dir = os.path.dirname(rel)
            base_name = os.path.basename(rel)
            name_noext, ext = os.path.splitext(base_name)

            # 미리보기/컴포지트 디렉토리
            prev_dir = os.path.join("./cs_test", device_type, "previews", rel_dir)
            comp_dir = os.path.join("./cs_test", device_type, "composites", rel_dir)
            os.makedirs(prev_dir, exist_ok=True)
            os.makedirs(comp_dir, exist_ok=True)

            # 탐색 스냅샷/최종 저장
            # 1) 탐색 중간들: trace 순서대로 라벨 포함 저장
            for sc, sc_score, sc_thr, ok in trace:
                k = f"{sc:.4f}"
                img_mid = snapshots.get(k, None)
                if img_mid is None:
                    # 없으면 즉석 리사이즈
                    W0, H0 = orig_img.size
                    img_mid = orig_img.resize((max(2, int(W0 * sc)), max(2, int(H0 * sc))), Image.BILINEAR)
                out_p = os.path.join(prev_dir, f"{name_noext}__scale{k}__{'OK' if ok else 'NO'}.jpg")
                save_scaled_preview(orig_img, sc, sc_score, sc_thr, out_p)

            # 2) 최종(best) 저장
            best_key = f"{best_scale:.4f}"
            ok_final = "OK" if final_score >= final_thresh else "NO"
            best_out = os.path.join(prev_dir, f"{name_noext}__BEST_scale{best_key}__{ok_final}.jpg")
            save_scaled_preview(orig_img, best_scale, final_score, final_thresh, best_out)

            # 3) 콤포지트 저장(원본 + 스냅샷 + 최종)
            comp_out = os.path.join(comp_dir, f"{name_noext}__composite.jpg")
            # snapshots dict에는 탐색 중간 이미지가 들어있음. 최종(best)도 보장 추가
            if best_key not in snapshots:
                W0, H0 = orig_img.size
                snapshots[best_key] = orig_img.resize((max(2, int(W0 * best_scale)),
                                                       max(2, int(H0 * best_scale))), Image.BILINEAR)
            save_composite(orig_img, snapshots, best_key, comp_out, cols=3, cell_max_w=540)

            # 로그 출력
            relative_path = os.path.relpath(image_path, start=os.getcwd())
            trace_str = "; ".join([f"{sc:.4f}:{scs:.4f}:{thr:.4f}:{'OK' if ok else 'NO'}"
                                   for (sc, scs, thr, ok) in trace])
            print(f"H: {orig_h:4} | W: {orig_w:4} | best_scale: {best_scale:.4f} | "
                  f"score: {final_score:.4f} / thr: {final_thresh:.4f} | "
                  f"time: {processing_time:.4f}s | {relative_path}")

            log_data.append([
                relative_path, orig_h, orig_w,
                f"{best_scale:.4f}", f"{final_score:.4f}", f"{final_thresh:.4f}",
                trace_str, f"{processing_time:.4f}"
            ])

        log_to_csv(device_type, log_data)
