# iter_logger.py
import os
import csv
import time

# fcntl은 UNIX에서만 동작. 필요 없으면 끄면 됨
try:
    import fcntl  # type: ignore
except Exception:
    fcntl = None

# 모듈 내부 상태
_CSV_PATH = None
_MD_PATH = None
_HEADERS = None
_WRITE_MD = True
_USE_FSYNC = True
_USE_LOCK = False

_DEFAULT_HEADERS = [
    "idx",                # 번호
    "success",            # 최종 성공여부 (1/0)
    "filename_wo_ext",    # 파일명(확장자 제외)
    "task",               # task 이름
    "num_crops",          # stage1 투입 crop 총 개수
    "s1_hit",             # stage1 topQ 안에 GT 포함 (1/0)
    "acc_s2_uptonow",     # 현재까지 누적 정확도 % (float)
    "stage1_sec",         # stage1 처리 시간
    "select_sec",         # topQ 선별 시간
    "total_s1_sec",       # stage1 전체 시간 (위 둘 합)
    "s1_topq_count",      # stage1 topQ 개수
]

def init_iter_logger(
    save_dir: str,
    csv_name: str = "iter_log.csv",
    md_name: str = "iter_log.md",
    headers=None,
    write_md: bool = True,
    use_fsync: bool = True,
    use_lock: bool = False,
):
    """
    시작 시 1회 호출.
    - 파일 없으면 헤더 생성
    - 모듈 내부 상태(경로, 헤더, 옵션) 설정
    """
    global _CSV_PATH, _MD_PATH, _HEADERS, _WRITE_MD, _USE_FSYNC, _USE_LOCK

    if not save_dir:
        raise ValueError("save_dir 비어있음")

    os.makedirs(save_dir, exist_ok=True)

    _CSV_PATH = os.path.join(save_dir, csv_name)
    _MD_PATH = os.path.join(save_dir, md_name)
    _HEADERS = list(headers) if headers is not None else list(_DEFAULT_HEADERS)
    _WRITE_MD = bool(write_md)
    _USE_FSYNC = bool(use_fsync)
    _USE_LOCK = bool(use_lock and (fcntl is not None))

    # CSV 헤더
    if not os.path.exists(_CSV_PATH):
        with open(_CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(_HEADERS)
            _maybe_fsync(f)

    # MD 헤더
    if _WRITE_MD and not os.path.exists(_MD_PATH):
        with open(_MD_PATH, "w", encoding="utf-8") as f:
            f.write("| " + " | ".join(_HEADERS) + " |\n")
            f.write("|" + " --- |" * len(_HEADERS) + "\n")
            _maybe_fsync(f)


def append_iter_log(row=None, **row_kwargs):
    """
    매 반복마다 한 줄 append.
    - row(list/tuple)로 전체를 전달하거나
    - 혹은 키워드 인자(**row_kwargs)로 부분만 전달 가능
      → 헤더 순서대로 매핑해서 빈 값은 ''로 채움
    """
    if _CSV_PATH is None or _HEADERS is None:
        raise RuntimeError("init_iter_logger(...) 먼저 호출 필요")

    # 값 구성
    if row is not None:
        row_vals = list(row)
        if len(row_vals) != len(_HEADERS):
            raise ValueError(f"row 길이 {len(row_vals)} != headers {len(_HEADERS)}")
    else:
        row_vals = [_to_str(row_kwargs.get(h, "")) for h in _HEADERS]

    # CSV append
    with open(_CSV_PATH, "a", newline="", encoding="utf-8") as f:
        _maybe_lock(f, lock=True)
        writer = csv.writer(f)
        writer.writerow([_to_str(v) for v in row_vals])
        _maybe_lock(f, lock=False)
        _maybe_fsync(f)

    # MD append (옵션)
    if _WRITE_MD and _MD_PATH:
        md_line = "| " + " | ".join(_md_escape(v) for v in row_vals) + " |\n"
        with open(_MD_PATH, "a", encoding="utf-8") as f:
            _maybe_lock(f, lock=True)
            f.write(md_line)
            _maybe_lock(f, lock=False)
            _maybe_fsync(f)


# ---------- helpers ----------

def _to_str(x):
    if isinstance(x, float):
        # 너무 긴 float 줄이기
        return f"{x:.4g}"
    return str(x)

def _md_escape(x):
    s = _to_str(x)
    # 파이프는 테이블 문법 깨지니 이스케이프
    return s.replace("|", "\\|")

def _maybe_fsync(f):
    if _USE_FSYNC:
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass

def _maybe_lock(f, lock: bool):
    if not _USE_LOCK or fcntl is None:
        return
    try:
        fcntl.flock(f, fcntl.LOCK_EX if lock else fcntl.LOCK_UN)
    except Exception:
        # 락 실패해도 기록은 진행
        pass
