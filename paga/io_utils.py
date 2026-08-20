"""
io_utils.py
===========
Ghi kết quả và in tiêu đề. Mỗi file JSON đầu ra đều nhúng sẵn:
  * toàn bộ tham số dòng lệnh (tái lập được),
  * mô tả phần cứng/phần mềm (`envinfo.collect()`),
  * dấu thời gian và phiên bản package,
để mọi con số trong bài truy vết được về đúng lần chạy đã sinh ra nó.
"""
import datetime
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT, "outputs")


def out_path(*parts):
    p = os.path.join(OUTPUT_DIR, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def _default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def _is_quick(payload):
    try:
        return bool(payload.get("config", {}).get("quick"))
    except Exception:
        return False


def save_json(obj, path, verbose=True):
    """
    Ghi kết quả, KHÔNG cho phép một lần chạy `--quick` đè lên kết quả thật.

    `--quick` chỉ dùng để thử pipeline (3 seed, budget 300). Nếu nó ghi đè lên
    file kết quả đầy đủ thì mất hàng giờ tính toán mà không có cảnh báo nào --
    đúng một lần như vậy đã xoá mất kết quả 10 seed của cả hai kịch bản budget.
    Khi phát hiện tình huống đó, bản quick được ghi sang tên `*.quick.json`.
    """
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    if _is_quick(obj) and os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as fh:
                existing = json.load(fh)
        except Exception:
            existing = {}
        if existing and not _is_quick(existing):
            alt = path[:-5] + ".quick.json" if path.endswith(".json") else path + ".quick"
            print(f"[CHẶN GHI ĐÈ] {os.path.basename(path)} đang chứa kết quả ĐẦY ĐỦ "
                  f"({existing.get('config', {}).get('seeds')} seed).\n"
                  f"              Lần chạy --quick này được ghi sang "
                  f"{os.path.basename(alt)} thay vì đè lên.")
            path = alt

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_default, ensure_ascii=False)
    if verbose:
        print(f"[saved] {path}")
    return path


def result_envelope(args, extra=None):
    """Khung metadata chuẩn cho mọi file kết quả."""
    from . import __version__
    from .envinfo import collect
    env = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "paga_version": __version__,
        "config": vars(args) if hasattr(args, "__dict__") else dict(args),
        "environment": collect(),
    }
    if extra:
        env.update(extra)
    return env


def banner(msg, char="=", width=78):
    line = char * width
    print(f"\n{line}\n{msg}\n{line}", flush=True)


def section(msg, width=78):
    print(f"\n--- {msg} " + "-" * max(0, width - len(msg) - 5), flush=True)
