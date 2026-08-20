"""
envinfo.py
==========
Ghi lại cấu hình PHẦN CỨNG và PHẦN MỀM của mỗi lần chạy.

Trả lời phản biện: "Bổ sung wall-clock runtime, hardware/software ... mô tả thiết bị
và cách đo."

Mọi script thực nghiệm gọi `collect()` một lần và nhúng kết quả vào file JSON đầu ra,
nên mỗi con số runtime/bộ nhớ trong bài đều truy vết được về đúng máy đã chạy.
"""
import os
import platform
import subprocess
import sys


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def _cpu_name():
    """Tên CPU đầy đủ (platform.processor() trên Windows chỉ trả chuỗi family)."""
    if platform.system() == "Windows":
        # wmic đã bị gỡ khỏi Windows 11 mới -> ưu tiên PowerShell CIM.
        ps = _safe(lambda: subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Processor).Name"],
            capture_output=True, text=True, timeout=20).stdout.strip())
        if ps:
            return ps.splitlines()[0].strip()
        wmic = _safe(lambda: subprocess.run(
            ["wmic", "cpu", "get", "name"], capture_output=True, text=True,
            timeout=10).stdout.strip().splitlines())
        if wmic and len(wmic) > 1 and wmic[1].strip():
            return wmic[1].strip()
        return os.environ.get("PROCESSOR_IDENTIFIER") or platform.processor()
    if platform.system() == "Linux":
        txt = _safe(lambda: open("/proc/cpuinfo").read(), "")
        for line in (txt or "").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    if platform.system() == "Darwin":
        return _safe(lambda: subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=10).stdout.strip())
    return platform.processor()


def _versions():
    out = {"python": sys.version.split()[0]}
    for mod in ("numpy", "scipy", "matplotlib", "tensorflow", "tf_keras", "psutil"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            out[mod] = None
    return out


def _gpus():
    """Danh sách GPU mà TensorFlow nhìn thấy (rỗng => chạy CPU)."""
    try:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        import tensorflow as tf
        devs = tf.config.list_physical_devices("GPU")
        out = []
        for d in devs:
            det = _safe(lambda: tf.config.experimental.get_device_details(d), {}) or {}
            out.append({"name": d.name, "device_name": det.get("device_name"),
                        "compute_capability": det.get("compute_capability")})
        return out
    except Exception:
        return []


def collect():
    """Trả dict mô tả đầy đủ môi trường chạy."""
    try:
        import psutil
        cpu_logical = psutil.cpu_count(logical=True)
        cpu_physical = psutil.cpu_count(logical=False)
        freq = psutil.cpu_freq()
        cpu_mhz = float(freq.max or freq.current) if freq else None
        ram_gb = round(psutil.virtual_memory().total / 1024 ** 3, 2)
    except Exception:
        cpu_logical = os.cpu_count()
        cpu_physical = None
        cpu_mhz = None
        ram_gb = None

    gpus = _gpus()
    return {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "machine": platform.machine(),
        "cpu": _cpu_name(),
        "cpu_cores_physical": cpu_physical,
        "cpu_cores_logical": cpu_logical,
        "cpu_max_mhz": cpu_mhz,
        "ram_total_gb": ram_gb,
        "gpus": gpus,
        "compute_device": "GPU" if gpus else "CPU",
        "versions": _versions(),
        "thread_env": {v: os.environ.get(v) for v in
                       ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                        "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")},
        "measurement_method": {
            "wall_clock": "time.perf_counter() bao quanh toàn bộ vòng tối ưu",
            "cpu_time": "time.process_time() (tổng thời gian CPU của tiến trình)",
            "peak_host_memory": "psutil peak_wset (Windows) / lấy max RSS bằng "
                                "luồng lấy mẫu 20 Hz (Linux, macOS)",
            "peak_gpu_memory": "tf.config.experimental.get_memory_info('GPU:0')['peak'] "
                               "(None nếu chạy CPU)",
            "query": "1 query = 1 lần đo BLER trên 1 batch của model đích/surrogate",
            "fitness_evaluation": "1 fitness evaluation = 1 query (mọi phương pháp dùng "
                                  "cùng một hàm fitness và cùng batch)",
            "throughput": "tổng số symbol đã mô phỏng / wall-clock (symbol/s)",
        },
    }


def describe(info=None):
    """Chuỗi mô tả gọn để in ra terminal và dán vào bài."""
    info = info or collect()
    v = info["versions"]
    gpu = ", ".join(g.get("device_name") or g["name"] for g in info["gpus"]) or "không có (chạy CPU)"
    return (
        f"Thiết bị      : {info['cpu']} "
        f"({info['cpu_cores_physical']} nhân vật lý / {info['cpu_cores_logical']} luồng"
        + (f", {info['cpu_max_mhz']:.0f} MHz" if info["cpu_max_mhz"] else "") + ")\n"
        f"RAM           : {info['ram_total_gb']} GB\n"
        f"GPU           : {gpu}\n"
        f"Hệ điều hành  : {info['os']}\n"
        f"Phần mềm      : Python {v['python']}, TensorFlow {v['tensorflow']}, "
        f"tf_keras {v['tf_keras']}, NumPy {v['numpy']}, SciPy {v['scipy']}"
    )


if __name__ == "__main__":
    print(describe())
