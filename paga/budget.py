"""
profiling.py
============
Đo CHI PHÍ của mỗi phương pháp tấn công dưới CÙNG một budget.

Trả lời phản biện:
  * "Bổ sung wall-clock runtime, fitness evaluations và query count dưới cùng budget"
  * "Bổ sung peak CPU/GPU memory, runtime, fitness evaluations và throughput;
     mô tả thiết bị và cách đo"

Định nghĩa đơn vị chi phí (dùng thống nhất cho MỌI phương pháp):
  1 query = 1 fitness evaluation = 1 lần đo BLER trên `fit_batch` symbol
            của model đích (hộp trắng) hoặc của 1 surrogate (hộp đen).
  Truy vấn gradient (chỉ hộp trắng: khởi tạo elite FGM) được đếm RIÊNG ở
  `n_grad_queries` để không giấu chi phí của PAGA vào trong số query.

`Budget` cho phép dừng theo SỐ EVALUATION hoặc theo WALL-CLOCK, nên có thể so sánh
các phương pháp ở cả hai chế độ "cùng số query" và "cùng thời gian".
"""
import os
import platform
import threading
import time


# --------------------------------------------------------------------------- #
# Bộ nhớ
# --------------------------------------------------------------------------- #
def _rss_bytes(proc):
    try:
        return proc.memory_info().rss
    except Exception:
        return 0


#
# KHÔNG dùng `memory_info().peak_wset` của Windows.
#
# `peak_wset` là đỉnh working-set của TOÀN BỘ VÒNG ĐỜI tiến trình và KHÔNG BAO GIỜ
# reset. Trước đây `ResourceMeter` lấy
#
#     peak = max(rss_hiện_tại, peak_wset)          # peak_wset = đỉnh cả đời
#     delta = peak - rss_lúc_bắt_đầu_đo
#
# nên `delta` thực chất đo "hiện tại thấp hơn đỉnh cả đời bao nhiêu", chứ không
# phải "thuật toán cấp thêm bao nhiêu". Hệ quả: phương pháp chạy SAU thừa hưởng
# đỉnh do một pha khác tạo ra (ví dụ lần đo BLER 50000x10 giữa các run), còn
# phương pháp chạy TRƯỚC thì gần 0. Số đo được là VỊ TRÍ TRONG DANH SÁCH, không
# phải thuật toán:
#
#     hộp trắng   CMA-ES chạy thứ 6 -> 289 MB
#     hộp đen     CMA-ES chạy thứ 5 -> 1.3 MB      (cùng thuật toán, cùng cài đặt)
#
# Thay bằng: lấy mẫu RSS trong lúc chạy (mọi hệ điều hành) + `tracemalloc` cho
# phần cấp phát ở tầng Python.


def gpu_peak_bytes():
    """Đỉnh bộ nhớ GPU do TensorFlow báo cáo; None nếu không có GPU."""
    try:
        import tensorflow as tf
        if not tf.config.list_physical_devices("GPU"):
            return None
        return int(tf.config.experimental.get_memory_info("GPU:0")["peak"])
    except Exception:
        return None


def gpu_reset_peak():
    try:
        import tensorflow as tf
        if tf.config.list_physical_devices("GPU"):
            tf.config.experimental.reset_memory_stats("GPU:0")
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Budget
# --------------------------------------------------------------------------- #
class Budget:
    """
    Ràng buộc dừng dùng chung. Hết budget khi VƯỢT bất kỳ giới hạn nào được đặt.

    max_evals : số fitness evaluation tối đa  (chế độ "cùng số query")
    max_wall  : số giây tối đa                (chế độ "cùng wall-clock")
    """

    def __init__(self, max_evals=None, max_wall=None):
        if max_evals is None and max_wall is None:
            raise ValueError("Budget cần ít nhất một trong max_evals / max_wall")
        self.max_evals = int(max_evals) if max_evals is not None else None
        self.max_wall = float(max_wall) if max_wall is not None else None

    def exhausted(self, meter):
        if self.max_evals is not None and meter.n_evals >= self.max_evals:
            return True
        if self.max_wall is not None and meter.wall >= self.max_wall:
            return True
        return False

    def remaining_evals(self, meter):
        """Số evaluation còn lại (vô hạn nếu chỉ giới hạn theo thời gian)."""
        if self.max_evals is None:
            return 1 << 30
        return max(0, self.max_evals - meter.n_evals)

    def fraction(self, meter):
        fr = 0.0
        if self.max_evals:
            fr = max(fr, meter.n_evals / self.max_evals)
        if self.max_wall:
            fr = max(fr, meter.wall / self.max_wall)
        return min(1.0, fr)

    def to_dict(self):
        return {"max_evals": self.max_evals, "max_wall_s": self.max_wall}

    def __repr__(self):
        return f"Budget(max_evals={self.max_evals}, max_wall={self.max_wall})"


def as_budget(b):
    """Cho phép truyền thẳng một số nguyên thay cho Budget."""
    return b if isinstance(b, Budget) else Budget(max_evals=int(b))


# --------------------------------------------------------------------------- #
# ResourceMeter
# --------------------------------------------------------------------------- #
class ResourceMeter:
    """
    Đếm query/evaluation và đo tài nguyên trong suốt một lần chạy optimizer.

    Cách dùng:
        with ResourceMeter("PAMFGA") as meter:
            ...  # optimizer gọi meter.tick(...) trong mỗi lần đánh giá fitness
        print(meter.to_dict())
    """

    def __init__(self, label="", track_memory=True, sample_hz=20.0):
        self.label = label
        self.track_memory = track_memory
        self._sample_interval = 1.0 / float(sample_hz)

        self.n_evals = 0          # = số query = số fitness evaluation
        self.n_grad_queries = 0   # truy vấn gradient (chỉ hộp trắng)
        self.n_symbols = 0        # tổng số symbol đã mô phỏng
        self.n_generations = 0

        self._t0 = None
        self._cpu0 = None
        self._wall = 0.0
        self._cpu = 0.0

        self._proc = None
        self._baseline_rss = 0
        self._peak_rss = 0
        self._gpu_peak_start = None
        self._stop = threading.Event()
        self._thread = None
        self._tracemalloc_owner = False    # meter này có tự bật tracemalloc không
        self._peak_pyalloc = 0

    # -- vòng đời ----------------------------------------------------------- #
    def start(self):
        if self.track_memory:
            try:
                import psutil
                self._proc = psutil.Process(os.getpid())
                self._baseline_rss = _rss_bytes(self._proc)
                self._peak_rss = self._baseline_rss
            except Exception:
                self._proc = None
            gpu_reset_peak()
            self._gpu_peak_start = gpu_peak_bytes()
            if self._proc is not None:
                # Lấy mẫu RSS trên MỌI hệ điều hành. Windows cũng phải lấy mẫu:
                # `peak_wset` là đỉnh cả đời tiến trình nên không dùng được (xem
                # ghi chú ở đầu file).
                self._thread = threading.Thread(target=self._sample_loop, daemon=True)
                self._thread.start()
            # `tracemalloc` đo phần cấp phát Ở TẦNG PYTHON trong đúng khoảng chạy
            # -- đây mới là con số phân biệt được thuật toán (quần thể, ma trận
            # hiệp phương sai của CMA-ES, ...). RSS bị session TensorFlow lấn át.
            try:
                import tracemalloc
                if not tracemalloc.is_tracing():
                    tracemalloc.start()
                    self._tracemalloc_owner = True
                else:
                    tracemalloc.reset_peak()
            except Exception:
                self._tracemalloc_owner = False
        self._t0 = time.perf_counter()
        self._cpu0 = time.process_time()
        return self

    def stop(self):
        if self._t0 is not None:
            self._wall = time.perf_counter() - self._t0
            self._cpu = time.process_time() - self._cpu0
            self._t0 = None
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._proc is not None:
            self._peak_rss = max(self._peak_rss, _rss_bytes(self._proc))
        try:
            import tracemalloc
            if tracemalloc.is_tracing():
                self._peak_pyalloc = int(tracemalloc.get_traced_memory()[1])
                if self._tracemalloc_owner:
                    tracemalloc.stop()
        except Exception:
            pass
        return self

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False

    def _sample_loop(self):
        while not self._stop.wait(self._sample_interval):
            self._peak_rss = max(self._peak_rss, _rss_bytes(self._proc))

    # -- đếm ---------------------------------------------------------------- #
    def tick(self, n=1, symbols=0):
        """Ghi nhận n fitness evaluation, mỗi cái mô phỏng `symbols/n` symbol."""
        self.n_evals += int(n)
        self.n_symbols += int(symbols)

    def tick_grad(self, n=1):
        self.n_grad_queries += int(n)

    def tick_generation(self, n=1):
        self.n_generations += int(n)

    # -- đọc ---------------------------------------------------------------- #
    @property
    def wall(self):
        if self._t0 is not None:
            return time.perf_counter() - self._t0
        return self._wall

    @property
    def cpu_time(self):
        if self._t0 is not None:
            return time.process_time() - self._cpu0
        return self._cpu

    @property
    def peak_host_mem_mb(self):
        if not self._peak_rss:
            return None
        return self._peak_rss / 1024 ** 2

    @property
    def peak_host_mem_delta_mb(self):
        """
        RSS đỉnh TRONG LÚC CHẠY trừ RSS lúc bắt đầu.

        Vẫn nhiễu: cấp phát của TensorFlow, bộ nhớ tạm của phép đo BLER theo lô,
        và độ trễ trả bộ nhớ về hệ điều hành đều rơi vào đây. Muốn so bộ nhớ giữa
        các THUẬT TOÁN thì đọc `peak_python_alloc_kb`.
        """
        if not self._peak_rss:
            return None
        return max(0.0, (self._peak_rss - self._baseline_rss) / 1024 ** 2)

    @property
    def peak_python_alloc_kb(self):
        """
        Đỉnh cấp phát ở tầng Python trong đúng khoảng chạy (`tracemalloc`).

        Đây là con số so sánh được giữa các thuật toán: nó chỉ tính bộ nhớ mà mã
        Python cấp (quần thể, ma trận hiệp phương sai, lịch sử...), không tính
        vùng nhớ dùng chung của session TensorFlow.
        """
        return self._peak_pyalloc / 1024.0 if self._peak_pyalloc else 0.0

    @property
    def peak_gpu_mem_mb(self):
        peak = gpu_peak_bytes()
        return None if peak is None else peak / 1024 ** 2

    @property
    def evals_per_second(self):
        w = self.wall
        return self.n_evals / w if w > 0 else float("nan")

    @property
    def symbols_per_second(self):
        w = self.wall
        return self.n_symbols / w if w > 0 else float("nan")

    def to_dict(self):
        return {
            "label": self.label,
            "n_queries": self.n_evals,
            "n_fitness_evals": self.n_evals,
            "n_grad_queries": self.n_grad_queries,
            "n_generations": self.n_generations,
            "n_symbols_simulated": self.n_symbols,
            "wall_s": self.wall,
            "cpu_s": self.cpu_time,
            "peak_host_mem_mb": self.peak_host_mem_mb,
            "peak_host_mem_delta_mb": self.peak_host_mem_delta_mb,
            "peak_python_alloc_kb": self.peak_python_alloc_kb,
            "peak_gpu_mem_mb": self.peak_gpu_mem_mb,
            "throughput_evals_per_s": self.evals_per_second,
            "throughput_symbols_per_s": self.symbols_per_second,
        }

    def __repr__(self):
        return (f"ResourceMeter({self.label}: {self.n_evals} evals, "
                f"{self.wall:.2f}s, peak {self.peak_host_mem_mb or 0:.0f} MB)")


# Tương thích ngược với `rebuttal_common.QueryMeter` của bộ script cũ.
class QueryMeter(ResourceMeter):
    def __init__(self, label=""):
        super().__init__(label=label, track_memory=False)
        self.start()

    @property
    def n_queries(self):
        return self.n_evals

    @n_queries.setter
    def n_queries(self, v):
        self.n_evals = int(v)
