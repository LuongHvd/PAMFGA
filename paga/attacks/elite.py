"""
elite.py
========
HEURISTIC ELITE dùng chung: hạt giống khởi tạo dựa trên gradient cho PAGA và PAMFGA.

Đây là NGUỒN DUY NHẤT sinh ra perturbation elite. Cả ba nơi dưới đây gọi chung
hàm ở file này, nên perturbation elite của PAGA/PAMFGA **bằng đúng** đầu ra của
baseline FGMA theo cấu tạo, không thể lệch nhau do cài đặt trùng lặp:

    algorithms.paga    -> hạt elite trong khởi tạo lai của PAGA   (Mục 4.1)
    algorithms.pamfga  -> hạt elite theo TỪNG task của PAMFGA     (Mục 5.1)

Nhân UAP-FGM nằm ở `models.WhiteBoxOracle.uap_fgm`; file này chỉ bọc nó lại và
chiếu kết quả về ràng buộc PSR.

Ý tưởng (Mục 4.1 của bài, "Elite (FGM-based) initialization"): chọn ngẫu nhiên m
symbol từ chòm sao, dùng phương pháp fast-gradient dựng perturbation làm sai lệch
các symbol đó, tích luỹ thành một universal perturbation, rồi chèn vào quần thể
ban đầu như cá thể "elite". Nhờ vậy quần thể vừa có đa dạng ngẫu nhiên vừa có một
điểm khởi đầu mang tri thức riêng của bài toán.

VÌ SAO PHẢI GOM VỀ MỘT CHỖ. Trước đây `exp_channel` chế tạo perturbation MATCHED
trên graph kênh mà không có oracle gradient, nên PAGA **âm thầm** suy biến thành
PAGA-BB (0 truy vấn gradient) và bảng matched-vs-mismatched so sai đối tượng.
Ở đây, nếu không dựng được oracle gradient thì hàm CẢNH BÁO rõ ràng và ghi lại
trong `extras`, thay vì bỏ qua trong im lặng.
"""
import numpy as np

from ..fitness import normalize_psr


class EliteUnavailable(RuntimeError):
    """Không dựng được hạt elite vì thiếu quyền truy cập gradient."""


def fgma_elite(whitebox, n, psr_db, ebnodb, num_samples=10, rng=None):
    """
    Perturbation elite theo heuristic FGMA (universal fast-gradient perturbation).

    whitebox : `models.WhiteBoxOracle` (hoặc lớp con cho graph kênh). Mỗi lần lấy
               gradient được đếm vào `whitebox.meter.tick_grad()` nếu có meter,
               nên chi phí hộp trắng không bị giấu vào số query fitness.
    Trả vector phẳng (2n,) đã chiếu về ràng buộc PSR.
    """
    if whitebox is None:
        raise EliteUnavailable(
            "cần oracle gradient (WhiteBoxOracle) để dựng hạt elite FGMA")
    rng = rng if rng is not None else np.random.default_rng(0)
    p = whitebox.uap_fgm(ebnodb, int(num_samples), psr_db, rng)
    return normalize_psr(p, n, psr_db).reshape(-1)


def elite_seeds(whitebox, n, psr_db, ebnodb, n_seeds=1, num_samples=10, rng=None,
                required=False, label=""):
    """
    Sinh `n_seeds` hạt elite độc lập bằng heuristic FGMA.

    required=False -> thiếu oracle thì trả list rỗng KÈM CẢNH BÁO in ra màn hình
                      (gọi từ PAGA: khi đó PAGA suy biến thành PAGA-BB và điều đó
                      phải hiện rõ trong log lẫn trong `extras`).
    required=True  -> thiếu oracle thì ném `EliteUnavailable`.

    Trả (list vector phẳng, thông tin chẩn đoán).
    """
    info = {"requested": int(n_seeds), "produced": 0, "available": whitebox is not None,
            "num_samples": int(num_samples), "source": "FGMA (fast-gradient UAP)"}

    if n_seeds <= 0:
        info["note"] = "không yêu cầu hạt elite (biến thể hộp đen)"
        return [], info

    if whitebox is None:
        msg = (f"[CẢNH BÁO] {label or 'PAGA'}: không có oracle gradient nên KHÔNG "
               f"dựng được hạt elite FGMA.\n"
               f"           Thuật toán đang chạy như biến thể hộp đen (PAGA-BB). "
               f"Nếu muốn đúng PAGA,\n"
               f"           hãy truyền ctx['whitebox'] là một WhiteBoxOracle của "
               f"chính model đang tấn công.")
        if required:
            raise EliteUnavailable(msg)
        print(msg, flush=True)
        info["note"] = "thiếu oracle gradient -> chạy như PAGA-BB"
        return [], info

    rng = rng if rng is not None else np.random.default_rng(0)
    seeds = []
    for _ in range(int(n_seeds)):
        sub = np.random.default_rng(int(rng.integers(1 << 30)))
        seeds.append(fgma_elite(whitebox, n, psr_db, ebnodb, num_samples, sub))
    info["produced"] = len(seeds)
    info["norms"] = [float(np.linalg.norm(s)) for s in seeds]
    return seeds, info
