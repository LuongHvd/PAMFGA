"""
attacks
=======
Mọi phương pháp sinh perturbation, chia sẻ CÙNG một giao diện và CÙNG một kế toán
budget, để bảng so sánh trong bài là công bằng.

Giao diện chung:

    run(fitness, D, budget, rng, init=None, ctx=None, **kw) -> AttackResult

    fitness : `paga.fitness.Fitness` (hoặc `MultiTaskFitness` với PAMFGA).
              Mỗi lần gọi = 1 query = 1 fitness evaluation, tự đếm vào meter.
    D       : số chiều bài toán = 2η
    budget  : `paga.profiling.Budget` -- giới hạn theo số evaluation và/hoặc
              wall-clock. MỌI phương pháp dừng theo cùng một tiêu chí.
    rng     : numpy Generator (cùng seed cho mọi phương pháp trong một trial)
    init    : vector khởi tạo dùng chung cho mọi phương pháp trong một trial
    ctx     : phụ thuộc từng phương pháp (n, psr_db, oracle hộp trắng, ...)

Mô hình tri thức của attacker:
    HỘP TRẮNG  PAGA, FGMA  -- được truy cập gradient của model đích
    HỘP ĐEN    PAMFGA và mọi baseline tối ưu -- chỉ dùng giá trị fitness
               (với PAMFGA, fitness còn KHÔNG được lấy từ target mà từ các
                surrogate do attacker tự huấn luyện)
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class AttackResult:
    """Kết quả một lần chạy tấn công."""
    best_vec: np.ndarray
    best_f: float
    history: list = field(default_factory=list)
    extras: dict = field(default_factory=dict)

    def to_dict(self, with_history=True):
        d = {"best_f": float(self.best_f), "extras": self.extras}
        if with_history:
            d["history"] = self.history
        return d


class HistoryLogger:
    """
    Ghi lại đường hội tụ: best-so-far theo SỐ EVALUATION và theo WALL-CLOCK.

    Ghi cả mean/worst của quần thể để dựng lại hình "best / average / worst fitness
    theo generation" như Fig. 5 của bài.
    """

    def __init__(self, meter):
        self.meter = meter
        self.records = []

    def log(self, best_f, pop_fitness=None, generation=None):
        rec = {"evals": int(self.meter.n_evals),
               "wall": float(self.meter.wall),
               "best_f": float(best_f)}
        if pop_fitness is not None and len(pop_fitness):
            vals = np.asarray([v for v in pop_fitness if v is not None], float)
            if vals.size:
                rec["mean_f"] = float(np.mean(vals))
                rec["worst_f"] = float(np.min(vals))
        rec["generation"] = int(generation if generation is not None
                                else self.meter.n_generations)
        self.records.append(rec)
        return rec

    def __len__(self):
        return len(self.records)


# --------------------------------------------------------------------------- #
# Đăng ký phương pháp
# --------------------------------------------------------------------------- #
def _registry():
    """
    CHỈ hai phương pháp đề xuất, cộng một đường tham chiếu.

    "PAGA"/"PAMFGA" là bản bám notebook `reference/gecco-2025.ipynb` -- tức đúng
    code đã sinh ra hình và bảng của bản thảo.

    Các biến thể đối chiếu (`PAGA-paper`, `PAMFGA-paper`, `PAGA-aac`, `PAMFGA-aac`,
    `PAGA-BB`) và toàn bộ baseline (Random Search, DE, PSO, CMA-ES, GA, FGMA,
    RSBA, RMAEP) nằm ở `legacy/`. Chúng KHÔNG được nạp ở đây: repo này chỉ công bố
    hai thuật toán đề xuất.
    """
    from . import algorithms, jamming as jamming_mod
    return {
        "PAGA":   {"fn": algorithms.paga, "threat": "white-box", "proposed": True,
                   "needs": ("n", "psr_db", "ae", "whitebox"),
                   "desc": "GA hộp trắng: khởi tạo lai có hạt elite FGM, elitist "
                           "top-K, PopSize 50, Crossover1/2 + Mutation1/2"},
        "PAMFGA": {"fn": algorithms.pamfga, "threat": "black-box", "proposed": True,
                   "needs": ("n", "psr_db", "ae", "multitask"),
                   "desc": "GA đa nhân tố hộp đen: skill factor + RMP thích nghi "
                           "(khởi tạo 0.3), PopSize 50, G_max 150, gộp SVD"},

        "Jamming": {"fn": jamming_mod.jamming, "threat": "none", "proposed": False,
                    "needs": ("n", "psr_db"),
                    "desc": "[tham chiếu] nhiễu Gauss cùng mức năng lượng"},
    }


def get(name):
    reg = _registry()
    if name not in reg:
        raise KeyError(f"phương pháp '{name}' chưa đăng ký. Có: {sorted(reg)}")
    return reg[name]


def run(name, fitness, D, budget, rng, init=None, ctx=None, **kw):
    """Chạy một phương pháp theo tên."""
    return get(name)["fn"](fitness, D, budget, rng, init=init, ctx=ctx or {}, **kw)


def available(threat=None, proposed=None):
    reg = _registry()
    return [n for n, m in reg.items()
            if (threat is None or m["threat"] == threat)
            and (proposed is None or m["proposed"] == proposed)]


def describe_all():
    lines = [f"{'phương pháp':<14}{'mô hình tri thức':<14}{'đề xuất':<9}mô tả",
             "-" * 92]
    for name, m in _registry().items():
        lines.append(f"{name:<14}{m['threat']:<14}"
                     f"{'có' if m['proposed'] else '':<9}{m['desc']}")
    return "\n".join(lines)


__all__ = ["AttackResult", "HistoryLogger", "run", "get", "available", "describe_all"]
