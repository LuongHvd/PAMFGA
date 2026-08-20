"""
fitness.py
==========
Hàm fitness THỐNG NHẤT cho mọi phương pháp tấn công + kế toán budget.

Trả lời phản biện: "Thiết kế lại comparison theo cùng số fitness evaluations/query
và wall-clock."

Nguyên tắc công bằng (áp dụng cho PAGA, PAMFGA và mọi baseline):
  * cùng ràng buộc năng lượng    -> `normalize_psr` được gọi trong CHÍNH hàm fitness,
                                    nên không phương pháp nào lách được ràng buộc PSR
  * cùng định nghĩa fitness      -> BLER trên `batch` symbol tại `ebnodb`
  * cùng đơn vị chi phí          -> 1 lần gọi fitness = 1 query = 1 fitness evaluation
  * cùng điều kiện dừng          -> `Budget` (số evaluation hoặc wall-clock)
  * cùng khởi tạo                -> vector init chung cho mỗi seed (do script truyền vào)

Hai chế độ fitness:
  "batch" (mặc định)  BLER trên `fit_batch` symbol ngẫu nhiên. Phương sai thấp,
                      phù hợp để so sánh optimizer một cách công bằng.
  "paper"             Xấp xỉ BLER bằng đúng M = 2^k symbol với một hiện thực nhiễu
                      mỗi symbol -- chính là Algorithm 1 của bài. Rẻ nhưng rất nhiễu
                      (độ phân giải 1/M).

Đánh giá CUỐI CÙNG luôn tách khỏi fitness: dùng batch lớn hơn và nhiều lần lặp
độc lập (`bler_curve`), tránh việc một phương pháp "thắng" chỉ vì khớp quá mức
vào batch dùng khi tìm kiếm.
"""
import numpy as np

from .budget import ResourceMeter


# --------------------------------------------------------------------------- #
# Ràng buộc năng lượng PSR
# --------------------------------------------------------------------------- #
def normalize_psr(vec, n, psr_db, eps=1e-8):
    """
    Chiếu perturbation về đúng ràng buộc PSR (giống hệt `AE_MLP.normalize` của bài):
        p <- sqrt(PSR·n / ||p||²) · p
    `vec` nhận dạng (2n,) hoặc (1,2,n). Trả (1,2,n).
    """
    p = np.asarray(vec, dtype=np.float64).reshape(1, 2, n)
    psr = 10 ** (psr_db / 10.0)
    scale = np.sqrt((psr * n) / (np.linalg.norm(p) ** 2 + eps))
    return scale * p


def flat_to_p(vec, n):
    return np.asarray(vec, dtype=np.float64).reshape(1, 2, n)


def p_to_flat(p):
    return np.asarray(p, dtype=np.float64).reshape(-1)


def random_perturbation(n, psr_db, rng, kind="normal", k=4):
    """Khởi tạo ngẫu nhiên trong không gian perturbation, đã chuẩn hoá PSR."""
    if kind == "normal":
        sigma = np.sqrt(2 * (k / n) * 10 ** (psr_db / 10.0))
        v = rng.normal(0.0, sigma, size=(1, 2, n))
    else:
        v = rng.uniform(-1.0, 1.0, size=(1, 2, n))
    return normalize_psr(v, n, psr_db)


def gaussian_jamming(p_ref, rng):
    """Jamming Gauss cùng NĂNG LƯỢNG với perturbation tham chiếu (như code gốc)."""
    z = rng.standard_normal(np.shape(p_ref))
    return np.linalg.norm(p_ref) * z / (np.linalg.norm(z) + 1e-12)


# --------------------------------------------------------------------------- #
# Oracle: mọi thứ có thể trả về BLER cho một perturbation
# --------------------------------------------------------------------------- #
class AWGNOracle:
    """Model AWGN gốc của bài (AE_MLP hoặc bất kỳ model nào có `test_step`)."""

    def __init__(self, model, n=None, name="AWGN"):
        self.model = model
        self.n = n if n is not None else model.n
        self.name = name

    def bler(self, p, batch, ebnodb):
        return float(self.model.test_step(p, int(batch), ebnodb))


class ZooOracle:
    """Model trong zoo kiến trúc (chữ ký test_step khác nhau -> qua adapter)."""

    def __init__(self, model, arch_name, n=None):
        self.model = model
        self.arch_name = arch_name
        self.n = n if n is not None else model.n
        self.name = arch_name

    def bler(self, p, batch, ebnodb):
        from . import oracles as models
        return models.bler_of(self.model, self.arch_name, p, int(batch), ebnodb)


class ChannelOracle:
    """
    Model với kênh mở rộng (fading / path loss / CSI / đồng bộ).

    `rng` được giữ nguyên giữa các lần gọi để mỗi truy vấn thấy một hiện thực
    kênh độc lập -- đúng với giả định kênh quasi-static thay đổi giữa các block.
    """

    def __init__(self, chan_model, spec, rng=None, name=None):
        self.model = chan_model
        self.spec = spec
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.n = chan_model.n
        self.name = name or spec.name

    def bler(self, p, batch, ebnodb):
        return self.model.bler_channel(p, int(batch), ebnodb, self.spec, self.rng)


# --------------------------------------------------------------------------- #
# Hàm fitness
# --------------------------------------------------------------------------- #
class Fitness:
    """
    Đối tượng fitness gọi được: `f(vec) -> BLER` (càng cao attacker càng lợi).

    Tự động: chuẩn hoá PSR, đếm query/symbol vào `meter`, và ghi nhớ cá thể tốt nhất
    đã từng được đánh giá (`best_vec` / `best_f`) để báo cáo.
    """

    def __init__(self, oracle, n, psr_db, ebnodb=1.0, batch=2000, meter=None,
                 mode="batch", k=4, budget=None, label=""):
        self.oracle = oracle
        self.n = int(n)
        self.psr_db = float(psr_db)
        self.ebnodb = float(ebnodb)
        self.mode = mode
        self.k = int(k)
        self.M = 2 ** self.k
        self.batch = int(self.M if mode == "paper" else batch)
        self.meter = meter if meter is not None else ResourceMeter(label)
        self.budget = budget
        self.label = label or getattr(oracle, "name", "")
        self.best_vec = None
        self.best_f = -np.inf

    def __call__(self, vec):
        p = normalize_psr(vec, self.n, self.psr_db)
        val = self.oracle.bler(p, self.batch, self.ebnodb)
        self.meter.tick(1, symbols=self.batch)
        if val > self.best_f:
            self.best_f = float(val)
            self.best_vec = np.asarray(vec, dtype=float).reshape(-1).copy()
        return float(val)

    # -- tiện ích ---------------------------------------------------------- #
    @property
    def n_evals(self):
        return self.meter.n_evals

    def exhausted(self):
        return self.budget is not None and self.budget.exhausted(self.meter)

    def remaining(self):
        return self.budget.remaining_evals(self.meter) if self.budget else 1 << 30

    def config(self):
        return {"oracle": getattr(self.oracle, "name", type(self.oracle).__name__),
                "n": self.n, "psr_db": self.psr_db, "ebnodb": self.ebnodb,
                "mode": self.mode, "batch": self.batch}


class MultiTaskFitness:
    """
    Bộ T hàm fitness dùng CHUNG một bộ đếm budget -- nền tảng cho PAMFGA hộp đen,
    nơi mỗi task tương ứng với một autoencoder thay thế (surrogate).

    Điều then chốt cho tính công bằng: mọi truy vấn tới BẤT KỲ surrogate nào đều
    tính vào cùng một `meter`, nên PAMFGA với T task KHÔNG được dùng nhiều query
    hơn phương pháp đơn task ở cùng budget.
    """

    def __init__(self, oracles, n, psr_db, ebnodb=1.0, batch=2000, meter=None,
                 mode="batch", k=4, budget=None):
        self.meter = meter if meter is not None else ResourceMeter("multitask")
        self.budget = budget
        self.tasks = [
            Fitness(o, n, psr_db, ebnodb, batch, self.meter, mode, k, budget,
                    label=getattr(o, "name", f"task{i}"))
            for i, o in enumerate(oracles)
        ]
        self.n = int(n)
        self.psr_db = float(psr_db)
        self.T = len(self.tasks)

    def __len__(self):
        return self.T

    def __getitem__(self, i):
        return self.tasks[i]

    def evaluate(self, vec, task):
        return self.tasks[int(task)](vec)

    @property
    def n_evals(self):
        return self.meter.n_evals

    def exhausted(self):
        return self.budget is not None and self.budget.exhausted(self.meter)

    def remaining(self):
        return self.budget.remaining_evals(self.meter) if self.budget else 1 << 30

    def config(self):
        return {"n_tasks": self.T, "tasks": [t.config() for t in self.tasks]}


def make_fitness(oracle, n, psr_db, ebnodb=1.0, batch=2000, meter=None,
                 mode="batch", k=4, budget=None, label=""):
    """Hàm tiện lợi tạo `Fitness`."""
    return Fitness(oracle, n, psr_db, ebnodb, batch, meter, mode, k, budget, label)


# --------------------------------------------------------------------------- #
# Đánh giá cuối cùng (KHÔNG tính vào budget tìm kiếm)
# --------------------------------------------------------------------------- #
def bler_curve(oracle, p, n, psr_db, ebnodbs, batch=20000, iters=20):
    """
    BLER-vs-Eb/N0 của một perturbation cố định, trung bình `iters` lần độc lập.
    Dùng để BÁO CÁO, luôn tách khỏi hàm fitness dùng khi tìm kiếm.
    """
    p = normalize_psr(p, n, psr_db)
    out = np.zeros(len(ebnodbs), dtype=float)
    for _ in range(int(iters)):
        out += np.array([oracle.bler(p, batch, e) for e in ebnodbs]) / float(iters)
    return out


def clean_bler_curve(oracle, n, ebnodbs, batch=20000, iters=20):
    """BLER khi KHÔNG tấn công (đường tham chiếu để tính marginal)."""
    z = np.zeros([1, 2, n])
    out = np.zeros(len(ebnodbs), dtype=float)
    for _ in range(int(iters)):
        out += np.array([oracle.bler(z, batch, e) for e in ebnodbs]) / float(iters)
    return out
