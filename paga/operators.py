"""
operators.py
============
Toán tử tiến hoá của PAGA / PAMFGA, cài đặt bám sát Mục 4.2 và 5.2 của bài.

Biểu diễn cá thể (Eq. 14): perturbation phức p ∈ C^η được mã hoá thành nhiễm sắc thể
thực 2η chiều `Chro = [c_1, ..., c_{2η}]`. Mọi cá thể phải thoả ràng buộc công suất
(Eq. 15) và được chiếu về miền khả thi bằng phép chuẩn hoá (Eq. 16).

Toán tử lai ghép:
  hx()   Hybrid crossover -- kế thừa theo đoạn + lấy trung bình đoạn giữa (Eq. 19-20)
  sbx()  Simulated binary crossover với chỉ số phân bố η_c (Eq. 21-24)
  dox()  Của riêng PAMFGA: trao đổi cửa sổ liên tục có độ dài điều biến theo RMP
         (Eq. 33); nếu cửa sổ quá ngắn thì áp hoán vị 2-opt cục bộ

Toán tử đột biến:
  cim()  Central inversion mutation -- đảo thứ tự gen quanh một điểm trục (Eq. 26)
  pm()   Polynomial mutation với chỉ số phân bố η_m (Eq. 27-30)

Mọi toán tử làm việc trên vector 1 chiều 2η và KHÔNG tự chuẩn hoá; việc chiếu về
ràng buộc công suất do thuật toán gọi thực hiện qua `fitness.normalize_psr`, để
mỗi phép chuẩn hoá đều tường minh.

Quy ước biên gen: LB = -1, UB = 1 (Mục 4.2 của bài).
"""
import numpy as np

LB = -1.0
UB = 1.0


# --------------------------------------------------------------------------- #
# Lai ghép
# --------------------------------------------------------------------------- #
def hx(p1, p2, rng):
    """
    Hybrid crossover (Eq. 19-20).

    Hai điểm cắt  μ1 ~ U(0, ⌊2η/3⌋),  μ2 ~ U(μ1, 2η) chia nhiễm sắc thể thành ba
    đoạn: đoạn đầu lấy của cha 1, đoạn giữa lấy TRUNG BÌNH hai cha, đoạn cuối lấy
    của cha 2. Trả 2 con (con thứ hai đổi vai trò hai cha).
    """
    p1 = np.asarray(p1, float).reshape(-1)
    p2 = np.asarray(p2, float).reshape(-1)
    D = p1.size
    mu1 = int(rng.integers(0, max(1, D // 3)))
    mu2 = int(rng.integers(mu1, D)) if mu1 < D else D

    c1 = p1.copy()
    c1[mu1:mu2] = 0.5 * (p1[mu1:mu2] + p2[mu1:mu2])
    c1[mu2:] = p2[mu2:]

    c2 = p2.copy()
    c2[mu1:mu2] = 0.5 * (p1[mu1:mu2] + p2[mu1:mu2])
    c2[mu2:] = p1[mu2:]
    return c1, c2


def sbx(p1, p2, rng, eta_c=2.0, lb=LB, ub=UB):
    """
    Simulated binary crossover (Eq. 21-24) với chỉ số phân bố η_c.
    η_c lớn -> con gần cha mẹ (khai thác); η_c nhỏ -> con trải rộng (thăm dò).
    """
    p1 = np.asarray(p1, float).reshape(-1)
    p2 = np.asarray(p2, float).reshape(-1)
    D = p1.size

    lo = np.minimum(p1, p2)
    hi = np.maximum(p1, p2)
    gap = hi - lo
    same = gap < 1e-14                      # hai gen trùng nhau -> giữ nguyên

    gap_safe = np.where(same, 1.0, gap)
    nu1 = 1.0 + 2.0 * (lo - lb) / gap_safe
    nu2 = 1.0 + 2.0 * (ub - hi) / gap_safe
    nu1 = np.maximum(nu1, 1.0)
    nu2 = np.maximum(nu2, 1.0)

    theta1 = 1.0 - nu1 ** (-(eta_c + 1.0))
    theta2 = 1.0 - nu2 ** (-(eta_c + 1.0))
    theta1 = np.clip(theta1, 1e-12, 1.0)
    theta2 = np.clip(theta2, 1e-12, 1.0)

    u = rng.random(D)

    def spread(theta):
        thr = 1.0 / (2.0 * theta)
        small = (2.0 * theta * u) ** (1.0 / (eta_c + 1.0))
        large = (np.maximum(2.0 - 2.0 * theta * u, 1e-12)) ** (1.0 / (eta_c + 1.0))
        return np.where(u <= thr, small, large)

    b1 = spread(theta1)
    b2 = spread(theta2)

    c1 = 0.5 * ((lo + hi) + b1 * (lo - hi))
    c2 = 0.5 * ((lo + hi) + b2 * (hi - lo))
    c1 = np.where(same, p1, c1)
    c2 = np.where(same, p2, c2)
    return c1, c2


def two_opt(chro, rng):
    """Hoán vị 2-opt cục bộ: đảo ngược một đoạn con ngẫu nhiên."""
    c = np.asarray(chro, float).reshape(-1).copy()
    D = c.size
    if D < 3:
        return c
    i, j = sorted(rng.choice(D, 2, replace=False))
    if j - i < 1:
        j = min(D - 1, i + 1)
    c[i:j + 1] = c[i:j + 1][::-1]
    return c


def dox(p1, p2, rng, window_ratio, min_window=2):
    """
    dOX của PAMFGA (Eq. 33): trao đổi một CỬA SỔ LIÊN TỤC độ dài w = W·(2η),
    trong đó tỉ lệ cửa sổ W được điều biến theo giá trị RMP hiện tại của cặp task.
    RMP cao (hai task liên quan chặt) -> cửa sổ lớn -> truyền tri thức mạnh hơn.

    Nếu cửa sổ quá ngắn để tạo đủ biến thiên, con được áp thêm hoán vị 2-opt cục bộ.
    """
    p1 = np.asarray(p1, float).reshape(-1)
    p2 = np.asarray(p2, float).reshape(-1)
    D = p1.size
    w = int(round(float(np.clip(window_ratio, 0.0, 1.0)) * D))

    c1, c2 = p1.copy(), p2.copy()
    if w >= 1:
        start = int(rng.integers(0, max(1, D - w + 1)))
        sl = slice(start, start + w)
        c1[sl], c2[sl] = p2[sl].copy(), p1[sl].copy()
    if w < min_window:
        c1 = two_opt(c1, rng)
        c2 = two_opt(c2, rng)
    return c1, c2


# --------------------------------------------------------------------------- #
# Đột biến
# --------------------------------------------------------------------------- #
def cim(chro, rng):
    """
    Central inversion mutation (Eq. 26).

    Chọn điểm trục k ngẫu nhiên rồi đảo ngược riêng hai phần:
        Chro' = [c_k, c_{k-1}, ..., c_1, c_{2η}, c_{2η-1}, ..., c_{k+1}]
    Giá trị gen được giữ nguyên, chỉ vị trí thay đổi -> khám phá cấu trúc mới của
    vector perturbation mà không đổi năng lượng.
    """
    c = np.asarray(chro, float).reshape(-1)
    D = c.size
    if D < 2:
        return c.copy()
    k = int(rng.integers(1, D))         # 1..D-1 -> cả hai phần đều khác rỗng
    return np.concatenate([c[:k][::-1], c[k:][::-1]])


def pm(chro, rng, eta_m=20.0, p_gene=1.0, lb=LB, ub=UB):
    """
    Polynomial mutation (Eq. 27-30) với chỉ số phân bố η_m.

    p_gene: xác suất đột biến TỪNG gen. Bài mô tả áp cho mọi gen (p_gene=1.0);
    đặt 1/D nếu muốn biến thể thưa theo thông lệ chung.
    """
    c = np.asarray(chro, float).reshape(-1).copy()
    D = c.size
    span = ub - lb

    cc = np.clip(c, lb, ub)
    delta = np.minimum(cc - lb, ub - cc) / span
    delta = np.clip(delta, 0.0, 1.0)

    mu = rng.random(D)
    pow_ = 1.0 / (eta_m + 1.0)
    lo_term = (2.0 * mu + (1.0 - 2.0 * mu) * (1.0 - delta) ** (eta_m + 1.0)) ** pow_ - 1.0
    hi_term = 1.0 - (2.0 - 2.0 * mu + (2.0 * mu - 1.0)
                     * (1.0 - delta) ** (eta_m + 1.0)) ** pow_
    dbar = np.where(mu <= 0.5, lo_term, hi_term)

    if p_gene < 1.0:
        dbar = np.where(rng.random(D) < p_gene, dbar, 0.0)
    return c + dbar * span


# --------------------------------------------------------------------------- #
# Khởi tạo (Mục 4.1: uniform + normal + elite FGM)
# --------------------------------------------------------------------------- #
def init_uniform(D, rng):
    """Khởi tạo ngẫu nhiên đều -> thăm dò rộng không gian tìm kiếm."""
    return rng.uniform(LB, UB, size=D)


def init_normal(D, rng, sigma=1.0):
    """Khởi tạo Gauss đối xứng vòng -> tương ứng Chro ~ CN(0, P_max) sau chuẩn hoá."""
    return rng.normal(0.0, sigma, size=D)


def hybrid_init_population(pop_size, D, rng, elites=(), frac_uniform=0.5):
    """
    Quần thể khởi tạo lai (Mục 4.1): các cá thể elite (FGM) + Gauss + đều.
    Trả list vector CHƯA chuẩn hoá; hàm gọi chịu trách nhiệm chiếu về ràng buộc PSR.
    """
    pop = [np.asarray(e, float).reshape(-1).copy() for e in elites][:pop_size]
    n_rest = max(0, pop_size - len(pop))
    n_uniform = int(round(frac_uniform * n_rest))
    for _ in range(n_rest - n_uniform):
        pop.append(init_normal(D, rng))
    for _ in range(n_uniform):
        pop.append(init_uniform(D, rng))
    return pop[:pop_size]


# --------------------------------------------------------------------------- #
# Gộp nhiều perturbation thành một hướng đồng thuận (Mục 5.3)
# --------------------------------------------------------------------------- #
def svd_consensus(perturbations):
    """
    Gộp SVD của PAMFGA (Eq. 35-37).

    Xếp các perturbation elite của T task đã chuẩn hoá đơn vị thành ma trận
    P_norm (T × 2η), lấy VECTOR SINGULAR PHẢI ĐẦU TIÊN v1 -- hướng cực đại hoá
    tổng bình phương độ khớp với mọi elite:
        v1 = argmax_{||v||=1} Σ_j |q_j^H v|²

    Trả (v1, singular_values, energy_fraction). `energy_fraction` = S0²/ΣS² là
    phần năng lượng mà một hướng đồng thuận duy nhất giải thích được; đây là bằng
    chứng thực nghiệm cho việc dùng SVD, chứ KHÔNG phải bảo đảm lý thuyết rằng v1
    là hướng mạnh nhất với target chưa biết.
    """
    P = np.array([np.asarray(p, float).reshape(-1) for p in perturbations])
    norms = np.linalg.norm(P, axis=1, keepdims=True) + 1e-12
    P = P / norms
    U, S, Vt = np.linalg.svd(P, full_matrices=False)
    energy = float(S[0] ** 2 / max(np.sum(S ** 2), 1e-12))
    v1 = Vt[0]
    # Chọn dấu sao cho v1 cùng chiều với đa số elite (SVD chỉ xác định tới dấu).
    if float(np.sum(P @ v1)) < 0:
        v1 = -v1
    return v1, S, energy


def mean_consensus(perturbations):
    """Gộp bằng TRUNG BÌNH (đã căn dấu) -- dùng làm baseline cho ablation gộp."""
    P = np.array([np.asarray(p, float).reshape(-1) for p in perturbations])
    P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
    ref = P[0]
    aligned = np.array([q if float(np.dot(q, ref)) >= 0 else -q for q in P])
    return aligned.mean(axis=0)


def svd_consensus_complex(perturbations, n):
    """
    Gộp SVD trong MIỀN PHỨC -- đúng Eq. 35-37 của bài, vốn dùng chuyển vị liên hợp:

        P_norm = U Σ V^H,   v1 = argmax_{||v||=1} Σ_j |q_j^H v|²

    KHÁC BIỆT QUAN TRỌNG so với `svd_consensus` (miền thực 2η):
    tiêu chí |q^H v|² BẤT BIẾN với xoay pha toàn cục, còn (q̃ᵀ ṽ)² của miền thực thì
    không. Perturbation elite lấy từ các surrogate huấn luyện độc lập thường cùng
    hướng nhưng LỆCH PHA nhau, nên bản miền thực bỏ sót phần lớn cấu trúc chung:
    trên pool tổng hợp cùng-hướng-lệch-pha, năng lượng đồng thuận đo được là 0.605
    (thực) so với 0.966 (phức).

    `perturbations`: list vector thực 2η dạng [phần thực (n), phần ảo (n)].
    Trả (v1 dạng thực 2η, singular values, energy_fraction).
    """
    P = np.array([np.asarray(q, float).reshape(-1) for q in perturbations])
    C = P[:, :n] + 1j * P[:, n:]
    C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    U, S, Vh = np.linalg.svd(C, full_matrices=False)
    v = Vh[0].conj()
    # SVD phức chỉ xác định tới một pha; căn pha cho nhất quán với đa số elite.
    phase = np.angle(np.sum(C @ v.conj()))
    v = v * np.exp(1j * phase)
    energy = float(S[0] ** 2 / max(np.sum(S ** 2), 1e-12))
    return np.concatenate([v.real, v.imag]), S, energy
