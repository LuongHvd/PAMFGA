"""
gecco.py
========
PAGA và PAMFGA **đúng như notebook `gecco-2025.ipynb`** -- tức code đã thực sự
sinh ra các hình/bảng trong bản thảo. Đây là ĐỊNH NGHĨA CHUẨN của repo.

Vì sao không dùng `All_Autoencoder_Classes.py`: file đó chứa `EAbasedAttack` và
`EAbasedAttack_MFEA`, là các phiên bản CŨ HƠN. Notebook dùng `EAbasedAttackII`
(PAGA cho Fig. 5), `EAbasedAttackIII` (biến thể PopSize 30) và
`EAbasedAttack_MFEAII` (PAMFGA). Ba hàm này KHÔNG có trong file .py.

Khác biệt then chốt của bản notebook so với `EAbasedAttack`:
  * quần thể con được gom vào `new_population` RIÊNG, không append thẳng vào
    `population` đang duyệt;
  * `top_population` được MANG SANG thế hệ sau kèm fitness đã tính -> fitness của
    cha mẹ được CACHE, không đánh giá lại; đây là elitism thật sự và là lý do
    quần thể tụ khít trong Fig. 5;
  * giữ đủ `POPSIZE` cá thể (không phải POPSIZE-1);
  * `parent2 = population[randint(0, POPSIZE)]` (đủ dải).

PAMFGA (`EAbasedAttack_MFEAII`):
  * RMP khởi tạo **0.3** (không phải 0.7/0.95), đường chéo 1.0;
  * RMP cập nhật theo TẦN SUẤT lai chéo, không theo chất lượng con:
        success_ratio = success_count[i][j] / trial_count[i][j]
        rmp[i][j] = (1-alpha)*rmp[i][j] + alpha*success_ratio,   alpha = 0.1
    trong đó `success_count` tăng mỗi khi lai chéo GIỮA HAI TASK KHÁC NHAU xảy ra
    -- tức nó đo "đã lai chéo bao nhiêu", không đo "lai chéo có lợi không";
  * POPSIZE 50, MAXGENERATION **150**;
  * KHÔNG có dOX, SBX, CIM, PM, scalar fitness, age-based selection;
  * lai ghép luôn được thử (không có p_cross riêng): điều kiện là
    `skill_factor1 == skill_factor2 or rand < rmp[s1][s2]`;
  * trả về T perturbation (mỗi task một cái). Bước gộp nằm NGOÀI hàm.

BƯỚC GỘP -- điểm cần đặc biệt lưu ý
-----------------------------------
Notebook gộp bằng:

    P_norm = np.array([p / np.linalg.norm(p) for p in perturbations])  # (T,1,2,n)
    U, S, Vt = np.linalg.svd(P_norm, full_matrices=False)
    MFEA = Vt.T[:, :, :, 0]

`P_norm` là mảng 4 chiều nên NumPy chạy **SVD THEO LÔ trên từng ma trận (2,n)**,
tức SVD RIÊNG từng perturbation. Kết quả KHÔNG phụ thuộc các perturbation khác
(đã kiểm chứng), và có shape `(n,2,1)` chứ không phải `(1,2,n)`. Nghĩa là phép
"gộp đồng thuận" mô tả ở Mục 5.3 KHÔNG phải thứ đã được tính cho EAB/MFEA.
Chỉ nhánh dùng cho UAP (cell 37) mới flatten thành `(T,2n)` rồi gộp đúng.

`aggregate_notebook()` tái lập ĐÚNG hành vi notebook (mặc định), còn
`aggregate_consensus()` là phép gộp đúng nghĩa. Giữ cả hai để so được.
"""
import numpy as np

from ..fitness import normalize_psr
from . import AttackResult, HistoryLogger, elite


def _vec(x):
    return np.asarray(x, dtype=float).reshape(-1)


def _p(x, n):
    return np.asarray(x, dtype=float).reshape(1, 2, n)


# --------------------------------------------------------------------------- #
# Bước gộp
# --------------------------------------------------------------------------- #
def aggregate_notebook(perturbations, n):
    """
    Tái lập ĐÚNG bước gộp của notebook (cell 39/50): SVD theo lô trên từng
    perturbation, rồi lấy `Vt.T[:, :, :, 0]`.

    KHÔNG phải phép gộp đồng thuận: kết quả của một perturbation không phụ thuộc
    các perturbation khác. Trả (vector 2n, thông tin chẩn đoán).
    """
    P = np.array([_p(q, n) for q in perturbations])                  # (T,1,2,n)
    P = np.array([q / (np.linalg.norm(q) + 1e-12) for q in P])
    U, S, Vt = np.linalg.svd(P, full_matrices=False)                 # theo lô
    out = Vt.T[:, :, :, 0]                                           # (n,2,1)
    v = _vec(out)[:2 * n]
    info = {"mode": "gecco_batched_svd",
            "note": "SVD theo lô trên từng perturbation, KHÔNG gộp qua T",
            "raw_shape": list(out.shape)}
    return v, info


def aggregate_consensus(perturbations, n):
    """
    Phép gộp ĐÚNG NGHĨA (cell 37 của notebook, và Eq. 35-37 nếu hiểu theo
    consensus): xếp T perturbation đã chuẩn hoá thành ma trận (T, 2n), lấy vector
    singular phải đầu tiên.
    """
    P = np.array([_vec(q) for q in perturbations])
    P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
    U, S, Vt = np.linalg.svd(P, full_matrices=False)
    v = Vt[0]
    if float(np.sum(P @ v)) < 0:
        v = -v
    energy = float(S[0] ** 2 / max(np.sum(S ** 2), 1e-12))
    return v, {"mode": "consensus_svd", "energy_fraction": energy,
               "singular_values": [float(s) for s in S]}


AGGREGATORS = {"gecco": aggregate_notebook, "consensus": aggregate_consensus}


# --------------------------------------------------------------------------- #
# PAGA = EAbasedAttackII
# --------------------------------------------------------------------------- #
def paga(fitness, D, budget, rng, init=None, ctx=None, pop_size=50,
               crossrate=0.8, mutrate=0.2, fgm_samples=10, **kw):
    """
    Tái lập `EAbasedAttackII` của notebook, dừng theo `Budget` dùng chung.

    ctx cần 'n', 'psr_db', 'ae'; tuỳ chọn 'whitebox' cho hạt giống elite FGM.
    Lịch sử ghi best / median / worst fitness của quần thể mỗi thế hệ -- đúng ba
    đường của Fig. 5 (notebook lấy TRUNG VỊ, không phải trung bình).
    """
    ctx = ctx or {}
    meter = fitness.meter
    logger = HistoryLogger(meter)
    n = int(ctx.get("n", getattr(fitness, "n", D // 2)))
    psr_db = float(ctx.get("psr_db", getattr(fitness, "psr_db", -6.0)))
    ebnodb = float(ctx.get("ebnodb", getattr(fitness, "ebnodb", 1.0)))
    ae = ctx.get("ae")
    if ae is None:
        raise ValueError("PAGA cần ctx['ae'] (model để dùng toán tử gốc)")

    POPSIZE = int(pop_size)
    best_vec, best_f = None, -np.inf

    def evaluate(ind):
        nonlocal best_vec, best_f
        if budget.exhausted(meter):
            return None
        v = _vec(ind)
        f = fitness(v)
        if f > best_f:
            best_f, best_vec = f, v.copy()
        return f

    # --- khởi tạo: 1 elite FGM + POPSIZE/2 Gauss + (POPSIZE/2 - 1) đều ---- #
    seeds, elite_info = elite.elite_seeds(
        ctx.get("whitebox"), n, psr_db, ebnodb, n_seeds=1,
        num_samples=fgm_samples, rng=rng, label="PAGA")
    if seeds:
        first = _p(seeds[0], n)
    elif init is not None:
        first = normalize_psr(init, n, psr_db)
    else:
        first = ae.normalize(rng.normal(0.0, ae.PSR2sigma(psr_db), (1, 2, n)), psr_db)

    population = [first]
    for _ in range(int(POPSIZE / 2)):
        population.append(ae.normalize(
            rng.normal(0.0, ae.PSR2sigma(psr_db), (1, 2, n)), psr_db))
    for _ in range(int(POPSIZE / 2) - 1):
        population.append(ae.normalize(rng.uniform(-1.0, 1.0, (1, 2, n)), psr_db))

    top_population = []          # [(fitness, perturbation)] mang sang thế hệ sau
    gen = 0
    while not budget.exhausted(meter):
        gen += 1
        new_population = []
        for parent1 in population:
            if rng.random() < crossrate:
                parent2 = population[int(rng.integers(0, len(population)))]
                c1, c2 = ae.Crossover1(parent1, parent2)
                new_population += [ae.normalize(c1, psr_db),
                                   ae.normalize(c2, psr_db),
                                   ae.normalize(ae.Crossover2(parent1, parent2),
                                                psr_db)]
            if rng.random() < mutrate:
                new_population.append(ae.normalize(ae.Mutation1(parent1), psr_db))
                new_population.append(ae.normalize(ae.Mutation2(parent1), psr_db))

        # CHỈ con mới được đánh giá; cha mẹ dùng lại fitness đã cache.
        scored, stop = [], False
        for ind in new_population:
            f = evaluate(ind)
            if f is None:
                stop = True
                break
            scored.append((f, ind))
        scored.extend(top_population)
        if not scored:
            break
        scored.sort(key=lambda t: -t[0])
        top_population = scored[:POPSIZE]
        population = [ind for _, ind in top_population]

        fits = [f for f, _ in top_population]
        meter.tick_generation()
        rec = logger.log(fits[0], fits, gen)
        # Fig. 5 của notebook vẽ best / TRUNG VỊ / worst, không phải trung bình.
        rec["median_f"] = float(fits[len(fits) // 2])
        rec["worst_f"] = float(fits[-1])
        if stop:
            break

    final = _vec(ae.normalize(population[0], psr_db)) if population else best_vec
    if final is None:
        final = np.zeros(D)
    return AttackResult(final, float(best_f), logger.records,
                        {"generations": gen, "pop_size": POPSIZE,
                         "variant": "EAbasedAttackII (notebook GECCO)",
                         "elite_info": elite_info,
                         "elitist_carryover": True, "caches_parent_fitness": True,
                         "n_grad_queries": int(meter.n_grad_queries)})


# --------------------------------------------------------------------------- #
# PAMFGA = EAbasedAttack_MFEAII
# --------------------------------------------------------------------------- #
def pamfga(fitness, D, budget, rng, init=None, ctx=None, pop_size=50,
                 rmp_init=0.3, rmp_alpha=0.1, fgm_samples=10,
                 aggregate="gecco", **kw):
    """
    Tái lập `EAbasedAttack_MFEAII` của notebook, dừng theo `Budget` dùng chung.

    fitness   : `MultiTaskFitness` (T surrogate của attacker).
    aggregate : "gecco" (mặc định, tái lập đúng notebook) hoặc "consensus".
    """
    ctx = ctx or {}
    meter = fitness.meter
    logger = HistoryLogger(meter)
    n = int(ctx.get("n", getattr(fitness, "n", D // 2)))
    psr_db = float(ctx.get("psr_db", getattr(fitness, "psr_db", -6.0)))
    ebnodb = float(ctx.get("ebnodb", 1.0))
    ae = ctx.get("ae")
    if ae is None:
        raise ValueError("PAMFGA cần ctx['ae'] (model để dùng toán tử gốc)")

    tasks = fitness.tasks if hasattr(fitness, "tasks") else [fitness]
    T = len(tasks)
    POPSIZE = int(pop_size)
    best_f = -np.inf

    # RMP khởi tạo 0.3 + bộ đếm tần suất, đúng notebook.
    rmp = np.full((T, T), float(rmp_init))
    np.fill_diagonal(rmp, 1.0)
    success_count = np.zeros((T, T))
    trial_count = np.ones((T, T))

    def evaluate(ind, skill):
        nonlocal best_f
        if budget.exhausted(meter):
            return None
        f = tasks[int(skill)](_vec(ind))
        best_f = max(best_f, f)
        return f

    wb_tasks = ctx.get("whitebox_per_task") or []
    population, elite_info = [], {"per_task": []}
    for t in range(T):
        wb_t = wb_tasks[t] if t < len(wb_tasks) else None
        seeds, info = elite.elite_seeds(wb_t, n, psr_db, ebnodb, n_seeds=1,
                                        num_samples=fgm_samples, rng=rng,
                                        label=f"PAMFGA task {t}")
        elite_info["per_task"].append(info)
        first = (_p(seeds[0], n) if seeds else
                 (normalize_psr(init, n, psr_db) if init is not None else
                  ae.normalize(rng.normal(0.0, ae.PSR2sigma(psr_db), (1, 2, n)),
                               psr_db)))
        batch = ([first]
                 + [ae.normalize(rng.normal(0.0, ae.PSR2sigma(psr_db), (1, 2, n)),
                                 psr_db) for _ in range(int(POPSIZE / 2))]
                 + [ae.normalize(rng.uniform(-1.0, 1.0, (1, 2, n)), psr_db)
                    for _ in range(int(POPSIZE / 2) - 1)])
        for ind in batch:
            f = evaluate(ind, t)
            if f is None:
                break
            population.append({"p": ind, "skill": t, "fit": f})

    if not population:
        return AttackResult(np.zeros(D), -np.inf, logger.records,
                            {"note": "hết budget khi khởi tạo", "T": T})

    gen = 0
    while not budget.exhausted(meter):
        gen += 1
        mut_rate = ae.adaptive_mutation_rate(
            gen, int(kw.get("max_generation_ref", 150)))
        new_pop = []
        for ind in list(population):
            if budget.exhausted(meter):
                break
            p1, s1 = ind["p"], ind["skill"]
            other = population[int(rng.integers(0, len(population)))]
            p2, s2 = other["p"], other["skill"]

            # Notebook: lai luôn được thử; điều kiện là cùng task HOẶC rand < RMP.
            if s1 == s2 or rng.random() < rmp[s1][s2]:
                c1, c2 = ae.Crossover1(p1, p2)
                c3 = ae.Crossover2(p1, p2)
                for child, sk in ((c1, s1), (c2, s2), (c3, s1)):
                    cn = ae.normalize(child, psr_db)
                    f = evaluate(cn, sk)
                    if f is None:
                        break
                    new_pop.append({"p": cn, "skill": sk, "fit": f})
                trial_count[s1][s2] += 1
                trial_count[s2][s1] += 1
                if s1 != s2:   # notebook đếm "thành công" khi CÓ lai chéo task
                    success_count[s1][s2] += 1
                    success_count[s2][s1] += 1

            if rng.random() < mut_rate:
                for child in (ae.Mutation1(p1), ae.Mutation2(p1)):
                    cn = ae.normalize(child, psr_db)
                    f = evaluate(cn, s1)
                    if f is None:
                        break
                    new_pop.append({"p": cn, "skill": s1, "fit": f})

        population.extend(new_pop)
        updated = []
        for t in range(T):
            grp = sorted([i for i in population if i["skill"] == t],
                         key=lambda i: -i["fit"])
            updated.extend(grp[:POPSIZE])
        population = updated

        # Cập nhật RMP theo tần suất lai chéo (đúng notebook).
        for i in range(T):
            for j in range(T):
                if i != j:
                    ratio = success_count[i][j] / (trial_count[i][j] + 1e-8)
                    rmp[i][j] = (1 - rmp_alpha) * rmp[i][j] + rmp_alpha * ratio

        meter.tick_generation()
        rec = logger.log(best_f, [i["fit"] for i in population], gen)
        rec["rmp_mean"] = (float(rmp[~np.eye(T, dtype=bool)].mean())
                           if T > 1 else 1.0)

    # --- elite theo task + bước gộp ---------------------------------------- #
    elites, fits, task_ids = [], [], []
    for t in range(T):
        grp = [i for i in population if i["skill"] == t]
        if grp:
            b = max(grp, key=lambda i: i["fit"])
            elites.append(_vec(b["p"]))
            fits.append(b["fit"])
            task_ids.append(t)

    extras = {"T": T, "generations": gen, "pop_size": POPSIZE,
              "variant": "EAbasedAttack_MFEAII (notebook GECCO)",
              "rmp_init": rmp_init,
              "rmp_update": "success_ratio (tần suất lai chéo)",
              "rmp_final": rmp.tolist(), "elite_info": elite_info,
              "task_ids": task_ids,
              "task_elite_fitness": [float(f) for f in fits],
              "task_elite_vectors": [e.tolist() for e in elites],
              "n_grad_queries": int(meter.n_grad_queries)}
    if not elites:
        return AttackResult(np.zeros(D), float(best_f), logger.records, extras)

    agg_fn = AGGREGATORS.get(aggregate, aggregate_notebook)
    v, agg_info = agg_fn(elites, n)
    extras["aggregate"] = agg_info
    return AttackResult(normalize_psr(v, n, psr_db).reshape(-1), float(best_f),
                        logger.records, extras)
