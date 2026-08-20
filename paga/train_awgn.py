"""
train_models.py
===============
Huấn luyện và LƯU checkpoint cho hai nhóm model phụ trợ:

  --surrogates T   T autoencoder thay thế của attacker (các TASK của PAMFGA).
                   Chúng được khởi tạo và huấn luyện ĐỘC LẬP với nhau và với
                   target; không tham số hay truy vấn nào của target được dùng.

  --zoo NAME       kiến trúc đích cho thực nghiệm chuyển giao đa kiến trúc
                   (zoo gốc + zoo mở rộng: activation / normalization /
                   latent dim / depth).

Vì sao tách khỏi các script thực nghiệm: nếu huấn luyện tại chỗ thì thời gian
huấn luyện sẽ lẫn vào wall-clock của thuật toán tấn công và phá vỡ so sánh chi
phí công bằng. Ở đây huấn luyện một lần, các thực nghiệm chỉ NẠP checkpoint.

Ví dụ
-----
    python -m paga.train_models --surrogates 10
    python -m paga.train_models --zoo all
    python -m paga.train_models --zoo MLP_batchnorm --seeds 3
    python -m paga.train_models --zoo CNN --seeds 3        # chậm, chạy riêng
"""
import argparse
import os
import time

import numpy as np

from . import oracles as models

MODEL_CKPT_DIR = os.path.join(models.CKPT_DIR, "paper_models")


def model_ckpt_path(name, seed):
    return os.path.join(MODEL_CKPT_DIR, f"{name}_seed{seed}")
from .envinfo import describe


def train_surrogates(T, k, n, quick=True, force=False):
    os.makedirs(models.SURROGATE_CKPT_DIR, exist_ok=True)
    tp, vp = models.default_train_params(quick=quick)
    AE_MLP = models.AE_MLP

    for i in range(T):
        path = models.surrogate_ckpt_path(i, k, n)
        if os.path.exists(path + ".index") and not force:
            print(f"[đã có]  {path}")
            continue
        t0 = time.time()
        print(f"[train]  surrogate {i} (seed {100 + i}) ...", flush=True)
        # Seed khác nhau -> khởi tạo và quỹ đạo huấn luyện khác nhau, tức các
        # surrogate có biên quyết định khác nhau (đúng Mục 5.1 của bài).
        m = AE_MLP(k, n, 100 + i)
        m.train(np.zeros([1, 2, n]), tp, vp)
        m.save(path)
        z = np.zeros([1, 2, n])
        b0 = m.test_step(z, 50000, 0)
        b6 = m.test_step(z, 50000, 6)
        print(f"         lưu {path} | BLER sạch: EbNo0={b0:.4f} EbNo6={b6:.5f} "
              f"| {time.time() - t0:.0f}s", flush=True)


def train_paper_models(names, seeds, k, n, quick=True, skip_seed0=False,
                       force=False):
    """Huấn luyện các autoencoder của bài (MLP_base, MLP_relu, ..., CNN)."""
    os.makedirs(MODEL_CKPT_DIR, exist_ok=True)
    reg = models.PAPER_MODELS
    for name in names:
        if name not in reg:
            print(f"[bỏ qua] kiến trúc không có: {name}")
            continue
        tp, vp = models.default_train_params(quick=quick, name=name)
        for sd in range(seeds):
            if skip_seed0 and sd == 0:
                print(f"[bỏ qua] {name} seed0")
                continue
            path = model_ckpt_path(name, sd)
            if os.path.exists(path + ".index") and not force:
                print(f"[đã có]  {path}")
                continue
            t0 = time.time()
            print(f"[train]  {name} seed{sd} "
                  f"[{models.family_of(name)}, trục={models.axis_of(name)}] ...",
                  flush=True)
            m = models.build(name, k, n, seed=sd)
            models.train_model(m, name, tp, vp)
            m.save(path)
            z = np.zeros([1, 2, n])
            b0 = models.bler_of(m, name, z, 20000, 0)
            b6 = models.bler_of(m, name, z, 20000, 6)
            print(f"         lưu {path} | BLER sạch: EbNo0={b0:.4f} EbNo6={b6:.5f} "
                  f"| {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--surrogates", type=int, default=0,
                    help="số autoencoder thay thế cần huấn luyện (task của PAMFGA)")
    ap.add_argument("--models", nargs="*", default=None,
                    help="tên autoencoder của bài, hoặc 'all' "
                         "(MLP_base, MLP_relu, MLP_deep, MLP_widedec, CNN)")
    ap.add_argument("--seeds", type=int, default=3, help="số seed cho mỗi kiến trúc")
    ap.add_argument("--k", type=int, default=models.DEFAULT_K)
    ap.add_argument("--n", type=int, default=models.DEFAULT_N)
    ap.add_argument("--full_train", action="store_true",
                    help="lịch huấn luyện dài (chất lượng cao, chậm)")
    ap.add_argument("--skip_seed0", action="store_true")
    ap.add_argument("--force", action="store_true", help="ghi đè checkpoint đã có")
    args = ap.parse_args()

    print(describe())
    print()

    if args.surrogates:
        train_surrogates(args.surrogates, args.k, args.n,
                         quick=not args.full_train, force=args.force)

    if args.models:
        names = (models.model_names() if args.models == ["all"] else args.models)
        train_paper_models(names, args.seeds, args.k, args.n,
                           quick=not args.full_train, skip_seed0=args.skip_seed0,
                           force=args.force)

    if not args.surrogates and not args.models:
        ap.print_help()
        print("\nKiến trúc đích khả dụng:")
        for nm in models.model_names():
            have = sum(os.path.exists(model_ckpt_path(nm, s) + ".index")
                       for s in range(8))
            print(f"  {nm:<20} {models.family_of(nm):<15} "
                  f"trục={models.axis_of(nm):<22} ({have} checkpoint đã có)")


if __name__ == "__main__":
    main()
