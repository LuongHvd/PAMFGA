"""
train_channel_models.py
=======================
Huấn luyện autoencoder end-to-end **DƯỚI TỪNG KÊNH**, rồi mới dùng cho thực nghiệm
tấn công.

VÌ SAO BẮT BUỘC. Autoencoder end-to-end học chòm sao và bộ giải mã TỐI ƯU CHO KÊNH
mà nó được huấn luyện. Lấy AE huấn luyện trên AWGN rồi thả vào Rayleigh thì phần
lớn suy giảm quan sát được là do **lệch kênh huấn luyện/kiểm tra**, không phải do
kênh và càng không phải do tấn công. Triệu chứng rõ nhất: BLER KHÔNG TẤN CÔNG dưới
Rayleigh lên tới ~0.31, trong khi một AE huấn luyện đúng dưới Rayleigh thấp hơn
hẳn. Mọi "marginal" tính trên receiver hỏng sẵn đều vô nghĩa.

Quy trình đúng:
    1. với mỗi kịch bản kênh -> huấn luyện AE end-to-end DƯỚI kênh đó   (script này)
    2. huấn luyện T surrogate cũng DƯỚI kênh đó (cho kịch bản hộp đen)  (script này)
    3. chạy tấn công trên các model đã khớp kênh                        (exp_channel)

Checkpoint lưu ở `channel_ckpt/<scenario>/target_seed<s>` và
`channel_ckpt/<scenario>/surrogate_<i>`.

Ví dụ
-----
    python -m paga.train_channel_models --scenarios core
    python -m paga.train_channel_models --scenarios all --surrogates 4
    python -m paga.train_channel_models --scenarios all --surrogates 4  # lịch full (mặc định)
"""
import argparse
import os
import time

import numpy as np

from . import oracles as models
from .channels import AE_MLP_Channel, get_scenarios
from .envinfo import describe

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHANNEL_CKPT_DIR = os.path.join(ROOT, "checkpoints", "channel_ckpt")


def ckpt_path(scenario, role, idx=0):
    d = os.path.join(CHANNEL_CKPT_DIR, scenario)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{role}_{idx:02d}")


def ckpt_exists(scenario, role, idx=0):
    return os.path.exists(ckpt_path(scenario, role, idx) + ".index")


def default_channel_train_params(quick=True):
    """
    Lịch huấn luyện dưới kênh. Fading khó hơn AWGN nên cần nhiều vòng hơn lịch
    AWGN gốc. Lịch FULL là MẶC ĐỊNH vì đối chứng AWGN cho thấy lịch quick thiếu
    huấn luyện (0.2331 so với 0.2054 của checkpoint gốc); full đạt 0.2047.
    """
    if quick:
        tp = [[1000, 0.001, 8.5, 600],
              [5000, 0.0005, 8.5, 2000],
              [10000, 0.0001, 8.5, 2000]]
        vp = [[50000, 8.5, 1000]] * 3
    else:
        tp = [[1000, 0.001, 8.5, 2000],
              [5000, 0.0005, 8.5, 8000],
              [10000, 0.0001, 8.5, 8000],
              [10000, 0.00001, 8.5, 4000]]
        vp = [[100000, 8.5, 2000]] * 4
    return tp, vp


def train_one(scenario, spec, role, idx, k, n, quick, force, verbose=True):
    path = ckpt_path(scenario, role, idx)
    if os.path.exists(path + ".index") and not force:
        print(f"  [đã có] {role}_{idx:02d}")
        return None
    tp, vp = default_channel_train_params(quick)
    seed = (idx if role == "target" else 100 + idx)
    t0 = time.time()
    print(f"  [train] {scenario}/{role}_{idx:02d} ...", flush=True)
    m = AE_MLP_Channel(k, n, seed)          # khởi tạo mới, KHÔNG nạp AWGN
    m.train_channel(spec, tp, vp, rng=np.random.default_rng(1000 + idx),
                    verbose=False)
    m.save(path)
    rng = np.random.default_rng(7)
    z = np.zeros([1, 2, n])
    b0 = m.bler_channel(z, 50000, 0.0, spec, rng)
    b8 = m.bler_channel(z, 50000, 8.0, spec, rng)
    print(f"          lưu {os.path.relpath(path, ROOT)} | BLER sạch: "
          f"EbNo0={b0:.4f} EbNo8={b8:.5f} | {time.time() - t0:.0f}s", flush=True)
    return {"scenario": scenario, "role": role, "idx": idx,
            "bler_clean_ebno0": float(b0), "bler_clean_ebno8": float(b8)}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", choices=["core", "extended", "all"], default="core")
    ap.add_argument("--only", nargs="*", default=None, help="chỉ vài kịch bản")
    ap.add_argument("--target_seeds", type=int, default=1)
    ap.add_argument("--surrogates", type=int, default=0,
                    help="số surrogate huấn luyện dưới cùng kênh (cho hộp đen)")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n", type=int, default=7)
    # Lịch FULL là mặc định: đối chứng dưới AWGN cho thấy lịch quick THIẾU HUẤN
    # LUYỆN (BLER@0dB 0.2331 so với 0.2054 của checkpoint gốc), còn lịch full tái
    # lập đúng checkpoint gốc (0.2047). Dùng quick sẽ làm mọi so sánh kênh bị lẫn
    # với sai số huấn luyện.
    ap.add_argument("--quick_train", action="store_true",
                    help="lịch ngắn — CHỈ để thử pipeline, KHÔNG dùng cho số liệu")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    print(describe())
    if args.quick_train:
        print("\n[CẢNH BÁO] lịch quick THIẾU HUẤN LUYỆN: đối chứng dưới AWGN cho "
              "BLER@0dB 0.2331 so với\n            0.2054 của checkpoint gốc. Số "
              "liệu sẽ lẫn sai số huấn luyện với hiệu ứng\n            kênh. Chỉ "
              "dùng để thử pipeline.")
    scenarios = get_scenarios(args.scenarios)
    if args.only:
        scenarios = {k: v for k, v in scenarios.items() if k in args.only}

    print(f"\nHuấn luyện AE khớp kênh cho {len(scenarios)} kịch bản "
          f"({args.target_seeds} target + {args.surrogates} surrogate mỗi kịch bản)")
    rows = []
    for name, spec in scenarios.items():
        print(f"\n=== {name}: {spec.describe()}")
        for s in range(args.target_seeds):
            r = train_one(name, spec, "target", s, args.k, args.n,
                          args.quick_train, args.force)
            if r:
                rows.append(r)
        for i in range(args.surrogates):
            r = train_one(name, spec, "surrogate", i, args.k, args.n,
                          args.quick_train, args.force)
            if r:
                rows.append(r)

    if rows:
        print(f"\n{'kịch bản':<22}{'vai trò':<12}{'BLER@0dB':>11}{'BLER@8dB':>12}")
        print("-" * 57)
        for r in rows:
            print(f"{r['scenario']:<22}{r['role'] + '_' + str(r['idx']):<12}"
                  f"{r['bler_clean_ebno0']:>11.4f}{r['bler_clean_ebno8']:>12.5f}")
    print("\nxong.")


if __name__ == "__main__":
    main()
