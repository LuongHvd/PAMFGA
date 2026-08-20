"""
models.py
=========
Nạp model và tạo oracle, dùng chung cho mọi thực nghiệm.

Ba nhóm model:
  1. TARGET      autoencoder đích của bài (checkpoint `mlp_ae_k_4_n_7`).
                 Hộp trắng: attacker truy cập được cả gradient.
                 Hộp đen  : attacker KHÔNG được truy vấn model này khi tìm kiếm;
                            nó chỉ dùng ở bước ĐÁNH GIÁ cuối.
  2. SURROGATE   T autoencoder thay thế do attacker tự huấn luyện. Đây là các TASK
                 của PAMFGA. Chúng chỉ dùng chung cấu hình (η,k) và mô hình kênh
                 với target -- không dùng bất kỳ tham số/truy vấn nào của target.
  3. PAPER_MODELS  năm autoencoder mà bài nhắc tới (MLP_base, MLP_relu, MLP_deep,
                 MLP_widedec, CNN), dùng khi cần đổi kiến trúc đích. Định nghĩa
                 lớp nằm ở `autoencoders.py`.

`WhiteBoxOracle` dựng SẴN MỘT LẦN node gradient trong graph rồi tái sử dụng. Bản
gốc `AE_MLP.fgm_attack` gọi `tf.gradients` bên trong vòng lặp nên graph phình ra
và chậm dần; ở đây kết quả toán học không đổi nhưng chi phí giảm mạnh, và số lần
truy vấn gradient được đếm RIÊNG để chi phí hộp trắng của PAGA không bị giấu.
"""
import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.ERROR)

from .autoencoders import (AE_MLP, AE_CNN, AE_netOne_MLP,
                                     AE_netTwo_DeepMLP, AE_inf_rate)

from .channels import AE_MLP_Channel

DEFAULT_K = 4
DEFAULT_N = 7
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Checkpoint là SẢN PHẨM, không phải mã nguồn: thư mục `checkpoints/` được tạo
# khi huấn luyện và không commit. Sinh lại bằng `train_awgn.py` / `train_channel.py`.
# Bản đã huấn luyện trước đây được lưu ở `legacy/checkpoints/`.
CKPT_DIR = os.path.join(ROOT, "checkpoints")
DEFAULT_CKPT = os.path.join(CKPT_DIR, "mlp_ae_k_4_n_7")
SURROGATE_CKPT_DIR = os.path.join(CKPT_DIR, "surrogate_ckpt")


# --------------------------------------------------------------------------- #
# Các autoencoder của bài
# --------------------------------------------------------------------------- #
# Đây là toàn bộ kiến trúc đích mà bài báo nhắc tới; định nghĩa lớp nằm ở
# `autoencoders.py`. Cột `needs_training_flag` đánh dấu những lớp có dropout nên
# `test_step`/`train` của chúng nhận thêm cờ is_training.
#
#   tên            lớp                  họ               cờ     trục thay đổi
PAPER_MODELS = {
    "MLP_base":    (AE_MLP,            "in-family",     False, "gốc (surrogate)"),
    "MLP_relu":    (AE_netOne_MLP,     "in-family",     False, "activation"),
    "MLP_deep":    (AE_netTwo_DeepMLP, "in-family",     True,  "depth+dropout"),
    "MLP_widedec": (AE_inf_rate,       "in-family",     False, "decoder width"),
    "CNN":         (AE_CNN,            "out-of-family", True,  "convolutional"),
}


def model_names():
    return list(PAPER_MODELS)


def family_of(name):
    return PAPER_MODELS[name][1]


def axis_of(name):
    return PAPER_MODELS[name][3]


def build(name, k=DEFAULT_K, n=DEFAULT_N, seed=0, ckpt=None):
    """Dựng (và tuỳ chọn nạp checkpoint) một autoencoder của bài."""
    return PAPER_MODELS[name][0](k, n, seed, filename=ckpt)


def bler_of(model, name, p, batch, ebnodb):
    """Đo BLER, che đi khác biệt chữ ký `test_step` giữa các lớp."""
    if PAPER_MODELS[name][2]:
        return float(model.test_step(False, 0.0, p, int(batch), ebnodb))
    return float(model.test_step(p, int(batch), ebnodb))


def train_model(model, name, tp, vp, dr_out=0.1):
    p = np.zeros([1, 2, model.n])
    if PAPER_MODELS[name][2]:
        model.train(True, dr_out, p, tp, vp)
    else:
        model.train(p, tp, vp)


def default_train_params(quick=True, name=None):
    """
    Lịch huấn luyện. Mỗi phần tử: [batch_size, lr, EbNo(dB), số vòng lặp].
    CNN chậm hơn ~120x mỗi vòng nên có lịch ngắn riêng.
    """
    if name == "CNN":
        if quick:
            tp = [[400, 0.001, 8.5, 300], [800, 0.0005, 8.5, 800]]
            vp = [[50000, 8.5, 1000]] * 2
        else:
            tp = [[1000, 0.001, 8.5, 500], [3000, 0.0005, 8.5, 1000],
                  [4000, 0.0001, 8.5, 1000]]
            vp = [[100000, 8.5, 1000]] * 3
        return tp, vp
    if quick:
        tp = [[1000, 0.001, 8.5, 400], [5000, 0.0005, 8.5, 1500],
              [10000, 0.0001, 8.5, 1500]]
        vp = [[50000, 8.5, 1000]] * 3
    else:
        tp = [[1000, 0.001, 8.5, 1000], [1000, 0.0001, 8.5, 10000],
              [10000, 0.0001, 8.5, 10000], [10000, 0.00001, 8.5, 10000]]
        vp = [[100000, 8.5, 1000]] * 4
    return tp, vp


# --------------------------------------------------------------------------- #
# Nạp model
# --------------------------------------------------------------------------- #
def load_target(k=DEFAULT_K, n=DEFAULT_N, seed=0, ckpt=DEFAULT_CKPT):
    """Autoencoder đích của bài (AWGN)."""
    return AE_MLP(k, n, seed, filename=ckpt)


def load_channel_target(k=DEFAULT_K, n=DEFAULT_N, seed=0, ckpt=DEFAULT_CKPT):
    """Cùng trọng số pretrain nhưng graph có kênh mở rộng."""
    return AE_MLP_Channel(k, n, seed, filename=ckpt)


def surrogate_ckpt_path(idx, k=DEFAULT_K, n=DEFAULT_N):
    return os.path.join(SURROGATE_CKPT_DIR, f"surrogate_k{k}_n{n}_{idx:02d}")


def surrogate_exists(idx, k=DEFAULT_K, n=DEFAULT_N):
    return os.path.exists(surrogate_ckpt_path(idx, k, n) + ".index")


def load_surrogates(T, k=DEFAULT_K, n=DEFAULT_N, verbose=True):
    """
    Nạp T autoencoder thay thế đã huấn luyện độc lập (các TASK của PAMFGA).

    Nếu chưa có checkpoint -> báo lỗi kèm hướng dẫn, vì huấn luyện tại chỗ sẽ làm
    thời gian chạy của PAMFGA lẫn với thời gian huấn luyện và phá vỡ so sánh
    wall-clock công bằng.
    """
    missing = [i for i in range(T) if not surrogate_exists(i, k, n)]
    if missing:
        raise FileNotFoundError(
            f"Thiếu checkpoint surrogate {missing} trong {SURROGATE_CKPT_DIR}.\n"
            f"Chạy trước:  python -m paga.train_models --surrogates {T}")
    out = []
    for i in range(T):
        if verbose:
            print(f"  nạp surrogate {i}", flush=True)
        out.append(AE_MLP(k, n, 100 + i, filename=surrogate_ckpt_path(i, k, n)))
    return out


# --------------------------------------------------------------------------- #
# Oracle gradient (hộp trắng)
# --------------------------------------------------------------------------- #
class WhiteBoxOracle:
    """
    Truy cập gradient cho khởi tạo elite FGM của PAGA (Mục 4.1 của bài).

    Node gradient được thêm vào graph ĐÚNG MỘT LẦN trong hàm khởi tạo, sau đó
    tái sử dụng. Mỗi lần lấy gradient được đếm vào `meter.tick_grad()`, tách bạch
    khỏi số query fitness.
    """

    def __init__(self, model, meter=None):
        self.model = model
        self.n = model.n
        self.k = model.k
        self.M = model.M
        self.meter = meter
        with model.graph.as_default():
            self._grad_y = tf.gradients(model.vars['cross_entropy'], model.vars['y'])
            self._probs = tf.nn.softmax(model.vars['s_hat'])

    # -- nguyên thuỷ -------------------------------------------------------- #
    def extra_feed(self):
        """Feed bổ sung cần thiết để chạy tới `y`. Graph AWGN không cần gì thêm."""
        return {}

    def forward_y(self, s, p, ebnodb):
        """Tín hiệu thu y ứng với symbol s dưới perturbation p."""
        fd = {self.model.vars['s']: s, self.model.vars['p']: p,
              self.model.vars['noise_std']: self.model.EbNo2Sigma(ebnodb)}
        fd.update(self.extra_feed())
        return self.model.sess.run(self.model.vars['y'], feed_dict=fd
                                   ).reshape(1, 2, self.n)

    def predict_from_y(self, y):
        return self.model.sess.run(self._probs, feed_dict={self.model.vars['y']: y})

    def grad_from_y(self, y, label):
        """∇_y CE(y, label). Đếm 1 truy vấn gradient."""
        g = self.model.sess.run(
            self._grad_y,
            feed_dict={self.model.vars['y']: y,
                       self.model.vars['s']: np.asarray(label).reshape(1)})
        if self.meter is not None:
            self.meter.tick_grad(1)
        return -1.0 * np.asarray(g).reshape(1, 2, self.n)

    # -- FGM cho một symbol ------------------------------------------------- #
    def fgm_single(self, s, p, ebnodb, max_bisect=30):
        """
        Perturbation FGM chuẩn tắc cho một symbol (Sadeghi & Larsson), dùng tìm
        kiếm nhị phân trên biên độ để lấy ε nhỏ nhất gây phân loại sai.
        """
        y = self.forward_y(s, p, ebnodb)
        eps_acc = 1e-7 * np.linalg.norm(y)
        eps_vec = np.zeros([self.M])
        for cls in range(self.M):
            direction = self.grad_from_y(y, cls)
            direction = direction / (np.linalg.norm(direction) + 1e-12)
            e_max, e_min = float(np.linalg.norm(y)), 0.0
            eps = e_max
            for _ in range(max_bisect):
                if e_max - e_min <= eps_acc:
                    break
                eps = 0.5 * (e_max + e_min)
                probs = self.predict_from_y(y + eps * direction)
                if np.argmax(probs) == s:
                    e_min = eps
                else:
                    e_max = eps
            eps_vec[cls] = eps + eps_acc
        false_cls = int(np.argmin(eps_vec))
        d = self.grad_from_y(y, false_cls)
        d = d / (np.linalg.norm(d) + 1e-12)
        return float(np.min(eps_vec)) * d

    # -- UAP-FGM (khởi tạo elite của PAGA / baseline FGMA) ------------------ #
    def uap_fgm(self, ebnodb, num_samples, psr_db, rng=None):
        """
        Universal adversarial perturbation theo thuật toán FGM tích luỹ (Alg. 1
        của Sadeghi & Larsson) -- cũng chính là baseline FGMA và là hạt giống
        elite trong khởi tạo lai của PAGA.
        """
        rng = rng or np.random.default_rng(self.model.seed or 0)
        uap = np.zeros([1, 2, self.n])
        psr = 10 ** (psr_db / 10.0)
        for _ in range(int(num_samples)):
            s = np.asarray([int(rng.integers(0, self.M))]).reshape(1)
            y = self.forward_y(s, uap, ebnodb)
            if np.argmax(self.predict_from_y(y)) != s:
                continue                       # symbol đã sai -> không cần thêm
            adv = self.fgm_single(s, uap, ebnodb).reshape(1, 2, self.n)
            cand = uap + adv
            eps_uni = np.sqrt((psr * self.n) / (np.linalg.norm(cand) ** 2 + 1e-8))
            uap = cand if np.linalg.norm(cand) < eps_uni else eps_uni * cand
        return uap


class ChannelWhiteBoxOracle(WhiteBoxOracle):
    """
    Oracle gradient cho `AE_MLP_Channel`, để chế tạo perturbation MATCHED trực
    tiếp trên một kịch bản kênh vẫn có được hạt elite FGMA.

    Chỉ `forward_y` cần feed thêm phần kênh (fading/CSI/path loss); các nhánh
    `predict_from_y` và `grad_from_y` feed thẳng tensor `y` nên bỏ qua khối kênh.

    Không có lớp này thì PAGA chạy trên graph kênh sẽ âm thầm suy biến thành
    PAGA-BB, và bảng matched-vs-mismatched sẽ so hai thuật toán khác nhau.
    """

    def __init__(self, chan_model, spec, rng=None, meter=None):
        super().__init__(chan_model, meter)
        self.spec = spec
        self.rng = rng if rng is not None else np.random.default_rng(0)

    def extra_feed(self):
        return self.model.channel_feed(self.spec, self.rng, batch=1)
