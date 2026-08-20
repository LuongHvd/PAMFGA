"""
channels.py
===========
Mô hình tín hiệu thu MỞ RỘNG, có kênh attacker->receiver TÁCH RIÊNG khỏi kênh
transmitter->receiver.

Trả lời các phản biện:
  (a) "Mở rộng mô hình nhận tín hiệu có kênh attacker-receiver, path loss và fading"
  (b) "Thêm Rayleigh/Rician, sai lệch đồng bộ hoặc CSI"
  (c) "Thêm Rayleigh và Rician với thông số K-factor/path loss rõ ràng; đánh giá
       matched và mismatched channel conditions"
  (d) "AWGN quá đơn giản"

---------------------------------------------------------------------------
MÔ HÌNH TÍN HIỆU
---------------------------------------------------------------------------
Với mỗi channel use i = 0..n-1 (tín hiệu phức), tín hiệu thu tại receiver:

    r_i = sqrt(G_s)·h_s,i·x_i  +  sqrt(G_a)·h_a,i·p̃_i  +  n_i

trong đó
    x_i     : symbol phát của autoencoder (đã chuẩn hoá công suất trung bình)
    p̃       : perturbation của attacker SAU khi chịu sai lệch đồng bộ:
                  p̃ = shift_τ( p ) · exp(j(θ + 2π·ε·i))
              với  θ = lệch pha sóng mang, ε = CFO chuẩn hoá, τ = lệch định thời
              (số channel use bị trượt)
    h_s,h_a : hệ số fading Rician độc lập của tuyến tín hiệu và tuyến attacker,
              h = sqrt(K/(K+1)) + sqrt(1/(K+1))·CN(0,1);  K=0 -> Rayleigh,
              fading tắt -> h=1 (AWGN)
    G_s,G_a : độ lợi path loss (tuyến tín hiệu / tuyến attacker), khai báo bằng dB
    n_i     : AWGN CN(0, 2σ²) đúng như model gốc

Receiver cân bằng zero-forcing theo CSI ƯỚC LƯỢNG của tuyến tín hiệu:

    y_i = r_i / ( sqrt(G_s)·ĥ_s,i ),    ĥ_s = h_s + e,  e ~ CN(0, csi_sigma²)

Khi fading tắt, G_s=G_a=1, θ=ε=τ=0 và CSI hoàn hảo thì y = x + n + p, tức TRÙNG
KHỚP graph AWGN gốc của bài. `sanity_check_awgn()` kiểm tra đúng điều này.

---------------------------------------------------------------------------
CÔNG SUẤT NHIỄU LOẠN THỰC SỰ ĐẾN ĐƯỢC RECEIVER
---------------------------------------------------------------------------
PSR trong bài là ràng buộc tại MÁY PHÁT của attacker. Khi có path loss, PSR quan
sát được tại receiver là

    PSR_rx(dB) = PSR_tx(dB) + PL_a(dB) - PL_s(dB)

`received_psr_db()` tính đại lượng này; mọi bảng kết quả đều in kèm để không thổi
phồng hiệu lực tấn công khi tuyến attacker bị suy hao.

KHÔNG train lại model: ta dựng graph mới rồi nạp lại đúng bộ trọng số
encoder/decoder đã pretrain, nên mọi kết luận vẫn nói về đúng autoencoder của bài.
"""
import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from dataclasses import dataclass, field, asdict

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.ERROR)

from .autoencoders import AE_MLP


OTA_LIMITATION_NOTE = """\
GIỚI HẠN PHẢI NÊU TRONG BÀI
---------------------------
Toàn bộ kết quả "physical / over-the-air" trong công trình này là SIMULATED
OVER-THE-AIR (additive physical-layer attack được mô phỏng), KHÔNG có đo đạc
phần cứng:
  * không dùng SDR/USRP, không có RF front-end thực, không đo trong buồng
    hoặc thực địa;
  * suy hao, fading, lệch pha/CFO/định thời và lỗi CSI đều được SINH RA theo
    mô hình thống kê ở `channels.py`, không phải đo từ kênh thật;
  * các khiếm khuyết phần cứng (phi tuyến PA, méo I/Q, nhiễu pha dao động nội,
    lỗi lượng tử ADC/DAC) KHÔNG được mô hình hoá;
  * receiver được giả định cân bằng zero-forcing với CSI của tuyến tín hiệu;
    các sơ đồ thu khác (MMSE, ước lượng kênh chung với tín hiệu can nhiễu)
    chưa được xét.
Do đó mọi phát biểu nên dùng "simulated over-the-air" thay cho "over-the-air",
và hiệu lực tấn công nên báo cáo dưới dạng MARGINAL = BLER(có tấn công) -
BLER(không tấn công) để tách phần suy giảm do chính kênh gây ra.
"""


# --------------------------------------------------------------------------- #
# Đặc tả kênh
# --------------------------------------------------------------------------- #
@dataclass
class ChannelSpec:
    """
    Mọi tham số kênh đều khai báo TƯỜNG MINH để đưa thẳng vào bảng cấu hình của bài.

    fading        : "none" | "rayleigh" | "rician"
    K_s_db, K_a_db: K-factor Rician (dB) của tuyến tín hiệu / tuyến attacker.
                    Chỉ dùng khi fading="rician". Rayleigh <=> K = 0 (tức -inf dB).
    pathloss_s_db : suy hao tuyến transmitter->receiver (dB, <= 0 là suy hao)
    pathloss_a_db : suy hao tuyến attacker->receiver   (dB, <= 0 là suy hao)
    csi_sigma     : độ lệch chuẩn lỗi ước lượng kênh (0 = CSI hoàn hảo)
    phase_offset_deg : lệch pha sóng mang của attacker (độ)
    cfo_norm      : CFO chuẩn hoá -> quay pha thêm 2π·cfo_norm·i ở channel use i
    timing_shift  : lệch định thời, tính bằng số channel use (dịch vòng p)
    equalizer     : "zf" (zero-forcing theo ĥ_s) | "none" (không cân bằng)
    """
    name: str = "AWGN"
    fading: str = "none"
    K_s_db: float = 0.0
    K_a_db: float = 0.0
    pathloss_s_db: float = 0.0
    pathloss_a_db: float = 0.0
    csi_sigma: float = 0.0
    phase_offset_deg: float = 0.0
    cfo_norm: float = 0.0
    timing_shift: int = 0
    equalizer: str = "zf"
    note: str = ""

    # -- dẫn xuất ------------------------------------------------------------ #
    @property
    def fading_on(self):
        return self.fading != "none"

    @property
    def K_s(self):
        """K-factor tuyến tín hiệu ở thang tuyến tính (Rayleigh -> 0)."""
        return 10 ** (self.K_s_db / 10.0) if self.fading == "rician" else 0.0

    @property
    def K_a(self):
        return 10 ** (self.K_a_db / 10.0) if self.fading == "rician" else 0.0

    @property
    def amp_s(self):
        return 10 ** (self.pathloss_s_db / 20.0)

    @property
    def amp_a(self):
        return 10 ** (self.pathloss_a_db / 20.0)

    @property
    def theta(self):
        return np.deg2rad(self.phase_offset_deg)

    @property
    def perfect_csi(self):
        return self.csi_sigma <= 0.0

    def received_psr_db(self, psr_tx_db):
        """PSR quan sát tại receiver sau khi tính path loss hai tuyến."""
        return psr_tx_db + self.pathloss_a_db - self.pathloss_s_db

    def to_dict(self):
        d = asdict(self)
        d.update(K_s_linear=self.K_s, K_a_linear=self.K_a,
                 amp_s=self.amp_s, amp_a=self.amp_a)
        return d

    def describe(self):
        bits = [f"fading={self.fading}"]
        if self.fading == "rician":
            bits.append(f"K_s={self.K_s_db:g}dB (K={self.K_s:g})")
            bits.append(f"K_a={self.K_a_db:g}dB (K={self.K_a:g})")
        if self.pathloss_s_db:
            bits.append(f"PL_s={self.pathloss_s_db:g}dB")
        if self.pathloss_a_db:
            bits.append(f"PL_a={self.pathloss_a_db:g}dB")
        if self.csi_sigma:
            bits.append(f"CSI σ={self.csi_sigma:g}")
        else:
            bits.append("CSI hoàn hảo")
        if self.phase_offset_deg:
            bits.append(f"θ={self.phase_offset_deg:g}°")
        if self.cfo_norm:
            bits.append(f"CFO={self.cfo_norm:g}")
        if self.timing_shift:
            bits.append(f"τ={self.timing_shift} channel use")
        bits.append(f"eq={self.equalizer}")
        return ", ".join(bits)


# --------------------------------------------------------------------------- #
# Bộ kịch bản chuẩn
# --------------------------------------------------------------------------- #
def core_scenarios():
    """8 kịch bản cốt lõi (dùng cho bảng chính của bài)."""
    return {
        "AWGN": ChannelSpec(
            name="AWGN", fading="none",
            note="tham chiếu, trùng khớp graph gốc của bài"),
        "Rayleigh": ChannelSpec(
            name="Rayleigh", fading="rayleigh",
            note="fading Rayleigh cả hai tuyến, CSI hoàn hảo"),
        "Rician_K3dB": ChannelSpec(
            name="Rician_K3dB", fading="rician", K_s_db=3.0, K_a_db=3.0,
            note="LOS yếu"),
        "Rician_K10dB": ChannelSpec(
            name="Rician_K10dB", fading="rician", K_s_db=10.0, K_a_db=10.0,
            note="LOS mạnh, tiệm cận AWGN"),
        "Rayleigh_CSIerr0.1": ChannelSpec(
            name="Rayleigh_CSIerr0.1", fading="rayleigh", csi_sigma=0.1,
            note="lỗi ước lượng kênh vừa"),
        "Rayleigh_CSIerr0.2": ChannelSpec(
            name="Rayleigh_CSIerr0.2", fading="rayleigh", csi_sigma=0.2,
            note="lỗi ước lượng kênh nặng"),
        "Rayleigh_sync30deg": ChannelSpec(
            name="Rayleigh_sync30deg", fading="rayleigh", phase_offset_deg=30.0,
            note="lệch pha sóng mang của attacker"),
        "Rayleigh_PLa-6dB": ChannelSpec(
            name="Rayleigh_PLa-6dB", fading="rayleigh", pathloss_a_db=-6.0,
            note="tuyến attacker->receiver suy hao 6 dB"),
    }


def extended_scenarios():
    """Kịch bản bổ sung: CFO, lệch định thời, suy hao nặng, và kênh 'thực tế' gộp."""
    return {
        "Rayleigh_CFO0.02": ChannelSpec(
            name="Rayleigh_CFO0.02", fading="rayleigh", cfo_norm=0.02,
            note="CFO chuẩn hoá -> quay pha luỹ tiến theo channel use"),
        "Rayleigh_timing1": ChannelSpec(
            name="Rayleigh_timing1", fading="rayleigh", timing_shift=1,
            note="attacker lệch 1 channel use so với khung tín hiệu"),
        "Rayleigh_PLa-10dB": ChannelSpec(
            name="Rayleigh_PLa-10dB", fading="rayleigh", pathloss_a_db=-10.0,
            note="attacker ở xa receiver"),
        "Realistic_OTA": ChannelSpec(
            name="Realistic_OTA", fading="rician", K_s_db=6.0, K_a_db=3.0,
            pathloss_a_db=-6.0, csi_sigma=0.1, phase_offset_deg=20.0,
            cfo_norm=0.01, timing_shift=1,
            note="gộp mọi khiếm khuyết: fading + suy hao + CSI + pha + CFO + định thời"),
    }


def all_scenarios():
    d = core_scenarios()
    d.update(extended_scenarios())
    return d


def get_scenarios(which="core"):
    if which == "core":
        return core_scenarios()
    if which == "extended":
        return extended_scenarios()
    if which == "all":
        return all_scenarios()
    raise ValueError(f"bộ kịch bản không hợp lệ: {which}")


# --------------------------------------------------------------------------- #
# Sinh fading / méo đồng bộ
# --------------------------------------------------------------------------- #
def gen_fading(B, n, K, rng, fading_on=True):
    """
    Hệ số Rician K (K=0 -> Rayleigh), chuẩn hoá E[|h|²]=1.
    fading_on=False -> h = 1 (kênh AWGN).
    Trả (h_real, h_imag) dạng float32 shape (B, n).
    """
    if not fading_on:
        return np.ones((B, n), np.float32), np.zeros((B, n), np.float32)
    los = np.sqrt(K / (K + 1.0))
    sca = np.sqrt(1.0 / (K + 1.0)) / np.sqrt(2.0)
    hr = los + sca * rng.standard_normal((B, n))
    hi = sca * rng.standard_normal((B, n))
    return hr.astype(np.float32), hi.astype(np.float32)


def apply_sync_impairments(p, spec):
    """
    Áp sai lệch đồng bộ của attacker lên perturbation TRƯỚC khi phát:
    dịch định thời (dịch vòng), lệch pha sóng mang, và CFO (quay pha luỹ tiến).

    p: (1,2,n) [real; imag]. Trả mảng cùng shape.
    """
    p = np.asarray(p, dtype=np.float64).reshape(1, 2, -1)
    n = p.shape[2]
    pr, pi = p[0, 0, :].copy(), p[0, 1, :].copy()

    if spec.timing_shift:
        sh = int(spec.timing_shift) % n
        pr, pi = np.roll(pr, sh), np.roll(pi, sh)

    idx = np.arange(n, dtype=np.float64)
    ang = spec.theta + 2.0 * np.pi * spec.cfo_norm * idx
    if np.any(ang != 0.0):
        c, s = np.cos(ang), np.sin(ang)
        pr, pi = pr * c - pi * s, pr * s + pi * c

    return np.stack([pr, pi], axis=0).reshape(1, 2, n)


# --------------------------------------------------------------------------- #
# Graph autoencoder có kênh mở rộng
# --------------------------------------------------------------------------- #
class AE_MLP_Channel(AE_MLP):
    """
    AE_MLP với khối kênh đầy đủ (fading / path loss / CSI / đồng bộ).
    Nạp lại đúng trọng số encoder-decoder đã pretrain của bài.
    """

    def create_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.seed)
            batch_size = tf.placeholder(tf.int32, shape=())

            # --- Transmitter: dùng lại encoder gốc (tên biến trùng checkpoint) ---
            s = tf.random_uniform(shape=[batch_size], minval=0, maxval=self.M,
                                  dtype=tf.int64)
            x = self.encoder(s)                     # [B,2,n]
            xr, xi = x[:, 0, :], x[:, 1, :]

            # --- Perturbation (đã áp sai lệch đồng bộ ở phía numpy) ---
            p = tf.placeholder(tf.float32, shape=(None, 2, self.n))
            pr, pi = p[:, 0, :], p[:, 1, :]

            # --- Fading + CSI + path loss ---
            hs_r = tf.placeholder(tf.float32, shape=(None, self.n))
            hs_i = tf.placeholder(tf.float32, shape=(None, self.n))
            ha_r = tf.placeholder(tf.float32, shape=(None, self.n))
            ha_i = tf.placeholder(tf.float32, shape=(None, self.n))
            hhat_r = tf.placeholder(tf.float32, shape=(None, self.n))
            hhat_i = tf.placeholder(tf.float32, shape=(None, self.n))
            a_s = tf.placeholder(tf.float32, shape=())
            a_a = tf.placeholder(tf.float32, shape=())
            use_eq = tf.placeholder(tf.float32, shape=())   # 1 = ZF, 0 = không cân bằng

            noise_std = tf.placeholder(tf.float32, shape=())
            # MỘT op nhiễu duy nhất trên tensor [B,2,n] rồi mới tách I/Q, y hệt
            # graph gốc. Nếu dùng hai op riêng thì cả hai nhận cùng op-seed và
            # sinh RA CÙNG MỘT chuỗi -> nhiễu I và Q tương quan hoàn toàn, làm
            # BLER lệch hàng chục phần trăm so với model gốc.
            noise = tf.random_normal(tf.shape(x), mean=0.0, stddev=noise_std,
                                     seed=self.seed)
            nr, ni = noise[:, 0, :], noise[:, 1, :]

            # Tuyến tín hiệu: sqrt(G_s)·h_s ⊙ x
            sr = a_s * (hs_r * xr - hs_i * xi)
            si = a_s * (hs_r * xi + hs_i * xr)
            # Tuyến attacker: sqrt(G_a)·h_a ⊙ p̃   (p [1,n] broadcast lên [B,n])
            ar = a_a * (ha_r * pr - ha_i * pi)
            ai = a_a * (ha_r * pi + ha_i * pr)

            rr = sr + nr + ar
            ri = si + ni + ai

            # Zero-forcing theo CSI ước lượng: y = r / (sqrt(G_s)·ĥ_s)
            cr = a_s * hhat_r
            ci = a_s * hhat_i
            denom = cr * cr + ci * ci + 1e-12
            yr_eq = (rr * cr + ri * ci) / denom
            yi_eq = (ri * cr - rr * ci) / denom
            yr = use_eq * yr_eq + (1.0 - use_eq) * rr
            yi = use_eq * yi_eq + (1.0 - use_eq) * ri
            y = tf.stack([yr, yi], axis=1)          # [B,2,n]

            # --- Receiver: decoder gốc ---
            s_hat = self.decoder(y)
            correct = tf.equal(tf.argmax(tf.nn.softmax(s_hat), axis=1), s)
            bler = 1 - tf.reduce_mean(tf.cast(correct, tf.float32))
            # Cần cho oracle gradient (heuristic elite FGMA) khi chế tạo
            # perturbation MATCHED trực tiếp trên kênh này. Thiếu node này thì
            # PAGA trên graph kênh sẽ không dựng được hạt elite.
            cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=s,
                                                                   logits=s_hat)

            # --- Đường HUẤN LUYỆN qua chính khối kênh này ---------------- #
            # Bắt buộc phải có: một autoencoder end-to-end chỉ có nghĩa khi được
            # huấn luyện DƯỚI kênh mà nó sẽ hoạt động. Dùng AE huấn luyện trên
            # AWGN rồi thả vào Rayleigh thì phần lớn suy giảm đo được là do lệch
            # kênh huấn luyện/kiểm tra, không phải do kênh hay do tấn công.
            lr = tf.placeholder(tf.float32, shape=())
            train_op = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

            self.vars = {
                'batch_size': batch_size, 's': s, 'x': x, 'y': y, 'p': p,
                'hs_r': hs_r, 'hs_i': hs_i, 'ha_r': ha_r, 'ha_i': ha_i,
                'hhat_r': hhat_r, 'hhat_i': hhat_i, 'a_s': a_s, 'a_a': a_a,
                'use_eq': use_eq, 'noise_std': noise_std, 's_hat': s_hat,
                'bler': bler, 'cross_entropy': cross_entropy,
                'lr': lr, 'train_op': train_op,
                'init': tf.global_variables_initializer(),
            }
            # Chỉ lưu/khôi phục trọng số encoder-decoder (bỏ slot Adam).
            self.saver = tf.train.Saver(
                var_list=[v for v in tf.trainable_variables()
                          if "Adam" not in v.name and "beta" not in v.name])
        return

    # -- huấn luyện AE DƯỚI kênh này --------------------------------------- #
    def train_step_channel(self, batch_size, ebnodb, lr, spec, rng):
        """Một bước huấn luyện; kênh được rút lại mỗi bước (quasi-static theo block)."""
        fd = self.channel_feed(spec, rng, batch=int(batch_size))
        fd[self.vars['p']] = np.zeros((1, 2, self.n), np.float32)   # không tấn công
        fd[self.vars['noise_std']] = self.EbNo2Sigma(ebnodb)
        fd[self.vars['lr']] = float(lr)
        self.sess.run(self.vars['train_op'], feed_dict=fd)

    def train_channel(self, spec, training_params, validation_params, rng=None,
                      verbose=True):
        """
        Huấn luyện end-to-end dưới `spec`. `training_params` mỗi phần tử là
        [batch_size, lr, EbNo(dB), số vòng lặp]; `validation_params` là
        [val_size, val_EbNo, chu kỳ in].
        """
        rng = rng if rng is not None else np.random.default_rng(0)
        for idx, (batch_size, lr, ebnodb, iters) in enumerate(training_params):
            val_size, val_ebnodb, val_every = validation_params[idx]
            if verbose:
                print(f"    batch={batch_size} lr={lr} EbNo={ebnodb} iters={iters}",
                      flush=True)
            for i in range(int(iters)):
                self.train_step_channel(batch_size, ebnodb, lr, spec, rng)
                if verbose and val_every and i % val_every == 0:
                    z = np.zeros((1, 2, self.n))
                    b = self.bler_channel(z, val_size, val_ebnodb, spec, rng)
                    print(f"      it {i:>6}  val BLER={b:.5f}", flush=True)

    # -- một lần đo BLER dưới một kịch bản --------------------------------- #
    def bler_channel(self, p, batch_size, ebnodb, spec, rng):
        """BLER trên 1 batch dưới ChannelSpec `spec`. `rng` sinh fading + lỗi CSI."""
        B, n = int(batch_size), self.n
        hs_r, hs_i = gen_fading(B, n, spec.K_s, rng, spec.fading_on)
        ha_r, ha_i = gen_fading(B, n, spec.K_a, rng, spec.fading_on)

        if spec.perfect_csi:
            hhat_r, hhat_i = hs_r, hs_i
        else:
            cs = spec.csi_sigma / np.sqrt(2.0)
            hhat_r = (hs_r + cs * rng.standard_normal((B, n))).astype(np.float32)
            hhat_i = (hs_i + cs * rng.standard_normal((B, n))).astype(np.float32)

        p_tx = apply_sync_impairments(p, spec).astype(np.float32)

        fd = {
            self.vars['p']: p_tx.reshape(1, 2, n),
            self.vars['batch_size']: B,
            self.vars['noise_std']: self.EbNo2Sigma(ebnodb),
            self.vars['a_s']: np.float32(spec.amp_s),
            self.vars['a_a']: np.float32(spec.amp_a),
            self.vars['use_eq']: np.float32(1.0 if spec.equalizer == "zf" else 0.0),
            self.vars['hs_r']: hs_r, self.vars['hs_i']: hs_i,
            self.vars['ha_r']: ha_r, self.vars['ha_i']: ha_i,
            self.vars['hhat_r']: hhat_r, self.vars['hhat_i']: hhat_i,
        }
        return float(self.sess.run(self.vars['bler'], feed_dict=fd))

    def channel_feed(self, spec, rng, batch=1):
        """
        Feed dict cho phần KÊNH (fading, CSI, path loss, cân bằng) ứng với một
        batch cỡ `batch`. Dùng bởi oracle gradient khi cần chạy tới `y` trên graph
        kênh; các nhánh khác (predict/gradient theo y) feed thẳng `y` nên không cần.
        """
        n = self.n
        hs_r, hs_i = gen_fading(batch, n, spec.K_s, rng, spec.fading_on)
        ha_r, ha_i = gen_fading(batch, n, spec.K_a, rng, spec.fading_on)
        if spec.perfect_csi:
            hhat_r, hhat_i = hs_r, hs_i
        else:
            cs = spec.csi_sigma / np.sqrt(2.0)
            hhat_r = (hs_r + cs * rng.standard_normal((batch, n))).astype(np.float32)
            hhat_i = (hs_i + cs * rng.standard_normal((batch, n))).astype(np.float32)
        return {
            self.vars['batch_size']: int(batch),
            self.vars['a_s']: np.float32(spec.amp_s),
            self.vars['a_a']: np.float32(spec.amp_a),
            self.vars['use_eq']: np.float32(1.0 if spec.equalizer == "zf" else 0.0),
            self.vars['hs_r']: hs_r, self.vars['hs_i']: hs_i,
            self.vars['ha_r']: ha_r, self.vars['ha_i']: ha_i,
            self.vars['hhat_r']: hhat_r, self.vars['hhat_i']: hhat_i,
        }

    def bler_channel_curve(self, p, psr_db, ebnodbs, spec, seed=0,
                           batch_size=4000, iters=10):
        """BLER-vs-Eb/N0 (đã chuẩn hoá PSR) dưới 1 kịch bản, trung bình `iters` lần."""
        from .fitness import normalize_psr
        p = normalize_psr(p, self.n, psr_db)
        rng = np.random.default_rng(seed)
        out = np.zeros(len(ebnodbs), dtype=float)
        for _ in range(int(iters)):
            out += np.array([self.bler_channel(p, batch_size, e, spec, rng)
                             for e in ebnodbs]) / float(iters)
        return out


# --------------------------------------------------------------------------- #
# Kiểm tra tính đúng đắn
# --------------------------------------------------------------------------- #
def sanity_check_awgn(k=4, n=7, ckpt=None, batch=200000, tol=0.01,
                      ebnodbs=(0.0, 4.0, 8.0), psr_db=-6.0, verbose=True):
    """
    Xác nhận graph kênh mở rộng ở chế độ AWGN TRÙNG KHỚP graph gốc của bài.

    So sánh BLER (không tấn công và có tấn công) giữa `AE_MLP` và
    `AE_MLP_Channel` với ChannelSpec("AWGN"). Trả (ok, chi tiết).
    """
    from .fitness import normalize_psr
    from .oracles import DEFAULT_CKPT

    ckpt = DEFAULT_CKPT if ckpt is None else ckpt
    base = AE_MLP(k, n, 0, filename=ckpt)
    chan = AE_MLP_Channel(k, n, 0, filename=ckpt)
    spec = ChannelSpec(name="AWGN", fading="none")
    rng = np.random.default_rng(0)

    zero = np.zeros([1, 2, n])
    pert = normalize_psr(np.random.default_rng(1).standard_normal(2 * n), n, psr_db)

    rows, ok = [], True
    for label, p in (("no-attack", zero), ("attack", pert)):
        for e in ebnodbs:
            b_ref = float(np.mean([base.test_step(p, batch, e) for _ in range(3)]))
            b_new = float(np.mean([chan.bler_channel(p, batch, e, spec, rng)
                                   for _ in range(3)]))
            rel = abs(b_new - b_ref) / max(b_ref, 1e-9)
            good = rel <= tol or abs(b_new - b_ref) < 2e-4
            ok &= good
            rows.append({"case": label, "ebnodb": float(e), "bler_original": b_ref,
                         "bler_channel_graph": b_new, "rel_diff": rel, "ok": bool(good)})
            if verbose:
                print(f"  {label:<10} EbNo={e:>4.1f} dB | gốc={b_ref:.5f} "
                      f"| kênh={b_new:.5f} | lệch tương đối={rel:.2%} "
                      f"{'OK' if good else 'LỆCH!'}")
    if verbose:
        print(f"  => sanity check AWGN: {'ĐẠT' if ok else 'KHÔNG ĐẠT'}")
    return ok, rows


if __name__ == "__main__":
    print(OTA_LIMITATION_NOTE)
    print("\nCác kịch bản kênh:")
    for name, spec in all_scenarios().items():
        print(f"  {name:<22} {spec.describe()}")
        if spec.note:
            print(f"  {'':<22} ({spec.note})")
    print("\nKiểm tra graph kênh ở chế độ AWGN có trùng model gốc không:")
    sanity_check_awgn()
