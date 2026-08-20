# all aoutoencoder classes are here
# ---------------------------------------------------------------------------
# HAI HO LOP, HAI BINDING TENSORFLOW KHAC NHAU
#   tf   = tensorflow.compat.v1  -> cac lop graph TF1: AE_MLP, AE_CNN,
#          AE_netOne_MLP, AE_netTwo_DeepMLP, AE_inf_rate (dung tf.placeholder,
#          tf.Session, tf.set_random_seed, tf.layers, ...)
#   tf2  = tensorflow (TF2)      -> lop Keras 3: _AECore, AE_MLP1
#          (dung tf2.GradientTape, tf2.random.set_seed, ...)
# Truoc day ca hai cung dung ten `tf`, lenh import sau ghi de lenh truoc nen
# moi lop graph TF1 hong ngay khi khoi tao:
#     AttributeError: module 'tensorflow' has no attribute 'set_random_seed'
#
# TF_USE_LEGACY_KERAS phai duoc dat TRUOC lan import tensorflow dau tien thi
# `tf.layers` moi kha dung. Lop Keras 3 dung `import keras` doc lap nen khong
# bi anh huong boi bien nay.
# ---------------------------------------------------------------------------
import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


###############################  MLP of Table 1 ###############################
"""
AE_MLP_keras3.py
================
Bản chuyển AE_MLP từ TF1 graph-mode (tf.placeholder / tf.Session / tf.gradients /
tf.layers) sang **Keras 3 + TF2 eager**.

Giữ NGUYÊN API công khai của bản gốc:
    test_step, train, transmit, EbNo2Sigma, PSR2sigma, save, load,
    fgm_attack, UAPattack_fgm, RMAEP, hybrid_attack,
    EAbasedAttack, EAbasedAttack_MFEA, DEbasedAttack,
    bler_sim_attack_AWGN, calculate_BLER, bler_sim_attack_AWGN_proposed,
    fitness, normalize, Crossover1/2, Mutation1/2, adaptive_mutation_rate
THÊM 3 attack tối ưu hoá black-box mới:
    simpleGA, PSO, simple_CMAES

Nạp lại checkpoint TF1 cũ (mlp_ae_k_4_n_7): dùng
    ae = AE_MLP(4, 7, seed=0); ae.load_tf1_checkpoint('mlp_ae_k_4_n_7')
=> tái tạo đúng model gốc (đã kiểm chứng BLER sạch khớp bản TF1).

LƯU Ý: chạy KHÔNG đặt TF_USE_LEGACY_KERAS (để `import keras` là Keras 3).
"""
# import os  (da import o dau file)
# đảm bảo là Keras 3 (gỡ cờ legacy nếu môi trường lỡ đặt)
# (da dat TF_USE_LEGACY_KERAS=1 o dau file; KHONG duoc pop vi tf.layers can no)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import tensorflow as tf2
import keras


# ----------------------------------------------------------------------------- #
#  Mô hình autoencoder (Keras 3 Model subclass)                                  #
# ----------------------------------------------------------------------------- #
class _AECore(keras.Model):
    """Encoder + Decoder. Kênh (noise + perturbation) xử lý ngoài, ở AE_MLP."""

    def __init__(self, k, n, **kw):
        super().__init__(**kw)
        self.k = k
        self.n = n
        self.M = 2 ** k
        # embedding W [M, M]  (tương ứng biến 'Variable' trong checkpoint TF1)
        lim = np.sqrt(6.0 / (self.M + self.M))
        self.W = self.add_weight(
            shape=(self.M, self.M), trainable=True, name="embed_W",
            initializer=keras.initializers.RandomUniform(-lim, lim))
        # dense encoder -> 2n  (tương ứng 'dense')
        self.enc = keras.layers.Dense(2 * n, activation=None, name="enc_dense")
        # decoder: dense_1 (relu) + dense_2 (linear)  (tương ứng 'dense_1','dense_2')
        self.dec1 = keras.layers.Dense(self.M, activation="relu", name="dec_dense1")
        self.dec2 = keras.layers.Dense(self.M, activation=None, name="dec_dense2")

    def encode(self, s):
        """s: int tensor [B] -> x [B, 2, n] (đã chuẩn hoá công suất trung bình)."""
        x = tf2.nn.elu(tf2.gather(self.W, s))          # [B, M]
        x = self.enc(x)                              # [B, 2n]
        x = tf2.reshape(x, [-1, 2, self.n])           # [B, 2, n]
        x = x / tf2.sqrt(2.0 * tf2.reduce_mean(tf2.square(x)))
        return x

    def decode(self, y):
        """y: [B, 2, n] -> logits [B, M]."""
        y = tf2.reshape(y, [-1, 2 * self.n])
        y = self.dec1(y)
        y = self.dec2(y)
        return y


# ----------------------------------------------------------------------------- #
#  Lớp AE_MLP (giữ API cũ)                                                        #
# ----------------------------------------------------------------------------- #
class AE_MLP1(object):
    def __init__(self, k, n, seed=None, filename=None):
        self.k = k
        self.n = n
        self.bits_per_symbol = self.k / self.n
        self.M = 2 ** self.k
        self.seed = seed
        if seed is not None:
            tf2.random.set_seed(seed)
        self.core = _AECore(k, n)
        # build (khởi tạo trọng số) bằng 1 lần gọi thử
        self._build_once()
        self.optimizer = keras.optimizers.Adam(1e-3)
        if filename is not None:
            self.load(filename)

    # ---------- build / IO ----------
    def _build_once(self):
        s0 = tf2.zeros([2], dtype=tf2.int64)
        x0 = self.core.encode(s0)
        _ = self.core.decode(x0)

    def save(self, filename):
        self.core.save_weights(filename + ".weights.h5")
        return filename

    def load(self, filename):
        """Nạp trọng số Keras 3 (.weights.h5). Dùng load_tf1_checkpoint cho ckpt cũ."""
        path = filename if filename.endswith(".weights.h5") else filename + ".weights.h5"
        self.core.load_weights(path)
        return filename

    def load_tf1_checkpoint(self, ckpt_path):
        """Port trọng số từ checkpoint TF1 cũ (Saver) sang model Keras 3."""
        r = tf2.train.load_checkpoint(ckpt_path)
        W  = r.get_tensor("Variable")          # [M, M]
        ek = r.get_tensor("dense/kernel");   eb = r.get_tensor("dense/bias")
        d1k = r.get_tensor("dense_1/kernel"); d1b = r.get_tensor("dense_1/bias")
        d2k = r.get_tensor("dense_2/kernel"); d2b = r.get_tensor("dense_2/bias")
        self.core.W.assign(W)
        self.core.enc.set_weights([ek, eb])
        self.core.dec1.set_weights([d1k, d1b])
        self.core.dec2.set_weights([d2k, d2b])
        return self

    # ---------- tiện ích ----------
    def EbNo2Sigma(self, ebnodb):
        ebno = 10 ** (ebnodb / 10)
        return 1 / np.sqrt(2 * self.bits_per_symbol * ebno)

    def PSR2sigma(self, psr_db):
        ps = 10 ** (psr_db / 10)
        return np.sqrt(2 * self.bits_per_symbol * ps)

    def normalize(self, individual, PSR_dB):
        PSR = 10 ** (PSR_dB / 10)
        scale = np.sqrt((PSR * self.n) / (np.linalg.norm(individual) ** 2 + 1e-8))
        return scale * individual

    def _as_p(self, p):
        """Chuẩn hoá perturbation về tensor float32 [.,2,n]."""
        p = np.asarray(p, dtype=np.float32).reshape(-1, 2, self.n)
        return tf2.convert_to_tensor(p)

    # ---------- forward / BLER ----------
    def _forward_logits(self, s, p, sigma):
        x = self.core.encode(s)                                  # [B,2,n]
        noise = tf2.random.normal(tf2.shape(x), stddev=sigma, dtype=tf2.float32)
        y = x + noise + p                                        # broadcast p [1,2,n]
        return self.core.decode(y), y

    def test_step(self, p, batch_size, ebnodb):
        """BLER trên 1 batch tại 1 mức Eb/No (eager, vector hoá)."""
        sigma = np.float32(self.EbNo2Sigma(ebnodb))
        s = tf2.random.uniform([batch_size], 0, self.M, dtype=tf2.int64)
        logits, _ = self._forward_logits(s, self._as_p(p), sigma)
        pred = tf2.argmax(logits, axis=1)
        bler = 1.0 - tf2.reduce_mean(tf2.cast(tf2.equal(pred, s), tf2.float32))
        return float(bler.numpy())

    def transmit(self, s):
        s = tf2.convert_to_tensor(np.asarray(s).reshape(-1), dtype=tf2.int64)
        return self.core.encode(s).numpy()

    # ---------- train (GradientTape thay train_op) ----------
    def train_step(self, p, batch_size, ebnodb, lr):
        self.optimizer.learning_rate.assign(lr)
        sigma = np.float32(self.EbNo2Sigma(ebnodb))
        pt = self._as_p(p)
        with tf2.GradientTape() as tape:
            s = tf2.random.uniform([batch_size], 0, self.M, dtype=tf2.int64)
            logits, _ = self._forward_logits(s, pt, sigma)
            ce = tf2.reduce_mean(
                keras.losses.sparse_categorical_crossentropy(s, logits, from_logits=True))
        grads = tape.gradient(ce, self.core.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.core.trainable_variables))

    def train(self, p, training_params, validation_params):
        for index, params in enumerate(training_params):
            batch_size, lr, ebnodb, iterations = params
            print(f"\nBatch Size: {batch_size}, Learning Rate: {lr}, "
                  f"EbNodB: {ebnodb}, Iterations: {iterations}")
            val_size, val_ebnodb, val_steps = validation_params[index]
            for i in range(iterations):
                self.train_step(p, batch_size, ebnodb, lr)
                if i % val_steps == 0:
                    print(self.test_step(p, val_size, val_ebnodb))

    # ---------- fitness black-box (dùng chung cho mọi metaheuristic) ----------
    def fitness(self, individual, ebnodb):
        """Số ký hiệu bị phân loại sai trên M mẫu ngẫu nhiên (BLER count)."""
        sigma = np.float32(self.EbNo2Sigma(ebnodb))
        s = tf2.random.uniform([self.M], 0, self.M, dtype=tf2.int64)
        logits, _ = self._forward_logits(s, self._as_p(individual), sigma)
        pred = tf2.argmax(logits, axis=1)
        return int(tf2.reduce_sum(tf2.cast(tf2.not_equal(pred, s), tf2.int32)).numpy())

    # ======================================================================== #
    #  Tấn công gradient: FGM  (tf2.gradients -> GradientTape)                   #
    # ======================================================================== #
    def _grad_ce_wrt_y(self, y_np, label_cls):
        """dCE/dy tại y_np với nhãn label_cls (eager)."""
        y = tf2.convert_to_tensor(np.asarray(y_np, np.float32).reshape(1, 2, self.n))
        lbl = tf2.convert_to_tensor(np.asarray([label_cls]), dtype=tf2.int64)
        with tf2.GradientTape() as tape:
            tape.watch(y)
            logits = self.core.decode(y)
            ce = tf2.reduce_mean(
                keras.losses.sparse_categorical_crossentropy(lbl, logits, from_logits=True))
        return tape.gradient(ce, y).numpy().reshape(1, 2, self.n)

    def _logits_from_y(self, y_np):
        y = tf2.convert_to_tensor(np.asarray(y_np, np.float32).reshape(1, 2, self.n))
        return self.core.decode(y).numpy()

    def fgm_attack(self, s, p, ebnodb):
        """Adversarial example theo Sadeghi & Larsson (bisection theo từng lớp)."""
        np.random.seed(self.seed)
        num_class = self.M
        sigma = np.float32(self.EbNo2Sigma(ebnodb))
        # y = x + noise + p (một realization) rồi lấy gradient tại y
        s_t = tf2.convert_to_tensor(np.asarray(s).reshape(-1), dtype=tf2.int64)
        _, y = self._forward_logits(s_t, self._as_p(p), sigma)
        y_reshaped = y.numpy().reshape(1, 2, self.n)
        eps_acc = 1e-7 * np.linalg.norm(y_reshaped)
        epsilon_vector = np.zeros([num_class])
        for cls in range(num_class):
            g = -1 * self._grad_ce_wrt_y(y_reshaped, cls)
            norm_adv = g / (np.linalg.norm(g) + 1e-12)
            emax = np.linalg.norm(y_reshaped); emin = 0.0
            it = 0
            while (emax - emin > eps_acc) and (it < 30):
                it += 1
                eps = (emax + emin) / 2
                adv = y_reshaped + eps * norm_adv
                pred = np.argmax(self._logits_from_y(adv))
                if np.equal(pred, s):
                    emin = eps
                else:
                    emax = eps
            epsilon_vector[cls] = eps + eps_acc
        false_cls = int(np.argmin(epsilon_vector))
        min_eps = float(np.min(epsilon_vector))
        g = -1 * self._grad_ce_wrt_y(y_reshaped, false_cls)
        norm_adv = g / (np.linalg.norm(g) + 1e-12)
        adv_perturbation = min_eps * norm_adv
        return adv_perturbation, false_cls, min_eps

    def UAPattack_fgm(self, ebnodb, num_samples, PSR_dB):
        """Universal Adversarial Perturbation (Alg. 1, Sadeghi et al.)."""
        np.random.seed(self.seed)
        uap = np.zeros([1, 2, self.n])
        for _ in range(num_samples):
            s = np.asarray([np.floor(np.random.uniform(0, 2 ** self.k, 1))]).reshape(1)
            sigma = np.float32(self.EbNo2Sigma(ebnodb))
            logit = self._forward_logits(
                tf2.convert_to_tensor(s, tf2.int64), self._as_p(uap), sigma)[0].numpy()
            if np.argmax(logit) == s:
                adv, _, _ = self.fgm_attack(s, uap, ebnodb)
                adv = adv.reshape([1, 2, self.n])
                UAP = uap + adv
                PSR = 10 ** (PSR_dB / 10)
                eps_uni = np.sqrt((PSR * self.n) / (np.linalg.norm(UAP) ** 2 + 1e-8))
                if np.linalg.norm(uap + adv) < eps_uni:
                    uap = uap + adv
                else:
                    uap = eps_uni * (uap + adv)
        return uap

    def RMAEP(self, ebnodb, num_samples, PSR_dB, num_iterations):
        """RMAEP: PGD lặp trên gradient, chiếu theo ràng buộc PSR."""
        np.random.seed(self.seed)
        perturbation = np.zeros([1, 2, self.n])
        psr = 10 ** (PSR_dB / 10)
        epsilon = np.sqrt(2 * self.bits_per_symbol * psr)
        for _ in range(num_samples):
            s = np.asarray([np.floor(np.random.uniform(0, 2 ** self.k, 1))]).reshape(1)
            sigma = np.float32(self.EbNo2Sigma(ebnodb))
            logit = self._forward_logits(
                tf2.convert_to_tensor(s, tf2.int64), self._as_p(perturbation), sigma)[0].numpy()
            if np.argmax(logit) == s:
                for _ in range(num_iterations):
                    _, y = self._forward_logits(
                        tf2.convert_to_tensor(s, tf2.int64), self._as_p(perturbation), sigma)
                    g = -1 * self._grad_ce_wrt_y(y.numpy(), int(s[0]))
                    perturbation = perturbation + g / (np.linalg.norm(g) + 1e-10)
                    perturbation = np.clip(perturbation, -epsilon, epsilon)
        return perturbation

    # ---------- toán tử EA (numpy, giữ nguyên) ----------
    def Crossover1(self, parent1, parent2):
        child1 = np.zeros([1, 2, self.n]); child2 = np.zeros([1, 2, self.n])
        p1 = np.random.randint(0, self.n - 2); p2 = np.random.randint(p1, self.n - 1)
        for i in range(p1):
            child1[0][:, i] = parent1[0][:, i]; child2[0][:, i] = parent2[0][:, i]
        for i in range(p1, p2):
            child1[0][:, i] = parent2[0][:, i]; child2[0][:, i] = parent1[0][:, i]
        for i in range(p2, self.n):
            child1[0][:, i] = parent1[0][:, i]; child2[0][:, i] = parent2[0][:, i]
        return child1, child2

    def Crossover2(self, parent1, parent2):
        child = np.zeros([1, 2, self.n])
        p1 = np.random.randint(0, max(1, self.n // 3)); p2 = np.random.randint(p1, self.n - 1)
        for i in range(p1):
            child[0][:, i] = parent1[0][:, i]
        for i in range(p1, p2):
            child[0][0][i] = (parent2[0][0][i] + parent1[0][0][i]) / 2
            child[0][1][i] = (parent2[0][1][i] + parent1[0][0][i]) / 2
        for i in range(p2, self.n):
            child[0][:, i] = parent2[0][:, i]
        return child

    def Mutation1(self, parent):
        child = np.zeros([1, 2, self.n]); point = np.random.randint(0, self.n - 1)
        for i in range(point):
            child[0][:, i] = parent[0][:, point - i - 1]
        for i in range(point, self.n):
            child[0][:, i] = parent[0][:, self.n + point - i - 1]
        return child

    def Mutation2(self, parent):
        child = np.zeros([1, 2, self.n])
        p1 = np.random.randint(0, max(1, self.n // 3)); p2 = np.random.randint(p1, self.n - 1)
        for i in range(p1):
            child[0][:, i] = parent[0][:, i]
        for i in range(p1, p2):
            x = parent[0][0][i]; y = parent[0][1][i]
            child[0][0][i] = 1 + x + x * x + x * x * x
            child[0][1][i] = 1 + y + y * y + y * y * y
        for i in range(p2, self.n):
            child[0][:, i] = parent[0][:, i]
        return child

    def adaptive_mutation_rate(self, generation, max_generations):
        return 0.2 - (0.2 - 0.01) * (generation / max_generations)

    # ---------- EA / MFEA / DE / hybrid (port sang eager fitness) ----------
    def EAbasedAttack(self, ebnodb, num_samples, PSR_dB):
        np.random.seed(self.seed)
        population = [self.UAPattack_fgm(ebnodb, num_samples, PSR_dB)]
        POP, CR, MR, GEN = 50, 0.8, 0.2, 250
        for _ in range(POP // 2):
            population.append(self.normalize(
                np.random.normal(0, self.PSR2sigma(PSR_dB), (1, 2, self.n)), PSR_dB))
        for _ in range(POP // 2 - 1):
            population.append(self.normalize(np.random.uniform(-1, 1, (1, 2, self.n)), PSR_dB))
        for _ in range(GEN):
            for idx in range(POP):
                parent1 = population[idx]
                if np.random.uniform() < CR:
                    parent2 = population[np.random.randint(0, POP - 1)]
                    c1, c2 = self.Crossover1(parent1, parent2)
                    population.append(self.normalize(c1, PSR_dB))
                    population.append(self.normalize(c2, PSR_dB))
                    population.append(self.normalize(self.Crossover2(parent1, parent2), PSR_dB))
                if np.random.uniform() < MR:
                    population.append(self.normalize(self.Mutation1(parent1), PSR_dB))
                    population.append(self.normalize(self.Mutation2(parent1), PSR_dB))
            population.sort(key=lambda x: -self.fitness(x, ebnodb))
            population = population[:POP - 1]
        population.append(self.UAPattack_fgm(ebnodb, num_samples, PSR_dB))
        population.sort(key=lambda x: -self.fitness(x, ebnodb))
        return self.normalize(population[0], PSR_dB)

    def DEbasedAttack(self, ebnodb, num_samples, PSR_dB, F):
        np.random.seed(self.seed)
        population = [self.UAPattack_fgm(ebnodb, num_samples, PSR_dB)]
        POP, CR, GEN = 50, 0.8, 250
        for _ in range(POP // 2):
            population.append(self.normalize(
                np.random.normal(0, self.PSR2sigma(PSR_dB), (1, 2, self.n)), PSR_dB))
        for _ in range(POP // 2 - 1):
            population.append(self.normalize(np.random.uniform(-1, 1, (1, 2, self.n)), PSR_dB))
        for _ in range(GEN):
            for idx in range(POP):
                Ik = population[idx]
                I1 = population[np.random.randint(0, POP)]
                I2 = population[np.random.randint(0, POP)]
                I3 = population[np.random.randint(0, POP)]
                Vk = self.normalize(I1 + F * (I2 - I3), PSR_dB)
                Ok = np.array(Ik, copy=True)
                j = np.random.randint(0, self.n)
                for i in range(self.n):
                    if np.random.rand() < CR or i == j:
                        Ok[0][:, i] = Vk[0][:, i]
                Ok = self.normalize(Ok, PSR_dB)
                if self.fitness(Ok, ebnodb) > self.fitness(Ik, ebnodb):
                    population[idx] = Ok
        population.append(self.UAPattack_fgm(ebnodb, num_samples, PSR_dB))
        population.sort(key=lambda x: -self.fitness(x, ebnodb))
        return self.normalize(population[0], PSR_dB)

    def gradient_descent(self, individual, ebnodb, PSR_dB, learning_rate=0.01, iterations=100):
        """Tinh chỉnh gradient (dùng gradient BLER xấp xỉ qua fgm direction)."""
        for _ in range(iterations):
            s = np.asarray([np.floor(np.random.uniform(0, 2 ** self.k, 1))]).reshape(1)
            sigma = np.float32(self.EbNo2Sigma(ebnodb))
            _, y = self._forward_logits(
                tf2.convert_to_tensor(s, tf2.int64), self._as_p(individual), sigma)
            g = -1 * self._grad_ce_wrt_y(y.numpy(), int(s[0]))
            individual = individual - learning_rate * g
            individual = self.normalize(individual, PSR_dB)
        return individual

    def hybrid_attack(self, ebnodb, num_samples, PSR_dB):
        np.random.seed(self.seed)
        population = [self.UAPattack_fgm(ebnodb, num_samples, PSR_dB)]
        POP, CR, GEN = 50, 0.8, 250
        for _ in range(POP // 2):
            population.append(self.normalize(
                np.random.normal(0, self.PSR2sigma(PSR_dB), (1, 2, self.n)), PSR_dB))
        for _ in range(POP // 2 - 1):
            population.append(self.normalize(np.random.uniform(-1, 1, (1, 2, self.n)), PSR_dB))
        for gen in range(GEN):
            new_pop = []
            for idx in range(POP):
                parent1 = population[idx]
                if np.random.uniform() < CR:
                    parent2 = population[np.random.randint(0, POP - 1)]
                    c1, c2 = self.Crossover1(parent1, parent2)
                    new_pop += [self.normalize(c1, PSR_dB), self.normalize(c2, PSR_dB),
                                self.normalize(self.Crossover2(parent1, parent2), PSR_dB)]
                if np.random.uniform() < self.adaptive_mutation_rate(gen, GEN):
                    new_pop += [self.normalize(self.Mutation1(parent1), PSR_dB),
                                self.normalize(self.Mutation2(parent1), PSR_dB)]
            population.extend(new_pop)
            population.sort(key=lambda x: -self.fitness(x, ebnodb))
            population = population[:POP]
            population[0] = self.gradient_descent(population[0], ebnodb, PSR_dB)
        population.append(self.UAPattack_fgm(ebnodb, num_samples, PSR_dB))
        population.sort(key=lambda x: -self.fitness(x, ebnodb))
        return self.normalize(population[0], PSR_dB)

    def EAbasedAttack_MFEA(self, ebnodb, num_perturbations, num_samples, PSR_dB):
        np.random.seed(self.seed)
        POP, CR, MR, GEN = 50, 0.8, 0.2, 250
        population = []
        for i in range(num_perturbations):
            population.append({'perturbation': self.UAPattack_fgm(ebnodb, num_samples, PSR_dB),
                               'skill_factor': i})
            for _ in range(POP // 2):
                population.append({'perturbation': self.normalize(
                    np.random.normal(0, self.PSR2sigma(PSR_dB), (1, 2, self.n)), PSR_dB),
                    'skill_factor': i})
            for _ in range(POP // 2 - 1):
                population.append({'perturbation': self.normalize(
                    np.random.uniform(-1, 1, (1, 2, self.n)), PSR_dB), 'skill_factor': i})
        for gen in range(GEN):
            mr = self.adaptive_mutation_rate(gen, GEN)
            new_pop = []
            for ind in population:
                p1 = ind['perturbation']; sf1 = ind['skill_factor']
                if np.random.uniform() < CR:
                    info2 = population[np.random.randint(0, len(population))]
                    p2 = info2['perturbation']; sf2 = info2['skill_factor']
                    c1, c2 = self.Crossover1(p1, p2)
                    new_pop += [{'perturbation': self.normalize(c1, PSR_dB), 'skill_factor': sf1},
                                {'perturbation': self.normalize(c2, PSR_dB), 'skill_factor': sf2},
                                {'perturbation': self.normalize(self.Crossover2(p1, p2), PSR_dB),
                                 'skill_factor': sf1}]
                if np.random.uniform() < mr:
                    new_pop += [{'perturbation': self.normalize(self.Mutation1(p1), PSR_dB),
                                 'skill_factor': sf1},
                                {'perturbation': self.normalize(self.Mutation2(p1), PSR_dB),
                                 'skill_factor': sf1}]
            population.extend(new_pop)
            updated = []
            for tid in range(num_perturbations):
                tp = [ind for ind in population if ind['skill_factor'] == tid]
                tp.sort(key=lambda x: -self.fitness(x['perturbation'], ebnodb))
                updated.extend(tp[:POP])
            population = updated
        best = []
        for i in range(num_perturbations):
            tp = [ind for ind in population if ind['skill_factor'] == i]
            tp.append({'perturbation': self.UAPattack_fgm(ebnodb, num_samples, PSR_dB),
                       'skill_factor': i})
            tp.sort(key=lambda x: -self.fitness(x['perturbation'], ebnodb))
            best.append(self.normalize(tp[0]['perturbation'], PSR_dB))
        return best

    # ======================================================================== #
    #  3 ATTACK MỚI: simpleGA, PSO, simple_CMAES                                #
    #  (tất cả tối đa hoá fitness = số ký hiệu sai; chuẩn hoá PSR mỗi lần đánh giá)#
    # ======================================================================== #
    def _rand_indiv(self, PSR_dB):
        v = np.random.normal(0, self.PSR2sigma(PSR_dB), (1, 2, self.n))
        return self.normalize(v, PSR_dB)

    def simpleGA(self, ebnodb, PSR_dB, pop_size=40, generations=60,
                 tournament=3, cx_alpha=0.5, mut_sigma=0.3, mut_rate=0.2, elite=2):
        """
        GA thực (real-coded) tối giản:
          - chọn lọc giải đấu (tournament)
          - lai BLX-alpha
          - đột biến Gauss
          - giữ elite
        Mỗi cá thể là perturbation [1,2,n], chuẩn hoá PSR trước khi tính fitness.
        """
        np.random.seed(self.seed)
        D = 2 * self.n
        pop = [self._rand_indiv(PSR_dB).reshape(D) for _ in range(pop_size)]
        fit = [self.fitness(p.reshape(1, 2, self.n), ebnodb) for p in pop]

        def tsel():
            idx = np.random.randint(0, pop_size, tournament)
            return pop[idx[np.argmax([fit[i] for i in idx])]]

        for _ in range(generations):
            order = np.argsort(fit)[::-1]
            new_pop = [pop[order[e]].copy() for e in range(elite)]      # elitism
            while len(new_pop) < pop_size:
                a, b = tsel(), tsel()
                # BLX-alpha crossover
                lo = np.minimum(a, b); hi = np.maximum(a, b); rng = hi - lo
                child = np.random.uniform(lo - cx_alpha * rng, hi + cx_alpha * rng)
                # Gaussian mutation
                mask = np.random.rand(D) < mut_rate
                child = child + mask * np.random.normal(0, mut_sigma, D)
                child = self.normalize(child.reshape(1, 2, self.n), PSR_dB).reshape(D)
                new_pop.append(child)
            pop = new_pop
            fit = [self.fitness(p.reshape(1, 2, self.n), ebnodb) for p in pop]
        best = pop[int(np.argmax(fit))]
        return self.normalize(best.reshape(1, 2, self.n), PSR_dB)

    def PSO(self, ebnodb, PSR_dB, swarm=40, iters=60, w=0.72, c1=1.49, c2=1.49):
        """
        Particle Swarm Optimization trên R^{2n}:
          v <- w*v + c1*r1*(pbest-x) + c2*r2*(gbest-x);  x <- x+v
        Chuẩn hoá PSR trước mỗi lần tính fitness. Trả gbest.
        """
        np.random.seed(self.seed)
        D = 2 * self.n
        sig = self.PSR2sigma(PSR_dB)
        X = np.random.normal(0, sig, (swarm, D))
        V = np.random.normal(0, sig * 0.1, (swarm, D))

        def f(x):
            return self.fitness(self.normalize(x.reshape(1, 2, self.n), PSR_dB), ebnodb)

        pbest = X.copy()
        pbest_f = np.array([f(x) for x in X])
        g = int(np.argmax(pbest_f)); gbest = pbest[g].copy(); gbest_f = pbest_f[g]
        for _ in range(iters):
            r1 = np.random.rand(swarm, D); r2 = np.random.rand(swarm, D)
            V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
            X = X + V
            for i in range(swarm):
                fi = f(X[i])
                if fi > pbest_f[i]:
                    pbest[i] = X[i].copy(); pbest_f[i] = fi
                    if fi > gbest_f:
                        gbest = X[i].copy(); gbest_f = fi
        return self.normalize(gbest.reshape(1, 2, self.n), PSR_dB)

    def simple_CMAES(self, ebnodb, PSR_dB, iters=60, popsize=None, sigma0=None):
        """
        CMA-ES tối giản (thuần numpy) trên R^{2n}, tối đa hoá fitness.
        Cập nhật mean + ma trận hiệp phương sai C theo trọng số xếp hạng.
        Chuẩn hoá PSR trước mỗi lần tính fitness. Trả mean tốt nhất.
        """
        np.random.seed(self.seed)
        D = 2 * self.n
        lam = popsize or (4 + int(3 * np.log(D)))         # kích thước quần thể
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = 1.0 / np.sum(w ** 2)
        # tham số thích nghi
        cc = (4 + mueff / D) / (D + 4 + 2 * mueff / D)
        cs = (mueff + 2) / (D + mueff + 5)
        c1 = 2 / ((D + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((D + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (D + 1)) - 1) + cs
        chiN = np.sqrt(D) * (1 - 1 / (4 * D) + 1 / (21 * D ** 2))

        mean = np.zeros(D)
        sigma = sigma0 or self.PSR2sigma(PSR_dB)
        C = np.eye(D); pc = np.zeros(D); ps = np.zeros(D)

        def f(x):
            return self.fitness(self.normalize(x.reshape(1, 2, self.n), PSR_dB), ebnodb)

        best_x = mean.copy(); best_f = -1
        for _ in range(iters):
            # phân rã C = B diag(d^2) B^T
            C = np.triu(C) + np.triu(C, 1).T
            d2, B = np.linalg.eigh(C)
            d2 = np.clip(d2, 1e-14, None); d = np.sqrt(d2)
            Z = np.random.randn(lam, D)
            Y = Z @ (B * d).T                              # ~ N(0, C)
            X = mean + sigma * Y
            fvals = np.array([f(x) for x in X])
            order = np.argsort(fvals)[::-1]                # tối đa hoá
            if fvals[order[0]] > best_f:
                best_f = fvals[order[0]]; best_x = X[order[0]].copy()
            Xsel = X[order[:mu]]; Ysel = Y[order[:mu]]
            mean_old = mean.copy()
            mean = w @ Xsel
            yw = w @ Ysel
            # cập nhật đường tiến hoá
            invsqrtC = B @ np.diag(1 / d) @ B.T
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ yw)
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** 2) / chiN) < (1.4 + 2 / (D + 1))
            pc = (1 - cc) * pc + (hsig * np.sqrt(cc * (2 - cc) * mueff)) * yw
            # cập nhật C
            artmp = Ysel
            C = ((1 - c1 - cmu) * C
                 + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
                 + cmu * (artmp.T * w) @ artmp)
            sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = float(np.clip(sigma, 1e-8, 1e3))
        return self.normalize(best_x.reshape(1, 2, self.n), PSR_dB)

    # ---------- báo cáo BLER (giữ API cũ) ----------
    def calculate_BLER(self, p2, PSR_dB, ebnodbs, batch_size, iterations):
        np.random.seed(self.seed)
        p2 = self.normalize(p2, PSR_dB)
        out = np.zeros_like(np.asarray(ebnodbs, dtype=float))
        for _ in range(iterations):
            out += np.array([self.test_step(p2, batch_size, e) for e in ebnodbs]) / iterations
        return out

    def bler_sim_attack_AWGN(self, p, PSR_dB, ebnodbs, batch_size, iterations):
        np.random.seed(self.seed)
        p = self.normalize(p, PSR_dB)
        eb = np.asarray(ebnodbs, dtype=float)
        no = np.zeros_like(eb); adv = np.zeros_like(eb); jam = np.zeros_like(eb)
        for _ in range(iterations):
            no += np.array([self.test_step(np.zeros([1, 2, self.n]), batch_size, e) for e in eb]) / iterations
            adv += np.array([self.test_step(p, batch_size, e) for e in eb]) / iterations
            nz = np.random.normal(0, 1, p.shape)
            j = np.linalg.norm(p) / np.linalg.norm(nz) * nz
            jam += np.array([self.test_step(j, batch_size, e) for e in eb]) / iterations
        return no, adv, jam

    def bler_sim_attack_AWGN_proposed(self, p1, p2, p3, PSR_dB, ebnodbs, batch_size, iterations):
        no, adv, jam = self.bler_sim_attack_AWGN(p1, PSR_dB, ebnodbs, batch_size, iterations)
        eab = self.calculate_BLER(p2, PSR_dB, ebnodbs, batch_size, iterations)
        deb = self.calculate_BLER(p3, PSR_dB, ebnodbs, batch_size, iterations)
        return no, adv, jam, eab, deb
class AE_MLP(object):
    def __init__(self, k, n, seed=None, filename=None):
        self.k = k 
        self.n = n
        self.bits_per_symbol = self.k/self.n
        self.M = 2**self.k
        self.seed = seed            
        self.graph = None  
        self.sess = None  
        self.vars = None  
        self.saver = None 
        self.constellations = None
        self.blers = None
        self.create_graph() 
        self.create_session()
        if filename is not None:    
            self.load(filename)       
        return
    
    def create_graph(self):
        '''This function creates the computation graph of the autoencoder'''
        self.graph = tf.Graph()        
        with self.graph.as_default():  
            tf.set_random_seed(self.seed) 
            batch_size = tf.placeholder(tf.int32, shape=())
            
            # Transmitter
            s = tf.random_uniform(shape=[batch_size], minval=0, maxval=self.M, dtype=tf.int64)
            x = self.encoder(s)     
            
            # the attack vector
            p = tf.placeholder(tf.float32,shape=(None,2,self.n)) 
            
            # Channel
            noise_std = tf.placeholder(tf.float32, shape=()) # 
            noise = tf.random_normal(tf.shape(x), mean=0.0, stddev=noise_std, seed=self.seed)
            y = x + noise + p
    
            # Receiver
            s_hat = self.decoder(y)
            
            # Loss function
            cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=s, logits=s_hat)
            
            # Performance metrics
            correct_predictions = tf.equal(tf.argmax(tf.nn.softmax(s_hat), axis=1), s)
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
            bler = 1-accuracy
            
            # Optimizer
            lr = tf.placeholder(tf.float32, shape=())    
            train_op = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
        
            # References to graph variables we need to access later 
            self.vars = {
                'accuracy': accuracy,
                'batch_size': batch_size,
                'bler': bler,
                'cross_entropy': cross_entropy,
                'init': tf.global_variables_initializer(),
                'lr': lr,
                'noise_std': noise_std,
                'noise': noise,
                'p': p,
                's': s,
                's_hat': s_hat,
                'train_op': train_op,
                'x': x,
                'y': y,
            }            
            self.saver = tf.train.Saver()
        return
    
    def create_session(self):
        '''Create a session for the autoencoder instance with the compuational graph'''
        self.sess = tf.Session(graph=self.graph)        
        self.sess.run(self.vars['init'])
        return
    
    def encoder(self, input):
        '''The transmitter'''
        W = self.weight_variable((self.M,self.M))
        x = tf.nn.elu(tf.nn.embedding_lookup(W, input))
        x = tf.layers.dense(x, 2*self.n, activation=None)
        x = tf.reshape(x, shape=[-1,2,self.n])
        #Average power normalization
        x = x/tf.sqrt(2*tf.reduce_mean(tf.square(x)))
        return x
    
    def decoder(self, input):
        '''The Receiver'''
        y = tf.reshape(input, shape=[-1,2*self.n])
        y = tf.layers.dense(y, self.M, activation=tf.nn.relu)
        y = tf.layers.dense(y, self.M, activation=None)
        return y

    def EbNo2Sigma(self, ebnodb):
        '''Convert Eb/No in dB to noise standard deviation'''
        ebno = 10**(ebnodb/10)
        return 1/np.sqrt(2*self.bits_per_symbol*ebno) 
    
    def PSR2sigma(self, psr_db):
        '''Convert P/eb in dB to p standard deviation'''
        ps = 10**(psr_db/10)
        return np.sqrt(2*self.bits_per_symbol*ps) 
    
    def gen_feed_dict(self, perturbation, batch_size, ebnodb, lr):
        '''Generate a feed dictionary for training and validation'''        
        return {
            self.vars['p']: perturbation,
            self.vars['batch_size']: batch_size,
            self.vars['noise_std']: self.EbNo2Sigma(ebnodb),
            self.vars['lr']: lr,
        }           

    def load(self, filename):
        '''Load a pre_trained model'''
        return self.saver.restore(self.sess, filename)
    
    def save(self, filename):
        '''Save the current model'''
        return self.saver.save(self.sess, filename)  
    
    def test_step(self, p, batch_size, ebnodb):
        '''Compute the BLER over a single batch and Eb/No'''
        bler = self.sess.run(self.vars['bler'], feed_dict=self.gen_feed_dict(p, batch_size, ebnodb, lr=0))
        return bler
    
    def transmit(self, s):
        '''Returns the transmitted sigals corresponding to message indices'''
        return self.sess.run(self.vars['x'], feed_dict={self.vars['s']: s})
       
    def train(self, p, training_params, validation_params):  
        '''Training and validation loop'''
        for index, params in enumerate(training_params):            
            batch_size, lr, ebnodb, iterations = params            
            print('\nBatch Size: ' + str(batch_size) +
                  ', Learning Rate: ' + str(lr) +
                  ', EbNodB: ' + str(ebnodb) +
                  ', Iterations: ' + str(iterations))
            
            val_size, val_ebnodb, val_steps = validation_params[index]
            for i in range(iterations):
                self.train_step(p, batch_size, ebnodb, lr)    
                if (i%val_steps==0):
                    bler = self.sess.run(self.vars['bler'], feed_dict=self.gen_feed_dict(p,val_size, val_ebnodb, lr))
                    print(bler)                           
        return       
    
    def train_step(self, p, batch_size, ebnodb, lr):
        '''A single training step'''
        self.sess.run(self.vars['train_op'], feed_dict=self.gen_feed_dict(p, batch_size, ebnodb, lr)) #self.sess.run(train_op, feed_dict=self.gen_feed_dict(batch_size, ebnodb, lr))#s
        return 
    
    def weight_variable(self, shape):
        '''Xavier-initialized weights optimized for ReLU Activations'''
        (fan_in, fan_out) = shape
        low = np.sqrt(6.0/(fan_in + fan_out)) 
        high = -np.sqrt(6.0/(fan_in + fan_out))
        return tf.Variable(tf.random_uniform(shape, minval=low, maxval=high, dtype=tf.float32))
    

    def bler_sim_attack_AWGN(self, p, PSR_dB, ebnodbs, batch_size, iterations):
        '''Generate the BLER for 3 cases: 1) no attack, 2) adversarial attack, and 3) jamming attack'''
        np.random.seed(seed=self.seed)
        PSR = 10**(PSR_dB/10)
        scale_factor = np.sqrt( (PSR * self.n) / (np.linalg.norm(p)**2 +  0.00000001) ) # 
        p = scale_factor * p
        BLER_no_attack = np.zeros_like(ebnodbs)
        BLER_adv_attack = np.zeros_like(ebnodbs)
        BLER_jamming = np.zeros_like(ebnodbs)
        for i in range(iterations):
            # No attack - clean case
            print('This is interation: ', i)
            bler = np.array([self.sess.run(self.vars['bler'],
                            feed_dict=self.gen_feed_dict(np.zeros([1,2,self.n]), batch_size, ebnodb, lr=0)) for ebnodb in ebnodbs]) #bler = np.array([self.sess.run(self.vars['bler'],feed_dict=self.gen_feed_dict(p, batch_size, ebnodb, lr=0)) for ebnodb in ebnodbs])
            BLER_no_attack = BLER_no_attack + bler/iterations
            # adversarial attack
            bler_attack = np.array([self.sess.run(self.vars['bler'],
                            feed_dict=self.gen_feed_dict(  p ,batch_size, ebnodb, lr=0)) for ebnodb in ebnodbs]) # I think lr=0 is equal to is_training=False
            BLER_adv_attack = BLER_adv_attack + bler_attack/iterations
            # Jamming attack
            normal_noise_as_jammer = np.random.normal(0,1,p.shape)
            jamming = np.linalg.norm(p) * (1 / np.linalg.norm(normal_noise_as_jammer)) * normal_noise_as_jammer
            bler_jamming= np.array([self.sess.run(self.vars['bler'],
                            feed_dict=self.gen_feed_dict(jamming,batch_size, ebnodb, lr=0)) for ebnodb in ebnodbs]) # I think lr=0 is equal to is_training=False
            BLER_jamming = BLER_jamming + bler_jamming/iterations
        return BLER_no_attack, BLER_adv_attack, BLER_jamming

    # def bler_sim_attack_AWGN_proposed(self, p1, p2 , PSR_dB, ebnodbs, batch_size, iterations):
    #     '''Generate the BLER for 4 cases: 1) no attack, 2) adversarial attack, 3) jamming attack, and 4) ea attack'''
    #     np.random.seed(seed=self.seed)
    #     PSR = 10**(PSR_dB/10)
    #     scale_factor = np.sqrt( (PSR * self.n) / (np.linalg.norm(p1)**2 +  0.00000001) ) # 
    #     p1 = scale_factor * p1
    #     scale_factor = np.sqrt( (PSR * self.n) / (np.linalg.norm(p2)**2 +  0.00000001) ) # 
    #     p2 = scale_factor * p2
    #     BLER_no_attack = np.zeros_like(ebnodbs)
    #     BLER_adv_attack = np.zeros_like(ebnodbs)
    #     BLER_jamming = np.zeros_like(ebnodbs)
    #     BLER_ea = np.zeros_like(ebnodbs)
    #     for i in range(iterations):
    #         # No attack - clean case
    #         bler = np.array([self.sess.run(self.vars['bler'],
    #                         feed_dict=self.gen_feed_dict(np.zeros([1,2,self.n]), batch_size, ebnodb, lr=0)) for ebnodb in ebnodbs]) #bler = np.array([self.sess.run(self.vars['bler'],feed_dict=self.gen_feed_dict(p, batch_size, ebnodb, lr=0)) for ebnodb in ebnodbs])
    #         BLER_no_attack = BLER_no_attack + bler/iterations
    #         # adversarial attack
    #         bler_attack = np.array([self.sess.run(self.vars['bler'],
    #                         feed_dict=self.gen_feed_dict(  p1 ,batch_size, ebnodb, lr=0)) for ebnodb in ebnodbs]) # I think lr=0 is equal to is_training=False
    #         BLER_adv_attack = BLER_adv_attack + bler_attack/iterations
    #         # Jamming attack
    #         normal_noise_as_jammer = np.random.normal(0,1,p1.shape)
    #         jamming = np.linalg.norm(p1) * (1 / np.linalg.norm(normal_noise_as_jammer)) * normal_noise_as_jammer
    #         bler_jamming= np.array([self.sess.run(self.vars['bler'],
    #                         feed_dict=self.gen_feed_dict(jamming,batch_size, ebnodb, lr=0)) for ebnodb in ebnodbs]) # I think lr=0 is equal to is_training=False
    #         BLER_jamming = BLER_jamming + bler_jamming/iterations

    #         # EA attack
    #         bler_attack = np.array([self.sess.run(self.vars['bler'],
    #                         feed_dict=self.gen_feed_dict(  p2 ,batch_size, ebnodb, lr=0)) for ebnodb in ebnodbs]) # I think lr=0 is equal to is_training=False
    #         BLER_adv_attack = BLER_adv_attack + bler_attack/iterations
    #     return BLER_no_attack, BLER_adv_attack, BLER_jamming
    def calculate_BLER(self,  p2, PSR_dB, ebnodbs, batch_size, iterations):
        
        np.random.seed(seed=self.seed)
        PSR = 10**(PSR_dB/10)
        scale_factor = np.sqrt( (PSR * self.n) / (np.linalg.norm(p2)**2 +  0.00000001) ) # 
        p2 = scale_factor * p2
    
      
        BLER_eab_attack = np.zeros_like(ebnodbs)

        for i in range(iterations):
    
            # EA attack
            bler_attack = np.array([self.sess.run(self.vars['bler'],
                            feed_dict=self.gen_feed_dict(  p2 ,batch_size, ebnodb, lr=0)) for ebnodb in ebnodbs]) # I think lr=0 is equal to is_training=False
            BLER_eab_attack = BLER_eab_attack + bler_attack/iterations
            
            
        return  BLER_eab_attack
    def bler_sim_attack_AWGN_proposed(self, p1, p2,p3, PSR_dB, ebnodbs, batch_size, iterations):
        '''Generate the BLER for 4 cases: 1) no attack, 2) adversarial attack, 3) jamming attack, and 4) ea attack, 5) de'''
        np.random.seed(seed=self.seed)
        PSR = 10**(PSR_dB/10)

        
        scale_factor = np.sqrt( (PSR * self.n) / (np.linalg.norm(p1)**2 +  0.00000001) ) # 
        p1 = scale_factor * p1
        scale_factor = np.sqrt( (PSR * self.n) / (np.linalg.norm(p2)**2 +  0.00000001) ) # 
        p2 = scale_factor * p2
        scale_factor = np.sqrt( (PSR * self.n) / (np.linalg.norm(p3)**2 +  0.00000001) ) # 
        p3 = scale_factor * p3

        BLER_no_attack = np.zeros_like(ebnodbs)
        BLER_adv_attack = np.zeros_like(ebnodbs)
        BLER_jamming = np.zeros_like(ebnodbs)
        BLER_eab_attack = np.zeros_like(ebnodbs)
        BLER_deb_attack = np.zeros_like(ebnodbs)
        for i in range(iterations):
            # No attack - clean case
            bler = np.array([self.sess.run(self.vars['bler'],
                            feed_dict=self.gen_feed_dict(np.zeros([1,2,self.n]), batch_size, ebnodb, lr=0)) for ebnodb in ebnodbs]) #bler = np.array([self.sess.run(self.vars['bler'],feed_dict=self.gen_feed_dict(p, batch_size, ebnodb, lr=0)) for ebnodb in ebnodbs])
            BLER_no_attack = BLER_no_attack + bler/iterations
            # adversarial attack
            bler_attack = np.array([self.sess.run(self.vars['bler'],
                            feed_dict=self.gen_feed_dict(  p1 ,batch_size, ebnodb, lr=0)) for ebnodb in ebnodbs]) # I think lr=0 is equal to is_training=False
            BLER_adv_attack = BLER_adv_attack + bler_attack/iterations
            # Jamming attack
            normal_noise_as_jammer = np.random.normal(0,1,p1.shape)
            jamming = np.linalg.norm(p1) * (1 / np.linalg.norm(normal_noise_as_jammer)) * normal_noise_as_jammer
            bler_jamming= np.array([self.sess.run(self.vars['bler'],
                            feed_dict=self.gen_feed_dict(jamming,batch_size, ebnodb, lr=0)) for ebnodb in ebnodbs]) # I think lr=0 is equal to is_training=False
            BLER_jamming = BLER_jamming + bler_jamming/iterations

            # EA attack
            bler_attack = np.array([self.sess.run(self.vars['bler'],
                            feed_dict=self.gen_feed_dict(  p2 ,batch_size, ebnodb, lr=0)) for ebnodb in ebnodbs]) # I think lr=0 is equal to is_training=False
            BLER_eab_attack = BLER_eab_attack + bler_attack/iterations
            
            #de
            bler_attack = np.array([self.sess.run(self.vars['bler'],
                            feed_dict=self.gen_feed_dict(  p3 ,batch_size, ebnodb, lr=0)) for ebnodb in ebnodbs]) # I think lr=0 is equal to is_training=False
            BLER_deb_attack = BLER_deb_attack + bler_attack/iterations
        return BLER_no_attack, BLER_adv_attack, BLER_jamming, BLER_eab_attack, BLER_deb_attack
    #BLER_deb_attack
    

    def fgm_attack(self,s,p, ebnodb): #in_img,in_label,num_class
        '''Create an input specific adversarial example using the method proposed by Sadeghi and Larsson in [2] '''
        np.random.seed(seed=self.seed)
        num_class = self.M 
        y_reshaped = np.reshape(self.sess.run(self.vars['y'], feed_dict={self.vars['s']: s, self.vars['p']: p, self.vars['noise_std']: self.EbNo2Sigma(ebnodb)}), [1,2,self.n])  #[-1]         #print('y_reshaped',y_reshaped.shape)        
        eps_acc = 0.0000001 * np.linalg.norm(y_reshaped)
        epsilon_vector = np.zeros([num_class])
        predictions = tf.nn.softmax(self.vars['s_hat'], name = 'predictions')        
        for cls in range(num_class):
            s_target = np.array([cls])
            adv_per_needtoreshape  = -1 * np.asarray( self.sess.run(tf.gradients(self.vars['cross_entropy'],self.vars['y']), feed_dict={self.vars['y']: y_reshaped , self.vars['s']:s_target }) )
            adv_per = adv_per_needtoreshape.reshape(1,2,self.n)
            norm_adv_per = adv_per / (np.linalg.norm(adv_per) +  0.000000000001)
            epsilon_max = 1 * np.linalg.norm(y_reshaped)
            epsilon_min = 0
            num_iter = 0
            wcount = 0
            while (epsilon_max-epsilon_min > eps_acc) and (num_iter < 30):
                wcount = wcount+1
                num_iter = num_iter +1
                epsilon = (epsilon_max + epsilon_min)/2
                adv_img_givencls = y_reshaped + (epsilon * norm_adv_per)
                
                predicted_probabilities = self.sess.run(predictions, feed_dict={self.vars['y']: adv_img_givencls})
                compare = np.equal(np.argmax(predicted_probabilities),s)
                if compare:
                    epsilon_min = epsilon
                else:
                    epsilon_max = epsilon
            epsilon_vector[cls] = epsilon + eps_acc
        false_cls = np.argmin(epsilon_vector)
        minimum_epsilon = np.min(epsilon_vector)
        adv_dirc = -1 * np.asarray(self.sess.run(tf.gradients(self.vars['cross_entropy'],self.vars['y']), feed_dict={self.vars['y']: y_reshaped, self.vars['s']: np.asarray([false_cls]) })  ).reshape(1,2,self.n)
        norm_adv_dirc = adv_dirc / (np.linalg.norm(adv_dirc) + 0.000000000001)
        adv_perturbation = minimum_epsilon * norm_adv_dirc
        return adv_perturbation, false_cls, minimum_epsilon
    '''hybrid attack'''
    def gradient_descent(self, individual, ebnodb, PSR_dB, learning_rate=0.01, iterations=100):
        for _ in range(iterations):
            tf_individual = tf.convert_to_tensor(individual)
            with tf.GradientTape() as tape:
                tape.watch(tf_individual)
                loss = -self.fitness(individual, ebnodb)
            gradients = tape.gradient(loss, tf_individual)
            individual = individual - learning_rate * gradients
            individual = self.normalize(individual, PSR_dB)
        return individual

    def hybrid_attack(self, ebnodb, num_samples, PSR_dB):
        np.random.seed(seed=self.seed)
        universal_per_hybrid = np.zeros([1, 2, self.n])
        population = []
        POPSIZE = 50
        CROSSRATE = 0.8
        MAXGENERATION = 250

        # Initialization
        population.append(self.UAPattack_fgm(ebnodb, num_samples, PSR_dB))
        for i in range(int(POPSIZE / 2)):
            individual = np.random.normal(loc=0.0, scale=self.PSR2sigma(PSR_dB), size=(1, 2, self.n))
            population.append(self.normalize(individual, PSR_dB))
        
        for i in range(int(POPSIZE / 2) - 1):
            individual = np.random.uniform(-1, 1, size=(1, 2, self.n))
            population.append(self.normalize(individual, PSR_dB))

        for generation in range(MAXGENERATION):
            new_population = []
            for indi_index in range(POPSIZE):
                parent1 = population[indi_index]

                # Crossover
                if np.random.uniform() < CROSSRATE:
                    p2_index = np.random.randint(0, POPSIZE - 1)
                    parent2 = population[p2_index]
                    child1, child2 = self.Crossover1(parent1, parent2)
                    new_population.append(self.normalize(child1, PSR_dB))
                    new_population.append(self.normalize(child2, PSR_dB))
                    new_population.append(self.normalize(self.Crossover2(parent1, parent2), PSR_dB))

                # Mutation
                if np.random.uniform() < self.adaptive_mutation_rate(generation, MAXGENERATION):
                    new_population.append(self.normalize(self.Mutation1(parent1), PSR_dB))
                    new_population.append(self.normalize(self.Mutation2(parent1), PSR_dB))

            # Combine old and new populations and sort by fitness
            population.extend(new_population)
            population.sort(key=lambda x: -self.fitness(x, ebnodb))
            population = population[:POPSIZE]

            # Apply Gradient Descent to the best individual
            best_individual = population[0]
    
            best_individual = self.gradient_descent(best_individual, ebnodb,PSR_dB)
            population[0] = best_individual

        # Final evaluation and selection
        population.append(self.UAPattack_fgm(ebnodb, num_samples, PSR_dB))
        population.sort(key=lambda x: -self.fitness(x, ebnodb))
        universal_per_hybrid = self.normalize(population[0], PSR_dB)

        return universal_per_hybrid

    def UAPattack_fgm(self,ebnodb,num_samples,PSR_dB):
        k = self.k
        '''Create a Universal Adversarial Perturbation as suggested by Alg. 1 of Sadeghi et al in [2]'''
        np.random.seed(seed=self.seed)
        universal_per_fgm = np.zeros([1,2,self.n])
        for cnr_index in range(num_samples):#               
            s =  np.asarray([np.floor(np.random.uniform(0,2**k,1))]).reshape(1) 
            predicted_label = np.argmax( self.sess.run(self.vars['s_hat'], feed_dict={self.vars['s']:s, self.vars['p']:universal_per_fgm, self.vars['noise_std']: self.EbNo2Sigma(ebnodb)}) )
            if predicted_label == s:
                # First we need to find adverssarial direction for this instant  by solving eq. (1) of the paper
                adv_perturbation,_,_ = self.fgm_attack(s, universal_per_fgm,ebnodb)
                adv_perturbn_reshaped = adv_perturbation.reshape([1,2,self.n])
                UAP = universal_per_fgm + adv_perturbation.reshape([1,2,self.n])
                PSR = 10**(PSR_dB/10)
                Epsilon_uni = np.sqrt( (PSR * self.n) / (np.linalg.norm(UAP)**2 + 0.00000001) )
                # Second we need to revise the universal perturbation
                if np.linalg.norm(universal_per_fgm + adv_perturbn_reshaped) < Epsilon_uni: 
                    universal_per_fgm = universal_per_fgm + adv_perturbn_reshaped
                else:
                    universal_per_fgm =  Epsilon_uni * (universal_per_fgm + adv_perturbn_reshaped) 
        return universal_per_fgm
    #RMAEP
    def RMAEP(self, ebnodb, num_samples, PSR_dB, num_iterations):
        """
        Implements the RMAEP algorithm for generating adversarial perturbations.
        
        Args:
            ebnodb (float): Eb/No in dB for the channel.
            num_samples (int): Number of samples for attack evaluation.
            PSR_dB (float): Perturbation-to-Signal Ratio in dB.
            num_iterations (int): Number of iterations for PGD updates.
        
        Returns:
            numpy.ndarray: The adversarial perturbation vector.
        """
        np.random.seed(self.seed)
        perturbation = np.zeros([1, 2, self.n])  # Initialize perturbation
        psr = 10**(PSR_dB / 10)  # Convert PSR from dB to linear scale
        epsilon = np.sqrt(2 * self.bits_per_symbol * psr)  # Perturbation constraint

        for _ in range(num_samples):
            # Randomly sample a valid symbol
            s = np.asarray([np.floor(np.random.uniform(0, 2**self.k, 1))]).reshape(1)
            predicted_label = np.argmax(self.sess.run(
                self.vars['s_hat'],
                feed_dict={self.vars['s']: s, self.vars['p']: perturbation, self.vars['noise_std']: self.EbNo2Sigma(ebnodb)}
            ))
            
            # If the model predicts correctly, refine the perturbation
            if predicted_label == s:
                # Iterative PGD updates
                for _ in range(num_iterations):
                    gradients = -1 * np.asarray(self.sess.run(
                        tf.gradients(self.vars['cross_entropy'], self.vars['y']),
                        feed_dict={
                            self.vars['s']: s,
                            self.vars['p']: perturbation,
                            self.vars['noise_std']: self.EbNo2Sigma(ebnodb)
                        }
                    )).reshape(1, 2, self.n)

                    # Normalize the gradients to project onto the unit sphere
                    norm_grad = gradients / (np.linalg.norm(gradients) + 1e-10)

                    # Add perturbation step and enforce the PSR constraint
                    perturbation += norm_grad
                    perturbation = np.clip(
                        perturbation, -epsilon, epsilon
                    )
        return perturbation


    # EA BASED

    def Crossover1(self, parent1, parent2):
        child1 = np.zeros([1,2,self.n], dtype=float)
        child2 = np.zeros([1,2,self.n], dtype=float)
        point1 = np.random.randint(0, self.n -2, dtype= int)
        point2 = np.random.randint(point1, self.n -1, dtype= int)

        for i in range(point1):
            child1[0][0][i] = parent1[0][0][i]
            child1[0][1][i] = parent1[0][1][i]
            child2[0][0][i] = parent2[0][0][i]
            child2[0][1][i] = parent2[0][1][i]

        for i in range(point1, point2):
            child1[0][0][i] = parent2[0][0][i]
            child1[0][1][i] = parent2[0][1][i]
            child2[0][0][i] = parent1[0][0][i]
            child2[0][1][i] = parent1[0][1][i]

        for i in range(point2, self.n):
            child1[0][0][i] = parent1[0][0][i]
            child1[0][1][i] = parent1[0][1][i]
            child2[0][0][i] = parent2[0][0][i]
            child2[0][1][i] = parent2[0][1][i]
        return child1, child2
    
    def Crossover2(self, parent1, parent2):
        child = np.zeros([1,2,self.n], dtype=float)
        point1 = np.random.randint(0, self.n/3, dtype= int)
        point2 = np.random.randint(point1, self.n -1, dtype= int)

        for i in range(point1):
            child[0][0][i] = parent1[0][0][i]
            child[0][1][i] = parent1[0][1][i]

        for i in range(point1, point2):
            child[0][0][i] = (parent2[0][0][i] + parent1[0][0][i])/2
            child[0][1][i] = (parent2[0][1][i] + parent1[0][0][i])/2

        for i in range(point2, self.n):
            child[0][0][i] = parent2[0][0][i]
            child[0][1][i] = parent2[0][1][i]
        return child
    
    def Mutation1(self, parent):
        child = np.zeros([1,2,self.n])
        point = np.random.randint(0, self.n -1, dtype= int)

        for i in range(point):
            child[0][0][i] = parent[0][0][point-i-1]
            child[0][1][i] = parent[0][1][point-i-1]
        for i in range(point, self.n):
            child[0][0][i] = parent[0][0][self.n+ point-i-1]
            child[0][1][i] = parent[0][1][self.n+ point-i-1]
        return child
    
    def Mutation2(self, parent):
        child = np.zeros([1,2,self.n], dtype=float)
        point1 = np.random.randint(0, self.n/3, dtype= int)
        point2 = np.random.randint(point1, self.n -1, dtype= int)

        for i in range(point1):
            child[0][0][i] = parent[0][0][i]
            child[0][1][i] = parent[0][1][i]


        for i in range(point1, point2):
            x = parent[0][0][i]
            y = parent[0][1][i]
            child[0][0][i] = 1 + x + x*x + x*x*x
            child[0][1][i] = 1 + y + y*y + y*y*y

        for i in range(point2, self.n):
            child[0][0][i] = parent[0][0][i]
            child[0][1][i] = parent[0][1][i]
        return child

    def fitness(self, individual, ebnodb):
        k = self.k
        res = 0
        for i in range(2**k):
            s =  np.asarray([np.floor(np.random.uniform(0,2**k,1))]).reshape(1) 
            predicted_label = np.argmax(self.sess.run(self.vars['s_hat'], feed_dict={self.vars['s']:s, self.vars['p']:individual, self.vars['noise_std']: self.EbNo2Sigma(ebnodb)}) )
            if predicted_label != s:
                res+=1
        return res
    
    def normalize(self, individual, PSR_dB):
        PSR = 10**(PSR_dB/10)
        scale_factor = np.sqrt( (PSR * self.n) / (np.linalg.norm(individual)**2 +  0.00000001) ) # 
        # energy = 0
        # for i in range(self.n):
        #     x = individual[0][0][i]
        #     y = individual[0][1][i]
        #     energy += x*x + y*y
        return scale_factor * individual
    #let my try
    def adaptive_mutation_rate(self, generation, max_generations):
        initial_rate = 0.2
        final_rate = 0.01
        return initial_rate - (initial_rate - final_rate) * (generation / max_generations)
    
    def EAbasedAttack(self, ebnodb, num_samples, PSR_dB):
        '''Create a EAbased Adversarial Perturbation as suggested'''
        np.random.seed(seed=self.seed)
        universal_per_eab = np.zeros([1,2,self.n])
        population = []
        POPSIZE = 50
        CROSSRATE = 0.8
        MUTATIONRATE = 0.2
        MAXGENERATION = 250
        
        # Initialization
        population.append(self.UAPattack_fgm(ebnodb,num_samples,PSR_dB))
        for i in range(int(POPSIZE/2)):
            # individual = tf.random_normal([1,2,self.n], mean=0.0, stddev= self.PSR2sigma(PSR_dB), seed=self.seed)
            individual = np.random.normal(loc=0.0, scale=self.PSR2sigma(PSR_dB), size=(1, 2, self.n))
            population.append(self.normalize(individual,PSR_dB))
        
        for i in range(int(POPSIZE/2)-1):
            individual = np.random.uniform(-1, 1, size=(1, 2, self.n))
            population.append(self.normalize(individual, PSR_dB))

        # Main loop
        for iteration in range(MAXGENERATION):
            for indi_index in range(POPSIZE):
                parent1 = population[indi_index]

                # Crossover
                if (np.random.uniform() < CROSSRATE):
                    p2_index = np.random.randint(0, POPSIZE-1)
                    parent2 = population[p2_index]
                    child1, child2 =  self.Crossover1(parent1, parent2)
                    population.append(self.normalize(child1, PSR_dB))
                    population.append(self.normalize(child2, PSR_dB))
                    population.append(self.normalize(self.Crossover2(parent1, parent2), PSR_dB))

                # Mutation
                if (np.random.uniform() < MUTATIONRATE):
                    population.append(self.normalize(self.Mutation1(parent1), PSR_dB))
                    population.append(self.normalize(self.Mutation2(parent1),PSR_dB))

            # population.sort(key = -self.fitness)
            population.sort(key = lambda x: -self.fitness(x, ebnodb))
            population = population[0:POPSIZE-1]

        population.append(self.UAPattack_fgm(ebnodb,num_samples,PSR_dB))
        population.sort(key = lambda x: -self.fitness(x, ebnodb))
        universal_per_eab = self.normalize(population[0],PSR_dB)

        return universal_per_eab
    
    # def EAbasedAttack_MFEA(self, ebnodb,num_perturbations, num_samples, PSR_dB):
    #     '''MFEA-based Adversarial Perturbations Generation for multiple tasks'''
    #     np.random.seed(seed=self.seed)
    #     POPSIZE = 50
    #     CROSSRATE = 0.8
    #     MUTATIONRATE = 0.2
    #     MAXGENERATION = 250
    #     population = []

    #     # Initialization for each task
    #     for i in range(num_perturbations):
    #         initial_perturbation = self.UAPattack_fgm(ebnodb, num_samples, PSR_dB)
    #         print(f"Initial perturbation for task {i}: {initial_perturbation}")
    #         population.append({
    #             'perturbation': self.normalize(initial_perturbation, PSR_dB),
    #             'skill_factor': i
    #         })
    #         for _ in range(int(POPSIZE / 2) - 1):
    #             perturbation = np.random.normal(0.0, self.PSR2sigma(PSR_dB), (1, 2, self.n))
    #             population.append({
    #                 'perturbation': self.normalize(perturbation, PSR_dB),
    #                 'skill_factor': i
    #             })
    #         for _ in range(int(POPSIZE / 2)):
    #             perturbation = np.random.uniform(-1, 1, size=(1, 2, self.n))
    #             population.append({
    #                 'perturbation': self.normalize(perturbation, PSR_dB),
    #                 'skill_factor': i
    #             })

    #     # MFEA Main Loop
    #     for generation in range(MAXGENERATION):
    #         mutation_rate = self.adaptive_mutation_rate(generation, MAXGENERATION)
    #         new_population = []
    #         for individual in population:
    #             parent1 = individual['perturbation']
    #             skill_factor1 = individual['skill_factor']
               
    #             # MFEA Crossover with skill factor
    #             if np.random.uniform() < CROSSRATE:
    #                 parent2_info = population[np.random.randint(0,POPSIZE -1)]
    #                 parent2 = parent2_info['perturbation']
    #                 skill_factor2 = parent2_info['skill_factor']
                  
    #                 # Crossover
    #                 child1, child2 = self.Crossover1(parent1, parent2)
                   
    #                 new_population.extend([
    #                     {'perturbation': self.normalize(child1, PSR_dB), 'skill_factor': skill_factor1},
    #                     {'perturbation': self.normalize(child2, PSR_dB), 'skill_factor': skill_factor2}
    #                 ])
    #                 new_population.append({
    #                     'perturbation': self.normalize(self.Crossover2(parent1, parent2), PSR_dB),
    #                     'skill_factor': skill_factor1
    #                 })
                   

    #             # Mutation
    #             if np.random.uniform() < mutation_rate:
    #                 new_population.append({
    #                     'perturbation': self.normalize(self.Mutation1(parent1), PSR_dB),
    #                     'skill_factor': skill_factor1
    #                 })
    #                 new_population.append({
    #                     'perturbation': self.normalize(self.Mutation2(parent1), PSR_dB),
    #                     'skill_factor': skill_factor1
    #                 })
                   
           

    #         population.extend(new_population)
    #         n_population=[]
    #         for task_id in range(num_perturbations):
    #             task_population = [ind for ind in population if ind['skill_factor'] == task_id]
    #             task_population.sort(key=lambda x: -self.fitness(x['perturbation'], ebnodb))  
    #             n_population.append(task_population[:POPSIZE])

    #         # print(f"Generation {generation}: Best fitness for task 0: {self.fitness(population[0]['perturbation'], 0)}")

    #     best_perturbations = []
    #     for i in range(num_perturbations):
    #         task_population = [ind for ind in n_population if ind['skill_factor'] == i]

    #         task_population.append(self.UAPattack_fgm(ebnodb, num_samples, PSR_dB))
    #         task_population.sort(key=lambda x: -self.fitness(x['perturbation'], ebnodb))  
    #         population = task_population[:POPSIZE]
    #         best_perturbations.append(self.normalize(population[0]['perturbation']))   
            
    #     print(best_perturbations)
    #     return best_perturbations

    def EAbasedAttack_MFEA(self, ebnodb, num_perturbations, num_samples, PSR_dB):
        '''MFEA-based Adversarial Perturbations Generation for multiple tasks'''
        np.random.seed(seed=self.seed)
        POPSIZE = 50
        CROSSRATE = 0.8
        MUTATIONRATE = 0.2
        MAXGENERATION = 250
        population = []

        # Initialization for each task
        for i in range(num_perturbations):
    
            population.append({
                'perturbation':  self.UAPattack_fgm(ebnodb, num_samples, PSR_dB),
                'skill_factor': i
            })
            for _ in range(int(POPSIZE / 2) ):
                perturbation = np.random.normal(0.0, self.PSR2sigma(PSR_dB), (1, 2, self.n))
                population.append({
                    'perturbation': self.normalize(perturbation, PSR_dB),
                    'skill_factor': i
                })
            for _ in range(int(POPSIZE / 2)-1):
                perturbation = np.random.uniform(-1, 1, size=(1, 2, self.n))
                population.append({
                    'perturbation': self.normalize(perturbation, PSR_dB),
                    'skill_factor': i
                })

        # MFEA Main Loop
        for generation in range(MAXGENERATION):
            mutation_rate = self.adaptive_mutation_rate(generation, MAXGENERATION)
            new_population = []
            for individual in population:
                parent1 = individual['perturbation']
                skill_factor1 = individual['skill_factor']

                # Crossover with skill factor
                if np.random.uniform() < CROSSRATE:
                    parent2_info = population[np.random.randint(0, len(population))]
                    parent2 = parent2_info['perturbation']
                    skill_factor2 = parent2_info['skill_factor']

                    
                    child1, child2 = self.Crossover1(parent1, parent2)
                    new_population.extend([
                        {'perturbation': self.normalize(child1, PSR_dB), 'skill_factor': skill_factor1},
                        {'perturbation': self.normalize(child2, PSR_dB), 'skill_factor': skill_factor2}
                    ])
                    new_population.append({
                        'perturbation': self.normalize(self.Crossover2(parent1, parent2), PSR_dB),
                        'skill_factor': skill_factor1
                    })

                # Mutation
                if np.random.uniform() < mutation_rate:
                    new_population.append({
                        'perturbation': self.normalize(self.Mutation1(parent1), PSR_dB),
                        'skill_factor': skill_factor1
                    })
                    new_population.append({
                        'perturbation': self.normalize(self.Mutation2(parent1), PSR_dB),
                        'skill_factor': skill_factor1
                    })

            population.extend(new_population)
            updated_population = []
            for task_id in range(num_perturbations):
                task_population = [ind for ind in population if ind['skill_factor'] == task_id]
                task_population.sort(key=lambda x: -self.fitness(x['perturbation'], ebnodb))
                updated_population.extend(task_population[:POPSIZE])

            population = updated_population

        best_perturbations = []
        for i in range(num_perturbations):
            print("final")
            print(i)
            task_population = [ind for ind in population if ind['skill_factor'] == i]
            task_population.append({
                'perturbation': self.UAPattack_fgm(ebnodb, num_samples, PSR_dB),
                'skill_factor': i
            })
            task_population.sort(key=lambda x: -self.fitness(x['perturbation'], ebnodb))
            best_perturbations.append(self.normalize(task_population[0]['perturbation'], PSR_dB))

        
        return best_perturbations

    #### Differential Evolution
    def DEbasedAttack(self, ebnodb, num_samples, PSR_dB, F):
        '''Create a DEbased Adversarial Perturbation'''
        np.random.seed(seed=self.seed)
        universal_per_deb = np.zeros([1,2,self.n])
        population = []
        POPSIZE = 50
        CROSSRATE = 0.8
        MUTATIONRATE = 0.2
        MAXGENERATION = 250

        #Initialization
        population.append(self.UAPattack_fgm(ebnodb,num_samples,PSR_dB))
        for i in range(int(POPSIZE/2)):
            # individual = tf.random_normal([1,2,self.n], mean=0.0, stddev= self.PSR2sigma(PSR_dB), seed=self.seed)
            individual = np.random.normal(loc=0.0, scale=self.PSR2sigma(PSR_dB), size=(1, 2, self.n))
            population.append(self.normalize(individual,PSR_dB))
        
        for i in range(int(POPSIZE/2)-1):
            individual = np.random.uniform(-1, 1, size=(1, 2, self.n))
            population.append(self.normalize(individual, PSR_dB))

        for iteration in range(MAXGENERATION):
            for indi_index in range(POPSIZE):
                Ik = population[indi_index]

                I1 = population[np.random.randint(0,POPSIZE)]
                I2 = population[np.random.randint(0,POPSIZE)]
                I3 = population[np.random.randint(0,POPSIZE)]
                
                Vk = np.zeros([1,2,self.n], dtype=float)
                for i in range(self.n):
                    Vk[0][0][i] = I1[0][0][i] + F*(I2[0][0][i] - I3[0][0][i])
                    Vk[0][1][i] = I1[0][1][i] + F*(I2[0][1][i] - I3[0][1][i])
                    
                Vk = self.normalize(Vk, PSR_dB)
                
                Ok = np.zeros([1,2,self.n], dtype=float)
                j = np.random.randint(0, self.n)
    
                for i in range(self.n):
                    if np.random.rand() < CROSSRATE or i == j:
                        Ok[0][0][i] = Vk[0][0][i]
                        Ok[0][1][i] = Vk[0][1][i]
                    else:
                        Ok[0][0][i] = Ik[0][0][i]
                        Ok[0][1][i] = Ik[0][1][i]
                        
                Ok = self.normalize(Ok, PSR_dB)
                if self.fitness(Ok, ebnodb) > self.fitness(Ik, ebnodb):
                    population[indi_index] = Ok
                    
        population.append(self.UAPattack_fgm(ebnodb,num_samples,PSR_dB))
        population.sort(key = lambda x: -self.fitness(x, ebnodb))
        universal_per_deb = self.normalize(population[0],PSR_dB)

        return universal_per_deb
    
    


###############################  CNN of Table 1 ###############################
class AE_CNN(object):
    def __init__(self, k, n, seed=None, filename=None):
        self.k = k 
        self.n = n
        self.bits_per_symbol = self.k/self.n
        self.M = 2**self.k
        self.seed = seed
        self.graph = None  
        self.sess = None  
        self.vars = None 
        self.saver = None 
        self.constellations = None
        self.blers = None
        self.create_graph()
        self.create_session()
        if filename is not None:    
            self.load(filename)       
        return
    
    def create_graph(self):
        '''This function creates the computation graph of the autoencoder'''
        self.graph = tf.Graph()        
        with self.graph.as_default():  
            tf.set_random_seed(self.seed)
            batch_size = tf.placeholder(tf.int32, shape=(), name='batchsize')
            is_training = tf.placeholder(tf.bool, name='istraining')
            dr_out = tf.placeholder(tf.float32,shape=(), name='drout')
            # Transmitter
            s = tf.random_uniform(shape=[batch_size], minval=0, maxval=self.M, dtype=tf.int64)
            x = self.encoder(s)     

            # the attack vector
            p = tf.placeholder(tf.float32,shape=(None,2,self.n), name='pname') # batch * 2 * n is the shape of y and x.
            
            # Channel
            noise_std = tf.placeholder(tf.float32, shape=()) 
            noise = tf.random_normal(tf.shape(x), mean=0.0, stddev=noise_std)
            y = x + noise + p
            
            # Receiver
            s_hat = self.decoder(y, dr_out, is_training)
            
            # Loss function
            cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=s, logits=s_hat)
 
            # Performance metrics
            correct_predictions = tf.equal(tf.argmax(tf.nn.softmax(s_hat), axis=1), s)
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
            bler = 1-accuracy
            
            
            # Optimizer
            lr = tf.placeholder(tf.float32, shape=())      
            train_op = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
        
            # References to graph variables we need to access later 
            self.vars = {
                'accuracy': accuracy,
                'batch_size': batch_size,
                'bler': bler,
                'cross_entropy': cross_entropy,
                'dr_out':dr_out,
                'init': tf.global_variables_initializer(),
                'is_training':is_training,
                'lr': lr,
                'noise_std': noise_std,
                'noise': noise,
                'p': p,
                's': s,
                's_hat': s_hat,
                'train_op': train_op,
                'x': x,
                'y': y,
            }            
            self.saver = tf.train.Saver()
        return
    
    
    def create_session(self):
        '''Create a session for the autoencoder instance with the compuational graph'''
        self.sess = tf.Session(graph=self.graph)      
        self.sess.run(self.vars['init'])
        return
    
    
    def encoder(self, input):
        '''The transmitter'''
        W = self.weight_variable((self.M,self.M))
        x = tf.nn.elu(tf.nn.embedding_lookup(W, input)) 
        x = tf.reshape(x,[-1,1,self.M])
        conv0 = tf.layers.conv1d(x, 16, 6, strides=1, padding='same', data_format='channels_first',
                             activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=tf.glorot_uniform_initializer(seed=None, dtype=tf.float32),
                             trainable=True)
        flattened0 = tf.layers.flatten(conv0)
        x = tf.layers.dense(flattened0, 2*self.n, activation=None)
        x = tf.reshape(x, shape=[-1,2,self.n]) 
        #Average power normalization
        x = x/tf.sqrt(2*tf.reduce_mean(tf.square(x)))
        return x
    
    def decoder(self, input, dr_out, is_training):
        '''The Receiver'''
        reshaped = tf.reshape(input, shape=[-1,1,2,self.n])
        conv1 = tf.layers.conv2d(reshaped, 16, [2,3], strides=(1, 1), padding='same', data_format='channels_first',
                             activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=tf.glorot_uniform_initializer(seed=None, dtype=tf.float32),
                             trainable=True)
        conv2 = tf.layers.conv2d(conv1, 8, [2,3], strides=(1, 1), padding='same', data_format='channels_first',
                             activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=tf.glorot_uniform_initializer(seed=None, dtype=tf.float32),
                             trainable=True)
        drout = tf.layers.dropout(conv2, rate=dr_out, noise_shape=None, training=is_training, name='dropou1')
        flattened = tf.layers.flatten(drout)
        dense1 = tf.layers.dense(flattened, 2*self.M, activation=tf.nn.relu)
        y = tf.layers.dense(dense1, self.M, activation=None)
        return y

    def EbNo2Sigma(self, ebnodb):
        '''Convert Eb/No in dB to noise standard deviation'''
        ebno = 10**(ebnodb/10)
        return 1/np.sqrt(2*self.bits_per_symbol*ebno) 
    
    def gen_feed_dict(self, is_training,dr_out, perturbation, batch_size, ebnodb, lr):
        '''Generate a feed dictionary for training and validation'''      
        return {
            self.vars['is_training']: is_training,
            self.vars['dr_out']: dr_out,
            self.vars['p']: perturbation,
            self.vars['batch_size']: batch_size,
            self.vars['noise_std']: self.EbNo2Sigma(ebnodb),
            self.vars['lr']: lr,
        }    
    
    
    def load(self, filename):
        '''Load a pre_trained model'''
        return self.saver.restore(self.sess, filename)
    
    def save(self, filename):
        '''Save the current model'''
        return self.saver.save(self.sess, filename)  
    
    def test_step(self, is_training, dr_out, p, batch_size, ebnodb):
        '''Compute the BLER over a single batch and Eb/No'''
        bler = self.sess.run(self.vars['bler'], feed_dict=self.gen_feed_dict(is_training, dr_out, p, batch_size, ebnodb, lr=0))  
        return bler
    
    def transmit(self, s):
        '''Returns the transmitted sigals corresponding to message indices'''
        return self.sess.run(self.vars['x'], feed_dict={self.vars['s']: s})
       
    def train(self, is_training, dr_out ,p, training_params, validation_params):  
        '''Training and validation loop'''
        for index, params in enumerate(training_params):            
            batch_size, lr, ebnodb, iterations = params            
            print('\nBatch Size: ' + str(batch_size) +
                  ', Learning Rate: ' + str(lr) +
                  ', EbNodB: ' + str(ebnodb) +
                  ', Iterations: ' + str(iterations))
            
            val_size, val_ebnodb, val_steps = validation_params[index]
            
            for i in range(iterations):
                self.train_step(is_training, dr_out, p, batch_size, ebnodb, lr)    
                if (i%val_steps==0):
                    bler = self.sess.run(self.vars['bler'], feed_dict=self.gen_feed_dict(is_training, dr_out , p,val_size, val_ebnodb, lr))
                    print(bler)                           
        return       
    
    def train_step(self, is_training, dr_out , p, batch_size, ebnodb, lr):
        '''A single training step'''
        self.sess.run(self.vars['train_op'], feed_dict=self.gen_feed_dict(is_training, dr_out , p, batch_size, ebnodb, lr)) #self.sess.run(train_op, feed_dict=self.gen_feed_dict(batch_size, ebnodb, lr))#s
        return 
    
    def weight_variable(self, shape):
        '''Xavier-initialized weights optimized for ReLU Activations'''
        (fan_in, fan_out) = shape
        low = np.sqrt(6.0/(fan_in + fan_out)) 
        high = -np.sqrt(6.0/(fan_in + fan_out))
        return tf.Variable(tf.random_uniform(shape, minval=low, maxval=high, dtype=tf.float32))
    
    
    def bler_sim_attack_AWGN(self, is_training, dr_out , p, p_eab, p_mfea, PSR_dB, ebnodbs, batch_size, iterations):
        '''Generate the BLER for 4 cases: 1) no attack, 2) synchronous adversarial attack, 3) non-synchronous adversarial attack and 4) jamming attack'''
        PSR = 10**(PSR_dB/10)
        if PSR_dB == -10:
            p = 0.5*p
            p_eab = 0.5*p_eab
            p_mfea = 0.5*p_mfea
        else:
            scale_factor = np.sqrt( (PSR * self.n) / (np.linalg.norm(p)**2 + 0.00000001) ) # note that self.n is the power of the x, as designed by Jakob
            p = scale_factor * p

            scale_factor = np.sqrt( (PSR * self.n) / (np.linalg.norm(p_eab)**2 + 0.00000001) )
            p_eab = scale_factor * p_eab

            scale_factor = np.sqrt( (PSR * self.n) / (np.linalg.norm(p_mfea)**2 + 0.00000001) )
            p_mfea = scale_factor * p_mfea
        
        BLER_no_attack = np.zeros_like(ebnodbs)
        BLER_attack_rolled = np.zeros_like(ebnodbs)
        BLER_jamming = np.zeros_like(ebnodbs)
        BLER_attack_rolled_eab = np.zeros_like(ebnodbs)
        BLER_attack_rolled_mfea = np.zeros_like(ebnodbs)

        for i in range(iterations):
            # No attack - clean case
            print('This is interation: ', i)
            bler = np.array([self.sess.run(self.vars['bler'],
                            feed_dict=self.gen_feed_dict(is_training, dr_out , np.zeros([1,2,self.n]), batch_size, ebnodb, lr=0)) for ebnodb in ebnodbs]) #bler = np.array([self.sess.run(self.vars['bler'],feed_dict=self.gen_feed_dict(p, batch_size, ebnodb, lr=0)) for ebnodb in ebnodbs])
            BLER_no_attack = BLER_no_attack + bler/iterations
            # attack - rolled attack - nonsynchronous
            p_rolled = np.roll(p, int(np.ceil(np.random.uniform(0,self.n))))
            bler_attack_rolled = np.array([self.sess.run(self.vars['bler'],
                            feed_dict=self.gen_feed_dict(is_training, dr_out ,p_rolled,batch_size, ebnodb, lr=0)) for ebnodb in ebnodbs]) # I think lr=0 is equal to is_training=False
            BLER_attack_rolled = BLER_attack_rolled + bler_attack_rolled/iterations
            # Jamming attack
            normal_noise_as_jammer = np.random.normal(0,1,p.shape)
            jamming = np.linalg.norm(p) * (1 / np.linalg.norm(normal_noise_as_jammer)) * normal_noise_as_jammer
            bler_jamming= np.array([self.sess.run(self.vars['bler'],
                            feed_dict=self.gen_feed_dict(is_training, dr_out, jamming,batch_size, ebnodb, lr=0)) for ebnodb in ebnodbs]) # I think lr=0 is equal to is_training=False
            BLER_jamming = BLER_jamming + bler_jamming/iterations
            
            # EAB attack
            p_rolled_eab = np.roll(p_eab, int(np.ceil(np.random.uniform(0,self.n))))
            bler_attack_rolled_eab = np.array([self.sess.run(self.vars['bler'],
                            feed_dict=self.gen_feed_dict(is_training, dr_out ,p_rolled_eab,batch_size, ebnodb, lr=0)) for ebnodb in ebnodbs]) # I think lr=0 is equal to is_training=False
            BLER_attack_rolled_eab = BLER_attack_rolled_eab + bler_attack_rolled_eab/iterations

            #MFEA attack
            p_rolled_mfea = np.roll(p_mfea, int(np.ceil(np.random.uniform(0,self.n))))
            bler_attack_rolled_mfea = np.array([self.sess.run(self.vars['bler'],
                            feed_dict=self.gen_feed_dict(is_training, dr_out ,p_rolled_mfea,batch_size, ebnodb, lr=0)) for ebnodb in ebnodbs]) # I think lr=0 is equal to is_training=False
            BLER_attack_rolled_mfea = BLER_attack_rolled_mfea + bler_attack_rolled_mfea/iterations
        return BLER_no_attack, BLER_attack_rolled, BLER_jamming, BLER_attack_rolled_eab, BLER_attack_rolled_mfea
    
    def bler_sim(self, is_training, dr_out , p, p_eab , PSR_dB, ebnodbs, batch_size, iterations):
        '''Generate the BLER for 4 cases: 1) no attack, 2) synchronous adversarial attack, 3) non-synchronous adversarial attack and 4) jamming attack'''
        PSR = 10**(PSR_dB/10)
        scale_factor = np.sqrt( (PSR * self.n) / (np.linalg.norm(p)**2 + 0.00000001) ) # note that self.n is the power of the x, as designed by Jakob
        p = scale_factor * p

        scale_factor = np.sqrt( (PSR * self.n) / (np.linalg.norm(p_eab)**2 + 0.00000001) )
        p_eab = scale_factor * p_eab

  
        
        BLER_no_attack = np.zeros_like(ebnodbs)
        BLER_attack_rolled = np.zeros_like(ebnodbs)
        BLER_jamming = np.zeros_like(ebnodbs)
        BLER_attack_rolled_eab = np.zeros_like(ebnodbs)
        BLER_attack_rolled_mfea = np.zeros_like(ebnodbs)

        for i in range(iterations):
            # No attack - clean case
            print('This is interation: ', i)
            bler = np.array([self.sess.run(self.vars['bler'],
                            feed_dict=self.gen_feed_dict(is_training, dr_out , np.zeros([1,2,self.n]), batch_size, ebnodb, lr=0)) for ebnodb in ebnodbs]) #bler = np.array([self.sess.run(self.vars['bler'],feed_dict=self.gen_feed_dict(p, batch_size, ebnodb, lr=0)) for ebnodb in ebnodbs])
 
            # Jamming attack
            normal_noise_as_jammer = np.random.normal(0,1,p.shape)
            jamming = np.linalg.norm(p) * (1 / np.linalg.norm(normal_noise_as_jammer)) * normal_noise_as_jammer
            bler_jamming= np.array([self.sess.run(self.vars['bler'],
                            feed_dict=self.gen_feed_dict(is_training, dr_out, jamming,batch_size, ebnodb, lr=0)) for ebnodb in ebnodbs]) # I think lr=0 is equal to is_training=False
            BLER_jamming = BLER_jamming + bler_jamming/iterations
            
            # EAB attack
            p_rolled_eab = np.roll(p_eab, int(np.ceil(np.random.uniform(0,self.n))))
            bler_attack_rolled_eab = np.array([self.sess.run(self.vars['bler'],
                            feed_dict=self.gen_feed_dict(is_training, dr_out ,p_rolled_eab,batch_size, ebnodb, lr=0)) for ebnodb in ebnodbs]) # I think lr=0 is equal to is_training=False
            BLER_attack_rolled_eab = BLER_attack_rolled_eab + bler_attack_rolled_eab/iterations

            
        return  BLER_jamming, BLER_attack_rolled_eab
    
    
    
    
    
    
    
    
################################################################################
################################################################################
################################################################################
###############################################################################    
################################################################################
###############################################################################    
################################################################################
###############################################################################    
################################################################################
###############################################################################    
################################################################################
###############################################################################    
################################################################################
############################################################################### 
    
#################### MLP of Table 2 == netOne_MLP #############################
class AE_netOne_MLP(object):
    def __init__(self, k, n, seed=None, filename=None):
        self.k = k 
        self.n = n
        self.bits_per_symbol = self.k/self.n
        self.M = 2**self.k
        self.seed = seed         
        self.graph = None  
        self.sess = None  
        self.vars = None  
        self.saver = None 
        self.constellations = None
        self.blers = None
        self.create_graph() 
        self.create_session()
        if filename is not None:    
            self.load(filename)       
        return
    
    def create_graph(self):
        '''This function creates the computation graph of the autoencoder'''
        self.graph = tf.Graph()        
        with self.graph.as_default():  
            tf.set_random_seed(self.seed) 
            batch_size = tf.placeholder(tf.int32, shape=())
            
            # Transmitter
            s = tf.random_uniform(shape=[batch_size], minval=0, maxval=self.M, dtype=tf.int64)
            x = self.encoder(s)     
            
            # the attack vector
            p = tf.placeholder(tf.float32,shape=(None,2,self.n)) 
            
            # Channel
            noise_std = tf.placeholder(tf.float32, shape=()) # 
            noise = tf.random_normal(tf.shape(x), mean=0.0, stddev=noise_std, seed=self.seed)
            y = x + noise + p
    
            # Receiver
            s_hat = self.decoder(y)
            
            # Loss function
            cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=s, logits=s_hat)
            
            # Performance metrics
            correct_predictions = tf.equal(tf.argmax(tf.nn.softmax(s_hat), axis=1), s)
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
            bler = 1-accuracy
            
            # Optimizer
            lr = tf.placeholder(tf.float32, shape=())    
            train_op = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
        
            # References to graph variables we need to access later 
            self.vars = {
                'accuracy': accuracy,
                'batch_size': batch_size,
                'bler': bler,
                'cross_entropy': cross_entropy,
                'init': tf.global_variables_initializer(),
                'lr': lr,
                'noise_std': noise_std,
                'noise': noise,
                'p': p,
                's': s,
                's_hat': s_hat,
                'train_op': train_op,
                'x': x,
                'y': y,
            }            
            self.saver = tf.train.Saver()
        return
    
    def create_session(self):
        '''Create a session for the autoencoder instance with the compuational graph'''
        self.sess = tf.Session(graph=self.graph)        
        self.sess.run(self.vars['init'])
        return
    
    def encoder(self, input):
        '''The transmitter'''
        W = self.weight_variable((self.M,self.M))
        x = tf.nn.relu(tf.nn.embedding_lookup(W, input))
        x = tf.layers.dense(x, 2*self.n, activation=None)
        x = tf.reshape(x, shape=[-1,2,self.n])
        #Average power normalization
        x = x/tf.sqrt(2*tf.reduce_mean(tf.square(x)))
        return x
    
    def decoder(self, input):
        '''The Receiver'''
        y = tf.reshape(input, shape=[-1,2*self.n])
        y = tf.layers.dense(y, self.M, activation=tf.nn.relu)
        y = tf.layers.dense(y, self.M, activation=None)
        return y

    def EbNo2Sigma(self, ebnodb):
        '''Convert Eb/No in dB to noise standard deviation'''
        ebno = 10**(ebnodb/10)
        return 1/np.sqrt(2*self.bits_per_symbol*ebno) 
    
    def gen_feed_dict(self, perturbation, batch_size, ebnodb, lr):
        '''Generate a feed dictionary for training and validation'''        
        return {
            self.vars['p']: perturbation,
            self.vars['batch_size']: batch_size,
            self.vars['noise_std']: self.EbNo2Sigma(ebnodb),
            self.vars['lr']: lr,
        }           

    def load(self, filename):
        '''Load a pre_trained model'''
        return self.saver.restore(self.sess, filename)
    
    def save(self, filename):
        '''Save the current model'''
        return self.saver.save(self.sess, filename)  
    
    def test_step(self, p, batch_size, ebnodb):
        '''Compute the BLER over a single batch and Eb/No'''
        bler = self.sess.run(self.vars['bler'], feed_dict=self.gen_feed_dict(p, batch_size, ebnodb, lr=0))
        return bler
    
    def transmit(self, s):
        '''Returns the transmitted sigals corresponding to message indices'''
        return self.sess.run(self.vars['x'], feed_dict={self.vars['s']: s})
       
    def train(self, p, training_params, validation_params):  
        '''Training and validation loop'''
        for index, params in enumerate(training_params):            
            batch_size, lr, ebnodb, iterations = params            
            print('\nBatch Size: ' + str(batch_size) +
                  ', Learning Rate: ' + str(lr) +
                  ', EbNodB: ' + str(ebnodb) +
                  ', Iterations: ' + str(iterations))
            
            val_size, val_ebnodb, val_steps = validation_params[index]
            for i in range(iterations):
                self.train_step(p, batch_size, ebnodb, lr)    
                if (i%val_steps==0):
                    bler = self.sess.run(self.vars['bler'], feed_dict=self.gen_feed_dict(p,val_size, val_ebnodb, lr))
                    print(bler)                           
        return       
    
    def train_step(self, p, batch_size, ebnodb, lr):
        '''A single training step'''
        self.sess.run(self.vars['train_op'], feed_dict=self.gen_feed_dict(p, batch_size, ebnodb, lr)) #self.sess.run(train_op, feed_dict=self.gen_feed_dict(batch_size, ebnodb, lr))#s
        return 
    
    def weight_variable(self, shape):
        '''Xavier-initialized weights optimized for ReLU Activations'''
        (fan_in, fan_out) = shape
        low = np.sqrt(6.0/(fan_in + fan_out)) 
        high = -np.sqrt(6.0/(fan_in + fan_out))
        return tf.Variable(tf.random_uniform(shape, minval=low, maxval=high, dtype=tf.float32))
    

    def bler_sim_attack_AWGN(self, p, PSR_dB, ebnodbs, batch_size, iterations):
        '''Generate the BLER for 3 cases: 1) no attack, 2) adversarial attack, and 3) jamming attack'''
        np.random.seed(seed=self.seed)
        PSR = 10**(PSR_dB/10)
        scale_factor = np.sqrt( (PSR * self.n) / (np.linalg.norm(p)**2 +  0.00000001) ) # 
        p = scale_factor * p
        BLER_no_attack = np.zeros_like(ebnodbs)
        BLER_attack_rolled = np.zeros_like(ebnodbs)
        BLER_jamming = np.zeros_like(ebnodbs)
        for i in range(iterations):
            # No attack - clean case
            bler = np.array([self.sess.run(self.vars['bler'],
                            feed_dict=self.gen_feed_dict(np.zeros([1,2,self.n]), batch_size, ebnodb, lr=0)) for ebnodb in ebnodbs]) #bler = np.array([self.sess.run(self.vars['bler'],feed_dict=self.gen_feed_dict(p, batch_size, ebnodb, lr=0)) for ebnodb in ebnodbs])
            BLER_no_attack = BLER_no_attack + bler/iterations
            # attack - rolled attack - nonsynchronous
            p_rolled = np.roll(p, int(np.ceil(np.random.uniform(0,self.n))))
            bler_attack_rolled = np.array([self.sess.run(self.vars['bler'],
                            feed_dict=self.gen_feed_dict(p_rolled,batch_size, ebnodb, lr=0)) for ebnodb in ebnodbs]) # I think lr=0 is equal to is_training=False
            BLER_attack_rolled = BLER_attack_rolled + bler_attack_rolled/iterations
            # Jamming attack
            normal_noise_as_jammer = np.random.normal(0,1,p.shape)
            jamming = np.linalg.norm(p) * (1 / np.linalg.norm(normal_noise_as_jammer)) * normal_noise_as_jammer
            bler_jamming= np.array([self.sess.run(self.vars['bler'],
                            feed_dict=self.gen_feed_dict(jamming,batch_size, ebnodb, lr=0)) for ebnodb in ebnodbs]) # I think lr=0 is equal to is_training=False
            BLER_jamming = BLER_jamming + bler_jamming/iterations
        return BLER_no_attack, BLER_attack_rolled, BLER_jamming


#################### Deeper MLP of Table 2 == netTwo_DeepMLP #############################
class AE_netTwo_DeepMLP(object):
    def __init__(self, k, n, seed=None, filename=None):
        self.k = k 
        self.n = n
        self.bits_per_symbol = self.k/self.n
        self.M = 2**self.k
        self.seed = seed          
        self.graph = None  
        self.sess = None  
        self.vars = None  
        self.saver = None 
        self.constellations = None
        self.blers = None
        self.create_graph() 
        self.create_session()
        if filename is not None:    
            self.load(filename)       
        return
    
    def create_graph(self):
        '''This function creates the computation graph of the autoencoder'''
        self.graph = tf.Graph()        
        with self.graph.as_default():  
            tf.set_random_seed(self.seed) 
            batch_size = tf.placeholder(tf.int32, shape=())
            is_training = tf.placeholder(tf.bool, name='istraining')
            dr_out = tf.placeholder(tf.float32,shape=(), name='drout')
            # Transmitter
            s = tf.random_uniform(shape=[batch_size], minval=0, maxval=self.M, dtype=tf.int64)
            x = self.encoder(s,dr_out, is_training)     
            
            # the attack vector
            p = tf.placeholder(tf.float32,shape=(None,2,self.n)) 
            
            # Channel
            noise_std = tf.placeholder(tf.float32, shape=()) # 
            noise = tf.random_normal(tf.shape(x), mean=0.0, stddev=noise_std, seed=self.seed)
            y = x + noise + p
    
            # Receiver
            s_hat = self.decoder(y,dr_out, is_training)
            
            # Loss function
            cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=s, logits=s_hat)
            
            # Performance metrics
            correct_predictions = tf.equal(tf.argmax(tf.nn.softmax(s_hat), axis=1), s)
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
            bler = 1-accuracy
            
            # Optimizer
            lr = tf.placeholder(tf.float32, shape=())    
            train_op = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
        
            # References to graph variables we need to access later 
            self.vars = {
                'accuracy': accuracy,
                'batch_size': batch_size,
                'bler': bler,
                'cross_entropy': cross_entropy,
                'dr_out':dr_out,
                'init': tf.global_variables_initializer(),
                'is_training':is_training,
                'lr': lr,
                'noise_std': noise_std,
                'noise': noise,
                'p': p,
                's': s,
                's_hat': s_hat,
                'train_op': train_op,
                'x': x,
                'y': y,
            }            
            self.saver = tf.train.Saver()
        return
    
    def create_session(self):
        '''Create a session for the autoencoder instance with the compuational graph'''
        self.sess = tf.Session(graph=self.graph)       
        self.sess.run(self.vars['init'])
        return
    
    def encoder(self, input, dr_out, is_training):
        '''The transmitter'''
        W = self.weight_variable((self.M,self.M))
        x = tf.nn.relu(tf.nn.embedding_lookup(W, input))
        x = tf.layers.dense(x, 2*self.n, activation=tf.nn.relu)
        x = tf.layers.dropout(x,rate=dr_out, training = is_training)
        x = tf.layers.dense(x, 4*self.n, activation=tf.nn.relu)
        x = tf.layers.dropout(x,rate=dr_out, training = is_training)
#        x = tf.layers.dense(x, 4*self.n, activation=tf.nn.relu)
#        x = tf.layers.dropout(x,rate=dr_out, training = is_training)
        x = tf.layers.dense(x, 2*self.n, activation=None)
        x = tf.reshape(x, shape=[-1,2,self.n])
        #Average power normalization
        x = x/tf.sqrt(2*tf.reduce_mean(tf.square(x)))
        return x
    
    def decoder(self, input,  dr_out, is_training):
        '''The Receiver'''
        y = tf.reshape(input, shape=[-1,2*self.n])
        y = tf.layers.dense(y, 2*self.M, activation=tf.nn.relu)
        y = tf.layers.dropout(y,rate=dr_out, training=is_training)
        y = tf.layers.dense(y, 4*self.M, activation=tf.nn.relu)
        y = tf.layers.dropout(y,rate=dr_out, training=is_training)
#        y = tf.layers.dense(y, 4*self.M, activation=tf.nn.relu)
#        y = tf.layers.dropout(y,rate=dr_out, training=is_training)
        y = tf.layers.dense(y, 2*self.M, activation=tf.nn.relu)
        y = tf.layers.dropout(y,rate=dr_out, training=is_training)
        y = tf.layers.dense(y, self.M, activation=None)
        return y

    def EbNo2Sigma(self, ebnodb):
        '''Convert Eb/No in dB to noise standard deviation'''
        ebno = 10**(ebnodb/10)
        return 1/np.sqrt(2*self.bits_per_symbol*ebno) 
    
    def gen_feed_dict(self, is_training,dr_out, perturbation, batch_size, ebnodb, lr):
        '''Generate a feed dictionary for training and validation'''        
        return {
            self.vars['is_training']: is_training,
            self.vars['dr_out']: dr_out,
            self.vars['p']: perturbation,
            self.vars['batch_size']: batch_size,
            self.vars['noise_std']: self.EbNo2Sigma(ebnodb),
            self.vars['lr']: lr,
        }    
    
    def load(self, filename):
        '''Load a pre_trained model'''
        return self.saver.restore(self.sess, filename)
    
    def save(self, filename):
        '''Save the current model'''
        return self.saver.save(self.sess, filename)  
    
    def test_step(self, is_training, dr_out, p, batch_size, ebnodb):
        '''Compute the BLER over a single batch and Eb/No'''
        bler = self.sess.run(self.vars['bler'], feed_dict=self.gen_feed_dict(is_training, dr_out, p, batch_size, ebnodb, lr=0))
        return bler
    
    def transmit(self, s):
        '''Returns the transmitted sigals corresponding to message indices'''
        return self.sess.run(self.vars['x'], feed_dict={self.vars['s']: s})
       
    def train(self, is_training, dr_out , p, training_params, validation_params):  
        '''Training and validation loop'''
        for index, params in enumerate(training_params):            
            batch_size, lr, ebnodb, iterations = params            
            print('\nBatch Size: ' + str(batch_size) +
                  ', Learning Rate: ' + str(lr) +
                  ', EbNodB: ' + str(ebnodb) +
                  ', Iterations: ' + str(iterations))
            
            val_size, val_ebnodb, val_steps = validation_params[index]
            for i in range(iterations):
                self.train_step(is_training, dr_out, p, batch_size, ebnodb, lr)    
                if (i%val_steps==0):
                    bler = self.sess.run(self.vars['bler'], feed_dict=self.gen_feed_dict(is_training, dr_out, p,val_size, val_ebnodb, lr))
                    print(bler)                           
        return       
    
    def train_step(self, is_training, dr_out , p, batch_size, ebnodb, lr):
        '''A single training step'''
        self.sess.run(self.vars['train_op'], feed_dict=self.gen_feed_dict(is_training, dr_out , p, batch_size, ebnodb, lr)) 
        return 
    
    def weight_variable(self, shape):
        '''Xavier-initialized weights optimized for ReLU Activations'''
        (fan_in, fan_out) = shape
        low = np.sqrt(6.0/(fan_in + fan_out)) 
        high = -np.sqrt(6.0/(fan_in + fan_out))
        return tf.Variable(tf.random_uniform(shape, minval=low, maxval=high, dtype=tf.float32))
    

    def bler_sim_attack_AWGN(self, is_training, dr_out , p, PSR_dB, ebnodbs, batch_size, iterations):
        np.random.seed(seed=self.seed)
        PSR = 10**(PSR_dB/10)
        scale_factor = np.sqrt( (PSR * self.n) / (np.linalg.norm(p)**2 +  0.00000001) ) 
        p = scale_factor * p
        BLER_no_attack = np.zeros_like(ebnodbs)
        BLER_attack_rolled = np.zeros_like(ebnodbs)
        BLER_jamming = np.zeros_like(ebnodbs)
        for i in range(iterations):
            # No attack - clean case
            bler = np.array([self.sess.run(self.vars['bler'],
                            feed_dict=self.gen_feed_dict(is_training, dr_out , np.zeros([1,2,self.n]), batch_size, ebnodb, lr=0)) for ebnodb in ebnodbs]) 
            BLER_no_attack = BLER_no_attack + bler/iterations
            # attack - rolled attack - nonsynchronous
            p_rolled = np.roll(p, int(np.ceil(np.random.uniform(0,self.n))))
            bler_attack_rolled = np.array([self.sess.run(self.vars['bler'],
                            feed_dict=self.gen_feed_dict(is_training, dr_out , p_rolled,batch_size, ebnodb, lr=0)) for ebnodb in ebnodbs]) # I think lr=0 is equal to is_training=False
            BLER_attack_rolled = BLER_attack_rolled + bler_attack_rolled/iterations
            # Jamming attack
            normal_noise_as_jammer = np.random.normal(0,1,p.shape)
            jamming = np.linalg.norm(p) * (1 / np.linalg.norm(normal_noise_as_jammer)) * normal_noise_as_jammer
            bler_jamming= np.array([self.sess.run(self.vars['bler'],
                            feed_dict=self.gen_feed_dict(is_training, dr_out , jamming,batch_size, ebnodb, lr=0)) for ebnodb in ebnodbs]) 
            BLER_jamming = BLER_jamming + bler_jamming/iterations
        return BLER_no_attack, BLER_attack_rolled, BLER_jamming

################################################################################
################################################################################
################################################################################
###############################################################################    
################################################################################
###############################################################################    
################################################################################
###############################################################################    
################################################################################
###############################################################################    
################################################################################
###############################################################################    
################################################################################
############################################################################### 
class AE_inf_rate(object):
    def __init__(self, k, n, seed=None, filename=None):
        self.k = k 
        self.n = n
        self.bits_per_symbol = self.k/self.n
        self.M = 2**self.k
        self.seed = seed      
        self.graph = None # 
        self.sess = None  # 
        self.vars = None  # 
        self.saver = None # 
        self.constellations = None
        self.blers = None
        self.create_graph() #
        self.create_session()
        if filename is not None:    
            self.load(filename)       
        return
    
    def create_graph(self):
        '''This function creates the computation graph of the autoencoder'''
        self.graph = tf.Graph()        
        with self.graph.as_default():  #
            tf.set_random_seed(self.seed) # 
            batch_size = tf.placeholder(tf.int32, shape=())
            
            # Transmitter
            s = tf.random_uniform(shape=[batch_size], minval=0, maxval=self.M, dtype=tf.int64)
            x = self.encoder(s)     
                        
            # the attack vector
            p = tf.placeholder(tf.float32,shape=(None,2,self.n)) 
                        
            # Channel
            noise_std = tf.placeholder(tf.float32, shape=()) 
            noise = tf.random_normal(tf.shape(x), mean=0.0, stddev=noise_std, seed=self.seed)
            y = x + noise + p
            
            # Receiver
            s_hat = self.decoder(y)
            
            
            # Loss function
            cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=s, logits=s_hat)
            
            
            # Performance metrics
            correct_predictions = tf.equal(tf.argmax(tf.nn.softmax(s_hat), axis=1), s)
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
            bler = 1-accuracy
            
            
            # Optimizer
            lr = tf.placeholder(tf.float32, shape=()) #
            train_op = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
        
            # References to graph variables we need to access later 
            self.vars = {
                'accuracy': accuracy,
                'batch_size': batch_size,
                'bler': bler,
                'cross_entropy': cross_entropy,
                'init': tf.global_variables_initializer(),
                'lr': lr,
                'noise_std': noise_std,
                'noise': noise,
                'p': p,
                's': s,
                's_hat': s_hat,
                'train_op': train_op,
                'x': x,
                'y': y,
            }            
            self.saver = tf.train.Saver()
        return
    
    def create_session(self):
        '''Create a session for the autoencoder instance with the compuational graph'''
        self.sess = tf.Session(graph=self.graph) # this is how we load the exact graph of the object instance       
        self.sess.run(self.vars['init'])
        return
    
    def encoder(self, input):
        '''The transmitter'''
        W = self.weight_variable((self.M,self.M))
        x = tf.nn.elu(tf.nn.embedding_lookup(W, input)) 
        x = tf.layers.dense(x, 2*self.n, activation=None)
        x = tf.reshape(x, shape=[-1,2,self.n]) 
        #Average power normalization
        x = x/tf.sqrt(2*tf.reduce_mean(tf.square(x))) 
        return x
    
    def decoder(self, input):
        '''The Receiver'''
        y = tf.reshape(input, shape=[-1,2*self.n])
        y = tf.layers.dense(y, 4*self.n, activation=tf.nn.relu)
        y = tf.layers.dense(y, self.M, activation=None)
        return y
       
    def EbNo2Sigma(self, ebnodb):
        '''Convert Eb/No in dB to noise standard deviation'''
        ebno = 10**(ebnodb/10)
        return 1/np.sqrt(2*self.bits_per_symbol*ebno) 
    
    def gen_feed_dict(self, perturbation, batch_size, ebnodb, lr):
        '''Generate a feed dictionary for training and validation'''        
        return {
            self.vars['p']: perturbation,
            self.vars['batch_size']: batch_size,
            self.vars['noise_std']: self.EbNo2Sigma(ebnodb),
            self.vars['lr']: lr}       
    
    def load(self, filename):
        '''Load a pre_trained model'''
        return self.saver.restore(self.sess, filename)
        
    def save(self, filename):
        '''Save the current model'''
        return self.saver.save(self.sess, filename)  
    
    def test_step(self, p, batch_size, ebnodb):
        '''Compute the BLER over a single batch and Eb/No'''
        bler = self.sess.run(self.vars['bler'], feed_dict=self.gen_feed_dict(p, batch_size, ebnodb, lr=0))
        return bler
    
    def transmit(self, s):
        '''Returns the transmitted sigals corresponding to message indices'''
        return self.sess.run(self.vars['x'], feed_dict={self.vars['s']: s})
       
    def train(self, p, training_params, validation_params):  
        '''Training and validation loop'''
        for index, params in enumerate(training_params):            
            batch_size, lr, ebnodb, iterations = params            
            print('\nBatch Size: ' + str(batch_size) +
                  ', Learning Rate: ' + str(lr) +
                  ', EbNodB: ' + str(ebnodb) +
                  ', Iterations: ' + str(iterations))
            
            val_size, val_ebnodb, val_steps = validation_params[index]
            
            for i in range(iterations):
                self.train_step(p, batch_size, ebnodb, lr)    
                if (i%val_steps==0):
                    bler = self.sess.run(self.vars['bler'], feed_dict=self.gen_feed_dict(p,val_size, val_ebnodb, lr))
                    print(bler)                           
        return       
    
    def train_step(self, p, batch_size, ebnodb, lr):
        '''A single training step'''
        self.sess.run(self.vars['train_op'], feed_dict=self.gen_feed_dict(p, batch_size, ebnodb, lr)) 
        return 
    
    def weight_variable(self, shape):
        '''Xavier-initialized weights optimized for ReLU Activations'''
        (fan_in, fan_out) = shape
        low = np.sqrt(6.0/(fan_in + fan_out)) 
        high = -np.sqrt(6.0/(fan_in + fan_out))
        return tf.Variable(tf.random_uniform(shape, minval=low, maxval=high, dtype=tf.float32))
    

    def bler_sim_attack_AWGN(self, p, PSR_dB, ebnodbs, batch_size, iterations):
        np.random.seed(seed=self.seed)
        PSR = 10**(PSR_dB/10)
        scale_factor = np.sqrt( (PSR * self.n) / (np.linalg.norm(p)**2 +  0.00000001) ) # 
        p = scale_factor * p
        BLER_no_attack = np.zeros_like(ebnodbs)
        BLER_attack = np.zeros_like(ebnodbs)
        BLER_jamming = np.zeros_like(ebnodbs)
        for i in range(iterations):
            # No attack - clean case
            bler = np.array([self.sess.run(self.vars['bler'],
                            feed_dict=self.gen_feed_dict(np.zeros([1,2,self.n]), batch_size, ebnodb, lr=0)) for ebnodb in ebnodbs]) #
            BLER_no_attack = BLER_no_attack + bler/iterations
            # attack  - synchronous attack
            bler_attack = np.array([self.sess.run(self.vars['bler'],
                            feed_dict=self.gen_feed_dict(  p ,batch_size, ebnodb, lr=0)) for ebnodb in ebnodbs]) # 
            BLER_attack = BLER_attack + bler_attack/iterations
            # Jamming attack
            normal_noise_as_jammer = np.random.normal(0,1,p.shape)
            jamming = np.linalg.norm(p) * (1 / np.linalg.norm(normal_noise_as_jammer)) * normal_noise_as_jammer
            bler_jamming= np.array([self.sess.run(self.vars['bler'],
                            feed_dict=self.gen_feed_dict(jamming,batch_size, ebnodb, lr=0)) for ebnodb in ebnodbs]) 
            BLER_jamming = BLER_jamming + bler_jamming/iterations
        return BLER_no_attack, BLER_attack, BLER_jamming
   

    def fgm_attack(self,s,p, ebnodb): 
        '''Create an input specific adversarial example using the method proposed by Sadeghi and Larsson in [2] '''
        np.random.seed(seed=self.seed)
        num_class = self.M 
        y_reshaped = np.reshape(self.sess.run(self.vars['y'], feed_dict={self.vars['s']: s, self.vars['p']: p, self.vars['noise_std']: self.EbNo2Sigma(ebnodb)}), [1,2,self.n])         
        eps_acc = 0.0000001 * np.linalg.norm(y_reshaped)
        epsilon_vector = np.zeros([num_class])
        predictions = tf.nn.softmax(self.vars['s_hat'], name = 'predictions')        
        
        for cls in range(num_class):
            s_target = np.array([cls])
            adv_per_needtoreshape  = -1 * np.asarray( self.sess.run(tf.gradients(self.vars['cross_entropy'],self.vars['y']), feed_dict={self.vars['y']: y_reshaped , self.vars['s']:s_target }) )
            adv_per = adv_per_needtoreshape.reshape(1,2,self.n)
            norm_adv_per = adv_per / (np.linalg.norm(adv_per) +  0.000000000001)
            epsilon_max = 1 * np.linalg.norm(y_reshaped)
            epsilon_min = 0
            num_iter = 0
            wcount = 0
            while (epsilon_max-epsilon_min > eps_acc) and (num_iter < 30):
                wcount = wcount+1
                num_iter = num_iter +1
                epsilon = (epsilon_max + epsilon_min)/2
                adv_img_givencls = y_reshaped + (epsilon * norm_adv_per)
                
                predicted_probabilities = self.sess.run(predictions, feed_dict={self.vars['y']: adv_img_givencls})
            
                compare = np.equal(np.argmax(predicted_probabilities),s)
                if compare:
                    epsilon_min = epsilon
                else:
                    epsilon_max = epsilon
            epsilon_vector[cls] = epsilon + eps_acc
        false_cls = np.argmin(epsilon_vector)
        minimum_epsilon = np.min(epsilon_vector)
        adv_dirc = -1 * np.asarray(self.sess.run(tf.gradients(self.vars['cross_entropy'],self.vars['y']), feed_dict={self.vars['y']: y_reshaped, self.vars['s']: np.asarray([false_cls]) })  ).reshape(1,2,self.n)
        norm_adv_dirc = adv_dirc / (np.linalg.norm(adv_dirc) + 0.000000000001)
        adv_perturbation = minimum_epsilon * norm_adv_dirc
        return adv_perturbation, false_cls, minimum_epsilon
        

    
    def UAPattack_fgm(self,ebnodb,num_samples,PSR_dB):
        '''Create a Universal Adversarial Perturbation as suggested by Alg. 1 of Sadeghi et al in [2]'''
        np.random.seed(seed=self.seed)
        universal_per_fgm = np.zeros([1,2,self.n])
        for cnr_index in range(num_samples):#            
            s =  np.asarray([np.floor(np.random.uniform(0,16,1))]).reshape(1) 
            predicted_label = np.argmax( self.sess.run(self.vars['s_hat'], feed_dict={self.vars['s']:s, self.vars['p']:universal_per_fgm, self.vars['noise_std']: self.EbNo2Sigma(ebnodb)}) )
            if predicted_label == s:
                # First we need to find adverssarial direction for this instant  by solving eq. (1) of the paper
                adv_perturbation,_,_ = self.fgm_attack(s, universal_per_fgm,ebnodb)
                adv_perturbn_reshaped = adv_perturbation.reshape([1,2,self.n])
                UAP = universal_per_fgm + adv_perturbation.reshape([1,2,self.n])
                PSR = 10**(PSR_dB/10)
                Epsilon_uni = np.sqrt( (PSR * self.n) / (np.linalg.norm(UAP)**2 + 0.00000001) )
                # Second we need to revise the universal perturbation
                if np.linalg.norm(universal_per_fgm + adv_perturbn_reshaped) < Epsilon_uni: 
                    universal_per_fgm = universal_per_fgm + adv_perturbn_reshaped
                else:
                    universal_per_fgm =  Epsilon_uni * (universal_per_fgm + adv_perturbn_reshaped) 
        return universal_per_fgm