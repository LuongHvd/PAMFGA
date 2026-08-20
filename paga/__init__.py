"""
paga
=========
Khung thực nghiệm tái tổ chức cho bài báo

    "Evolutionary-based Transfer Learning for Physical Attack on
     End-to-End Autoencoder Communications"

Phương pháp đề xuất:
  * PAGA   (Physical Attack based on Genetic Algorithm)      -- kịch bản HỘP TRẮNG
  * PAMFGA (Multi-Factorial GA-based Physical Attack)        -- kịch bản HỘP ĐEN

Package chỉ chứa hai thuật toán đề xuất và những gì chúng cần. Baseline, biến thể
đối chiếu và toàn bộ script phân tích nằm ở `legacy/`.

  autoencoders.py  các autoencoder của bài: AE_MLP, AE_CNN, AE_netOne_MLP,
                   AE_netTwo_DeepMLP, AE_inf_rate
  channels.py      mô hình tín hiệu mở rộng: kênh attacker->receiver riêng biệt,
                   path loss, Rayleigh/Rician (K-factor), lỗi CSI, lệch pha/CFO/timing
  oracles.py       nạp model, oracle gradient hộp trắng (dùng cho hạt elite)
  fitness.py       hàm mục tiêu, ràng buộc PSR, đo BLER
  operators.py     toán tử tiến hoá và các cách gộp perturbation
  budget.py        ngân sách chung + đo tài nguyên
  attacks/         PAGA, PAMFGA, hạt elite, và jamming làm tham chiếu
  train_awgn.py    huấn luyện autoencoder dưới AWGN
  train_channel.py huấn luyện autoencoder dưới TỪNG kênh

GIỚI HẠN QUAN TRỌNG (phải nêu trong bài):
  Toàn bộ kết quả "over-the-air" ở đây là **simulated over-the-air / additive
  physical-layer attack**. Không có phần cứng SDR nào được dùng. Xem
  `channels.py::OTA_LIMITATION_NOTE`.
"""

__version__ = "1.0.0"

# Các lớp graph TF1 dùng `tf.layers`, chỉ tồn tại khi TensorFlow trỏ về legacy
# Keras. Biến này PHẢI được đặt TRƯỚC lần import tensorflow đầu tiên, nên nó nằm
# ngay đầu package. Lưu ý: `All_Autoencoder_Classes.py` xoá biến này khi được
# import, vì vậy `tf1_compat` đặt lại sau đó (xem chú thích ở module đó).
import os as _os

_os.environ["TF_USE_LEGACY_KERAS"] = "1"
_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
del _os

# Console Windows mặc định là cp1252 -> mọi chữ tiếng Việt có dấu sẽ ném
# UnicodeEncodeError. Ép stdout/stderr về UTF-8 ngay khi import package.
import sys as _sys

for _stream in ("stdout", "stderr"):
    _s = getattr(_sys, _stream, None)
    if _s is not None and hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
del _sys, _stream, _s

__all__ = [
    "channels",
    "envinfo",
    "fitness",
    "models",
    "operators",
    "plotting",
    "profiling",
    "stats",
]
