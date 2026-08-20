"""
reference.py
============
Đường THAM CHIẾU, không phải phương pháp cạnh tranh.

Chỉ có nhiễu Gauss cùng mức năng lượng ("jamming"). Nó tồn tại để trả lời câu hỏi
duy nhất mà mọi bảng kết quả tấn công đều phải trả lời:

    phần suy giảm quan sát được đến từ CẤU TRÚC đối kháng của perturbation,
    hay chỉ từ việc bơm thêm năng lượng vào kênh?

Vì vậy jamming luôn được chuẩn hoá về ĐÚNG ràng buộc PSR như perturbation đối
kháng. Khoảng cách giữa hai đường là phần quy được cho cấu trúc.

Mọi baseline tối ưu (Random Search, DE, PSO, CMA-ES, GA) và mọi tấn công của công
trình khác (FGMA, RSBA, RMAEP) đã được chuyển sang `legacy/baselines/`; repo này
chỉ giữ hai phương pháp đề xuất.
"""
import numpy as np

from . import AttackResult, HistoryLogger


def jamming(fitness, D, budget, rng, init=None, ctx=None, **kw):
    """
    Nhiễu Gauss cùng năng lượng, không tối ưu gì.

    Chỉ tốn 1 evaluation, nên khi đặt cạnh các phương pháp tiêu hết ngân sách thì
    phải nêu rõ đây là THAM CHIẾU chứ không phải đối thủ ở cùng budget.
    """
    meter = fitness.meter
    logger = HistoryLogger(meter)
    vec = rng.standard_normal(D)
    f = fitness(vec)
    logger.log(f, [f], 0)
    return AttackResult(np.asarray(vec, float).reshape(-1), float(f),
                        logger.records,
                        {"note": "tham chiếu, không lặp, không tối ưu"})
