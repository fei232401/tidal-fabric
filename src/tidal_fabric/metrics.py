"""指标收集:L1 的证据输出。队列/流指标就近挂在对象上,这里只放全局级。"""
from dataclasses import dataclass


@dataclass
class SimMetrics:
    storms: int = 0            # watchdog 触发的长暂停事件(全局累计)
    delivered_chunks: int = 0

    def summary(self):
        return {"storms": self.storms, "delivered_chunks": self.delivered_chunks}
