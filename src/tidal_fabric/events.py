"""确定性离散事件队列(学习清单 ⑫)。

决策留痕(设计文档 §三):
- 同刻事件按入队顺序(seq 单调递增)执行 → 任何场景重放结果逐位一致。
- 不做事件取消:PFC PAUSE/RESUME 语义幂等(重复置位无害),避免撤销状态机。
"""
import heapq
from itertools import count


class EventQueue:
    def __init__(self):
        self._heap = []
        self._seq = count()

    def schedule(self, time, action):
        heapq.heappush(self._heap, (time, next(self._seq), action))

    def peek_time(self):
        return self._heap[0][0] if self._heap else None

    def pop(self):
        """弹出最早事件,返回 (time, action)。"""
        time, _, action = heapq.heappop(self._heap)
        return time, action

    def __len__(self):
        return len(self._heap)
