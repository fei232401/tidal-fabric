"""流:NCCL 大象流(批量)与 KV 搬运流(一次性,延迟敏感)。

学习清单 ⑦⑧⑨ 的代码对应物。M1 只做"单次批量传输"两种形态;
NCCL ring 逐步推进、潮汐到达、控制器让渡策略在 M2/M3 加。
源 pacing:生产者按 NIC 队列剩余容量推进(NIC 满则挂等待者,腾出空间唤醒)。
"""
from dataclasses import dataclass
from typing import Optional, Tuple

from .pfc import LOSSLESS

DEFAULT_CHUNK_SIZE = 256 * 1024


@dataclass
class Chunk:
    flow_id: str
    size: int
    path: Tuple
    priority: int = LOSSLESS
    hop: int = 0
    ingress: Optional[str] = None
    created: float = 0.0


class Flow:
    def __init__(self, sim, flow_id, src, dst, total_bytes,
                 priority=LOSSLESS, chunk_size=DEFAULT_CHUNK_SIZE, key=None,
                 rate_limit=None, path=None):
        if total_bytes <= 0:
            raise ValueError("total_bytes 必须为正")
        self.sim = sim
        self.id = flow_id
        self.src, self.dst = src, dst
        self.total_bytes = total_bytes
        self.priority = priority
        self.chunk_size = chunk_size
        self.key = key or flow_id
        self.rate_limit = rate_limit
        self._next_slot = 0.0
        self.path = path if path is not None else sim.topo.route(src, dst, self.key)
        self.nic = sim.queue(self.path[0].name, priority)
        self.remaining = total_bytes
        self.bytes_done = 0
        self.started = None
        self.finish = None
        self.on_finish = None
        self.chunk_latencies = []
        self._waiter_registered = False
        sim.flows[flow_id] = self

    def start(self):
        self.started = self.sim.now
        self._push_more()

    def _push_more(self):
        self._waiter_registered = False
        while self.remaining > 0:
            if self.rate_limit is not None and self.sim.now < self._next_slot:
                self.sim.schedule(self._next_slot, self._push_more)
                return
            size = min(self.chunk_size, self.remaining)
            if self.nic.remaining_capacity() < size:
                if not self._waiter_registered:
                    self.nic.register_space_waiter(self._push_more)
                    self._waiter_registered = True
                return
            chunk = Chunk(self.id, size, self.path, self.priority,
                          hop=0, ingress=None, created=self.sim.now)
            self.remaining -= size
            self.nic.push(chunk)
            if self.rate_limit is not None:
                self._next_slot = self.sim.now + size / self.rate_limit

    def on_delivered(self, chunk):
        self.bytes_done += chunk.size
        self.chunk_latencies.append(self.sim.now - chunk.created)
        if self.bytes_done >= self.total_bytes:
            self.finish = self.sim.now
            if self.on_finish is not None:
                self.on_finish()

    @property
    def done(self):
        return self.bytes_done >= self.total_bytes

    @property
    def latency(self):
        """完成时延(创建→最后一字节送达)。KV 流即 TTFT 代理。"""
        if self.finish is None or self.started is None:
            return None
        return self.finish - self.started


class BulkFlow(Flow):
    """NCCL ring 段的 M1 形态:一次性批量字节(M2 升级为逐步迭代)。"""


class KVFlow(Flow):
    """KV 搬运:一次请求一整块,完成时延 = TTFT 代理。"""

    def __init__(self, sim, flow_id, src, dst, total_bytes,
                 deadline=None, **kw):
        super().__init__(sim, flow_id, src, dst, total_bytes, **kw)
        self.deadline = deadline

    @property
    def violated(self):
        return (self.deadline is not None and self.finish is not None
                and self.finish - self.started > self.deadline)
