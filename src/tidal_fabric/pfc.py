"""PFC 语义:水位阈值 + 逐跳 PAUSE/RESUME + 边沿触发 watchdog。

学习清单 ②③④ 的代码对应物。建模口径(设计文档 §三,简化全部可推翻):
- 队列挂在 (有向链路, 优先级) 上 = chunk"从该链路发出前"的驻留缓冲;
  水位 ≥ xoff → 向队内 chunk 的来路(ingress 链路)发 PAUSE。
  (真实交换机按 ingress 队列水位触发;本简化在缓冲依赖图上等价,足以成环。)
- PAUSE 抵达 → 上游队列停止"开始新 chunk";在途 chunk 发完(≈pause quanta)。
- 滞回:xoff 触发、xon 恢复(0.8/0.5 × capacity,典型比例待扫)。
- 同一 ingress 可能被多条下游队列要求暂停 → sim 层引用计数仲裁
  (0→1 发帧、1→0 才 RESUME,防误恢复),本模块只表达"本队列想要谁停"。
- NIC(服务器出口)无更上游:只被 PAUSE,不发 PAUSE;容量兼作源 pacing 窗口。
- watchdog = 边沿触发:被暂停持续 ≥ 阈值记一次 storm(对齐 SONiC 语义,F2)。
"""
from collections import deque

LOSSLESS = 3                              # 单一无损优先级:训/推 RDMA 同类 → PFC 连坐(H1 机理)
DEFAULT_CAPACITY = 16 * 1024 * 1024       # 交换机共享缓冲量级(估算,待扫)
DEFAULT_XOFF_RATIO = 0.8
DEFAULT_XON_RATIO = 0.5
NIC_CAPACITY = 4 * 1024 * 1024            # NIC 驻留上限 = 源 pacing 窗口(估算,待扫)


class QueueStats:
    def __init__(self):
        self.pushes = 0
        self.departures = 0
        self.max_occupancy = 0
        self.pause_frames_sent = 0     # 因本队列满而发出的 PAUSE 帧(引用计数 0→1)
        self.resume_frames_sent = 0
        self.pauses_received = 0       # 被下游 PAUSE 的次数
        self.paused_total = 0.0        # 累计被暂停时长(秒)
        self.storms = 0                # watchdog 触发(≥阈值的长暂停)

    def summary(self):
        return {
            "pushes": self.pushes,
            "departures": self.departures,
            "max_occupancy": self.max_occupancy,
            "pause_frames_sent": self.pause_frames_sent,
            "resume_frames_sent": self.resume_frames_sent,
            "pauses_received": self.pauses_received,
            "paused_total": round(self.paused_total, 9),
            "storms": self.storms,
        }


class PortQueue:
    """(有向链路, 优先级) 上的一级驻留队列。"""

    def __init__(self, sim, link, priority=LOSSLESS, capacity=None,
                 xoff_ratio=DEFAULT_XOFF_RATIO, xon_ratio=DEFAULT_XON_RATIO,
                 is_nic=False):
        self.sim = sim
        self.link = link
        self.priority = priority
        self.is_nic = is_nic
        if capacity is None:
            capacity = NIC_CAPACITY if is_nic else DEFAULT_CAPACITY
        self.capacity = capacity
        self.xoff = capacity * xoff_ratio
        self.xon = capacity * xon_ratio
        self.chunks = deque()
        self.occupancy = 0
        self.contributors = {}        # ingress 链路名 -> 队内 chunk 数
        self._paused_ingress = set()  # 本队列已要求暂停的 ingress 链路名
        self.paused = False           # 被下游 PAUSE
        self.paused_since = None
        self._storm_flagged = False
        self.transmitting = False
        self._space_waiters = []      # 等空间的生产者(NIC 源 pacing)
        self.stats = QueueStats()

    # ---- 生产者接口 ----
    def remaining_capacity(self):
        return self.capacity - self.occupancy

    def register_space_waiter(self, cb):
        if cb not in self._space_waiters:
            self._space_waiters.append(cb)

    def push(self, chunk):
        self.stats.pushes += 1
        self.chunks.append(chunk)
        self.occupancy += chunk.size
        if chunk.ingress is not None:
            self.contributors[chunk.ingress] = self.contributors.get(chunk.ingress, 0) + 1
        if self.occupancy > self.stats.max_occupancy:
            self.stats.max_occupancy = self.occupancy
        # 已处于暂停水位(含本次越限)→ 立即 PAUSE 该来路(NIC 除外)
        if (not self.is_nic) and chunk.ingress is not None \
                and self.occupancy >= self.xoff \
                and chunk.ingress not in self._paused_ingress:
            self._paused_ingress.add(chunk.ingress)
            self.sim.request_pause(self, chunk.ingress)
        self._try_service()

    # ---- 发送 ----
    def _try_service(self):
        if self.paused or self.transmitting or not self.chunks:
            return
        self.transmitting = True
        duration = self.chunks[0].size / self.link.rate
        self.sim.schedule(self.sim.now + duration, self._complete_service)

    def _complete_service(self):
        self.transmitting = False
        chunk = self.chunks.popleft()
        self.occupancy -= chunk.size
        self.stats.departures += 1
        if chunk.ingress is not None:
            n = self.contributors.get(chunk.ingress, 0) - 1
            if n > 0:
                self.contributors[chunk.ingress] = n
            else:
                self.contributors.pop(chunk.ingress, None)
        # 滞回恢复:降到 xon 才放行全部来路
        if (not self.is_nic) and self.occupancy <= self.xon and self._paused_ingress:
            for name in sorted(self._paused_ingress):
                self.sim.request_resume(self, name)
            self._paused_ingress.clear()
        # 释放空间 → 唤醒等空间的生产者(源 pacing)
        if self._space_waiters:
            waiters, self._space_waiters = self._space_waiters, []
            for cb in waiters:
                cb()
        # 在途 chunk 传播到下一跳
        self.sim.deliver_after(self, chunk)
        self._try_service()

    # ---- 被下游控制 ----
    def set_paused(self, flag):
        if flag and not self.paused:
            self.paused = True
            self.paused_since = self.sim.now
            self.stats.pauses_received += 1
            self._storm_flagged = False
            token = self.paused_since

            def _watchdog_check():
                if self.paused and self.paused_since == token and not self._storm_flagged:
                    self._storm_flagged = True
                    self.stats.storms += 1
                    self.sim.metrics.storms += 1

            self.sim.schedule(self.sim.now + self.sim.watchdog_threshold,
                              _watchdog_check)
        elif (not flag) and self.paused:
            self.stats.paused_total += self.sim.now - self.paused_since
            self.paused = False
            self.paused_since = None
            self._try_service()

    def __repr__(self):
        return (f"<PQ {self.link.name} occ={self.occupancy} "
                f"paused={self.paused} chunks={len(self.chunks)}>")
