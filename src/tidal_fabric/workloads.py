"""组合工作负载:NCCL ring 训练(逐步推进)+ KV 潮汐到达。

学习清单 ⑦(ring all-reduce 2(N-1)/N)⑧⑨(PD 分离/KV 搬运/潮汐)的代码对应物。
决策留痕(影子复现可推翻):
- ring 段的 ECMP key 按"段"稳定:真实通信环建好后链路固定,不随 step 重哈希;
  让渡重建(M4)才换 key → 那正是 H3"流-路映射重排"的来源。
- 训练 chunk 量子默认 2MB(大象流延迟不敏感,量子放大只影响自身时延统计),
  KV 默认 256KB(延迟敏感);两类量子分开做敏感性扫描(实验脚本)。
- 潮汐到达 = 分段常率窗口 + 指数间隔(泊松);确定性靠显式传入的 rng。
- 段间同步:一个 step 的全部段流完成才算步完成(同步 SGD 语义),
  compute_time 可选模拟通信间计算(默认 0,纯通信口径)。
"""
from .flows import BulkFlow, KVFlow


def ring_segment_paths(servers, route_fn, gen, prefix="ring"):
    """纯函数:给定 rank 列表与路由函数 → [(seg_key, path), ...]。

    预检器与 RingTraining 共用——保证"预检算的路径"与"实际跑的路径"
    严格同源(否则预检就是自欺)。
    """
    n = len(servers)
    out = []
    for i in range(n):
        key = f"{prefix}-g{gen}-seg{i}"
        out.append((key, route_fn(servers[i], servers[(i + 1) % n], key)))
    return out


class RingTraining:
    """N 个 rank 的 ring all-reduce,逐步推进,step 时间序列即训练慢化证据。

    弹性支持(M3/M4):
    - set_rate_limit:限速变化在下一 step 生效(安全点,不打断在跑的段);
    - request_rebuild:成员变更在步边界生效 + 重建窗口停摆(时长为 A2 占位常数,
      MultiWorld 实证重建窗口存在但幅度未标定);重建换 generation → 新 ECMP key
      → 流-路映射重排(M4 死锁研究的入口);
    - progress_bytes:按步累计 2(N-1)×G(跨 N 可比的公平进度口径);
    - route_fn:路由函数注入(最小 ECMP / 自适应 detour)——key 内嵌 generation,
      自适应路由可按代切换(让渡重哈希的受控来源)。
    """

    def __init__(self, sim, servers, grad_bytes, chunk_size=2 * 1024 * 1024,
                 compute_time=0.0, rate_limit=None, prefix="ring",
                 route_fn=None):
        if len(servers) < 2:
            raise ValueError("ring 至少 2 个 rank")
        self.sim = sim
        self.servers = list(servers)
        self.n = len(servers)
        self.grad_bytes = grad_bytes
        self.seg_bytes = int(2.0 * (self.n - 1) / self.n * grad_bytes)
        if self.seg_bytes <= 0:
            raise ValueError("grad_bytes 过小")
        self.chunk_size = chunk_size
        self.compute_time = compute_time
        self.rate_limit = rate_limit
        self.prefix = prefix
        self.route_fn = route_fn or (lambda s, d, k: sim.topo.route(s, d, k))
        self.step_count = 0
        self.step_times = []
        self.progress_bytes = 0
        self.rebuilds = 0
        self.stall_total = 0.0
        self._gen = 0
        self._pending_servers = None
        self._pending_rebuild_time = 0.0
        self._pending = set()
        self._step_start = None

    def _make_keys(self):
        return [f"{self.prefix}-g{self._gen}-seg{i}" for i in range(self.n)]

    def segment_paths(self, servers=None, gen=None):
        """当前(或指定)成员/代际的段路径——预检器的输入。"""
        return ring_segment_paths(servers or self.servers, self.route_fn,
                                   self._gen if gen is None else gen,
                                   self.prefix)

    def set_rate_limit(self, v):
        """限速变化在下一 step 生效(安全点语义)。"""
        self.rate_limit = v

    def request_rebuild(self, servers, rebuild_time=0.0):
        """请求重建:步边界生效(安全点);重建期间 ring 停摆 rebuild_time。"""
        self._pending_servers = list(servers)
        self._pending_rebuild_time = rebuild_time

    def start(self):
        self._begin_step()

    def _begin_step(self):
        if self._pending_servers is not None:
            self.servers = self._pending_servers
            self.n = len(self.servers)
            self.seg_bytes = int(2.0 * (self.n - 1) / self.n * self.grad_bytes)
            self._gen += 1
            self.rebuilds += 1
            self._pending_servers = None
        self._step_start = self.sim.now
        self._pending = set()
        for key, path in ring_segment_paths(self.servers, self.route_fn,
                                            self._gen, self.prefix):
            i = int(key.rsplit("seg", 1)[1])
            fid = f"{self.prefix}-g{self._gen}-s{self.step_count}-seg{i}"
            flow = BulkFlow(self.sim, fid, self.servers[i],
                            self.servers[(i + 1) % self.n], self.seg_bytes,
                            chunk_size=self.chunk_size, key=key,
                            rate_limit=self.rate_limit, path=path)
            flow.on_finish = self._finish_cb(fid)
            flow.start()
            self._pending.add(fid)

    def _finish_cb(self, fid):
        def cb():
            self._pending.discard(fid)
            if not self._pending:
                self.step_times.append(self.sim.now - self._step_start)
                self.step_count += 1
                self.progress_bytes += 2 * (self.n - 1) * self.grad_bytes
                delay = self.compute_time
                if self._pending_servers is not None:
                    # 重建停摆在步间(不进 step_times);若请求落在停摆窗口内,
                    # 下一轮步间才补停摆(边角情形,影响 < 一步)
                    delay = self._pending_rebuild_time
                    self.stall_total += self._pending_rebuild_time
                self.sim.schedule(self.sim.now + delay, self._begin_step)
        return cb

    @property
    def step_mean(self):
        return (sum(self.step_times) / len(self.step_times)
                if self.step_times else None)


class TidalKVWorkload:
    """KV 搬运潮汐到达:分段常率窗口 + 泊松间隔,每请求一个 KVFlow(TTFT 代理)。"""

    def __init__(self, sim, src, dst, size_bytes, deadline, windows, rng,
                 chunk_size=256 * 1024, prefix="kv", route_fn=None):
        self.sim = sim
        self.src, self.dst = src, dst
        self.size_bytes = size_bytes
        self.deadline = deadline
        self.windows = [(float(a), float(b), float(r)) for a, b, r in windows]
        prev_end = 0.0
        for t0, t1, r in self.windows:
            if t0 != prev_end or t1 <= t0:
                raise ValueError("windows 必须从 0 起连续覆盖")
            prev_end = t1
        self.end = self.windows[-1][1]
        self.rng = rng
        self.chunk_size = chunk_size
        self.prefix = prefix
        self.route_fn = route_fn or (lambda s, d, k: sim.topo.route(s, d, k))
        self.flows = []
        self._count = 0

    def start(self):
        self._schedule_next(0.0)

    def _rate_at(self, t):
        for t0, t1, r in self.windows:
            if t0 <= t < t1:
                return r, t1
        return 0.0, None

    def _schedule_next(self, t):
        """从 t 安排下一个到达;采样间隔跨窗口边界则跳到边界重采样。"""
        while t < self.end:
            rate, wend = self._rate_at(t)
            if rate > 0:
                dt = self.rng.expovariate(rate)
                if t + dt < wend:
                    self.sim.schedule(t + dt, self._arrive)
                    return
                t = wend
            else:
                t = wend
        return

    def _arrive(self):
        fid = f"{self.prefix}-{self._count}"
        flow = KVFlow(self.sim, fid, self.src, self.dst, self.size_bytes,
                      deadline=self.deadline, chunk_size=self.chunk_size,
                      path=self.route_fn(self.src, self.dst, fid))
        flow.start()
        self.flows.append(flow)
        self._count += 1
        self._schedule_next(self.sim.now)

    def latencies(self):
        return [f.latency for f in self.flows if f.finish is not None]

    @property
    def violation_rate(self):
        if not self.flows:
            return 0.0
        return sum(1 for f in self.flows if f.violated) / len(self.flows)

    @property
    def pending(self):
        return sum(1 for f in self.flows if f.finish is None)
