"""让渡控制器:监控 KV SLO → 滞回决策 → 步边界安全点执行(H4)。

决策留痕(影子复现可推翻;项目一 Rule 同款哲学):
- 监控:每 monitor_window 一拍;可见流 = finish ≤ now−feedback_delay 的
  (视图滞后建模 = metrics 管道延迟,H4 竞态源;项目一 L3 实证
  "现场 36.9% vs 理想 3.5%"的同款机理,这里受控复现);
- 判据:窗口内 P95 > deadline → 让渡一档;p95 ≤ healthy_ratio×deadline
  且健康持续 hold_healthy → 归还一档;中间为中性带(滞回防振荡);
- 防振荡:min_interval 动作间隔;
- 安全点:动作在 ring 步边界生效(限速下一 step 生效/重建停摆一拍),
  不打断在跑的 collective = 不撞 NCCL(F4 机理,H4 解法);
- gpu_only = 空间盲缩员(从末位移除 rank):不感知网络拓扑——这正是
  "让卡≠让路"的策略级复现;拓扑感知选 rank = 项目三思想迁移(future work)。
  net_aware = 直接限速共享链路上的段(H2 机理动态化)。
"""
from .stats import percentile


class ConcedeController:
    def __init__(self, sim, kv, ring, deadline, policy="net_aware",
                 monitor_window=0.5, feedback_delay=0.0, hold_healthy=1.0,
                 min_interval=0.5, min_samples=3, healthy_ratio=0.95,
                 cap_levels=(None, 12.5e9, 6.25e9, 3.125e9),
                 min_ranks=2, rebuild_time=0.05):
        if policy not in ("net_aware", "gpu_only"):
            raise ValueError(f"未知策略: {policy}")
        self.sim = sim
        self.kv = kv
        self.ring = ring
        self.deadline = deadline
        self.policy = policy
        self.monitor_window = monitor_window
        self.feedback_delay = feedback_delay
        self.hold_healthy = hold_healthy
        self.min_interval = min_interval
        self.min_samples = min_samples
        self.healthy_ratio = healthy_ratio
        self.cap_levels = tuple(cap_levels)
        self.min_ranks = min_ranks
        self.rebuild_time = rebuild_time
        self.cap_idx = 0
        self.members = list(ring.servers)
        self._removed = []
        self.actions = []   # (time, kind, detail)
        self._healthy_since = None
        self._last_action_t = float("-inf")
        self._ticks = 0
        self.sim.schedule(self.monitor_window, self._tick)

    def _tick(self):
        self._ticks += 1
        horizon = self.sim.now - self.feedback_delay
        visible = [f for f in self.kv.flows
                   if f.finish is not None
                   and horizon - self.monitor_window < f.finish <= horizon]
        if len(visible) >= self.min_samples:
            p95 = percentile([f.latency for f in visible], 95)
            if p95 > self.deadline:
                self._healthy_since = None
                self._act("concede", p95)
            elif p95 <= self.deadline * self.healthy_ratio:
                if self._healthy_since is None:
                    self._healthy_since = self.sim.now
                # epsilon:时刻为浮点累积,严判会把整拍漂移误判为未满
                if self.sim.now - self._healthy_since >= self.hold_healthy - 1e-9:
                    self._act("restore", p95)
            else:
                self._healthy_since = None   # 中性带:健康计时作废
        self.sim.schedule((self._ticks + 1) * self.monitor_window, self._tick)

    def _act(self, kind, p95):
        if self.sim.now - self._last_action_t < self.min_interval - 1e-9:
            return
        changed = self._concede() if kind == "concede" else self._restore()
        if changed:
            self._last_action_t = self.sim.now
            self._healthy_since = None

    def _concede(self):
        if self.policy == "net_aware":
            if self.cap_idx >= len(self.cap_levels) - 1:
                return False
            self.cap_idx += 1
            cap = self.cap_levels[self.cap_idx]
            self.ring.set_rate_limit(cap)
            self.actions.append((round(self.sim.now, 6), "concede",
                                 f"cap_idx={self.cap_idx} cap={cap}"))
            return True
        if len(self.members) <= self.min_ranks:
            return False
        removed = self.members.pop()
        self._removed.append(removed)
        self.ring.request_rebuild(list(self.members), self.rebuild_time)
        self.actions.append((round(self.sim.now, 6), "concede",
                             f"remove={removed} n={len(self.members)}"))
        return True

    def _restore(self):
        if self.policy == "net_aware":
            if self.cap_idx <= 0:
                return False
            self.cap_idx -= 1
            cap = self.cap_levels[self.cap_idx]
            self.ring.set_rate_limit(cap)
            self.actions.append((round(self.sim.now, 6), "restore",
                                 f"cap_idx={self.cap_idx} cap={cap}"))
            return True
        if not self._removed:
            return False
        back = self._removed.pop()
        self.members.append(back)
        self.ring.request_rebuild(list(self.members), self.rebuild_time)
        self.actions.append((round(self.sim.now, 6), "restore",
                             f"add={back} n={len(self.members)}"))
        return True

    @property
    def direction_changes(self):
        kinds = [k for _, k, _ in self.actions]
        return sum(1 for a, b in zip(kinds, kinds[1:]) if a != b)
