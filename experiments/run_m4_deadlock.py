#!/usr/bin/env python3
"""M4 实验(H3 主战场):死锁出现率扫描 + 静态预检有效性 + 检测延迟对比。

运行:cd ~/projects/tidal-fabric && ~/miniconda3/bin/python experiments/run_m4_deadlock.py
输出:experiments/results/m4_deadlock.json + 控制台摘要

场景设计(决策留痕,详见复习包 02_设计文档/05_L1_M4实验设计.md):
- 4L×2S×2srv,spine 25GB/s = 接入速率(每 leaf 双服务器 → 每条上联结构性
  2× 过载——第一版构造教训:入不敷出的队列会自解,死锁需要持续过载);
- ring 4 rank(一 leaf 一 rank,s0_0..s3_0),KV 受害流在 s0_1→s1_1(固定路径);
- 自适应路由:每段以概率 p 走 detour(经随机中转 leaf"下再上")——
  真实依据:AI/HPC fabric 自适应非最短路由(Slingshot/Spectrum-X/Valiant);
- 每 0.1s 强制重建(同成员)→ 段路由重掷骰子 = 让渡重哈希的受控代理
  (gen 递增,ScanRouter 按代规划);
- 预检模式:候选映射(含 KV 路径)有环 → 重掷(≤5 次)→ 全最小兜底;
- 死锁判定:仿真末端存在"paused && 占用>0 && 持续>0.3s"的队列(永久冻结)。
"""
import json
import pathlib
import random
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tidal_fabric.topology import CLOS
from tidal_fabric.sim import Simulator
from tidal_fabric.workloads import RingTraining, TidalKVWorkload
from tidal_fabric.precheck import precheck_routes

SPINE_RATE = 25e9
GRAD_BYTES = 100 * 1024 * 1024
RING_SERVERS = ["s0_0", "s1_0", "s2_0", "s3_0"]
KV_SRC, KV_DST = "s0_1", "s1_1"
KV_BYTES = 128 * 1024 * 1024
KV_RATE = 8.0
DURATION = 1.2
REROLL_INTERVAL = 0.1
P_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]
SEEDS = list(range(12))
FREEZE_MIN = 0.3


class ScanRouter:
    """按代规划段路由:每段以 p 走 detour;预检模式下联合验环、重掷、兜底。"""

    def __init__(self, topo, servers, p, rng, use_precheck, kv_path,
                 prefix="ring", max_retry=5):
        self.topo = topo
        self.servers = list(servers)
        self.n = len(servers)
        self.p = p
        self.rng = rng
        self.use_precheck = use_precheck
        self.kv_path = kv_path
        self.prefix = prefix
        self.max_retry = max_retry
        self.planned_gens = 0
        self.rejected_gens = 0
        self.fallback_gens = 0
        self._cache = {}

    def _candidate(self, gen):
        paths = []
        for i in range(self.n):
            key = f"{self.prefix}-g{gen}-seg{i}"
            s = self.servers[i]
            d = self.servers[(i + 1) % self.n]
            la, lb = self.topo._leaf_of[s], self.topo._leaf_of[d]
            via = [f"L{k}" for k in range(self.topo.n_leaf)
                   if f"L{k}" not in (la, lb)]
            if self.rng.random() < self.p and via:
                paths.append(self.topo.route_detour(
                    s, d, key, via_leaf=self.rng.choice(via)))
            else:
                paths.append(self.topo.route(s, d, key))
        return paths

    def _minimal(self, gen):
        return [self.topo.route(self.servers[i],
                                self.servers[(i + 1) % self.n],
                                f"{self.prefix}-g{gen}-seg{i}")
                for i in range(self.n)]

    def __call__(self, src, dst, key):
        _, gstr, sstr = key.split("-")
        gen = int(gstr[1:])
        if gen not in self._cache:
            self.planned_gens += 1
            paths = self._candidate(gen)
            if self.use_precheck:
                ok = precheck_routes(list(paths) + [self.kv_path])["ok"]
                if not ok:
                    self.rejected_gens += 1
                    for _ in range(self.max_retry):
                        paths = self._candidate(gen)
                        if precheck_routes(list(paths) + [self.kv_path])["ok"]:
                            break
                    else:
                        paths = self._minimal(gen)
                        self.fallback_gens += 1
            self._cache[gen] = paths
        return self._cache[gen][int(sstr[3:])]


def run_trial(p, seed, use_precheck):
    topo = CLOS(n_leaf=4, n_spine=2, servers_per_leaf=2, spine_rate=SPINE_RATE)
    sim = Simulator(topo, seed=seed)
    rng = random.Random(seed * 1000 + int(p * 100))
    kv_path = topo.route(KV_SRC, KV_DST, "kv")
    router = ScanRouter(topo, RING_SERVERS, p, rng, use_precheck, kv_path)
    kv = TidalKVWorkload(sim, KV_SRC, KV_DST, KV_BYTES, deadline=0.05,
                         windows=[(0.0, DURATION, KV_RATE)], rng=rng,
                         route_fn=lambda s, d, k: kv_path)
    ring = RingTraining(sim, RING_SERVERS, GRAD_BYTES,
                        chunk_size=2 * 1024 * 1024, route_fn=router)

    def reroll():
        ring.request_rebuild(RING_SERVERS, 0.0)
        if sim.now < DURATION - REROLL_INTERVAL / 2:
            sim.schedule(sim.now + REROLL_INTERVAL, reroll)

    sim.schedule(REROLL_INTERVAL, reroll)
    kv.start()
    ring.start()
    sim.run(until=DURATION + 0.1)

    frozen = [name for (name, _), q in sim.queues.items()
              if q.paused and q.occupancy > 0
              and (sim.now - q.paused_since) > FREEZE_MIN]
    storms = sum(q.stats.storms for q in sim.queues.values())
    return {
        "deadlock": bool(frozen),
        "frozen_queues": frozen,
        "storms": storms,
        "ring_steps": ring.step_count,
        "kv_done": len([f for f in kv.flows if f.finish is not None]),
        "kv_pending": kv.pending,
        "planned_gens": router.planned_gens,
        "rejected_gens": router.rejected_gens,
        "fallback_gens": router.fallback_gens,
    }


def bench_precheck_latency():
    """预检单次耗时(本实验规模图)vs watchdog 阈值(100ms,F2)。"""
    topo = CLOS(n_leaf=4, n_spine=2, servers_per_leaf=2)
    router = ScanRouter(topo, RING_SERVERS, 1.0, random.Random(1), False,
                        topo.route(KV_SRC, KV_DST, "kv"))
    paths = router._candidate(0) + [topo.route(KV_SRC, KV_DST, "kv")]
    precheck_routes(paths)
    t0 = time.perf_counter()
    n = 1000
    for _ in range(n):
        precheck_routes(paths)
    us = (time.perf_counter() - t0) / n * 1e6
    return {"precheck_us_per_call": round(us, 1), "watchdog_ms": 100.0,
            "speedup_vs_watchdog": round(100e3 / us, 0)}


def main():
    results = {"config": {"spine_rate_GBs": SPINE_RATE / 1e9,
                          "grad_MB": GRAD_BYTES // (1024 * 1024),
                          "kv_MB": KV_BYTES // (1024 * 1024),
                          "kv_rate_per_s": KV_RATE,
                          "duration_s": DURATION,
                          "reroll_interval_s": REROLL_INTERVAL,
                          "p_values": P_VALUES, "seeds": len(SEEDS),
                          "freeze_min_s": FREEZE_MIN},
               "scan": {}, "latency": bench_precheck_latency()}

    for mode, use_precheck in (("raw", False), ("precheck", True)):
        for p in P_VALUES:
            key = f"{mode}_p{int(p * 100):02d}"
            print(f"running {key} ...", flush=True)
            trials = [run_trial(p, s, use_precheck) for s in SEEDS]
            dl = sum(1 for t in trials if t["deadlock"])
            results["scan"][key] = {
                "p_detour": p, "mode": mode,
                "deadlocks": dl, "total": len(trials),
                "deadlock_rate": round(dl / len(trials), 3),
                "mean_storms": round(sum(t["storms"] for t in trials)
                                     / len(trials), 1),
                "mean_ring_steps": round(sum(t["ring_steps"] for t in trials)
                                         / len(trials), 1),
                "mean_kv_done": round(sum(t["kv_done"] for t in trials)
                                      / len(trials), 1),
                "mean_kv_pending": round(sum(t["kv_pending"] for t in trials)
                                         / len(trials), 1),
                "rejected_gens": sum(t["rejected_gens"] for t in trials),
                "fallback_gens": sum(t["fallback_gens"] for t in trials),
                "planned_gens": sum(t["planned_gens"] for t in trials),
                "example_frozen": next((t["frozen_queues"] for t in trials
                                        if t["deadlock"]), None),
            }

    out = pathlib.Path(__file__).parent / "results" / "m4_deadlock.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\n结果已写入 {out}\n")

    print(f"{'场景':<16} {'死锁/总':>8} {'死锁率':>7} {' storms':>8} "
          f"{'步数':>6} {'拒掷':>5} {'兜底':>5}")
    for key, m in results["scan"].items():
        print(f"{key:<16} {m['deadlocks']:>4}/{m['total']:<3} "
              f"{m['deadlock_rate']:>7} {m['mean_storms']:>8} "
              f"{m['mean_ring_steps']:>6} {m['rejected_gens']:>5} "
              f"{m['fallback_gens']:>5}")
    lat = results["latency"]
    print(f"\n预检延迟: {lat['precheck_us_per_call']}µs/次 vs watchdog "
          f"{lat['watchdog_ms']}ms → 快 {lat['speedup_vs_watchdog']:.0f}×")


if __name__ == "__main__":
    main()
