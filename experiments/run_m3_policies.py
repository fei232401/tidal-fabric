#!/usr/bin/env python3
"""M3 实验:让渡策略对照(none / gpu_only / net_aware)+ H4 视图滞后。

运行:cd ~/projects/tidal-fabric && ~/miniconda3/bin/python experiments/run_m3_policies.py
输出:experiments/results/m3_policies.json + 控制台摘要

场景设计(决策留痕,详见复习包 02_设计文档/04_L1_M3实验设计.md):
- 与 M2 同底座(n_spine=1 受控最坏情形,同潮汐/同 deadline),只换控制面;
- gpu_only = 空间盲缩员(末位移除):让卡≠让路 的策略级复现;
- net_aware = 段限速动态化(H2 机理 + 滞回控制器);
- feedback_delay = 视图滞后(H4):0 / 200ms / 500ms 三档;
- 训练公平口径 = progress_bytes(Σ 2(N-1)×G,跨 N 可比)。
"""
import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tidal_fabric.topology import CLOS
from tidal_fabric.sim import Simulator
from tidal_fabric.workloads import RingTraining, TidalKVWorkload
from tidal_fabric.controller import ConcedeController
from tidal_fabric.stats import percentile, mean

SPINE_RATE = 30e9
GRAD_BYTES = 100 * 1024 * 1024
RING_CHUNK = 2 * 1024 * 1024
RING_SERVERS = ["s0_0", "s1_0", "s2_0", "s3_0"]
KV_SRC, KV_DST = "s0_1", "s1_1"
KV_BYTES = 256 * 1024 * 1024
KV_CHUNK = 256 * 1024
KV_DEADLINE = 0.020
WINDOWS = [(0.0, 3.0, 2.0), (3.0, 7.0, 12.0), (7.0, 10.0, 2.0)]
FAST_WINDOWS = [(0.0, 2.0, 2.0), (2.0, 4.0, 12.0), (4.0, 6.0, 2.0),
                (6.0, 8.0, 12.0), (8.0, 10.0, 2.0)]
DURATION = 10.0
SEED = 42
CTRL_KW = dict(monitor_window=0.5, min_samples=3, min_interval=0.3,
               hold_healthy=1.0, healthy_ratio=0.95,
               cap_levels=(None, 12.5e9, 6.25e9, 3.125e9),
               min_ranks=2, rebuild_time=0.05)


def run_scenario(policy=None, feedback_delay=0.0, static_cap=None,
                 train_only=False, no_kv=False, windows=None):
    topo = CLOS(n_leaf=4, n_spine=1, servers_per_leaf=2, spine_rate=SPINE_RATE)
    sim = Simulator(topo, seed=SEED)
    kv = ring = ctrl = None
    if not train_only:
        kv = TidalKVWorkload(sim, KV_SRC, KV_DST, KV_BYTES, KV_DEADLINE,
                             windows or WINDOWS, random.Random(SEED),
                             chunk_size=KV_CHUNK)
    cap = static_cap
    ring = RingTraining(sim, RING_SERVERS, GRAD_BYTES,
                        chunk_size=RING_CHUNK, rate_limit=cap)
    if policy is not None:
        ctrl = ConcedeController(sim, kv, ring, KV_DEADLINE,
                                 policy=policy, feedback_delay=feedback_delay,
                                 **CTRL_KW)
    if kv:
        kv.start()
    ring.start()
    sim.run(until=DURATION + 1.0)

    m = {"delivered_chunks": sim.metrics.delivered_chunks}
    total_pauses = sum(q.stats.pauses_received for q in sim.queues.values())
    m["pfc_pause_episodes"] = total_pauses
    if kv is not None:
        lats = [l * 1000 for l in kv.latencies()]
        m["kv"] = {"count": len(kv.flows), "pending": kv.pending,
                   "p50_ms": round(percentile(lats, 50), 3),
                   "p95_ms": round(percentile(lats, 95), 3),
                   "violation_rate": round(kv.violation_rate, 4)}
    m["ring"] = {"steps": ring.step_count,
                 "progress_MB": round(ring.progress_bytes / (1024 * 1024), 1),
                 "rebuilds": ring.rebuilds,
                 "stall_ms": round(ring.stall_total * 1000, 1)}
    if ctrl is not None:
        m["ctrl"] = {"actions": len(ctrl.actions),
                     "direction_changes": ctrl.direction_changes,
                     "timeline": ctrl.actions}
    return m


def main():
    results = {
        "config": {"spine_rate_GBs": SPINE_RATE / 1e9,
                   "grad_MB": GRAD_BYTES // (1024 * 1024),
                   "kv_deadline_ms": KV_DEADLINE * 1000,
                   "windows": WINDOWS, "duration_s": DURATION, "seed": SEED,
                   "ctrl": {k: (str(v) if isinstance(v, tuple) else v)
                            for k, v in CTRL_KW.items()}},
        "scenarios": {},
    }
    scenarios = [
        ("ring_only", dict(train_only=True)),
        ("none", dict()),
        ("static_cap25", dict(static_cap=6.25e9)),
        ("gpu_only", dict(policy="gpu_only")),
        ("net_aware", dict(policy="net_aware")),
        ("net_aware_d200", dict(policy="net_aware", feedback_delay=0.2)),
        ("net_aware_d500", dict(policy="net_aware", feedback_delay=0.5)),
        ("fast_tide", dict(policy="net_aware", windows=FAST_WINDOWS)),
        ("fast_tide_d500", dict(policy="net_aware", feedback_delay=0.5,
                                windows=FAST_WINDOWS)),
    ]
    for name, kw in scenarios:
        print(f"running {name} ...", flush=True)
        results["scenarios"][name] = run_scenario(**kw)

    base = results["scenarios"]["ring_only"]["ring"]["progress_MB"]
    for m in results["scenarios"].values():
        m["ring"]["progress_ratio"] = round(m["ring"]["progress_MB"] / base, 3)

    out = pathlib.Path(__file__).parent / "results" / "m3_policies.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\n结果已写入 {out}\n")

    print(f"{'场景':<16} {'KV p95':>8} {'违约率':>7} {'进度比':>7} "
          f"{'动作':>4} {'换向':>4} {'重建':>4} {'PFC':>6}")
    for name, m in results["scenarios"].items():
        kv = m.get("kv", {})
        ctrl = m.get("ctrl", {})
        print(f"{name:<16} {kv.get('p95_ms', '-'):>8} "
              f"{kv.get('violation_rate', '-'):>7} "
              f"{m['ring']['progress_ratio']:>7} "
              f"{ctrl.get('actions', '-'):>4} "
              f"{ctrl.get('direction_changes', '-'):>4} "
              f"{m['ring']['rebuilds']:>4} {m['pfc_pause_episodes']:>6}")
    for name, m in results["scenarios"].items():
        if "ctrl" in m and m["ctrl"]["timeline"]:
            print(f"\n{name} 动作时间线: " +
                  "; ".join(f"{t:.1f}s {k}({d})" for t, k, d
                            in m["ctrl"]["timeline"]))


if __name__ == "__main__":
    main()
