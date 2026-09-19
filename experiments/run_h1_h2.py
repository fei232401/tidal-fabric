#!/usr/bin/env python3
"""M2 实验:H1(让卡没让路)与 H2(通信让渡)主矩阵 + chunk 敏感性。

运行:cd ~/projects/tidal-fabric && ~/miniconda3/bin/python experiments/run_h1_h2.py
输出:experiments/results/h1_h2.json + 控制台摘要

场景设计(决策留痕,详见复习包 02_设计文档/03_L1_M2实验设计.md):
- n_spine=1 = 受控最坏情形:全部跨 leaf 流量共享同一 spine 路径;
  KV(s0_1→s1_1)与 ring 段 A(s0_0→s1_0)恰好共享 leaf0 上联 + spine→leaf1 两条链路。
  ECMP 分散效应(2 spine 天然降低冲突)在 M4 讨论。
- cap = ring 每段流的源侧限速(H2 让渡手段);满速率参照 = NIC 25GB/s。
"""
import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tidal_fabric.topology import CLOS
from tidal_fabric.sim import Simulator
from tidal_fabric.workloads import RingTraining, TidalKVWorkload
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
DURATION = 10.0
SEED = 42
CAPS = {"cap_75": 0.75 * 25e9, "cap_50": 0.5 * 25e9, "cap_25": 0.25 * 25e9}


def run_scenario(train=True, cap=None, kv_chunk=KV_CHUNK, ring_chunk=RING_CHUNK,
                 train_only=False):
    topo = CLOS(n_leaf=4, n_spine=1, servers_per_leaf=2, spine_rate=SPINE_RATE)
    sim = Simulator(topo, seed=SEED)
    kv = ring = None
    if not train_only:
        kv = TidalKVWorkload(sim, KV_SRC, KV_DST, KV_BYTES, KV_DEADLINE,
                             WINDOWS, random.Random(SEED), chunk_size=kv_chunk)
    if train:
        ring = RingTraining(sim, RING_SERVERS, GRAD_BYTES,
                            chunk_size=ring_chunk, rate_limit=cap)
    if kv:
        kv.start()
    if ring:
        ring.start()
    sim.run(until=DURATION + 1.0)

    m = {"delivered_chunks": sim.metrics.delivered_chunks}
    total_pauses = sum(q.stats.pauses_received for q in sim.queues.values())
    total_storms = sum(q.stats.storms for q in sim.queues.values())
    max_paused = max((q.stats.paused_total for q in sim.queues.values()), default=0.0)
    m["pfc"] = {"pause_episodes": total_pauses, "storms": total_storms,
                "max_paused_ms": round(max_paused * 1000, 3)}
    if kv is not None:
        lats = [l * 1000 for l in kv.latencies()]
        m["kv"] = {
            "count": len(kv.flows),
            "pending": kv.pending,
            "p50_ms": round(percentile(lats, 50), 3),
            "p95_ms": round(percentile(lats, 95), 3),
            "p99_ms": round(percentile(lats, 99), 3),
            "violation_rate": round(kv.violation_rate, 4),
        }
    if ring is not None:
        steps = [t * 1000 for t in ring.step_times]
        m["ring"] = {
            "steps": ring.step_count,
            "step_mean_ms": round(mean(steps), 3),
            "step_p95_ms": round(percentile(steps, 95), 3),
        }
    return m


def main():
    results = {
        "config": {
            "spine_rate_GBs": SPINE_RATE / 1e9,
            "grad_bytes_MB": GRAD_BYTES // (1024 * 1024),
            "ring_chunk_MB": RING_CHUNK // (1024 * 1024),
            "kv_bytes_MB": KV_BYTES // (1024 * 1024),
            "kv_chunk_KB": KV_CHUNK // 1024,
            "kv_deadline_ms": KV_DEADLINE * 1000,
            "windows": WINDOWS,
            "duration_s": DURATION,
            "seed": SEED,
            "caps_GBs": {k: round(v / 1e9, 2) for k, v in CAPS.items()},
        },
        "scenarios": {},
        "sensitivity": {},
    }

    scenarios = [
        ("kv_only", dict(train=False)),
        ("ring_only", dict(train=True, train_only=True)),
        ("colocated", dict(train=True)),
        ("cap_75", dict(train=True, cap=CAPS["cap_75"])),
        ("cap_50", dict(train=True, cap=CAPS["cap_50"])),
        ("cap_25", dict(train=True, cap=CAPS["cap_25"])),
    ]
    for name, kw in scenarios:
        print(f"running {name} ...", flush=True)
        results["scenarios"][name] = run_scenario(**kw)

    base = results["scenarios"]["ring_only"]["ring"]["step_mean_ms"]
    for m in results["scenarios"].values():
        if "ring" in m:
            m["ring"]["slowdown_vs_ring_only"] = round(
                m["ring"]["step_mean_ms"] / base, 3)

    print("running sensitivity: kv_chunk ...", flush=True)
    for c in (128 * 1024, 512 * 1024):
        results["sensitivity"][f"kv_{c // 1024}K"] = run_scenario(
            train=True, kv_chunk=c)
    print("running sensitivity: ring_chunk ...", flush=True)
    for c in (1 * 1024 * 1024, 4 * 1024 * 1024):
        results["sensitivity"][f"ring_{c // (1024 * 1024)}M"] = run_scenario(
            train=True, ring_chunk=c)

    out = pathlib.Path(__file__).parent / "results" / "h1_h2.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\n结果已写入 {out}\n")

    hdr = f"{'场景':<12} {'KV p50':>8} {'KV p95':>8} {'KV p99':>8} {'违约率':>7} {'步均值':>8} {'慢化':>6} {'PFC风暴':>8}"
    print(hdr)
    for name, m in results["scenarios"].items():
        kv = m.get("kv", {})
        ring = m.get("ring", {})
        print(f"{name:<12} "
              f"{kv.get('p50_ms', '-'):>8} {kv.get('p95_ms', '-'):>8} "
              f"{kv.get('p99_ms', '-'):>8} {kv.get('violation_rate', '-'):>7} "
              f"{ring.get('step_mean_ms', '-'):>8} "
              f"{ring.get('slowdown_vs_ring_only', '-'):>6} "
              f"{m['pfc']['storms']:>8}")
    print("\n敏感性(colocated 基准 KV p95 = "
          f"{results['scenarios']['colocated']['kv']['p95_ms']} ms):")
    for name, m in results["sensitivity"].items():
        print(f"  {name:<10} KV p95={m['kv']['p95_ms']:>7} ms  "
              f"步均值={m['ring']['step_mean_ms']:>7} ms")


if __name__ == "__main__":
    main()
