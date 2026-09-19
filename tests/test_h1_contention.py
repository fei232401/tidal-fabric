"""H1 机理在 toy 规模的复现:让卡没让路 → 共置 KV 搬运显著恶化。

数字依据:上联 20GB/s 为共享瓶颈。
- KV 独占:512MB/20GB/s ≈ 25.6ms(+少量背压开销);
- KV 与训练共置:FIFO 公平分享 → ~512MB/10GB/s ≈ 51ms(≈2×)。
真实 H1 实验矩阵(潮汐到达 + ring + 违约率)在 M2 交付。
"""
from tidal_fabric.topology import CLOS
from tidal_fabric.sim import Simulator
from tidal_fabric.flows import BulkFlow, KVFlow

MB = 1024 * 1024
TRAIN_BYTES = 1 * 1024 * MB
KV_BYTES = 512 * MB


def _kv_latency(with_training: bool) -> float:
    topo = CLOS(n_leaf=2, n_spine=1, servers_per_leaf=2, spine_rate=20e9)
    sim = Simulator(topo)
    kv = KVFlow(sim, "kv", "s0_1", "s1_1", KV_BYTES)
    if with_training:
        BulkFlow(sim, "train", "s0_0", "s1_0", TRAIN_BYTES)
    for f in sim.flows.values():
        f.start()
    sim.run(until=2.0)
    assert all(f.done for f in sim.flows.values())
    return kv.latency


def test_h1_kv_degrades_when_colocated():
    alone = _kv_latency(False)
    shared = _kv_latency(True)
    assert alone < 0.035
    assert shared > 1.5 * alone
