"""M2 组件测试:ring 逐步推进/路由稳定/潮汐到达/源限速/H2 机理。"""
import random

from tidal_fabric.topology import CLOS
from tidal_fabric.sim import Simulator
from tidal_fabric.flows import BulkFlow, KVFlow
from tidal_fabric.workloads import RingTraining, TidalKVWorkload

MB = 1024 * 1024


def test_ring_steps_and_timing():
    """2 rank ring:步数推进 + 步时间 ≈ seg/NIC 带宽(两段反向无争抢)。"""
    topo = CLOS(n_leaf=2, n_spine=1, servers_per_leaf=2)
    sim = Simulator(topo)
    ring = RingTraining(sim, ["s0_0", "s1_0"], 16 * MB, chunk_size=256 * 1024)
    ring.start()
    sim.run(until=0.004)
    assert ring.step_count >= 3
    # seg = 2×(1/2)×16MB = 16MB;瓶颈 NIC 25GB/s → 0.64ms + 传播 ≈ 0.70ms
    assert all(0.0006 < t < 0.0009 for t in ring.step_times)


def test_ring_routing_stable_across_steps():
    """段 key 稳定 → ECMP 路由跨 step 不变(通信环固定;重建才重哈希,M4)。"""
    topo = CLOS(n_leaf=2, n_spine=2, servers_per_leaf=1)
    sim = Simulator(topo)
    ring = RingTraining(sim, ["s0_0", "s1_0"], 4 * MB, chunk_size=256 * 1024)
    ring.start()
    sim.run(until=0.002)
    assert ring.step_count >= 2
    assert sim.flows["ring-g0-s0-seg0"].path == sim.flows["ring-g0-s1-seg0"].path


def test_tidal_arrival_counts_deterministic():
    """泊松到达:数量在期望量级 + 双跑一致(确定性)。"""
    def build():
        topo = CLOS(n_leaf=2, n_spine=1, servers_per_leaf=2)
        sim = Simulator(topo)
        kv = TidalKVWorkload(sim, "s0_0", "s1_0", 1 * MB, 1.0,
                             [(0.0, 1.0, 20.0)], random.Random(42))
        kv.start()
        sim.run(until=2.0)
        return kv
    a, b = build(), build()
    assert 5 <= len(a.flows) <= 50
    assert len(a.flows) == len(b.flows)
    assert all(f.done for f in a.flows)


def test_rate_limit_pacing():
    """源限速:完成时间 ≈ bytes/rate(而非 NIC 线速)。"""
    topo = CLOS(n_leaf=2, n_spine=1, servers_per_leaf=2)
    sim = Simulator(topo)
    f = BulkFlow(sim, "capped", "s0_0", "s1_0", 64 * MB,
                 rate_limit=10e9, chunk_size=256 * 1024)
    f.start()
    sim.run(until=0.05)
    assert f.done
    assert 0.0055 < f.finish < 0.0075   # 64MB/10GB/s = 6.4ms


def test_h2_cap_protects_kv():
    """H2 机理 toy 验证:训练限速 → KV 延迟改善(恢复),训练自身变慢(代价)。"""
    def scenario(cap):
        topo = CLOS(n_leaf=2, n_spine=1, servers_per_leaf=2, spine_rate=20e9)
        sim = Simulator(topo)
        train = BulkFlow(sim, "train", "s0_0", "s1_0", 512 * MB, rate_limit=cap)
        kv = KVFlow(sim, "kv", "s0_1", "s1_1", 64 * MB)
        train.start()
        kv.start()
        sim.run(until=1.0)
        assert train.done and kv.done
        return kv.latency, train.finish

    kv_full, train_full = scenario(None)
    kv_capped, train_capped = scenario(4e9)
    assert kv_capped < kv_full        # H2:通信让渡 → KV 恢复
    assert train_capped > train_full  # 代价:训练变慢
