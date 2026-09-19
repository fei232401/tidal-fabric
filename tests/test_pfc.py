"""PFC 语义测试:逐跳反压、跨级传播、watchdog。

数字依据(设计文档 §四):容量 16MB、xoff 12.8MB、xon 8MB;
场景参数使"入 > 出"成立 → 队列必然越过 xoff(无竞态,确定性)。
"""
from tidal_fabric.topology import CLOS
from tidal_fabric.sim import Simulator
from tidal_fabric.flows import BulkFlow

MB = 1024 * 1024


def test_pfc_backpressure_first_hop():
    """两源挤一条较慢上联:上联队列越过 xoff → 两个源 NIC 都被 PAUSE。"""
    topo = CLOS(n_leaf=2, n_spine=1, servers_per_leaf=2,
                server_rate=25e9, spine_rate=30e9)
    sim = Simulator(topo, watchdog_threshold=1e9)
    BulkFlow(sim, "a", "s0_0", "s1_0", 64 * MB)
    BulkFlow(sim, "b", "s0_1", "s1_1", 64 * MB)
    for f in sim.flows.values():
        f.start()
    sim.run(until=10.0)
    assert all(f.done for f in sim.flows.values())
    nic_a = sim.queue("s0_0->L0")
    nic_b = sim.queue("s0_1->L0")
    uplink = sim.queue("L0->S0")
    assert nic_a.stats.pauses_received >= 1
    assert nic_b.stats.pauses_received >= 1
    assert nic_a.stats.paused_total > 0
    assert uplink.stats.pause_frames_sent >= 2
    assert uplink.stats.max_occupancy >= uplink.xoff


def test_pfc_multihop_backpressure_chain():
    """慢接收端 → 反压链跨 3 级队列(L1 出口 → S0 出口 → L0 出口)。

    链深达 NIC 与参数相关(交换机排空快于上游填满)——这正是
    "PFC 死锁稀有、只在特定流-路组合下成环"的旁证(F1),成环条件
    的系统扫描是 M4(H3 主战场)的工作。
    """
    topo = CLOS(n_leaf=2, n_spine=1, servers_per_leaf=3,
                server_rate=10e9, spine_rate=50e9)
    sim = Simulator(topo)
    for i in range(3):
        BulkFlow(sim, f"f{i}", f"s0_{i}", "s1_0", 64 * MB)
    for f in sim.flows.values():
        f.start()
    sim.run(until=60.0)
    assert all(f.done for f in sim.flows.values())
    assert sim.queue("L1->s1_0").stats.pause_frames_sent >= 1
    assert sim.queue("S0->L1").stats.pauses_received >= 1
    assert sim.queue("L0->S0").stats.pauses_received >= 1


def test_watchdog_storm_detection():
    """暂停持续 ≥ 阈值 → 记 storm(SONiC watchdog 对照组,F2)。

    拥塞起源 = leaf1 出口队列(L1->s1_0,入 50 出 25);被它 PAUSE 的是
    spine 出口队列(S0->L1)——storm 记在被暂停的队列上,持续时长 =
    起源队列 xoff→xon 排空时间(4.8MB/25GB/s = 192µs > 100µs 阈值)。
    """
    topo = CLOS(n_leaf=2, n_spine=1, servers_per_leaf=2)
    sim = Simulator(topo, watchdog_threshold=100e-6)
    BulkFlow(sim, "a", "s0_0", "s1_0", 64 * MB)
    BulkFlow(sim, "b", "s0_1", "s1_0", 64 * MB)
    for f in sim.flows.values():
        f.start()
    sim.run(until=60.0)
    assert all(f.done for f in sim.flows.values())
    assert sim.queue("L1->s1_0").stats.pause_frames_sent >= 1
    assert sim.queue("S0->L1").stats.pauses_received >= 1
    assert sim.queue("S0->L1").stats.storms >= 1
    assert sim.metrics.storms >= 1
