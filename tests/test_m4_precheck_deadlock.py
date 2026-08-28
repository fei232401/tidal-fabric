"""M4 测试:静态预检(依赖图/环检测)+ PFC 死锁复现与破解(H3 核心)。

死锁构造(设计文档 05,决策留痕):4 leaf × 2 spine × 2 server,七条流——
两条直连 + 两条 detour(经中转 leaf"下再上")闭合成 4-环:
  L0→S0 ⇒ S0→L1 ⇒ L1→S1 ⇒ S1→L0 ⇒ L0→S0
三条补流制造**全链过载**(每环节入>出,队列永久顶在 xoff 之上——
第一版构造失败教训:入不敷出的队列在暂停传播间隙就排空,级联自解)。
过载 + 环 = 引用互粘 → 四队列互相 PAUSE 且全满 → 无逃逸点 → 永久冻结。
"""
from tidal_fabric.topology import CLOS
from tidal_fabric.sim import Simulator
from tidal_fabric.flows import BulkFlow
from tidal_fabric.precheck import find_cycle, precheck_routes
from tidal_fabric.workloads import ring_segment_paths

MB = 1024 * 1024

# (fid, src, dst, path) —— 显式路径的七条流
_FLOW_SPECS = [
    ("f1", "s0_0", "s1_0", ["s0_0->L0", "L0->S0", "S0->L1", "L1->s1_0"]),
    ("f2", "s0_1", "s2_0", ["s0_1->L0", "L0->S0", "S0->L1", "L1->S1",
                            "S1->L2", "L2->s2_0"]),       # detour via L1
    ("f3", "s1_0", "s0_0", ["s1_0->L1", "L1->S1", "S1->L0", "L0->s0_0"]),
    ("f4", "s1_1", "s2_0", ["s1_1->L1", "L1->S1", "S1->L0", "L0->S0",
                            "S0->L2", "L2->s2_0"]),       # detour via L0
    ("f5", "s3_0", "s1_0", ["s3_0->L3", "L3->S1", "S1->L1", "L1->s1_0"]),
    ("f6", "s3_1", "s1_0", ["s3_1->L3", "L3->S0", "S0->L1", "L1->s1_0"]),
    ("f7", "s2_1", "s0_0", ["s2_1->L2", "L2->S1", "S1->L0", "L0->s0_0"]),
]

# 破解版:f2 改经 L3 中转(切断 S0→L1 ⇒ L1→S1 续接边)→ 依赖图无环
_BROKEN_F2 = ["s0_1->L0", "L0->S0", "S0->L3", "L3->S1", "S1->L2", "L2->s2_0"]

CYCLE_Q = ["L0->S0", "S0->L1", "L1->S1", "S1->L0"]


def _paths(topo, broken=False):
    out = []
    for fid, src, dst, names in _FLOW_SPECS:
        if broken and fid == "f2":
            names = _BROKEN_F2
        out.append(tuple(topo.links[n] for n in names))
    return out


def test_find_cycle_basics():
    assert find_cycle({("a", "b"), ("b", "c"), ("c", "a")}) is not None
    assert find_cycle({("a", "b"), ("b", "c")}) is None
    assert find_cycle(set()) is None
    cyc = find_cycle({("a", "b"), ("b", "c"), ("c", "d"), ("d", "b")})
    assert set(cyc) == {"b", "c", "d"}


def test_precheck_detects_canonical_cycle():
    topo = CLOS(n_leaf=4, n_spine=2, servers_per_leaf=2)
    r = precheck_routes(_paths(topo))
    assert not r["ok"]
    assert set(r["cycle"]) == set(CYCLE_Q)


def test_precheck_accepts_broken_mapping():
    topo = CLOS(n_leaf=4, n_spine=2, servers_per_leaf=2)
    r = precheck_routes(_paths(topo, broken=True))
    assert r["ok"] and r["cycle"] is None


def test_minimal_routing_structurally_acyclic():
    """健康最小 ECMP(上行-下行)结构无环:随机重建怎么重哈希都不成环。"""
    import random
    topo = CLOS(n_leaf=4, n_spine=2, servers_per_leaf=2)
    rng = random.Random(7)
    servers = [f"s{i}_0" for i in range(4)]
    route_fn = lambda s, d, k: topo.route(s, d, k)
    for trial in range(50):
        k = rng.randint(2, 4)
        members = rng.sample(servers, k)
        paths = [p for _, p in ring_segment_paths(members, route_fn, gen=trial)]
        assert precheck_routes(paths)["ok"], f"trial {trial} 意外出环"


def test_deadlock_freezes_and_breaks():
    """引擎级死锁:含环+过载 → 四队列互锁永久冻结;无环 → 全部送达。"""

    def scenario(broken):
        topo = CLOS(n_leaf=4, n_spine=2, servers_per_leaf=2, spine_rate=40e9)
        sim = Simulator(topo)
        for (fid, src, dst, names), path in zip(_FLOW_SPECS, _paths(topo, broken)):
            if broken and fid == "f2":
                continue
            BulkFlow(sim, fid, src, dst, 128 * MB, chunk_size=1 * MB, path=path)
        if broken:
            f2_names = _BROKEN_F2
            path = tuple(topo.links[n] for n in f2_names)
            BulkFlow(sim, "f2", "s0_1", "s2_0", 128 * MB,
                     chunk_size=1 * MB, path=path)
        for f in sim.flows.values():
            f.start()
        sim.run(until=1.0)
        return sim

    # 含环:冻结(四队列互 PAUSE 且非空,流不完成,watchdog 记 storm)
    sim_bad = scenario(False)
    assert not all(f.done for f in sim_bad.flows.values())
    for q in CYCLE_Q:
        queue = sim_bad.queue(q)
        assert queue.paused and queue.occupancy > 0, f"{q} 未冻结"
    assert sim_bad.metrics.storms >= 4

    # 无环(换一条 detour):全部送达,无冻结
    sim_good = scenario(True)
    assert all(f.done for f in sim_good.flows.values())
    assert not any(q.paused and q.occupancy > 0
                   for q in sim_good.queues.values())
