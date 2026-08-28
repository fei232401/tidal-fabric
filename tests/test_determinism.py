"""确定性重放:同场景双跑逐位一致(项目一方法论,攻防可辩护的根基)。"""
from tidal_fabric.topology import CLOS
from tidal_fabric.sim import Simulator
from tidal_fabric.flows import BulkFlow, KVFlow

MB = 1024 * 1024


def _scenario_summary():
    topo = CLOS(n_leaf=2, n_spine=1, servers_per_leaf=2, spine_rate=20e9)
    sim = Simulator(topo, seed=42)
    BulkFlow(sim, "train", "s0_0", "s1_0", 512 * MB)
    KVFlow(sim, "kv", "s0_1", "s1_1", 256 * MB)
    for f in sim.flows.values():
        f.start()
    sim.run(until=1.0)
    queues = tuple(sorted(
        (name, q.stats.max_occupancy, q.stats.pauses_received,
         q.stats.pause_frames_sent, q.stats.storms)
        for (name, _), q in sim.queues.items()
    ))
    flows = tuple(sorted(
        (fid, round(f.latency, 12), f.bytes_done)
        for fid, f in sim.flows.items()
    ))
    return sim.metrics.delivered_chunks, flows, queues


def test_deterministic_replay():
    assert _scenario_summary() == _scenario_summary()
