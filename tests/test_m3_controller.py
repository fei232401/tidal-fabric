"""M3 组件测试:控制器滞回决策/视图滞后/gpu_only 缩员/ring 弹性安全点。"""
from tidal_fabric.topology import CLOS
from tidal_fabric.sim import Simulator
from tidal_fabric.workloads import RingTraining
from tidal_fabric.controller import ConcedeController

MB = 1024 * 1024


class _StubFlow:
    def __init__(self, finish, latency):
        self.finish = finish
        self.latency = latency


class _StubKV:
    """控制器只依赖 .flows(finish/latency),用桩隔离决策逻辑。"""

    def __init__(self, flows):
        self.flows = flows


def test_controller_net_aware_concede_restore():
    """违约 → 连降三档到底;健康持续 → 逐档归还(滞回)。"""
    topo = CLOS(n_leaf=2, n_spine=1, servers_per_leaf=1)
    sim = Simulator(topo)
    ring = RingTraining(sim, ["s0_0", "s1_0"], 2 * MB)
    flows = [_StubFlow(t, 0.05) for t in (0.05, 0.15, 0.25, 0.35)]
    flows += [_StubFlow(t, 0.005) for t in (0.45, 0.55, 0.65, 0.75, 0.85, 0.95)]
    ctrl = ConcedeController(sim, _StubKV(flows), ring, deadline=0.02,
                             policy="net_aware", monitor_window=0.1,
                             min_samples=1, min_interval=0.0, hold_healthy=0.2)
    sim.run(until=1.05)
    kinds = [k for _, k, _ in ctrl.actions]
    assert kinds == ["concede", "concede", "concede", "restore", "restore"]
    assert ring.rate_limit == 12.5e9


def test_controller_view_lag_visibility():
    """H4 视图滞后:feedback_delay 内完成的流对决策不可见。"""
    def first_action_time(delay):
        topo = CLOS(n_leaf=2, n_spine=1, servers_per_leaf=1)
        sim = Simulator(topo)
        ring = RingTraining(sim, ["s0_0", "s1_0"], 2 * MB)
        ctrl = ConcedeController(sim, _StubKV([_StubFlow(0.5, 0.05)]), ring,
                                 deadline=0.02, policy="net_aware",
                                 monitor_window=0.1, min_samples=1,
                                 min_interval=0.0, feedback_delay=delay)
        sim.run(until=0.75)
        assert len(ctrl.actions) == 1
        return ctrl.actions[0][0]

    assert first_action_time(0.0) == 0.5
    assert first_action_time(0.2) == 0.7


def test_gpu_only_concede_restore():
    """gpu_only:违约 → 末位缩员(4→2 到下限);健康 → 归还(2→4)。"""
    topo = CLOS(n_leaf=4, n_spine=1, servers_per_leaf=1)
    sim = Simulator(topo)
    ring = RingTraining(sim, ["s0_0", "s1_0", "s2_0", "s3_0"], 2 * MB)
    flows = [_StubFlow(0.05, 0.05), _StubFlow(0.15, 0.05)]
    flows += [_StubFlow(t, 0.005) for t in (0.45, 0.55, 0.65, 0.75,
                                            0.85, 0.95, 1.05, 1.15)]
    ctrl = ConcedeController(sim, _StubKV(flows), ring, deadline=0.02,
                             policy="gpu_only", monitor_window=0.1,
                             min_samples=1, min_interval=0.0,
                             hold_healthy=0.2, rebuild_time=0.01)
    ring.start()
    sim.run(until=1.5)
    kinds = [k for _, k, _ in ctrl.actions]
    assert kinds == ["concede", "concede", "restore", "restore"]
    assert ring.n == 4
    assert ring.rebuilds == 4
    assert ring.servers == ["s0_0", "s1_0", "s2_0", "s3_0"]


def test_ring_rebuild_safe_point():
    """重建在步边界生效:停摆在步间(不进 step_times),成员/代际切换。"""
    topo = CLOS(n_leaf=4, n_spine=1, servers_per_leaf=1)
    sim = Simulator(topo)
    ring = RingTraining(sim, ["s0_0", "s1_0", "s2_0", "s3_0"], 4 * MB)
    ring.start()
    sim.run(until=0.0002)
    ring.request_rebuild(["s0_0", "s1_0", "s2_0"], rebuild_time=0.001)
    sim.run(until=0.02)
    assert ring.rebuilds == 1
    assert ring.n == 3
    assert ring.stall_total == 0.001
    assert ring.servers == ["s0_0", "s1_0", "s2_0"]
    assert all(t < 0.002 for t in ring.step_times)


def test_ring_rate_limit_applies_next_step():
    """限速变化在下一 step 生效(安全点)。"""
    topo = CLOS(n_leaf=2, n_spine=1, servers_per_leaf=2)
    sim = Simulator(topo)
    ring = RingTraining(sim, ["s0_0", "s1_0"], 16 * MB, chunk_size=256 * 1024)
    ring.start()
    sim.run(until=0.001)
    full = ring.step_times[0]
    ring.set_rate_limit(10e9)
    sim.run(until=0.02)
    capped = ring.step_times[-1]
    assert capped > full * 2
