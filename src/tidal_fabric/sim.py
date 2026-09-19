"""Simulator:把拓扑/PFC/流装进同一个确定性事件循环(学习清单 ⑫)。

决策留痕(设计文档 §三):
- 单一 EventQueue;now 只在事件弹出时推进。
- PAUSE 帧引用计数 _pause_refs:同一 (ingress 链路, 优先级) 可能被多条
  下游队列要求暂停 → 0→1 发 PAUSE 帧、1→0 才发 RESUME(防误恢复)。
- 队列懒建:首次访问 (链路, 优先级) 时按 NIC/交换机参数实例化。
"""
import random

from .events import EventQueue
from .pfc import PortQueue, LOSSLESS
from .metrics import SimMetrics


class Simulator:
    def __init__(self, topology, seed=42, watchdog_threshold=0.1):
        self.topo = topology
        self.events = EventQueue()
        self.now = 0.0
        self.rng = random.Random(seed)
        self.watchdog_threshold = watchdog_threshold
        self.queues = {}
        self._pause_refs = {}
        self.flows = {}
        self.metrics = SimMetrics()
        self._event_count = 0

    def schedule(self, time, action):
        if time < self.now:
            time = self.now
        self.events.schedule(time, action)

    def queue(self, link_name, priority=LOSSLESS):
        key = (link_name, priority)
        q = self.queues.get(key)
        if q is None:
            link = self.topo.links[link_name]
            is_nic = self.topo.nodes[link.src].kind == "server"
            q = PortQueue(self, link, priority, is_nic=is_nic)
            self.queues[key] = q
        return q

    def request_pause(self, downstream_queue, ingress_name):
        key = (ingress_name, downstream_queue.priority)
        refs = self._pause_refs.setdefault(key, set())
        if downstream_queue.link.name in refs:
            return
        refs.add(downstream_queue.link.name)
        if len(refs) == 1:
            downstream_queue.stats.pause_frames_sent += 1
            prop = self.topo.links[ingress_name].prop_delay

            def deliver_pause():
                self.queue(ingress_name, downstream_queue.priority).set_paused(True)

            self.schedule(self.now + prop, deliver_pause)

    def request_resume(self, downstream_queue, ingress_name):
        key = (ingress_name, downstream_queue.priority)
        refs = self._pause_refs.get(key)
        if not refs or downstream_queue.link.name not in refs:
            return
        refs.discard(downstream_queue.link.name)
        if not refs:
            downstream_queue.stats.resume_frames_sent += 1
            prop = self.topo.links[ingress_name].prop_delay

            def deliver_resume():
                self.queue(ingress_name, downstream_queue.priority).set_paused(False)

            self.schedule(self.now + prop, deliver_resume)

    def deliver_after(self, queue, chunk):
        link = queue.link

        def arrive():
            chunk.hop += 1
            if chunk.hop >= len(chunk.path):
                self.record_delivery(chunk)
            else:
                chunk.ingress = link.name
                self.queue(chunk.path[chunk.hop].name, chunk.priority).push(chunk)

        self.schedule(self.now + link.prop_delay, arrive)

    def record_delivery(self, chunk):
        self.metrics.delivered_chunks += 1
        flow = self.flows.get(chunk.flow_id)
        if flow is not None:
            flow.on_delivered(chunk)

    def run(self, until=None, max_events=10_000_000):
        while len(self.events):
            t = self.events.peek_time()
            if until is not None and t > until:
                break
            time, action = self.events.pop()
            self.now = time
            action()
            self._event_count += 1
            if self._event_count > max_events:
                raise RuntimeError("事件数超上限:疑似活锁或参数失当")
