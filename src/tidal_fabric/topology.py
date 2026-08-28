"""CLOS 拓扑与 ECMP 路由(学习清单 ⑤⑥ 的代码对应物)。

决策留痕(设计文档 §三/§四):
- 默认 4 leaf × 2 spine × 每 leaf 2 server:PFC 死锁环需 ≥3 台交换机的
  循环缓冲依赖(F1 三交换机三角环),4 leaf 给成环留足冗余。
- ECMP = crc32(flow_key) % spine 数:同流同路(不乱序),跨流摊开;
  真实 ECMP 哈希五元组,此处单键简化(影子复现可升级)。
- 链路双向独立建模;速率为数据中心典型量级(200G 接入/400G 上联,估算待扫)。
"""
from dataclasses import dataclass
import zlib

SERVER_RATE = 25e9   # 200GbE
SPINE_RATE = 50e9    # 400GbE
PROP_DELAY = 20e-6   # 跨架传播延迟量级


@dataclass(frozen=True)
class Node:
    name: str
    kind: str  # 'server' | 'leaf' | 'spine'


@dataclass(frozen=True)
class Link:
    name: str         # 'src->dst'
    src: str
    dst: str
    rate: float       # bytes/s
    prop_delay: float # 秒


class CLOS:
    def __init__(self, n_leaf=4, n_spine=2, servers_per_leaf=2,
                 server_rate=SERVER_RATE, spine_rate=SPINE_RATE,
                 prop_delay=PROP_DELAY):
        if n_leaf < 1 or n_spine < 1 or servers_per_leaf < 1:
            raise ValueError("n_leaf/n_spine/servers_per_leaf 必须 ≥1")
        self.n_leaf, self.n_spine = n_leaf, n_spine
        self.servers_per_leaf = servers_per_leaf
        self.nodes = {}
        self.links = {}
        self._leaf_of = {}
        for i in range(n_leaf):
            leaf = f"L{i}"
            self.nodes[leaf] = Node(leaf, "leaf")
            for j in range(servers_per_leaf):
                srv = f"s{i}_{j}"
                self.nodes[srv] = Node(srv, "server")
                self._leaf_of[srv] = leaf
                self._add_pair(srv, leaf, server_rate, prop_delay)
        for i in range(n_spine):
            spine = f"S{i}"
            self.nodes[spine] = Node(spine, "spine")
        for i in range(n_leaf):
            for j in range(n_spine):
                self._add_pair(f"L{i}", f"S{j}", spine_rate, prop_delay)

    def _add_pair(self, a, b, rate, prop_delay):
        self.links[f"{a}->{b}"] = Link(f"{a}->{b}", a, b, rate, prop_delay)
        self.links[f"{b}->{a}"] = Link(f"{b}->{a}", b, a, rate, prop_delay)

    def route(self, src, dst, flow_key):
        """server→server 的有向链路序列;跨 leaf 时按 flow_key 选 spine。"""
        if src not in self._leaf_of or dst not in self._leaf_of:
            raise ValueError(f"route 仅支持 server↔server: {src}->{dst}")
        if src == dst:
            raise ValueError("src 与 dst 不能相同")
        la, lb = self._leaf_of[src], self._leaf_of[dst]
        if la == lb:
            return (self.links[f"{src}->{la}"], self.links[f"{la}->{dst}"])
        spine = f"S{zlib.crc32(flow_key.encode()) % self.n_spine}"
        return (self.links[f"{src}->{la}"], self.links[f"{la}->{spine}"],
                self.links[f"{spine}->{lb}"], self.links[f"{lb}->{dst}"])

    def reverse_link(self, link_name):
        a, b = link_name.split("->")
        return self.links[f"{b}->{a}"]

    def route_detour(self, src, dst, key, via_leaf, spine_a=None, spine_b=None):
        """非最短(自适应/Valiant 式)路由:src→leaf→SA→via_leaf→SB→leaf→dst。

        真实依据:AI/HPC fabric 的自适应非最短路由(Cray Slingshot、
        NVIDIA Spectrum-X、Valiant LB);F1 记载"重路由可造出环形路径"。
        关键性质:流量在 via_leaf "下再上"——产生交换机级续接边
        (S→Lvia ⇒ Lvia→S'),这是缓冲依赖环的原料(单调路由没有)。
        """
        if src not in self._leaf_of or dst not in self._leaf_of:
            raise ValueError(f"route 仅支持 server↔server: {src}->{dst}")
        la, lb, lv = self._leaf_of[src], self._leaf_of[dst], via_leaf
        if la == lb:
            raise ValueError("同 leaf 直连即可,无需 detour")
        if lv in (la, lb):
            raise ValueError("via_leaf 不能是源/目的 leaf")
        sa = spine_a or f"S{zlib.crc32((key + ':a').encode()) % self.n_spine}"
        sb = spine_b or f"S{zlib.crc32((key + ':b').encode()) % self.n_spine}"
        return (self.links[f"{src}->{la}"], self.links[f"{la}->{sa}"],
                self.links[f"{sa}->{lv}"], self.links[f"{lv}->{sb}"],
                self.links[f"{sb}->{lb}"], self.links[f"{lb}->{dst}"])
