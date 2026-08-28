"""H3 核心:让渡前静态预检——缓冲依赖图 + 环检测。

机理(学习清单 ④,决策链已拍板):
- 队列 (X, e) 被下游满队列 PAUSE;满队列又 PAUSE 自己的来路 →
  依赖边 = "活跃流路径上的相邻链路对"(li ⇒ lj:某流连续经过 li、lj)。
- 依赖图有环 + 环上队列同时充满 = PFC 死锁(Duato 信道依赖理论的应用);
  服务器收包是逃逸点(终链不续接)——所以"上行-下行"单调路径结构无环,
  环只能来自交换机级续接(中转/自适应非最短路由、重路由异常,F1)。
- 预检 = 让渡/重建前对新流-路映射建图验环,有环 → 拒绝/换路;
  O(V+E) 一次 DFS,µs 级(对比:真死锁后 watchdog 恢复要 100-400ms,F2)。
"""


def _name(link):
    return getattr(link, "name", link)


def dependency_edges(paths):
    """paths: 链路序列(可迭代) → 依赖边集合 {(li, lj)}(确定性:集合)。"""
    edges = set()
    for path in paths:
        names = [_name(l) for l in path]
        for a, b in zip(names, names[1:]):
            edges.add((a, b))
    return edges


def find_cycle(edges):
    """有向图环检测(迭代 DFS 三色标记)。

    返回环的节点序列(首尾同概念,不重复首节点)或 None。
    确定性:起点与邻居均按名排序。
    """
    adj = {}
    nodes = set()
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
        nodes.add(a)
        nodes.add(b)
    color = {n: 0 for n in nodes}   # 0 白 / 1 灰(在栈上) / 2 黑
    parent = {}
    for start in sorted(nodes):
        if color[start]:
            continue
        color[start] = 1
        stack = [(start, iter(sorted(adj.get(start, ()))))]
        while stack:
            node, it = stack[-1]
            pushed = False
            for nxt in it:
                if color[nxt] == 0:
                    color[nxt] = 1
                    parent[nxt] = node
                    stack.append((nxt, iter(sorted(adj.get(nxt, ())))))
                    pushed = True
                    break
                if color[nxt] == 1:
                    # 灰邻居 = 当前 DFS 路径上的祖先 → 回边成环
                    cyc = [node]
                    cur = node
                    while cur != nxt:
                        cur = parent[cur]
                        cyc.append(cur)
                    cyc.reverse()
                    return cyc
            if not pushed:
                color[node] = 2
                stack.pop()
    return None


def precheck_routes(paths):
    """对一组活跃流路径做静态预检。

    返回 {"ok": 无环?, "cycle": 环链路序列或 None, "n_edges": 边数}。
    """
    edges = dependency_edges(paths)
    cycle = find_cycle(edges)
    return {"ok": cycle is None, "cycle": cycle, "n_edges": len(edges)}


def precheck_rebuild(active_paths, candidate_paths):
    """让渡重建预检:活跃流(如 KV 搬运)∪ 候选新 ring 段路径 → 验环。

    active_paths: 让渡后仍存活的流路径;candidate_paths: 重建后的 ring 段路径。
    """
    return precheck_routes(list(active_paths) + list(candidate_paths))
