from tidal_fabric.events import EventQueue


def test_same_time_events_fifo():
    """同刻事件按入队顺序执行(确定性基石)。"""
    q = EventQueue()
    out = []
    q.schedule(1.0, lambda: out.append("a"))
    q.schedule(1.0, lambda: out.append("b"))
    q.schedule(0.5, lambda: out.append("c"))
    while len(q):
        _, action = q.pop()
        action()
    assert out == ["c", "a", "b"]


def test_time_ordering():
    q = EventQueue()
    out = []
    for t in (3, 1, 2):
        q.schedule(float(t), lambda t=t: out.append(t))
    while len(q):
        _, action = q.pop()
        action()
    assert out == [1, 2, 3]
