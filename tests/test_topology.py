from tidal_fabric.topology import CLOS


def test_route_same_leaf():
    topo = CLOS()
    path = topo.route("s0_0", "s0_1", "f1")
    assert [l.name for l in path] == ["s0_0->L0", "L0->s0_1"]


def test_route_ecmp_deterministic_and_spread():
    topo = CLOS(n_spine=2)
    p1 = topo.route("s0_0", "s1_0", "k1")
    p2 = topo.route("s0_0", "s1_0", "k1")
    assert p1 == p2
    names = [l.name for l in p1]
    assert names[0] == "s0_0->L0"
    assert names[1] in ("L0->S0", "L0->S1")
    assert names[3] == "L1->s1_0"
    spines = {topo.route("s0_0", "s1_0", f"k{i}")[1].name for i in range(50)}
    assert spines == {"L0->S0", "L0->S1"}


def test_reverse_link():
    topo = CLOS()
    assert topo.reverse_link("s0_0->L0").name == "L0->s0_0"
