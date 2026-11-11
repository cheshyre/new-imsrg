"""Orbits, kets and channels against tests/golden/basis/*.txt."""

import pytest

import imsrg

CASES = ["emax2_He4", "emax2_O16", "emax4_O16", "emax2_sd-shell", "emax4_p-shell"]


def parse_golden(path):
    g = {"orbits": [], "kets": [], "channels": []}
    for line in path.read_text().splitlines():
        t = line.split()
        if not t or t[0].startswith("#"):
            continue
        if t[0] == "basis":
            g["emax"] = int(t[2])
            g["hw"] = float(t[t.index("hw") + 1])
        elif t[0] == "reference":
            g["reference"], g["valence"] = t[1], t[3]
        elif t[0] in ("norbits", "nkets", "ntwobody_channels"):
            g[t[0]] = int(t[1])
        elif t[0] == "orbit":
            g["orbits"].append(tuple(int(x) for x in t[2:6]))
        elif t[0] == "ket":
            g["kets"].append((int(t[2]), int(t[3])))
        elif t[0] == "channel":
            g["channels"].append(tuple(int(x) for x in t[2:6]))
    return g


@pytest.mark.parametrize("case", CASES)
def test_basis_matches_golden(golden, case):
    g = parse_golden(golden / "basis" / f"{case}.txt")
    ms = imsrg.ModelSpace(g["emax"], g["reference"], g["valence"], hw=g["hw"])
    assert ms.num_orbits == g["norbits"]
    assert ms.num_kets == g["nkets"]
    assert ms.num_channels == g["ntwobody_channels"]
    assert ms.emax == g["emax"]
    orbits = [(o.n, o.l, o.two_j, o.two_tz) for o in ms.orbits()]
    assert orbits == g["orbits"]
    kets = ms.kets()
    assert kets.shape == (g["nkets"], 2)
    assert [tuple(map(int, k)) for k in kets] == g["kets"]
    # golden stores the pair J and Tz (two_j / 2, two_tz / 2)
    chans = [(c.two_j // 2, c.parity, c.two_tz // 2, c.num_kets) for c in ms.channels()]
    assert chans == g["channels"]


def test_channel_lookup_and_kets(o16_emax2):
    ms = o16_emax2
    c = ms.channel(0)
    assert ms.find_channel(c.two_j, c.parity, c.two_tz).index == 0
    assert ms.channel_kets(0).shape == (c.num_kets, 2)
    assert ms.find_channel(99, 0, 0) is None
    with pytest.raises(imsrg.InputError):
        ms.channel(10_000)


def test_orbit_properties_and_equality(o16_emax2):
    o = o16_emax2.orbits()[0]
    assert o == imsrg.Orbit(o.n, o.l, o.two_j, o.two_tz)
    assert o.is_proton == (o.two_tz == -1)
    assert o.e == 2 * o.n + o.l


def test_model_space_equality(o16_emax2):
    assert o16_emax2 == o16_emax2
    assert o16_emax2 != imsrg.ModelSpace(2, "O16", "O16", hw=16.0)
