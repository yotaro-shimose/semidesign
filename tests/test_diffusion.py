import numpy as np
import pytest

from conftest import depth_variance, gaussian_layer, make
from semidesign import Grid, Simulation, fv
from semidesign.dopants import get_dopant
from semidesign.materials import OXIDE


def test_variance_grows_by_exactly_2Dt_in_both_directions():
    sim = make(width=3000, depth=3000, h=10)
    sim.conc["P"] = gaussian_layer(sim, 1500, 60, 1e13, x_range=(1400, 1600))
    c0 = sim.conc["P"].copy()
    temp, tmin = 1050.0, 20.0
    sim.anneal(temp, tmin, steps=10)
    c1 = sim.conc["P"]
    two_dt = 2 * get_dopant("P").diffusivity(temp) * 1e14 * tmin * 60
    assert c1.sum() == pytest.approx(c0.sum(), rel=1e-9)
    assert depth_variance(sim, c1) - depth_variance(sim, c0) == pytest.approx(two_dt, rel=1e-3)
    x = sim.grid.lateral_centers(0)[:, None]
    var_x = lambda c: (c * (x - (c * x).sum() / c.sum()) ** 2).sum() / c.sum()
    assert var_x(c1) - var_x(c0) == pytest.approx(two_dt, rel=1e-3)


def test_oxide_filled_trench_blocks_dopant():
    sim = make(width=600, depth=1200, h=10)
    sim.etch(1000, openings=[[280, 320]])
    sim.deposit("oxide", 30)
    sim.planarize(0)
    sim.conc["P"] = gaussian_layer(sim, 100, 30, 1e14, x_range=(0, 250))
    sim.anneal(1100, 30)
    c = sim.conc["P"]
    right = c[sim.grid.lateral_index(330):, :100]
    left = c[: sim.grid.lateral_index(250), :100]
    assert right.max() < 1e-3 * left.max()


def test_static_interface_reaches_segregation_equilibrium():
    # 20 nm of oxide on boron-doped silicon, long hot anneal: C_Si / C_ox -> m
    sim = make(width=20, depth=400, h=10, headroom=100, substrate_dopant="B", substrate_conc=1e17)
    top = sim.grid.z_centers < 0
    sim.mat[:, top & (sim.grid.z_centers > -25)] = OXIDE
    sim.anneal(1100, 600, steps=20)
    c = sim.conc["B"][0]
    z = sim.grid.z_centers
    c_ox = c[(z > -20) & (z < 0)].mean()
    c_si = c[(z > 0) & (z < 20)].mean()
    assert c_si / c_ox == pytest.approx(get_dopant("B").segregation, rel=0.05)


def test_periodic_edges_wrap():
    sim = make(width=1000, depth=1000, periodic=True)
    sim.conc["P"] = gaussian_layer(sim, 100, 30, 1e14, x_range=(0, 100))
    sim.anneal(1100, 30)
    c = sim.conc["P"]
    assert c[-1].max() == pytest.approx(c[0].max(), rel=0.2)


def test_3d_uses_conjugate_gradients_and_conserves_dose(monkeypatch):
    monkeypatch.setattr(fv, "DIRECT_SOLVER_MAX_CELLS", 10)
    sim = Simulation(Grid((200, 200), 400, 20, headroom_nm=100), keep_snapshots=False)
    layer = gaussian_layer(sim, 60, 20, 1e14)
    sim.conc["P"] = layer
    sim.anneal(1050, 10, steps=5)
    assert sim.history[-1].info["solver"] == "cg"
    assert sim.conc["P"].sum() == pytest.approx(layer.sum(), rel=1e-6)


def test_face_pairs_and_region_are_consistent():
    i, j = fv.face_pairs((3, 4), periodic_axes=(0,))
    assert len(i) == 3 * 3 + 2 * 4 + 4  # vertical + horizontal + wrap
    member = np.zeros((3, 4), bool)
    member[1] = True
    r = fv.Region.of(member, i, j)
    assert r.n == 4 and r.inner_a.size == 3
    assert r.edge_cell.size == 8  # each of the 4 cells has an upper and lower neighbour
