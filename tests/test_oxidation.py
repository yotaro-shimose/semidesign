import numpy as np
import pytest

from conftest import make
from semidesign.materials import GAS, NITRIDE, OXIDE, SI
from semidesign.oxidation import SI_CONSUMED_PER_OXIDE, DealGrove


@pytest.mark.parametrize("ambient,temp,minutes", [("wet", 1000, 60), ("dry", 1000, 120)])
def test_flat_wafer_follows_deal_grove(ambient, temp, minutes):
    h = 2
    sim = make(width=20, depth=600, h=h, headroom=500)
    sim.oxidize(temp, minutes, ambient)
    expected = DealGrove.at(temp, ambient).thickness(0, minutes)
    col = sim.mat[0]
    assert (col == OXIDE).sum() * h == pytest.approx(expected, abs=2 * h)
    si_surface = sim.grid.z_centers[np.argmax(col == SI)] - h / 2
    assert si_surface == pytest.approx(SI_CONSUMED_PER_OXIDE * expected, abs=2 * h)


def test_deal_grove_restart_is_consistent():
    dg = DealGrove.at(1000, "dry")
    assert dg.thickness(dg.thickness(0, 30), 30) == pytest.approx(dg.thickness(0, 60), rel=1e-9)


def test_wet_faster_than_dry_and_111_faster_than_100():
    grow = lambda amb, o: DealGrove.at(1000, amb, o).thickness(0, 60)
    assert grow("wet", "100") > 5 * grow("dry", "100")
    assert grow("dry", "111") > grow("dry", "100")


def test_trench_sidewalls_oxidize_and_volume_leaves_through_the_opening():
    sim = make(width=400, depth=800, h=5, headroom=300)
    sim.etch(500, openings=[[150, 250]])
    sim.oxidize(1000, 10, "wet")
    k = sim.grid.z_centers.searchsorted(250)
    row = sim.mat[:, k]
    x = sim.grid.lateral_centers(0)
    wall = x[(row == SI) & (x < 200)].max()
    assert wall < 150 - 20  # the side wall was consumed sideways
    assert (row[(x > wall) & (x < 400 - (400 - 250))] == OXIDE).all()  # trench closed by oxide
    assert not sim.history[-1].info["headroom_exceeded"]


def test_nitride_masks_oxidation():
    sim = make(width=600, depth=400, h=5, headroom=300)
    sim.deposit("nitride", 50, conformal=False)
    sim.etch(100, openings=[[300, 600]], materials=["nitride"])
    sim.oxidize(1000, 30, "wet")
    assert sim.oxide_thickness(50) == 0
    assert sim.oxide_thickness(500) > 150
    assert (sim.mat == NITRIDE).sum() > 0


@pytest.mark.parametrize("dopant,piles_up", [("P", True), ("B", False)])
def test_moving_interface_segregation(dopant, piles_up):
    sim = make(width=20, depth=600, h=5, headroom=300, substrate_dopant=dopant, substrate_conc=1e17)
    sim.oxidize(1000, 30, "wet")
    col = sim.conc[dopant][0]
    si = sim.mat[0] == SI
    surface = col[si][:2].mean()
    bulk = col[si][-20:].mean()
    assert (surface > 1.2 * bulk) if piles_up else (surface < 0.9 * bulk)
    total_before = 1e17 * 600 * 1e-7
    total_after = sim.total_dose(dopant)
    assert total_after == pytest.approx(total_before, rel=1e-6)


def test_headroom_exceeded_is_reported():
    sim = make(width=20, depth=500, h=5, headroom=20)
    sim.oxidize(1000, 60, "wet")
    assert sim.history[-1].info["headroom_exceeded"]
