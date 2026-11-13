import math

import numpy as np
import pytest

from conftest import depth_variance, gaussian_layer
from semidesign import Grid, Simulation
from semidesign.defects import DefectModel
from semidesign.dopants import get_dopant
from semidesign.grid import DEPTH, Refinement, graded_edges
from semidesign.materials import OXIDE, SI
from semidesign.oxidation import SI_CONSUMED_PER_OXIDE, DealGrove


def test_graded_edges_refine_and_grow_gently():
    zone = Refinement(DEPTH, -10, 30, 1.0)
    e = graded_edges(-200, 3000, 20, [zone], growth=1.3)
    w = np.diff(e)
    assert e[0] == -200 and e[-1] == 3000
    inside = (e[:-1] >= -10) & (e[1:] <= 30)
    assert w[inside].max() <= 1.0 + 1e-9
    assert w.max() <= 20 + 1e-9
    ratio = w[1:] / w[:-1]
    assert np.all((ratio < 1.3 + 1e-6) & (ratio > 1 / 1.3 - 1e-6))


def graded(width=400, depth=1500, base=20, headroom=200, refine=(), **kw):
    grid = Grid((width,), depth, base, headroom_nm=headroom, refine=refine)
    return Simulation(grid, keep_snapshots=False, **kw)


def test_diffusion_on_graded_grid_spreads_by_2Dt():
    sim = graded(width=3000, depth=3000, base=20, refine=(
        Refinement(DEPTH, 1300, 1700, 4), Refinement(0, 1300, 1700, 4)))
    assert not sim.grid.is_uniform
    layer = gaussian_layer(sim, 1500, 60, 1e13, x_range=(1400, 1600))
    sim.conc["P"] = layer
    total0 = (layer * sim.grid.volumes).sum()
    v0 = depth_variance(sim, layer)
    sim.anneal(1050, 20, steps=10)
    two_dt = 2 * get_dopant("P").diffusivity(1050) * 1e14 * 1200
    assert (sim.conc["P"] * sim.grid.volumes).sum() == pytest.approx(total0, rel=1e-9)
    assert depth_variance(sim, sim.conc["P"]) - v0 == pytest.approx(two_dt, rel=0.02)


def test_implant_dose_is_the_same_on_graded_and_uniform_grids():
    doses = []
    for refine in ((), (Refinement(DEPTH, -20, 200, 2), Refinement(0, 150, 250, 2))):
        sim = graded(width=600, depth=1000, base=10, refine=refine)
        sim.implant("B", 1e14, 20, openings=[[100, 300]], damage_factor=0)
        doses.append(sim.total_dose("B"))
    assert doses[1] == pytest.approx(doses[0], rel=1e-3)


def test_oxidation_on_graded_grid_follows_deal_grove():
    sim = graded(width=20, depth=600, base=20, headroom=600,
                 refine=(Refinement(DEPTH, -450, 200, 2),))
    sim.oxidize(1000, 60, "wet")
    expected = DealGrove.at(1000, "wet").thickness(0, 60)
    assert sim.oxide_thickness() == pytest.approx(expected, abs=4)
    col = sim.mat[0]
    si_surface = sim.grid.z_edges[np.argmax(col == SI)]
    assert si_surface == pytest.approx(SI_CONSUMED_PER_OXIDE * expected, abs=4)


def test_supersaturation_on_graded_grid_matches_slab_solution():
    sim = graded(width=20, depth=2000, base=20, headroom=50,
                 refine=(Refinement(DEPTH, 450, 550, 2),))
    z, h = sim.grid.z_centers, sim.grid.widths(-1)
    k = np.searchsorted(z, 500)
    layer = np.zeros(sim.grid.shape)
    layer[:, k] = 1e14 / (h[k] * 1e-7)
    sim.defects.add_implant_damage(layer, sim.mat)
    s = sim.defects.advance(sim.mat, 900.0, 1.0)
    emission = 1e14 * (1 - math.exp(-1.0 / DefectModel().dissolution_time(900.0)))
    expected = emission * z[k] * 1e-7 / DefectModel().transport_capacity(900.0)
    assert s[0, -1] - 1 == pytest.approx(expected, rel=1e-6)


def test_graded_grid_reproduces_uniform_junction_depth():
    depths = []
    for refine in ((), (Refinement(DEPTH, -20, 300, 2),)):
        sim = graded(width=200, depth=1500, base=10, refine=refine)
        sim.implant("P", 5e13, 100, damage_factor=0)
        sim.anneal(1000, 10)
        depths.append(sim.junctions_along_depth()[0])
    assert depths[1] == pytest.approx(depths[0], abs=5)
