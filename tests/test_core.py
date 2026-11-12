import math

import numpy as np
import pytest

from semidesign import Grid, Simulation, load_recipe, run_recipe
from semidesign import physics
from semidesign.physics import GAS, OXIDE, SI


def make(width=400, depth=1500, h=10, periodic=False, headroom=200, **kw):
    return Simulation(Grid((width,), depth, h, headroom_nm=headroom, periodic=periodic),
                      keep_snapshots=False, **kw)


def in_si_fraction(rp, drp):
    return 0.5 * (1 + math.erf(rp / (math.sqrt(2) * drp)))


# ------------------------------------------------------------------ implant

def test_implant_dose_conserved_with_mask():
    sim = make(width=1000)
    sim.implant("P", 1e13, 50, openings=[[200, 600]])
    rp, drp = physics.get_dopant("P").projected_range(50)
    expected = 1e13 * 0.4 * in_si_fraction(rp, drp)  # 40% of the box is open
    assert sim.total_dose("P") == pytest.approx(expected, rel=1e-6)


def test_mask_blocks_and_lateral_spread_is_limited():
    sim = make(width=2000, depth=800)
    sim.implant("As", 1e15, 30, openings=[[0, 500]])
    c = sim.conc["As"]
    near = c[sim.grid.lateral_index(450)].max()
    far = c[sim.grid.lateral_index(800)].max()
    assert far < 1e-6 * near


def test_implant_skips_gas_and_follows_trench_bottom():
    sim = make(width=400, depth=1500)
    sim.etch(500, openings=[[0, 100]])
    sim.implant("B", 1e13, 30, openings=[[0, 100]], lateral_sigma_nm=0)
    assert np.all(sim.conc["B"][sim.mat == GAS] == 0)
    col = sim.conc["B"][sim.grid.lateral_index(50)]
    peak_z = sim.grid.z_centers[np.argmax(col)]
    rp, _ = physics.get_dopant("B").projected_range(30)
    assert peak_z == pytest.approx(500 + rp, abs=sim.grid.spacing_nm)


# ---------------------------------------------------------------- diffusion

def _moments(sim, name, axis):
    c = sim.conc[name]
    g = sim.grid
    pos = g.z_centers if axis == -1 else g.lateral_centers(axis)
    shape = [1] * c.ndim
    shape[axis] = -1
    p = pos.reshape(shape)
    m = c.sum()
    mean = (c * p).sum() / m
    var = (c * (p - mean) ** 2).sum() / m
    return m, var


def test_diffusion_conserves_dose_and_spreads_exactly_2Dt_both_directions():
    sim = make(width=3000, depth=3000, h=10)
    sim.implant("P", 1e13, 200, openings=[[1400, 1600]], rp_nm=1500, drp_nm=60)  # away from edges; substrate is B
    m0, vz0 = _moments(sim, "P", -1)
    _, vx0 = _moments(sim, "P", 0)
    temp, tmin = 1050.0, 20.0
    sim.anneal(temp, tmin, steps=10)
    m1, vz1 = _moments(sim, "P", -1)
    _, vx1 = _moments(sim, "P", 0)
    d = physics.get_dopant("P").diffusivity(temp) * 1e14
    two_dt = 2 * d * tmin * 60
    assert m1 == pytest.approx(m0, rel=1e-9)
    # backward Euler on this grid grows the variance by exactly 2Dt
    assert vz1 - vz0 == pytest.approx(two_dt, rel=1e-3)
    assert vx1 - vx0 == pytest.approx(two_dt, rel=1e-3)


def test_dopant_does_not_cross_oxide_trench():
    sim = make(width=600, depth=1200, h=10)
    sim.etch(1000, openings=[[280, 320]])
    sim.deposit("oxide", 30)
    sim.planarize(0)
    assert np.all(sim.mat[sim.grid.lateral_index(300), 25:110] == OXIDE)
    sim.implant("P", 1e14, 30, openings=[[0, 250]])
    sim.anneal(1100, 30)
    right = sim.conc["P"][sim.grid.lateral_index(400):, : 100]
    left = sim.conc["P"][: sim.grid.lateral_index(200), : 100]
    assert right.max() < 1e-3 * left.max()


def test_periodic_edges_let_dopant_wrap_around():
    sim = make(width=1000, depth=1000, periodic=True)
    sim.implant("P", 1e14, 50, openings=[[0, 100]])
    sim.anneal(1100, 30)
    c = sim.conc["P"]
    assert c[-1].max() == pytest.approx(c[0].max(), rel=0.2)


def test_3d_uses_iterative_solver_and_conserves_dose():
    sim = Simulation(Grid((200, 200), 400, 20, headroom_nm=100), keep_snapshots=False)
    sim.implant("P", 1e14, 50, openings=[[60, 140, 60, 140]])
    before = sim.total_dose("P")
    import semidesign.diffusion as dif
    old = dif.DIRECT_SOLVER_MAX_CELLS
    dif.DIRECT_SOLVER_MAX_CELLS = 10
    try:
        sim.anneal(1050, 10, steps=5)
    finally:
        dif.DIRECT_SOLVER_MAX_CELLS = old
    assert sim.history[-1].info["solver"] == "cg"
    assert sim.total_dose("P") == pytest.approx(before, rel=1e-6)


# ---------------------------------------------------------- geometry / oxide

def test_conformal_deposit_closes_narrow_trench():
    sim = make(width=400, depth=1000)
    sim.etch(600, openings=[[150, 250]])
    sim.deposit("oxide", 60)
    sim.planarize(0)
    col = sim.mat[sim.grid.lateral_index(200)]
    z = sim.grid.z_centers
    assert np.all(col[(z > 0) & (z < 600)] == OXIDE)
    assert np.all(col[z < 0] == GAS)


def test_oxidation_matches_deal_grove_on_flat_wafer():
    sim = make(width=200, depth=1000, h=2, headroom=400)
    sim.oxidize(1000, 60, ambient="wet", steps=40)
    b, ba = physics.deal_grove_coefficients(1000, "wet", "100")
    expected = physics.deal_grove_grow(0.0, 60, b, ba)
    col = sim.mat[0]
    tox = (col == OXIDE).sum() * 2
    assert tox == pytest.approx(expected, abs=3 * 2)
    si_top = sim.grid.z_centers[np.argmax(col == SI)]
    assert si_top == pytest.approx(physics.SI_CONSUMED_PER_OXIDE * expected, abs=2 * 2)


def test_oxidation_reports_missing_headroom():
    sim = make(width=100, depth=500, h=5, headroom=50)
    sim.oxidize(1000, 60, ambient="wet")
    assert sim.history[-1].info["headroom_exceeded"]


def test_masked_columns_do_not_oxidize():
    sim = make(width=400, depth=500, h=5)
    sim.oxidize(1000, 30, ambient="wet", openings=[[0, 200]])
    assert (sim.mat[sim.grid.lateral_index(100)] == OXIDE).sum() > 10
    assert (sim.mat[sim.grid.lateral_index(300)] == OXIDE).sum() == 0


# ------------------------------------------------------------------ recipe

def test_example_recipe_runs(tmp_path):
    from pathlib import Path
    from semidesign.output import write_outputs
    recipe = load_recipe(Path(__file__).parent.parent / "examples" / "cis_photodiode_2d.toml")
    sim = run_recipe(recipe)
    summary = write_outputs(sim, tmp_path, recipe["name"], plot=False)
    # lid (p) on top of the bucket (n) on the substrate (p): two junctions
    assert len(summary["junction_depths_nm"]) == 2
    top, bottom = summary["junction_depths_nm"]
    assert 30 < top < 300 and 600 < bottom < 2000
    assert (tmp_path / "summary.json").exists()
    assert len(summary["files"]["snapshots"]) == len(recipe["step"]) + 1


def test_bad_step_gives_clear_error():
    with pytest.raises(ValueError, match="missing"):
        run_recipe({"grid": {"width_nm": 100, "depth_nm": 100}, "step": [{"type": "implant"}]})
