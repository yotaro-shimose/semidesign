from pathlib import Path

import numpy as np
import pytest

from semidesign import Grid, Simulation, load_recipe, run_recipe
from semidesign.pixel import PixelAnalysis, PixelSpec
from semidesign.recipe import pixel_spec

EXAMPLE = Path(__file__).parent.parent / "examples" / "cis_photodiode_2d.toml"


@pytest.fixture(scope="module")
def example():
    recipe = load_recipe(EXAMPLE)
    sim = run_recipe(recipe, keep_snapshots=False)
    analysis = PixelAnalysis(sim, pixel_spec(recipe))
    return analysis, analysis.run()


def test_example_pixel_figures_are_physical(example):
    analysis, r = example
    assert 0.5 < r["sensor_potential_v"] < 3.0
    assert r["tg_off"]["barrier_v"] > 0.3  # holds electrons with the TG off
    assert r["tg_on"]["barrier_v"] < r["tg_off"]["barrier_v"]  # the TG opens the way
    assert r["tg_on"]["saddle_v"] > r["tg_off"]["saddle_v"]
    assert r["full_well"]["electrons_per_um"] > 50
    assert r["residual_electrons_per_um"] < 0.1 * r["full_well"]["electrons_per_um"]


def test_storage_point_and_path(example):
    analysis, r = example
    x, z = r["storage_point_nm"]
    cell = analysis._cell_at((x, z))
    assert analysis.pd[cell]
    pot, path = analysis.states["empty, TG on"]
    assert analysis.fd[analysis._cell_at(tuple(path.points_nm[-1]))]
    assert path.psi[-1] > path.psi[0]  # electrons end up higher in psi, in the FD


def test_full_well_grows_as_the_pd_fills(example):
    _, r = example
    phi, electrons = zip(*r["charge_vs_phi_pd"])
    assert np.all(np.diff(phi) > 0) and np.all(np.diff(electrons) < 0)


def test_bad_points_and_merged_regions_are_reported():
    g = Grid((400,), 600, 10, headroom_nm=50)
    sim = Simulation(g, substrate_dopant="P", substrate_conc=1e16, keep_snapshots=False)
    with pytest.raises(ValueError, match="one n-type region"):
        PixelAnalysis(sim, PixelSpec(pd_at=(100, 300), fd_at=(300, 20)))
    sim = Simulation(g, substrate_dopant="B", substrate_conc=1e16, keep_snapshots=False)
    with pytest.raises(ValueError, match="not in n-type silicon"):
        PixelAnalysis(sim, PixelSpec(pd_at=(100, 300), fd_at=(300, 20)))
