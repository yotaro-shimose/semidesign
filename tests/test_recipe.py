from pathlib import Path

import numpy as np
import pytest

from conftest import make
from semidesign import load_recipe, run_recipe
from semidesign.materials import GAS, OXIDE
from semidesign.output import write_outputs

EXAMPLE = Path(__file__).parent.parent / "examples" / "cis_photodiode_2d.toml"


def test_example_recipe_builds_a_pinned_photodiode(tmp_path):
    recipe = load_recipe(EXAMPLE)
    sim = run_recipe(recipe)
    summary = write_outputs(sim, tmp_path, recipe["name"], plot=False)
    # p+ lid on top of the n bucket on the p substrate: two junctions
    top, bottom = summary["junction_depths_nm"]
    assert 30 < top < 300 and 1000 < bottom < 3000
    assert (tmp_path / "summary.json").exists()
    assert len(summary["files"]["snapshots"]) == len(recipe["step"]) + 1
    # the trench is lined and filled with oxide down to its bottom
    edge = sim.mat[0]
    assert (edge[(sim.grid.z_centers > 0) & (sim.grid.z_centers < 2400)] == OXIDE).all()


def test_planarize_and_etch_clear_dopant_and_damage():
    sim = make(width=200, depth=500)
    sim.implant("P", 1e14, 30)
    sim.etch(50)
    removed = sim.mat == GAS
    assert sim.conc["P"][removed].max() == 0
    assert sim.defects.clusters[removed].max() == 0


def test_bad_steps_give_clear_errors():
    base = {"grid": {"width_nm": 100, "depth_nm": 100}}
    with pytest.raises(ValueError, match="missing"):
        run_recipe(base | {"step": [{"type": "implant"}]})
    with pytest.raises(ValueError, match="nitride"):
        run_recipe(base | {"step": [{"type": "oxidize", "temp_c": 1000, "time_min": 1,
                                     "openings": [[0, 50]]}]})
    with pytest.raises(ValueError, match="unknown type"):
        run_recipe(base | {"step": [{"type": "bake"}]})
