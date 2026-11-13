"""Parameter sets and comparison with measured profiles."""

import numpy as np
import pytest

from conftest import depth_variance, gaussian_layer, make
from semidesign.compare import Comparison
from semidesign.oxidation import DealGrove, OxidationModel
from semidesign.parameters import Parameters
from semidesign.recipe import parameters


def test_parameters_round_trip_through_a_file(tmp_path):
    p = Parameters().updated({"dopant": {"P": {"d0": 7.7}}, "defects": {"tau_800c_s": 500},
                              "oxidation": {"wet": {"b_um2_h": 400}},
                              "interface_charge_cm2": {"al2o3": -1e12}})
    assert p.dopants["P"].d0 == 7.7 and p.dopants["B"].d0 == Parameters().dopants["B"].d0
    assert p.interface_charge_cm2 == {"Al2O3": -1e12}
    f = tmp_path / "fab.toml"
    f.write_text(p.to_toml())
    assert Parameters.load(f) == p


def test_parameter_errors_name_the_problem():
    with pytest.raises(ValueError, match="unknown key 'd00'"):
        Parameters().updated({"dopant": {"B": {"d00": 1}}})
    with pytest.raises(ValueError, match="describe the element"):
        Parameters().updated({"dopant": {"B": {"mass_amu": 10}}})
    with pytest.raises(ValueError, match="unknown dopant"):
        Parameters().updated({"dopant": {"Ga": {"d0": 1}}})
    with pytest.raises(ValueError, match="unknown parameter sections"):
        Parameters().updated({"implant": {}})


def test_recipe_parameters_file_then_inline(tmp_path):
    (tmp_path / "fab.toml").write_text("[dopant.P]\nd0 = 5.0\nea = 3.5\n")
    recipe = {"parameters": {"file": str(tmp_path / "fab.toml"), "dopant": {"P": {"d0": 6.0}}}}
    p = parameters(recipe)
    assert (p.dopants["P"].d0, p.dopants["P"].ea) == (6.0, 3.5)


def test_simulation_uses_its_parameters():
    def spread(params):
        sim = make(width=40, depth=1500, params=params)
        sim.conc["P"] = gaussian_layer(sim, 600, 30, 1e13)
        before = depth_variance(sim, sim.conc["P"])
        sim.anneal(1050, 10)
        return depth_variance(sim, sim.conc["P"]) - before

    base = Parameters()
    faster = base.updated({"dopant": {"P": {"d0": 4 * base.dopants["P"].d0}}})
    assert spread(faster) == pytest.approx(4 * spread(base), rel=0.03)

    slow = OxidationModel(linear_rate_100_over_111=0.3)
    assert DealGrove.at(1000, "dry", "100", slow).thickness(0, 30) < DealGrove.at(1000, "dry").thickness(0, 30)


def _measured(tmp_path, sim, scale=1.0):
    prof = sim.depth_profile(200)
    si = prof["material"] == 1
    z, c = prof["depth_nm"][si], prof["B"][si] * scale
    f = tmp_path / "sims.csv"
    f.write_text("depth (nm), B (atoms/cm3)\n" + "\n".join(f"{a:.2f},{b:.4e}" for a, b in zip(z, c)))
    return f


def test_comparison_with_a_measured_profile(tmp_path):
    sim = make(width=400, depth=1000)
    sim.implant("B", 1e14, 20)
    sim.anneal(1000, 0.2)
    same = Comparison.read(_measured(tmp_path, sim), probe=200, dopant="B", floor_cm3=1e16).evaluate(sim)
    assert same["rms_log10"] < 0.01
    assert same["dose_cm2"]["simulated"] == pytest.approx(same["dose_cm2"]["measured"], rel=0.01)
    assert same["peak"]["simulated"] == same["peak"]["measured"]
    double = Comparison.read(_measured(tmp_path, sim, 2.0), probe=200, dopant="B", floor_cm3=1e16).evaluate(sim)
    assert double["rms_log10"] == pytest.approx(np.log10(2), abs=0.02)
    assert double["typical_factor"] == pytest.approx(2, rel=0.05)


def test_comparison_depth_origin_follows_the_silicon(tmp_path):
    sim = make(width=400, depth=1000)
    sim.implant("B", 1e14, 20)
    f = _measured(tmp_path, sim)  # written on the initial-surface scale, silicon starts at 0
    sim.oxidize(1000, 20, "wet")  # consumes silicon: the silicon surface moves down
    on_silicon = Comparison.read(f, 200, "B", floor_cm3=1e16)
    z_si, _ = on_silicon.simulated(sim)
    assert z_si[0] == pytest.approx(sim.grid.widths(-1)[np.argmax(sim.mat[sim.column(200)] == 1)] / 2)
    initial = Comparison.read(f, 200, "B", surface="initial", floor_cm3=1e16)
    assert initial.simulated(sim)[0][0] > 50


def test_outputs_for_measurement(tmp_path):
    from semidesign.output import write_outputs
    from semidesign.recipe import comparisons, probes, run_recipe
    sim0 = make(width=400, depth=1000)
    sim0.implant("B", 1e14, 20)
    sims = _measured(tmp_path, sim0)
    recipe = {"grid": {"width_nm": 400, "depth_nm": 1000},
              "output": {"probes": [100, 300]},
              "compare": [{"probe": 200, "dopant": "B", "file": str(sims), "floor_cm3": 1e16}],
              "step": [{"type": "implant", "dopant": "B", "dose": 1e14, "energy_kev": 20}]}
    sim = run_recipe(recipe, keep_snapshots=False)
    out = tmp_path / "out"
    summary = write_outputs(sim, out, "t", probes(recipe), comparisons=comparisons(recipe))
    assert [p["at_nm"] for p in summary["probes"]] == [100, 300]
    assert summary["compare"][0]["rms_log10"] < 0.01
    csv = (out / "profiles" / "probe_1.csv").read_text().splitlines()
    assert csv[1] == "depth_nm,material,B,net" and len(csv) > 50
    assert Parameters.load(out / "parameters.toml") == sim.params
    assert (out / "compare_1.png").exists()
