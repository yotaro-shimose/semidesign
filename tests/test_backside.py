"""Back-illuminated flow: thinning, back-side steps, starting profile, interface charge."""

from pathlib import Path

import numpy as np
import pytest

from conftest import make
from semidesign import Grid, Simulation, load_recipe, run_recipe
from semidesign.electrostatics import CHARGE_FACTOR, PoissonSolver
from semidesign.grid import DEPTH, Refinement
from semidesign.materials import AL2O3, GAS, OXIDE, SI
from semidesign.parameters import Parameters
from semidesign.pixel import PixelAnalysis
from semidesign.recipe import pixel_spec

BSI = Path(__file__).parent.parent / "examples" / "cis_bsi_2d.toml"


def test_flipped_grid_is_the_same_grid_upside_down():
    g = Grid((300,), 1000, 10, refine=(Refinement(DEPTH, 900, 950, 2),))
    f = g.flipped()
    assert np.allclose(f.z_edges, -g.z_edges[::-1])
    assert np.allclose(f.volumes, g.volumes[:, ::-1])
    assert np.allclose(f.flipped().z_edges, g.z_edges)
    assert f.shape == g.shape and np.allclose(f.edges[0], g.edges[0])


def test_thin_then_back_side_steps():
    sim = make(width=300, depth=1200)
    sim.deposit("oxide", 50, conformal=False)  # front film
    front_gas = sim.mat == GAS
    sim.thin(1000)
    z = sim.grid.z_centers
    assert not (sim.mat[:, z > 1000] != GAS).any() and (sim.mat[:, (z > 0) & (z < 1000)] == SI).all()
    # a trench from the back, 600 nm up from the new back surface
    sim.etch(600, openings=[[0, 50]], side="back")
    col = sim.mat[sim.column(20)]
    assert (col[(z > 420) & (z < 1000)] == GAS).all() and (col[(z > 0) & (z < 380)] == SI).all()
    # films from the back line the back and the trench, never the front
    sim.deposit("Al2O3", 10, side="back")
    sim.deposit("oxide", 60, side="back")
    assert (sim.mat[front_gas & (z < 0)] == GAS).all()
    assert (sim.mat[sim.column(200)][(z > 1000) & (z < 1008)] == AL2O3).all()
    assert (sim.mat[sim.column(20)][(z > 500) & (z < 990)] != GAS).all()  # trench filled
    layers = [l["material"] for l in sim.layers(200)]
    assert layers == ["SiO2", "Si", "Al2O3", "SiO2"]


def test_sealed_voids_are_not_filled():
    sim = make(width=200, depth=500)
    sim.mat[:, :] = np.where(sim.grid.z_centers < -40, GAS, sim.mat)
    z = sim.grid.z_centers
    sim.mat[:, (z < 0) & (z > -40)] = OXIDE
    sim.mat[5:10, (z < -10) & (z > -30)] = GAS  # a void inside the oxide
    sim.deposit("nitride", 30)
    assert (sim.mat[5:10, (z < -10) & (z > -30)] == GAS).all()


def test_thin_needs_room_in_the_box():
    sim = make(width=100, depth=500)
    with pytest.raises(ValueError, match="depth_nm"):
        sim.thin(600)
    with pytest.raises(ValueError, match="side"):
        sim.etch(10, side="top")


def test_starting_profile_is_interpolated_on_a_log_scale():
    g = Grid((20,), 3000, 10)
    sim = Simulation(g, substrate_profile=[(0, 1e15), (2000, 1e15), (3000, 1e17)], keep_snapshots=False)
    c = sim.conc["B"][0]
    z = g.z_centers
    at = lambda d: c[np.argmin(abs(z - d))]
    assert at(1000) == pytest.approx(1e15)
    assert at(2500) == pytest.approx(1e16, rel=0.05)
    assert at(2995) == pytest.approx(1e17, rel=0.05)
    assert (c[z < 0] == 0).all()


def test_interface_charge_is_balanced_by_the_silicon():
    # Si below a thin Al2O3 film with gas above: every boundary is closed, so the
    # silicon must hold exactly the opposite of the sheet charge
    g = Grid((10,), 1500, 10, headroom_nm=40, refine=(Refinement(DEPTH, -20, 50, 1),))
    sim = Simulation(g, substrate_dopant="B", substrate_conc=1e15, keep_snapshots=False)
    z = g.z_centers
    sim.mat[:, (z < 0) & (z > -10)] = AL2O3
    q = -2e12
    with_charge = PoissonSolver(g, sim.mat, sim.net_doping(), interface_charge_cm2={"Al2O3": q})
    sol = with_charge.solve(0.0)
    si = sim.mat[0] == SI
    rho = (sol.p - sol.n + sim.net_doping())[0][si]
    held = (rho * g.widths(-1)[si]).sum() * 1e-7  # cm^-2
    assert held == pytest.approx(-q, rel=1e-3)
    # the negative film gathers holes at the surface (accumulation)
    plain = PoissonSolver(g, sim.mat, sim.net_doping()).solve(0.0)
    k = np.flatnonzero(si)[0]
    assert sol.p[0, k] > 100 * plain.p[0, k]
    with pytest.raises(ValueError, match="insulator"):
        PoissonSolver(g, sim.mat, sim.net_doping(), interface_charge_cm2={"Si": 1e12})


@pytest.fixture(scope="module")
def bsi():
    recipe = load_recipe(BSI)
    sim = run_recipe(recipe, keep_snapshots=False)
    analysis = PixelAnalysis(sim, pixel_spec(recipe))
    return sim, analysis, analysis.run()


def test_bsi_example_structure(bsi):
    sim, _, _ = bsi
    assert [l["material"] for l in sim.layers(450)] == ["SiO2", "Si", "Al2O3", "SiO2"]
    si = [l for l in sim.layers(450) if l["material"] == "Si"][0]
    assert si["thickness_nm"] == pytest.approx(3000, abs=5)
    # the back trench reaches 400 nm below the front surface and is lined with the charged film
    trench = sim.mat[sim.column(20)]
    z = sim.grid.z_centers
    assert not (trench[(z > 450) & (z < 3000)] == SI).any()
    assert (trench == AL2O3).any()


def test_bsi_example_pixel(bsi):
    sim, analysis, r = bsi
    assert r["full_well"]["electrons_per_um"] > 100
    assert r["tg_on"]["barrier_v"] < r["tg_off"]["barrier_v"]
    col = r["collection_from_bottom"]
    assert col["barrier_v"] < 0.05  # electrons made at the back surface reach the PD
    assert col["silicon_bottom_nm"] == pytest.approx(3000, abs=5)
    assert 1000 < col["pd_bottom_nm"] < 3000
    # the charged film holds the back surface with holes; without it, far fewer
    assert col["bottom_surface_holes_cm3_min"] > 1e17
    no_film = PoissonSolver(sim.grid, sim.mat, sim.net_doping())
    pot = no_film.solve(pixel_spec(load_recipe(BSI)).tg_off_v, analysis.phi_n(20.0))
    back = analysis.collection(pot, analysis.storage_cell(pot))[0]
    assert back["bottom_surface_holes_cm3_min"] < 0.1 * col["bottom_surface_holes_cm3_min"]
