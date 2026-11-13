import math

import numpy as np
import pytest

from semidesign import Grid, Simulation
from semidesign.electrostatics import PERMITTIVITY, PoissonSolver, intrinsic_density, thermal_voltage
from semidesign.grid import DEPTH, Refinement
from semidesign.materials import OXIDE, POLY, SI


def test_pn_junction_built_in_potential():
    g = Grid((20,), 2000, 5, headroom_nm=20)
    sim = Simulation(g, substrate_dopant="B", substrate_conc=1e16, keep_snapshots=False)
    z = g.z_centers
    sim.conc["P"] = np.where((sim.mat == SI) & (z > 500), 2e16, 0.0)  # net n = 1e16 below 500 nm
    sol = PoissonSolver(g, sim.mat, sim.net_doping()).solve(0.0)
    vbi = thermal_voltage() * math.log(1e16 * 1e16 / intrinsic_density() ** 2)
    step = sol.psi[0, z > 1500].mean() - sol.psi[0, (z > 50) & (z < 200)].mean()
    assert step == pytest.approx(vbi, abs=2e-3)
    # depletion approximation: total width sqrt(2 eps Vbi (Na + Nd) / (q Na Nd))
    w = math.sqrt(2 * 11.7 * 8.854e-14 * vbi * 2e16 / (1.602e-19 * 1e32)) * 1e7
    holes, electrons = sol.p[0], sol.n[0]
    depleted = (holes < 0.5e16) & (electrons < 0.5e16) & (z > 0)
    assert depleted.sum() * 5 == pytest.approx(w, rel=0.15)


def test_gate_field_is_continuous_across_the_oxide():
    # MOS capacitor: D = eps E is continuous at the Si/SiO2 interface
    g = Grid((10,), 1000, 10, headroom_nm=40, refine=(Refinement(DEPTH, -30, 30, 1),))
    sim = Simulation(g, substrate_dopant="B", substrate_conc=1e17, keep_snapshots=False)
    z = g.z_centers
    sim.mat[:, (z < 0) & (z > -10)] = OXIDE
    sim.mat[:, (z < -10) & (z > -30)] = POLY
    # no electrons (deep depletion), so the silicon charge near the surface is smooth
    sol = PoissonSolver(g, sim.mat, sim.net_doping()).solve(1.0, phi_n=10.0)
    psi = sol.psi[0]
    ox = np.flatnonzero((z < 0) & (z > -10))
    e_ox = (psi[ox[-1]] - psi[ox[0]]) / (z[ox[-1]] - z[ox[0]])
    k = np.flatnonzero(z > 0)[0]
    e_si = (psi[k + 1] - psi[k]) / (z[k + 1] - z[k])
    assert PERMITTIVITY[OXIDE] * e_ox == pytest.approx(PERMITTIVITY[SI] * e_si, rel=0.05)
    assert psi[k] > 0.3  # the surface is depleted toward inversion at 1 V


def test_higher_gate_voltage_raises_the_surface_potential():
    g = Grid((10,), 800, 10, headroom_nm=40, refine=(Refinement(DEPTH, -30, 30, 1),))
    sim = Simulation(g, substrate_dopant="B", substrate_conc=1e17, keep_snapshots=False)
    z = g.z_centers
    sim.mat[:, (z < 0) & (z > -10)] = OXIDE
    sim.mat[:, (z < -10) & (z > -30)] = POLY
    solver = PoissonSolver(g, sim.mat, sim.net_doping())
    k = np.flatnonzero(z > 0)[0]
    surface = [solver.solve(v, phi_n=10.0).psi[0, k] for v in (-1.0, 0.0, 1.0, 2.0)]
    assert np.all(np.diff(surface) > 0)
