import math

import numpy as np
import pytest

from conftest import depth_variance, gaussian_layer, make
from semidesign.defects import cluster_dissolution_time, interstitial_transport_capacity
from semidesign.dopants import get_dopant


def test_supersaturation_matches_slab_solution():
    # damage in a thin layer at depth d, surface sink, closed bottom:
    # S - 1 = (emission per area) x d / (D_I C_I*) everywhere below the layer
    sim = make(width=20, depth=2000, h=10, headroom=50)
    z = sim.grid.z_centers
    k = np.searchsorted(z, 500)
    dose = 1e14
    layer = np.zeros(sim.grid.shape)
    layer[:, k] = dose / (sim.grid.spacing_nm * 1e-7)
    sim.defects.add_implant_damage(layer, sim.mat)
    temp, dt = 900.0, 1.0
    s = sim.defects.advance(sim.mat, temp, dt)
    emission = dose * (1 - math.exp(-dt / cluster_dissolution_time(temp))) / dt
    expected = emission * z[k] * 1e-7 / interstitial_transport_capacity(temp)
    assert s[0, k + 50] - 1 == pytest.approx(expected, rel=1e-6)
    assert s[0, k // 2] - 1 == pytest.approx(expected * z[k // 2] / z[k], rel=1e-6)


def _anneal_spread(dopant, damage, temp=900.0, minutes=10.0):
    sim = make(width=20, depth=2000, h=5, headroom=50)
    layer = gaussian_layer(sim, 300, 40, 1e14)
    sim.conc[dopant] = layer.copy()
    if damage:
        sim.defects.add_implant_damage(layer, sim.mat, factor=damage)
    v0 = depth_variance(sim, sim.conc[dopant])
    sim.anneal(temp, minutes)
    return depth_variance(sim, sim.conc[dopant]) - v0, sim


def test_damage_enhances_boron_diffusion_transiently():
    plain, _ = _anneal_spread("B", damage=0)
    ted, sim = _anneal_spread("B", damage=1)
    assert ted > 10 * plain
    info = sim.history[-1].info
    assert info["peak_supersaturation"] > 10
    assert info["damage_left_fraction"] < 1e-3  # 10 min >> tau(900 C)
    # once the damage is gone, diffusion is back to normal
    v0 = depth_variance(sim, sim.conc["B"])
    sim.anneal(900, 10)
    two_dt = 2 * get_dopant("B").diffusivity(900) * 1e14 * 600
    assert depth_variance(sim, sim.conc["B"]) - v0 == pytest.approx(two_dt, rel=0.05)


def test_moderate_supersaturation_retards_antimony():
    # Sb diffuses mostly through vacancies; with C_V / C_V* = 1 / S it slows
    # down while 1 < S < (1 - f_I) / f_I (very high S speeds it up again)
    plain, _ = _anneal_spread("Sb", damage=0, temp=1000, minutes=0.1)
    ted, sim = _anneal_spread("Sb", damage=0.01, temp=1000, minutes=0.1)
    assert 2 < sim.history[-1].info["peak_supersaturation"] < 40
    assert ted < 0.8 * plain


def test_damage_is_removed_with_the_silicon():
    sim = make(width=200, depth=500, h=10)
    sim.defects.add_implant_damage(gaussian_layer(sim, 30, 10, 1e14), sim.mat)
    sim.etch(100)
    assert sim.defects.clusters.sum() < 1e-3 * 1e14 / 1e-7
