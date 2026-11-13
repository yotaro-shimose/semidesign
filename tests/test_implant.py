import numpy as np
import pytest
from scipy import integrate, optimize

from conftest import make
from semidesign import bca
from semidesign.implant import calibrated, depth_cdf, range_at
from semidesign.materials import GAS

SRIM = bca.read_srim_boron()


def _srim(key, energy):
    return SRIM[key][np.argmin(abs(SRIM["energy_kev"] - energy))]


@pytest.mark.parametrize("energy", [10, 30, 100, 300, 1000, 3000])
def test_boron_table_agrees_with_srim(energy):
    d = range_at("B", energy)
    assert d.rp_nm == pytest.approx(_srim("rp_nm", energy), rel=0.05)
    assert d.drp_nm == pytest.approx(_srim("drp_nm", energy), rel=0.25)
    assert d.lateral_nm == pytest.approx(_srim("lateral_nm", energy), rel=0.25)


def test_monte_carlo_reproduces_srim_boron_range():
    d = bca.range_distribution("B", 100, n_ions=800, seed=1)
    assert d.rp_nm == pytest.approx(_srim("rp_nm", 100), rel=0.06)


@pytest.mark.parametrize("eps,b", [(1.0, 0.5), (1.0, 2.0), (10.0, 0.3), (10.0, 2.0)])
def test_magic_matches_exact_scattering_integral(eps, b):
    phi = lambda r: (bca._ZBL_C * np.exp(-bca._ZBL_D * r)).sum()
    g = lambda r: 1 - phi(r) / (r * eps) - (b / r) ** 2
    r0 = optimize.brentq(g, 1e-8, 1e4)
    val, _ = integrate.quad(lambda u: b / r0 / np.sqrt(max(g(r0 / u), 1e-300)), 0, 1, limit=400)
    exact = np.sin((np.pi - 2 * val) / 2) ** 2
    assert bca.sin2_half_angle(np.array([eps]), np.array([b]))[0] == pytest.approx(exact, rel=0.1)


def test_ranges_grow_with_energy_and_shrink_with_mass():
    for ion in ("B", "P", "As", "Sb"):
        rp = [range_at(ion, e).rp_nm for e in (10, 100, 1000)]
        assert rp[0] < rp[1] < rp[2]
    at_100 = [range_at(ion, 100).rp_nm for ion in ("B", "P", "As", "Sb")]
    assert at_100 == sorted(at_100, reverse=True)


def test_deep_boron_profile_has_a_shallow_tail():
    d = range_at("B", 1000)
    assert d.skewness < -0.5
    below = 1 - depth_cdf(d, np.array([d.rp_nm + 2 * d.drp_nm]))[0]
    above = depth_cdf(d, np.array([d.rp_nm - 2 * d.drp_nm]))[0]
    assert above > 2 * below


def test_dose_is_conserved_through_a_mask():
    sim = make(width=1000, depth=1000)
    sim.implant("P", 1e13, 50, openings=[[200, 600]], damage_factor=0)
    retained = range_at("P", 50).retained
    assert sim.total_dose("P") == pytest.approx(1e13 * 0.4 * retained, rel=1e-6)


def test_mask_edge_spread_is_limited():
    sim = make(width=2000, depth=800)
    sim.implant("As", 1e15, 30, openings=[[0, 500]], damage_factor=0)
    c = sim.conc["As"]
    assert c[sim.grid.lateral_index(800)].max() < 1e-6 * c[sim.grid.lateral_index(450)].max()


def test_implant_follows_trench_bottom_and_skips_gas():
    sim = make(width=400, depth=1500)
    sim.etch(500, openings=[[0, 100]])
    sim.implant("B", 1e13, 30, openings=[[0, 100]], lateral_sigma_nm=0, damage_factor=0)
    assert np.all(sim.conc["B"][sim.mat == GAS] == 0)
    col = sim.conc["B"][sim.grid.lateral_index(50)]
    peak = sim.grid.z_centers[np.argmax(col)]
    assert peak == pytest.approx(500 + range_at("B", 30).rp_nm, abs=2 * sim.grid.spacing_nm)


def test_calibration_shifts_and_stretches_keeping_shape():
    d = range_at("P", 200)
    c = calibrated(d, rp_nm=d.rp_nm + 100, drp_nm=2 * d.drp_nm)
    for p in (0.1, 0.5, 0.9):
        z_old = np.interp(p, depth_cdf(d, np.linspace(0, 2000, 20001)), np.linspace(0, 2000, 20001))
        z_new = np.interp(p, depth_cdf(c, np.linspace(0, 3000, 30001)), np.linspace(0, 3000, 30001))
        assert z_new == pytest.approx(d.rp_nm + 100 + 2 * (z_old - d.rp_nm), abs=1.0)


def test_energy_outside_table_is_an_error():
    with pytest.raises(ValueError, match="outside the range table"):
        range_at("B", 20000)


def test_implant_records_excess_interstitials():
    sim = make(width=200, depth=800)
    sim.implant("B", 1e14, 20)
    info = sim.history[-1].info
    assert info["excess_interstitials_cm2"] == pytest.approx(info["dose_in_structure"], rel=1e-9)
