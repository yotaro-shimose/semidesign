import math

import numpy as np
import pytest

from semidesign import Wafer, physics


def test_implant_dose_conserved():
    w = Wafer(background_conc=1e15, depth_nm=1000, dx_nm=0.5, oxide_nm=0.0)
    w.implant("P", 1e14, 50)
    # The Gaussian tail above the surface (x < 0) is not in silicon.
    rp, drp = physics.get_dopant("P").projected_range(50)
    in_si = 0.5 * (1 + math.erf(rp / (math.sqrt(2) * drp)))
    assert w.sheet_dose("P") == pytest.approx(1e14 * in_si, rel=1e-3)


def test_implant_through_oxide_loses_dose():
    w = Wafer(depth_nm=1000, dx_nm=0.5, oxide_nm=50.0)
    w.implant("As", 1e15, 30)  # Rp ~23 nm, almost all stops in the oxide
    assert w.sheet_dose("As") < 1e13


def test_diffusion_conserves_dose_and_matches_gaussian_spreading():
    # A Gaussian of variance s0^2 diffusing for time t has variance s0^2 + 2Dt.
    w = Wafer(background_dopant="P", background_conc=1e14, depth_nm=3000, dx_nm=1, oxide_nm=0.0)
    w.implant("B", 1e14, 200)  # Rp ~529 nm keeps the profile away from the surface
    temp, tmin = 1000.0, 30.0
    w.anneal(temp, tmin, steps=300)
    assert w.sheet_dose("B") == pytest.approx(1e14, rel=1e-3)

    c = w.total("B")
    mean = np.trapezoid(c * w.x, w.x) / np.trapezoid(c, w.x)
    var = np.trapezoid(c * (w.x - mean) ** 2, w.x) / np.trapezoid(c, w.x)
    _, drp = physics.get_dopant("B").projected_range(200)
    d_nm2_s = physics.get_dopant("B").diffusivity(temp) * 1e14
    expected = drp**2 + 2 * d_nm2_s * tmin * 60
    assert var == pytest.approx(expected, rel=0.02)


def test_deal_grove_stepwise_equals_single_step():
    b, ba = physics.deal_grove_coefficients(1000, "dry", "100")
    one = physics.deal_grove_grow(1.0, 60, b, ba)
    many = 1.0
    for _ in range(60):
        many = physics.deal_grove_grow(many, 1, b, ba)
    assert many == pytest.approx(one, rel=1e-9)


def test_wet_faster_than_dry_and_111_faster_than_100():
    def grow(ambient, orient):
        b, ba = physics.deal_grove_coefficients(1000, ambient, orient)
        return physics.deal_grove_grow(0.0, 60, b, ba)

    assert grow("wet", "100") > 5 * grow("dry", "100")
    assert grow("dry", "111") > grow("dry", "100")


def test_oxidation_consumes_silicon():
    w = Wafer(background_conc=1e15, depth_nm=500, dx_nm=1, oxide_nm=0.0)
    w.implant("P", 1e14, 10)  # shallow, near the surface
    before = w.sheet_dose("P")
    w.oxidize(1000, 60, ambient="wet")
    assert w.oxide_nm > 300
    assert w.sheet_dose("P") < 0.5 * before


def test_junction_depth_found():
    w = Wafer(background_dopant="B", background_conc=1e16, depth_nm=600, dx_nm=0.5, oxide_nm=0.0)
    w.implant("As", 5e15, 50)
    xjs = w.junction_depths()
    assert len(xjs) == 1
    assert 50 < xjs[0] < 200
