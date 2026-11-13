import numpy as np
import pytest

from semidesign import Grid, Simulation


def make(width=400, depth=1500, h=10, headroom=200, periodic=False, **kw):
    """A small 2D simulation without snapshots."""
    return Simulation(Grid((width,), depth, h, headroom_nm=headroom, periodic=periodic),
                      keep_snapshots=False, **kw)


def gaussian_layer(sim, depth_nm, sigma_nm, dose_cm2, x_range=None):
    """Concentration field (cm^-3) of a Gaussian layer, in silicon only."""
    g = sim.grid
    z = g.z_centers
    prof = np.exp(-0.5 * ((z - depth_nm) / sigma_nm) ** 2)
    prof *= dose_cm2 / (prof.sum() * g.spacing_nm * 1e-7)
    field = np.broadcast_to(prof, g.shape).copy()
    if x_range is not None:
        x = g.lateral_centers(0)
        field[(x < x_range[0]) | (x >= x_range[1])] = 0.0
    field[sim.mat != 1] = 0.0
    return field


def depth_variance(sim, c):
    z = sim.grid.z_centers
    w = c.sum(axis=tuple(range(c.ndim - 1)))
    mean = (w * z).sum() / w.sum()
    return (w * (z - mean) ** 2).sum() / w.sum()


@pytest.fixture
def make_sim():
    return make
