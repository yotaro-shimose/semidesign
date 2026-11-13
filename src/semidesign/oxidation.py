"""Thermal oxidation on a cell grid, in any direction.

The Deal-Grove model written as a field problem:

- Oxidant diffuses through the oxide that is connected to the ambient. With
  u = C / C* it satisfies lap(u) = 0, u = 1 at the oxide surface, and at each
  oxide/silicon face the reaction flux  D du/dn = k u,  k / D = 2 / A.
- Each oxidizing face advances at  v = (B/A) u_face  (oxide thickness per
  time). Bare silicon touching the ambient grows at B/A. In 1D this
  reproduces x^2 + A x = B t exactly.
- A fraction 0.44 of the new oxide thickness consumes silicon; the remaining
  volume (1.27 x the consumed silicon) must go somewhere. The oxide is treated
  as an incompressible fluid with potential flow: lap(phi) = -q with the
  volume sources q at the oxidizing faces, phi = 0 at the free surface. The
  outflow through each surface face fills the gas cell beyond it. Oxide in a
  narrow trench is therefore pushed up and out, not into the silicon.

Cells convert when their consumed (silicon) or filled (gas) fraction reaches
one; the excess carries over to the neighbouring cells the front reaches next.
Silicon nitride blocks the oxidant, so it works as an oxidation mask
(LOCOS), and the bird's beak under its edge comes out of the same equations.

Not modelled: stress, the orientation of each face (the wafer orientation is
used everywhere), and the fast initial regime of thin dry oxides.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy import ndimage

from . import fv
from .constants import kt_ev
from .grid import Grid
from .materials import GAS, OXIDE, OXIDIZABLE

SI_CONSUMED_PER_OXIDE = 0.44
EXPANSION = 1 / SI_CONSUMED_PER_OXIDE - 1  # extra oxide volume per consumed silicon volume
MAX_FRACTION_PER_STEP = 0.25


@dataclass(frozen=True)
class DealGroveConstants:
    """Arrhenius constants for <111> Si: B in um^2/h, B/A in um/h."""

    b_um2_h: float
    b_ea_ev: float
    b_over_a_um_h: float
    b_over_a_ea_ev: float


@dataclass(frozen=True)
class OxidationModel:
    """Calibration knobs of oxidation (defaults: Plummer, Deal & Griffin)."""

    dry: DealGroveConstants = DealGroveConstants(7.72e2, 1.23, 6.23e6, 2.0)
    wet: DealGroveConstants = DealGroveConstants(3.86e2, 0.78, 1.63e8, 2.05)
    linear_rate_100_over_111: float = 1 / 1.68


@dataclass(frozen=True)
class DealGrove:
    b_um2_h: float
    b_over_a_um_h: float

    @classmethod
    def at(cls, temp_c: float, ambient: str, orientation: str = "100",
           model: OxidationModel = OxidationModel()) -> "DealGrove":
        if ambient not in ("dry", "wet"):
            raise ValueError("ambient must be 'dry' or 'wet'")
        factor = {"111": 1.0, "100": model.linear_rate_100_over_111}.get(orientation)
        if factor is None:
            raise ValueError("orientation must be '100' or '111'")
        p: DealGroveConstants = getattr(model, ambient)
        kt = kt_ev(temp_c)
        return cls(p.b_um2_h * np.exp(-p.b_ea_ev / kt),
                   p.b_over_a_um_h * np.exp(-p.b_over_a_ea_ev / kt) * factor)

    @property
    def a_nm(self) -> float:
        return 1000 * self.b_um2_h / self.b_over_a_um_h

    @property
    def linear_rate_nm_s(self) -> float:
        return self.b_over_a_um_h * 1000 / 3600

    def thickness(self, start_nm, minutes: float):
        """1D oxide thickness (nm) after `minutes`, from a starting thickness."""
        a = self.b_um2_h / self.b_over_a_um_h
        x0 = np.asarray(start_nm, dtype=float) / 1000
        x1 = (-a + np.sqrt(a**2 + 4 * (x0**2 + a * x0 + self.b_um2_h * minutes / 60))) / 2
        return x1 * 1000


class OxidationFront:
    """Oxide growth state during one oxidation step."""

    def __init__(self, grid: Grid, model: DealGrove):
        self.grid = grid
        self.model = model
        self.i, self.j = grid.faces
        n = grid.n_cells
        self.consumed = np.zeros(n)  # fraction of each Si/poly cell turned into oxide
        self.filled = np.zeros(n)  # fraction of each gas cell filled by expanding oxide
        self.headroom_exceeded = False

    def rates(self, mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-cell rates (fraction of the cell per second): silicon consumed, gas filled."""
        m = mat.ravel()
        volume = self.grid.volumes.ravel()
        half_i, half_j, area = self.grid.face_geometry
        consume = np.zeros(m.size)  # consumed silicon volume per second, then / volume
        fill = np.zeros(m.size)
        oxidizable = np.isin(m, OXIDIZABLE)
        gas = m == GAS
        v0 = self.model.linear_rate_nm_s

        # bare silicon in contact with the ambient grows at the linear rate
        for a, b in ((self.i, self.j), (self.j, self.i)):
            bare = oxidizable[a] & gas[b]
            rate = SI_CONSUMED_PER_OXIDE * v0 * area[bare]
            np.add.at(consume, a[bare], rate)
            np.add.at(fill, b[bare], EXPANSION * rate)

        region = fv.Region.of(self._oxide_open_to_ambient(mat), self.i, self.j)
        if region.n > 0:
            out = m[region.edge_out]
            to_gas, to_si = out == GAS, np.isin(out, OXIDIZABLE)
            half = region.edge_side(half_i, half_j)  # oxide half cell at each edge face
            face_area = area[region.edge_face]
            lap = region.laplacian(self.grid.conductance(np.ones(m.size))[region.inner_face])
            g_surface = np.where(to_gas, face_area / half, 0.0)  # u = 1 at the oxide surface

            # oxidant: reaction resistance A/2 in series with the oxide half cell
            a_half = 0.5 * self.model.a_nm
            g_reaction = np.where(to_si, face_area / (half + a_half), 0.0)
            u = fv.SPDSolver(-lap + sp.diags(region.edge_sum(g_surface + g_reaction))).solve(
                region.edge_sum(g_surface))

            # growth of each oxide/silicon face
            u_face = u[region.edge_cell[to_si]] * a_half / (half[to_si] + a_half)
            rate = SI_CONSUMED_PER_OXIDE * v0 * u_face * face_area[to_si]
            np.add.at(consume, region.edge_out[to_si], rate)
            source = np.bincount(region.edge_cell[to_si], weights=EXPANSION * rate, minlength=region.n)

            # expansion: potential flow out through the free surface
            if source.any():
                phi = fv.SPDSolver(-lap + sp.diags(region.edge_sum(g_surface))).solve(source)
                outflow = g_surface[to_gas] * phi[region.edge_cell[to_gas]]
                np.add.at(fill, region.edge_out[to_gas], outflow)
        return consume / volume, fill / volume

    def stable_step(self, consume: np.ndarray, fill: np.ndarray) -> float:
        fastest = max(consume.max(initial=0.0), fill.max(initial=0.0))
        return MAX_FRACTION_PER_STEP / fastest if fastest > 0 else np.inf

    def advance(self, mat: np.ndarray, consume: np.ndarray, fill: np.ndarray, dt_s: float) -> np.ndarray:
        """Grow for dt (mat is updated in place). Returns the Si/poly cells that became oxide."""
        m = mat.reshape(-1)
        self.consumed += consume * dt_s
        self.filled += fill * dt_s
        converted = self._convert(m, self.consumed, np.isin(m, OXIDIZABLE))
        grown = self._convert(m, self.filled, m == GAS)
        self.headroom_exceeded |= bool(grown.reshape(mat.shape)[..., 0].any())
        return converted.reshape(mat.shape)

    def _convert(self, m: np.ndarray, progress: np.ndarray, kind: np.ndarray) -> np.ndarray:
        """Turn cells of `kind` whose progress reached 1 into oxide."""
        done = kind & (progress >= 1.0)
        if not done.any():
            return done
        volume = self.grid.volumes.ravel()
        excess = np.where(done, progress - 1.0, 0.0) * volume
        m[done] = OXIDE
        progress[done] = 0.0
        # the excess volume carries over to same-kind neighbours the front has not reached yet
        untouched = kind & ~done & (progress == 0.0)
        for a, b in ((self.i, self.j), (self.j, self.i)):
            sel = done[a] & untouched[b]
            np.maximum.at(progress, b[sel], excess[a[sel]] / volume[b[sel]])
        return done

    def _oxide_open_to_ambient(self, mat: np.ndarray) -> np.ndarray:
        """Oxide cells connected (through oxide) to a gas cell."""
        oxide = mat == OXIDE
        labels, count = ndimage.label(oxide)
        if count == 0:
            return oxide
        gas_near = ndimage.binary_dilation(mat == GAS) & oxide
        open_labels = np.unique(labels[gas_near])
        return np.isin(labels, open_labels[open_labels > 0])
