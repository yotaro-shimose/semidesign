"""Implant damage and transient enhanced diffusion (TED).

Model
1. "+1": each implanted ion that stops in silicon leaves one excess silicon
   interstitial (the rest of the damage recombines within picoseconds). These
   excess interstitials condense into {311} clusters at the implant depth.
2. During a thermal step the clusters dissolve with a time constant tau(T),
   emitting interstitials at rate G = -dQ/dt.
3. Free interstitials move so fast that their field is quasi-steady:
       D_I C_I* lap(S - 1) = -G   in silicon,
       S = 1 on every silicon face toward another material (surfaces and
       Si/SiO2 interfaces recombine interstitials),
   with S = C_I / C_I* the supersaturation. Only the product D_I C_I* enters.
4. Dopant diffusivities in silicon are multiplied by
   f_I S + (1 - f_I) / S (dopants.Dopant.point_defect_factor).

TED therefore lasts a few tau and is stronger when the damage sits deeper
(farther from the surface sink), as observed.

Parameters
- D_I C_I* / C_Si = 2980 exp(-4.95 eV / kT) cm^2/s: H. Bracht, N. A. Stolwijk,
  H. Mehrer, Phys. Rev. B 52, 16542 (1995).
These are the defaults of DefectModel, the knobs for calibration.
- {311} dissolution: activation energy 3.8 eV (D. J. Eaglesham et al.,
  Appl. Phys. Lett. 65, 2305 (1994)); time constant set to 1000 s at 800 C,
  i.e. about 30 s at 900 C and 2 s at 1000 C, matching the usual TED
  durations of roughly an hour, minutes and seconds. Real cluster lifetimes
  also depend on dose; treat tau as a calibration knob.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from . import fv
from .constants import NM2_PER_CM2, SI_ATOMS_PER_CM3, kt_ev
from .grid import Grid
from .materials import SI

NEGLIGIBLE_SUPERSATURATION = 1e-3  # S - 1 below this does not change diffusion


@dataclass(frozen=True)
class DefectModel:
    """Calibration knobs of the damage model (defaults: see the module notes)."""

    tau_800c_s: float = 1000.0  # {311} dissolution time constant at 800 C
    cluster_ea_ev: float = 3.8  # its activation energy
    di_ci_prefactor_cm2_s: float = 2980.0  # D_I C_I* / C_Si = prefactor exp(-Ea / kT)
    di_ci_ea_ev: float = 4.95

    def dissolution_time(self, temp_c: float) -> float:
        """Time constant (s) of {311} cluster dissolution."""
        return self.tau_800c_s * math.exp(self.cluster_ea_ev * (1 / kt_ev(temp_c) - 1 / kt_ev(800.0)))

    def transport_capacity(self, temp_c: float) -> float:
        """D_I C_I* in cm^-1 s^-1."""
        return self.di_ci_prefactor_cm2_s * SI_ATOMS_PER_CM3 * math.exp(-self.di_ci_ea_ev / kt_ev(temp_c))


class Defects:
    """Excess interstitials stored in clusters, and the supersaturation they drive."""

    def __init__(self, grid: Grid, model: DefectModel = DefectModel()):
        self.grid = grid
        self.model = model
        self.clusters = np.zeros(grid.shape)  # cm^-3 excess interstitials
        self.enhancing = False  # last advance() changed diffusion noticeably
        self._faces = grid.faces
        self._si_key: np.ndarray | None = None
        self._region: fv.Region | None = None
        self._solver: fv.SPDSolver | None = None

    def add_implant_damage(self, implanted: np.ndarray, mat: np.ndarray, factor: float = 1.0) -> None:
        """'+1' damage: excess interstitials equal to `factor` x the ions stopped in Si."""
        if factor > 0:
            self.clusters += factor * np.where(mat == SI, implanted, 0.0)
            self.enhancing = bool(self.clusters.any())

    def remove(self, cells: np.ndarray) -> None:
        """Drop damage in cells that are no longer silicon (etched, oxidized)."""
        self.clusters[cells] = 0.0

    def advance(self, mat: np.ndarray, temp_c: float, dt_s: float) -> np.ndarray | None:
        """Dissolve clusters for dt; return the supersaturation S, or None if S ~ 1."""
        if not self.clusters.any():
            self.enhancing = False
            return None
        before = self.clusters
        self.clusters = before * math.exp(-dt_s / self.model.dissolution_time(temp_c))
        emission = (before - self.clusters) / dt_s  # cm^-3 s^-1
        excess = self._quasi_steady_excess(mat, emission, temp_c)
        self.enhancing = bool(excess.max() > NEGLIGIBLE_SUPERSATURATION)
        return 1.0 + excess if self.enhancing else None

    def _quasi_steady_excess(self, mat, emission, temp_c) -> np.ndarray:
        """Solve -lap(s) = G / (D_I C_I*) on silicon with s = 0 at its boundaries."""
        si = mat == SI
        if self._si_key is None or not np.array_equal(self._si_key, si):
            self._build(si)
        r = self._region
        source = emission.ravel()[r.cells] * self.grid.volumes.ravel()[r.cells]
        rhs = source / (self.model.transport_capacity(temp_c) * NM2_PER_CM2)
        s = np.zeros(self.grid.n_cells)
        s[r.cells] = self._solver.solve(rhs)
        return np.maximum(s, 0.0).reshape(self.grid.shape)

    def _build(self, si: np.ndarray) -> None:
        g = self.grid.conductance(np.ones(self.grid.n_cells))  # area / distance per face
        half_i, half_j, area = self.grid.face_geometry
        r = fv.Region.of(si, *self._faces)
        sink = r.edge_sum(area[r.edge_face] / r.edge_side(half_i, half_j))  # s = 0 at the interface
        a = -r.laplacian(g[r.inner_face]) + sp.diags(sink)
        self._si_key, self._region, self._solver = si.copy(), r, fv.SPDSolver(a)
