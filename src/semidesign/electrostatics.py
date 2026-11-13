"""Electrostatic potential of a finished structure.

Poisson's equation on the process grid, finite volumes:

    -div(eps grad psi) = q (p - n + N_D - N_A)   in silicon
    -div(eps grad psi) = 0                       in oxide and nitride

psi is the electrostatic potential (V), measured so that holes in equilibrium
with the grounded p-type substrate have quasi-Fermi potential 0. Carriers are
in quasi-equilibrium (Boltzmann statistics):

    n = n_i exp((psi - phi_n) / V_T),   p = n_i exp((0 - psi) / V_T)

The electron quasi-Fermi potential phi_n is given per cell by the caller:
0 where electrons are in equilibrium with the substrate, the applied voltage
in a region tied to a contact (a floating diffusion), any value for a
floating region (a photodiode holding a given charge). Electrons see psi
upside down: they collect where psi is highest.

Poly-Si cells are gate electrodes held at V_gate + 0.56 V (n+ poly: Fermi
level at the conduction band edge, half a band gap above the intrinsic
level). Gas cells are left out (no field crosses the free surface). Outer
edges are closed; lateral edges optionally periodic.

Fixed charge at an interface between silicon and an insulator (the negative
charge of an Al2O3 or HfO2 film on the back surface, the positive fixed
charge of thermal oxide) is a sheet charge on the faces between them, given
per insulator material in cm^-2 (electrons' charge = -1). Half of it is put in
each of the two cells of a face, so the jump in eps grad psi across the
interface equals the sheet charge.

Newton's method, each step capped at 1 V and shortened until the residual drops.
Not modelled: Fermi-Dirac statistics and band-gap narrowing at very high
doping, incomplete ionization, interface traps (charge that depends on bias).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from . import fv
from .constants import NM_PER_CM
from .grid import Grid
from .materials import AL2O3, COUNT, GAS, HFO2, NITRIDE, OXIDE, POLY, SI, material_code

Q_OVER_EPS0 = 1.602176634e-19 / 8.8541878128e-12  # V m
CHARGE_FACTOR = Q_OVER_EPS0 * 1e-12  # charge term with cm^-3, nm and relative permittivity
GATE_OFFSET_V = 0.56  # n+ poly gate: psi = V_gate + E_g / 2
CONDUCTOR = 1e6  # "permittivity" of an electrode: its half cell adds no resistance
MAX_STEP_V = 1.0  # largest potential change per Newton step


def thermal_voltage(temp_k: float = 300.0) -> float:
    return 8.617333262e-5 * temp_k


def intrinsic_density(temp_k: float = 300.0) -> float:
    """n_i in cm^-3 (9.65e9 at 300 K, Altermatt et al. 2003), scaled with T^1.5 exp(-Eg/2kT)."""
    kt = thermal_voltage(temp_k)
    return 9.65e9 * (temp_k / 300.0) ** 1.5 * math.exp(-1.12 / (2 * kt) + 1.12 / (2 * thermal_voltage(300.0)))


PERMITTIVITY = np.zeros(COUNT)
PERMITTIVITY[SI], PERMITTIVITY[OXIDE], PERMITTIVITY[NITRIDE] = 11.7, 3.9, 7.5
PERMITTIVITY[AL2O3], PERMITTIVITY[HFO2] = 9.0, 22.0
PERMITTIVITY[POLY] = CONDUCTOR


@dataclass
class Potential:
    """Solution: psi (V) and carrier densities (cm^-3), grid-shaped; NaN in gas."""

    psi: np.ndarray
    n: np.ndarray
    p: np.ndarray
    iterations: int


class PoissonSolver:
    """Electrostatics of one structure; solve() for different biases and phi_n."""

    def __init__(self, grid: Grid, mat: np.ndarray, net_doping: np.ndarray, temp_k: float = 300.0,
                 interface_charge_cm2: Mapping[str, float] | None = None):
        self.grid = grid
        self.mat = mat.ravel()
        self.doping = np.where(self.mat == SI, net_doping.ravel(), 0.0)
        self.vt = thermal_voltage(temp_k)
        self.ni = intrinsic_density(temp_k)
        self.si = self.mat == SI
        self.gate = self.mat == POLY
        self.region = fv.Region.of((self.mat != GAS) & ~self.gate, *grid.faces)
        g = grid.conductance(PERMITTIVITY[self.mat])
        r = self.region
        self._lap = r.laplacian(g[r.inner_face])
        to_gate = self.gate[r.edge_out]
        self._g_gate = np.where(to_gate, g[r.edge_face], 0.0)
        self._volume = grid.volumes.ravel()[r.cells]
        self._si_local = self.si[r.cells]
        self._fixed = CHARGE_FACTOR * self._sheet_charge(interface_charge_cm2 or {})[r.cells]

    def _sheet_charge(self, per_material: Mapping[str, float]) -> np.ndarray:
        """Interface charges as cell charges (cm^-3 x cell size), half on each side of a face."""
        q = np.zeros(self.mat.size)
        i, j = self.grid.faces
        _, _, area = self.grid.face_geometry
        for name, density in per_material.items():
            code = material_code(name)
            if code in (GAS, SI, POLY):
                raise ValueError(f"interface charge needs an insulator next to silicon, not {name!r}")
            face = ((self.mat[i] == SI) & (self.mat[j] == code)) | ((self.mat[j] == SI) & (self.mat[i] == code))
            half = 0.5 * density * NM_PER_CM * area[face]  # cm^-2 over the face = cm^-3 x nm^ndim
            np.add.at(q, i[face], half)
            np.add.at(q, j[face], half)
        return q

    def solve(self, gate_v: float, phi_n: np.ndarray | float = 0.0,
              initial: Potential | None = None, tol_v: float = 1e-7, max_iter: int = 200) -> Potential:
        r, vt, ni = self.region, self.vt, self.ni
        phi = np.broadcast_to(np.asarray(phi_n, dtype=float), self.grid.shape).ravel()[r.cells]
        dop = self.doping[r.cells]
        psi_gate = gate_v + GATE_OFFSET_V
        if initial is None:
            psi = self._neutral_guess(dop, phi)
        else:
            psi = initial.psi.ravel()[r.cells].copy()
        diag_gate = r.edge_sum(self._g_gate)
        lap = self._lap - sp.diags(diag_gate)  # electrode faces tie cells to psi_gate
        gate_term = diag_gate * psi_gate
        si = self._si_local
        weight = CHARGE_FACTOR * self._volume

        def carriers(x):
            n = np.where(si, ni * np.exp(np.minimum((x - phi) / vt, 700.0)), 0.0)
            p = np.where(si, ni * np.exp(np.minimum(-x / vt, 700.0)), 0.0)
            return n, p

        def residual(x):
            n, p = carriers(x)
            return lap @ x + gate_term + weight * (p - n + dop) + self._fixed

        f = residual(psi)
        for it in range(1, max_iter + 1):
            n, p = carriers(psi)
            jac = -lap + sp.diags(weight * (n + p) / vt)  # minus the Jacobian: SPD
            delta = fv.SPDSolver(jac.tocsc()).solve(f)
            delta *= min(1.0, MAX_STEP_V / np.abs(delta).max())
            norm = np.linalg.norm(f)
            alpha = 1.0
            for _ in range(30):  # backtracking line search on the residual
                trial = psi + alpha * delta
                f_trial = residual(trial)
                if np.linalg.norm(f_trial) <= (1 - 1e-4 * alpha) * norm:
                    break
                alpha *= 0.5
            psi, f = trial, f_trial
            if alpha * np.abs(delta).max() < tol_v:
                break
        else:
            raise RuntimeError(f"Poisson solve did not converge (last update {alpha * np.abs(delta).max():.2e} V)")
        return self._expand(psi, phi, psi_gate, it)

    def _neutral_guess(self, dop, phi):
        """psi of charge-neutral silicon in equilibrium with the substrate (Newton takes it from there)."""
        return np.where(self._si_local, self.vt * np.arcsinh(dop / (2 * self.ni)), 0.0)

    def _expand(self, psi_local, phi_local, psi_gate, iterations) -> Potential:
        shape = self.grid.shape
        psi = np.full(self.mat.size, np.nan)
        psi[self.region.cells] = psi_local
        psi[self.gate] = psi_gate
        n = np.zeros(self.mat.size)
        p = np.zeros(self.mat.size)
        cells = self.region.cells[self._si_local]
        psi_si = psi_local[self._si_local]
        n[cells] = self.ni * np.exp(np.minimum((psi_si - phi_local[self._si_local]) / self.vt, 700.0))
        p[cells] = self.ni * np.exp(np.minimum(-psi_si / self.vt, 700.0))
        return Potential(psi.reshape(shape), n.reshape(shape), p.reshape(shape), iterations)
