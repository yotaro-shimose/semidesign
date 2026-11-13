"""Pixel figures of merit from the electrostatics.

The photodiode (PD) and floating diffusion (FD) are the n-type regions of
silicon that contain two points given by the user. Everything is
quasi-static: each state is an equilibrium of the potential with fixed
quasi-Fermi potentials. Holes are at 0 (the grounded p-type substrate).
Electrons are at a free value phi_PD in the floating PD and at V_FD
everywhere else: apart from the PD, the only supply of free electrons is the
FD contact, so a channel that opens under the TG fills from the FD, as during
reset and transfer. (Putting those electrons at the substrate level instead
would pin the surface under the gate at inversion and hide the TG.)

Defined as features of the potential landscape, not of drawn regions:

- sensor potential: the highest psi in the empty PD (the bottom of the
  electron valley; close to what is measured as the pinning voltage).
- transfer path and saddle: of all paths through silicon from the bottom of
  the PD valley to the FD, the one whose lowest psi is highest (found with a
  maximum spanning tree); its lowest psi is the saddle, the top of the barrier
  electrons have to cross. Barrier = sensor potential - saddle (> 0: electrons
  must climb).
- full well: electrons in the PD as it fills (phi_PD falls) until either its
  quasi-Fermi level reaches the TG saddle (electrons start spilling into the
  FD, TG off) or the PD becomes forward biased to the substrate (phi_PD = 0,
  the equilibrium full well). Whichever comes first.
- transfer (TG on): electrons start where they were stored, the bottom of
  the valley with the TG off; the TG-on barrier is measured from there.
  Every other point of the PD is checked too: its bottleneck to the FD is
  the highest saddle over all its paths, and a point above its bottleneck
  is a pocket (worst pocket = largest psi - bottleneck).
- residual after transfer: each PD cell drains until its quasi-Fermi level
  reaches min(V_FD, its bottleneck); what is left below is counted. A
  quasi-static estimate: a finite transfer time leaves more behind if a
  barrier is a few V_T or more.

In 2D the structure is taken to be uniform along the third axis, so charges
are per micrometre of width.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage
from scipy.optimize import brentq
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order, minimum_spanning_tree

from .electrostatics import Potential, PoissonSolver
from .materials import SI
from .simulation import Simulation

EMPTY_PHI_V = 20.0  # a PD quasi-Fermi potential this high holds no electrons


@dataclass(frozen=True)
class PixelSpec:
    pd_at: tuple[float, ...]  # a point inside the photodiode (lateral..., depth), nm
    fd_at: tuple[float, ...]  # a point inside the floating diffusion
    tg_off_v: float = 0.0
    tg_on_v: float = 2.8
    fd_v: float = 2.8
    temp_k: float = 300.0


@dataclass
class TransferPath:
    psi: np.ndarray  # V along the path
    distance_nm: np.ndarray
    points_nm: np.ndarray  # (n, ndim) cell centres
    sensor_v: float
    saddle_v: float

    @property
    def barrier_v(self) -> float:
        return self.sensor_v - self.saddle_v


class PixelAnalysis:
    def __init__(self, sim: Simulation, spec: PixelSpec):
        self.sim, self.spec, self.grid = sim, spec, sim.grid
        self.solver = PoissonSolver(sim.grid, sim.mat, sim.net_doping(), spec.temp_k)
        self.pd, self.fd = self._find_regions()
        self._si = (sim.mat == SI).ravel()
        self._centers = np.stack([c.ravel() for c in np.meshgrid(
            *(self.grid.centers(a) for a in range(self.grid.ndim)), indexing="ij")], 1)
        self._last: Potential | None = None
        self._last_state = (None, None)

    # ------------------------------------------------------------ regions

    def _cell_at(self, point) -> tuple[int, ...]:
        if len(point) != self.grid.ndim:
            raise ValueError(f"point {point} needs {self.grid.ndim} coordinates (lateral..., depth)")
        return tuple(int(np.clip(np.searchsorted(self.grid.edges[a], v, side="right") - 1,
                                 0, self.grid.shape[a] - 1)) for a, v in enumerate(point))

    def _find_regions(self) -> tuple[np.ndarray, np.ndarray]:
        n_type = (self.sim.mat == SI) & (self.sim.net_doping() > 0)
        labels, _ = ndimage.label(n_type)
        found = {}
        for name, point in (("photodiode", self.spec.pd_at), ("floating diffusion", self.spec.fd_at)):
            label = labels[self._cell_at(point)]
            if label == 0:
                raise ValueError(f"the {name} point {point} is not in n-type silicon")
            found[name] = label
        if found["photodiode"] == found["floating diffusion"]:
            raise ValueError("the photodiode and the floating diffusion are one n-type region: "
                             "nothing separates them")
        return labels == found["photodiode"], labels == found["floating diffusion"]

    def phi_n(self, phi_pd: float) -> np.ndarray:
        phi = np.full(self.grid.shape, self.spec.fd_v)
        phi[self.pd] = phi_pd
        return phi

    # ------------------------------------------------------------ states

    def solve(self, tg_v: float, phi_pd: float) -> Potential:
        """Equilibrium with the TG at tg_v and the PD quasi-Fermi potential at phi_pd.

        The previous solution seeds Newton only for a nearby state; from a far
        one (say an empty PD) the electron density of the old potential would be
        astronomically wrong, and a fresh start converges faster.
        """
        near = (self._last is not None and self._last_state[0] == tg_v
                and abs(self._last_state[1] - phi_pd) <= 0.25)
        self._last = self.solver.solve(tg_v, self.phi_n(phi_pd), initial=self._last if near else None)
        self._last_state = (tg_v, phi_pd)
        return self._last

    def electrons(self, pot: Potential) -> float:
        """Electrons in the PD (per um of width in 2D)."""
        count = (pot.n * self.grid.volumes_cm)[self.pd].sum()
        return float(count * 1e-4) if self.grid.ndim == 2 else float(count)

    def _maximin_tree(self, pot: Potential):
        """Spanning tree of silicon whose paths are maximin paths of psi."""
        psi = pot.psi.ravel()
        i, j = self.grid.faces
        keep = self._si[i] & self._si[j]
        i, j = i[keep], j[keep]
        weight = (np.nanmax(psi) + 1.0) - np.minimum(psi[i], psi[j])
        n = psi.size
        return minimum_spanning_tree(coo_matrix((weight, (i, j)), shape=(n, n)).tocsr())

    def _fd_seed(self, psi: np.ndarray) -> int:
        fd = np.flatnonzero(self.fd.ravel())
        return int(fd[np.argmax(psi[fd])])

    def storage_cell(self, pot: Potential) -> int:
        """Where electrons collect: the highest psi in the PD."""
        pd = np.flatnonzero(self.pd.ravel())
        return int(pd[np.argmax(pot.psi.ravel()[pd])])

    def transfer_path(self, pot: Potential, start: int | None = None) -> TransferPath:
        psi = pot.psi.ravel()
        start = self.storage_cell(pot) if start is None else start
        goal = self._fd_seed(psi)
        _, pred = breadth_first_order(self._maximin_tree(pot), start, directed=False,
                                      return_predecessors=True)
        path = [goal]
        while path[-1] != start:
            path.append(int(pred[path[-1]]))
            if path[-1] < 0:
                raise RuntimeError("no silicon path from the photodiode to the floating diffusion")
        path = np.array(path[::-1])
        pts = self._centers[path]
        dist = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
        before_fd = ~self.fd.ravel()[path]  # the saddle: lowest psi before the path enters the FD
        return TransferPath(psi[path], dist, pts, float(psi[start]), float(psi[path][before_fd].min()))

    def bottlenecks(self, pot: Potential) -> np.ndarray:
        """Per cell, the highest saddle over all silicon paths to the FD (NaN elsewhere)."""
        psi = pot.psi.ravel()
        seed = self._fd_seed(psi)
        order, pred = breadth_first_order(self._maximin_tree(pot), seed, directed=False,
                                          return_predecessors=True)
        b = np.full(psi.size, np.nan)
        b[seed] = psi[seed]
        fd = self.fd.ravel()
        for node in order[1:]:  # parents come first in breadth-first order
            parent = pred[node]
            b[node] = b[parent] if fd[node] else min(b[parent], psi[node])
        return b.reshape(self.grid.shape)

    # ------------------------------------------------------------ figures

    def run(self) -> dict:
        s = self.spec
        empty_off = self.solve(s.tg_off_v, EMPTY_PHI_V)
        storage = self.storage_cell(empty_off)
        off = self.transfer_path(empty_off, storage)
        sensor = off.sensor_v

        # full well: lower phi_PD until the PD reaches the TG saddle or the substrate level
        def overflow(phi):
            return phi - self.transfer_path(self.solve(s.tg_off_v, phi), storage).saddle_v

        if overflow(0.0) < 0:  # the TG barrier is reached before forward bias
            phi_full, limit = brentq(overflow, 0.0, sensor, xtol=1e-3), "transfer gate"
        else:
            phi_full, limit = 0.0, "junction (equilibrium full well)"
        full = self.solve(s.tg_off_v, phi_full)
        qv = [(round(float(phi), 4), self.electrons(self.solve(s.tg_off_v, phi)))
              for phi in np.linspace(phi_full, sensor, 8)]

        empty_on = self.solve(s.tg_on_v, EMPTY_PHI_V)
        on = self.transfer_path(empty_on, storage)
        bottleneck = self.bottlenecks(empty_on)
        pockets = (empty_on.psi - bottleneck)[self.pd]
        drained = np.full(self.grid.shape, s.fd_v)
        drained[self.pd] = np.minimum(s.fd_v, bottleneck[self.pd])
        self._last = None
        left = self.solver.solve(s.tg_on_v, drained)
        residual = self.electrons(left)

        self.states = {"empty, TG off": (empty_off, off), "empty, TG on": (empty_on, on)}
        unit = "per_um" if self.grid.ndim == 2 else ""
        return {
            "conditions": asdict(s),
            "sensor_potential_v": round(sensor, 4),
            "tg_off": {"saddle_v": round(off.saddle_v, 4), "barrier_v": round(off.barrier_v, 4)},
            "storage_point_nm": [round(float(v), 1) for v in self._centers[storage]],
            "tg_on": {"saddle_v": round(on.saddle_v, 4), "barrier_v": round(max(on.barrier_v, 0.0), 4),
                      "worst_pocket_v": round(max(float(pockets.max()), 0.0), 4)},
            "full_well": {f"electrons{'_' + unit if unit else ''}": round(self.electrons(full), 1),
                          "limited_by": limit, "phi_pd_v": round(phi_full, 4)},
            f"residual_electrons{'_' + unit if unit else ''}": float(f"{residual:.3g}"),
            "charge_vs_phi_pd": qv,
            "units": "electrons per um of width (2D)" if unit else "electrons",
        }
