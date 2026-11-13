"""Finite-volume building blocks shared by every transport model.

Unknowns live in cells; neighbouring cells exchange through the face between
them. A model supplies a property per cell; the face conductance is the
series combination of the two half cells (exact for a layered medium), so the
same code works on uniform and graded grids. This module enumerates faces,
assembles the sparse operator, restricts it to a region of cells, and solves
the symmetric positive definite systems that result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

DIRECT_SOLVER_MAX_CELLS = 400_000


def face_pairs(shape: tuple[int, ...], periodic_axes: tuple[int, ...] = ()
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flat indices (i, j) of every pair of cells that share a face, and each face's axis."""
    idx = np.arange(int(np.prod(shape))).reshape(shape)
    first, second, axes = [], [], []

    def add(a, b, ax):
        first.append(a.ravel())
        second.append(b.ravel())
        axes.append(np.full(a.size, ax))

    for ax, n in enumerate(shape):
        if n < 2:
            continue
        add(np.take(idx, range(n - 1), axis=ax), np.take(idx, range(1, n), axis=ax), ax)
        if ax in periodic_axes and n > 2:
            add(np.take(idx, [n - 1], axis=ax), np.take(idx, [0], axis=ax), ax)
    return np.concatenate(first), np.concatenate(second), np.concatenate(axes)


def series_conductance(k_i: np.ndarray, k_j: np.ndarray, half_i: np.ndarray,
                       half_j: np.ndarray, area: np.ndarray) -> np.ndarray:
    """Face conductance area / (half_i / k_i + half_j / k_j); zero if either k is zero."""
    k_i, k_j, half_i, half_j, area = np.broadcast_arrays(
        *(np.asarray(v, dtype=float) for v in (k_i, k_j, half_i, half_j, area)))
    out = np.zeros(k_i.shape)
    ok = (k_i > 0) & (k_j > 0)
    out[ok] = area[ok] / (half_i[ok] / k_i[ok] + half_j[ok] / k_j[ok])
    return out


def laplacian(n: int, i: np.ndarray, j: np.ndarray, g: np.ndarray) -> sp.csr_matrix:
    """Operator L with (L u)_k = sum over the faces of k of g (u_other - u_k)."""
    g = np.broadcast_to(np.asarray(g, dtype=float), i.shape)
    keep = g > 0
    i, j, g = i[keep], j[keep], g[keep]
    rows = np.concatenate([i, j, i, j])
    cols = np.concatenate([j, i, i, j])
    vals = np.concatenate([g, g, -g, -g])
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


@dataclass
class Region:
    """A subset of cells, with its faces split into inner and edge faces.

    inner_*: faces with both cells in the region (local indices).
    edge_*:  faces with one cell in the region (local index) and one outside
             (global index of the outside cell).
    *_face:  index of each face in the full face arrays, to look up conductances.
    """

    cells: np.ndarray
    local: np.ndarray
    inner_a: np.ndarray
    inner_b: np.ndarray
    inner_face: np.ndarray
    edge_cell: np.ndarray
    edge_out: np.ndarray
    edge_face: np.ndarray
    edge_on_i: np.ndarray  # the region cell is the face's i cell

    @classmethod
    def of(cls, member: np.ndarray, i: np.ndarray, j: np.ndarray) -> "Region":
        member = member.ravel()
        cells = np.flatnonzero(member)
        local = np.full(member.size, -1)
        local[cells] = np.arange(cells.size)
        mi, mj = member[i], member[j]
        inner = np.flatnonzero(mi & mj)
        edge_i = np.flatnonzero(mi & ~mj)  # region cell on the i side
        edge_j = np.flatnonzero(~mi & mj)  # region cell on the j side
        return cls(
            cells=cells, local=local,
            inner_a=local[i[inner]], inner_b=local[j[inner]], inner_face=inner,
            edge_cell=np.concatenate([local[i[edge_i]], local[j[edge_j]]]),
            edge_out=np.concatenate([j[edge_i], i[edge_j]]),
            edge_face=np.concatenate([edge_i, edge_j]),
            edge_on_i=np.concatenate([np.ones(edge_i.size, bool), np.zeros(edge_j.size, bool)]),
        )

    @property
    def n(self) -> int:
        return self.cells.size

    def laplacian(self, g_inner) -> sp.csr_matrix:
        return laplacian(self.n, self.inner_a, self.inner_b, g_inner)

    def edge_side(self, on_i: np.ndarray, on_j: np.ndarray) -> np.ndarray:
        """Per edge face, the value belonging to the region-side cell."""
        f = self.edge_face
        return np.where(self.edge_on_i, on_i[f], on_j[f])

    def edge_sum(self, values: np.ndarray, where: np.ndarray | None = None) -> np.ndarray:
        """Sum per region cell of a quantity given per edge face."""
        values = np.broadcast_to(np.asarray(values, dtype=float), self.edge_cell.shape)
        cells = self.edge_cell
        if where is not None:
            cells, values = cells[where], values[where]
        return np.bincount(cells, weights=values, minlength=self.n)


class SPDSolver:
    """Solver for a fixed symmetric positive definite matrix.

    factorize=True: direct sparse factorization (fits typical 2D sizes). Right
    for steady-state problems, where iterative methods converge slowly, and
    for any system solved many times.
    factorize=False: conjugate gradients with a diagonal preconditioner. Right
    for a time-step matrix used once: it is close to the identity, so a few
    iterations suffice.
    Systems too large to factorize (typical 3D) always use conjugate gradients,
    whose memory stays proportional to the number of cells.
    """

    def __init__(self, a: sp.spmatrix, factorize: bool = True):
        n = a.shape[0]
        self.n = n
        if factorize and n <= DIRECT_SOLVER_MAX_CELLS:
            self.method = "direct"
            self._lu = spla.splu(a.tocsc())
        else:
            self.method = "cg"
            self._a = a.tocsr()
            inv_diag = 1.0 / self._a.diagonal()
            self._m = spla.LinearOperator((n, n), matvec=lambda x: inv_diag * x)

    def solve(self, b: np.ndarray, x0: np.ndarray | None = None) -> np.ndarray:
        if self.n == 0:
            return np.zeros(0)
        if self.method == "direct":
            return self._lu.solve(b)
        x, info = spla.cg(self._a, b, x0=x0, rtol=1e-10, atol=0.0, M=self._m, maxiter=10_000)
        if info != 0:
            raise RuntimeError(f"conjugate gradient did not converge (info={info})")
        return x
