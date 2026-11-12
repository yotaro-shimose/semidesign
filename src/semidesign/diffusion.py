"""Implicit dopant diffusion on a 2D/3D grid with material-dependent diffusivity.

Finite-volume form of dC/dt = div(D grad C) solved with backward Euler.
Face diffusivity is the harmonic mean of the two cells, so flux into gas
(D = 0) is zero and flux into oxide is strongly suppressed. Outer boundaries
are zero-flux; lateral edges optionally wrap around (periodic pixels).

Solver: sparse direct LU for small problems (typical 2D), conjugate gradient
with a diagonal preconditioner for large ones (typical 3D), so memory stays
proportional to the number of cells.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

DIRECT_SOLVER_MAX_CELLS = 400_000


def laplacian(d_cell: np.ndarray, h_nm: float, periodic_axes: tuple[int, ...] = ()) -> sp.csr_matrix:
    """Sparse operator L with (L c)_i = sum_faces g_f (c_j - c_i), g in 1/s."""
    shape = d_cell.shape
    n = d_cell.size
    idx = np.arange(n).reshape(shape)
    rows, cols, vals = [], [], []

    def add_faces(d0, d1, i0, i1):
        g = np.zeros_like(d0)
        ok = (d0 > 0) & (d1 > 0)
        g[ok] = 2 * d0[ok] * d1[ok] / (d0[ok] + d1[ok])
        g /= h_nm**2
        g, i0, i1 = g.ravel(), i0.ravel(), i1.ravel()
        keep = g > 0
        g, i0, i1 = g[keep], i0[keep], i1[keep]
        rows.extend([i0, i1, i0, i1])
        cols.extend([i1, i0, i0, i1])
        vals.extend([g, g, -g, -g])

    for ax in range(len(shape)):
        if shape[ax] < 2:
            continue
        lo = [slice(None)] * len(shape)
        hi = [slice(None)] * len(shape)
        lo[ax] = slice(0, -1)
        hi[ax] = slice(1, None)
        add_faces(d_cell[tuple(lo)], d_cell[tuple(hi)], idx[tuple(lo)], idx[tuple(hi)])
        if ax in periodic_axes and shape[ax] > 2:
            first = [slice(None)] * len(shape)
            last = [slice(None)] * len(shape)
            first[ax] = slice(0, 1)
            last[ax] = slice(-1, None)
            add_faces(d_cell[tuple(last)], d_cell[tuple(first)], idx[tuple(last)], idx[tuple(first)])

    if not rows:
        return sp.csr_matrix((n, n))
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    v = np.concatenate(vals)
    return sp.csr_matrix((v, (r, c)), shape=(n, n))


class ImplicitDiffuser:
    """Backward-Euler stepper for one diffusivity field and fixed time step."""

    def __init__(self, d_cell_nm2_s: np.ndarray, h_nm: float, dt_s: float,
                 periodic_axes: tuple[int, ...] = (), force: str | None = None):
        self.shape = d_cell_nm2_s.shape
        lap = laplacian(d_cell_nm2_s, h_nm, periodic_axes)
        n = d_cell_nm2_s.size
        self.a = (sp.identity(n, format="csr") - dt_s * lap).tocsc()
        method = force or ("direct" if n <= DIRECT_SOLVER_MAX_CELLS else "cg")
        self.method = method
        if method == "direct":
            self._lu = spla.splu(self.a)
        else:
            inv_diag = 1.0 / self.a.diagonal()
            self._m = spla.LinearOperator((n, n), matvec=lambda x: inv_diag * x)

    def step(self, c: np.ndarray) -> np.ndarray:
        b = c.ravel()
        if self.method == "direct":
            x = self._lu.solve(b)
        else:
            x, info = spla.cg(self.a, b, x0=b, rtol=1e-10, atol=0.0, M=self._m, maxiter=5000)
            if info != 0:
                raise RuntimeError(f"diffusion solver did not converge (info={info})")
        return x.reshape(self.shape)
