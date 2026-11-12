"""Process simulation on a 2D/3D grid: structure + dopants + process steps."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import geometry, physics
from .diffusion import ImplicitDiffuser
from .grid import Grid
from .implant import implant_profile
from .physics import GAS, OXIDE, SI, SI_CONSUMED_PER_OXIDE, get_dopant, material_code


@dataclass
class StepRecord:
    index: int
    kind: str
    params: dict
    info: dict
    seconds: float

    def to_dict(self) -> dict:
        return {"index": self.index, "type": self.kind, "params": self.params,
                "info": self.info, "seconds": round(self.seconds, 3)}


@dataclass
class Snapshot:
    label: str
    mat: np.ndarray
    conc: dict[str, np.ndarray]


@dataclass
class Simulation:
    grid: Grid
    substrate_dopant: str = "B"
    substrate_conc: float = 1e15
    orientation: str = "100"
    keep_snapshots: bool = True
    mat: np.ndarray = field(init=False)
    conc: dict[str, np.ndarray] = field(init=False)
    history: list[StepRecord] = field(init=False, default_factory=list)
    snapshots: list[Snapshot] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        get_dopant(self.substrate_dopant)
        z = self.grid.z_centers
        self.mat = np.where(z >= 0, SI, GAS).astype(np.int8) * np.ones(self.grid.shape, dtype=np.int8)
        bg = np.where(self.mat == SI, self.substrate_conc, 0.0)
        self.conc = {self.substrate_dopant: bg}
        self._snapshot("initial")

    # ------------------------------------------------------------------ steps

    def etch(self, depth_nm: float, openings=None, materials=None) -> None:
        def run():
            mats = None if materials is None else [material_code(m) for m in materials]
            removed = geometry.etch(self.grid, self.mat, openings, depth_nm, mats)
            self._clear_gas()
            return {"cells_removed": int(removed.sum())}
        self._run("etch", {"depth_nm": depth_nm, "openings": openings, "materials": materials}, run)

    def deposit(self, material: str, thickness_nm: float, conformal: bool = True) -> None:
        def run():
            new = geometry.deposit(self.grid, self.mat, material_code(material),
                                   thickness_nm, conformal)
            return {"cells_added": int(new.sum())}
        self._run("deposit", {"material": material, "thickness_nm": thickness_nm,
                              "conformal": conformal}, run)

    def planarize(self, level_nm: float = 0.0) -> None:
        def run():
            removed = geometry.planarize(self.grid, self.mat, level_nm)
            self._clear_gas()
            return {"cells_removed": int(removed.sum())}
        self._run("planarize", {"level_nm": level_nm}, run)

    def implant(self, dopant: str, dose: float, energy_kev: float, openings=None,
                rp_nm=None, drp_nm=None, lateral_sigma_nm=None) -> None:
        def run():
            add, info = implant_profile(self.grid, self.mat, dopant, dose, energy_kev,
                                        openings, rp_nm, drp_nm, lateral_sigma_nm)
            self.conc[dopant] = self.conc.get(dopant, np.zeros(self.grid.shape)) + add
            info["dose_in_structure"] = float(add.sum() * self.grid.cell_volume_cm
                                              / self._lateral_area_cm())
            return info
        self._run("implant", {"dopant": dopant, "dose": dose, "energy_kev": energy_kev,
                              "openings": openings}, run)

    def anneal(self, temp_c: float, time_min: float, steps: int = 40) -> None:
        def run():
            method = self._diffuse(temp_c, time_min * 60, steps)
            return {"solver": method}
        self._run("anneal", {"temp_c": temp_c, "time_min": time_min, "steps": steps}, run)

    def oxidize(self, temp_c: float, time_min: float, ambient: str = "dry",
                openings=None, steps: int = 40) -> None:
        """Thermal oxidation, simplified to vertical growth in each column.

        Deal-Grove growth from the oxide already on each column; 44% of the new
        oxide thickness consumes silicon, the rest grows upward. Dopant in the
        consumed silicon stays where it was (now oxide). Oxidation of trench
        side walls and the bird's beak under mask edges are not modelled.
        Columns outside ``openings`` (e.g. under a nitride mask) do not oxidize.
        """
        def run():
            b, ba = physics.deal_grove_coefficients(temp_c, ambient, self.orientation)
            open_ = self.grid.mask(openings).ravel()
            h = self.grid.spacing_nm
            mat2 = self.mat.reshape(-1, self.mat.shape[-1])
            tox = self._oxide_on_columns().ravel().astype(float)
            start = tox.copy()
            dt_min = time_min / steps
            hit_top = False
            for _ in range(steps):
                new = np.where(open_, physics.deal_grove_grow(tox, dt_min, b, ba), tox)
                hit_top |= self._apply_oxidation(mat2, tox, new, h)
                tox = new
                self._diffuse(temp_c, dt_min * 60, 1)
            grown = tox - start
            return {"oxide_grown_nm_mean": float(grown[open_].mean()) if open_.any() else 0.0,
                    "oxide_grown_nm_max": float(grown.max()),
                    "headroom_exceeded": bool(hit_top)}
        self._run("oxidize", {"temp_c": temp_c, "time_min": time_min, "ambient": ambient,
                              "openings": openings}, run)

    # -------------------------------------------------------------- internals

    def _run(self, kind: str, params: dict, fn) -> None:
        t0 = time.perf_counter()
        info = fn()
        rec = StepRecord(len(self.history) + 1, kind, params, info, time.perf_counter() - t0)
        self.history.append(rec)
        self._snapshot(f"{rec.index}: {kind}")

    def _snapshot(self, label: str) -> None:
        if self.keep_snapshots:
            self.snapshots.append(Snapshot(label, self.mat.copy(),
                                           {k: v.copy() for k, v in self.conc.items()}))

    def _clear_gas(self) -> None:
        gas = self.mat == GAS
        for c in self.conc.values():
            c[gas] = 0.0

    def _lateral_area_cm(self) -> float:
        return float(np.prod([w * 1e-7 for w in self.grid.lateral_nm]))

    def _diffuse(self, temp_c: float, total_s: float, steps: int) -> str:
        factor = np.vectorize(physics.DIFFUSIVITY_FACTOR.get)(self.mat).astype(float)
        dt = total_s / steps
        periodic = tuple(range(self.grid.ndim - 1)) if self.grid.periodic else ()
        method = "none"
        for name, c in self.conc.items():
            d_nm2_s = get_dopant(name).diffusivity(temp_c) * 1e14 * factor
            stepper = ImplicitDiffuser(d_nm2_s, self.grid.spacing_nm, dt, periodic)
            method = stepper.method
            for _ in range(steps):
                c = stepper.step(c)
            self.conc[name] = np.maximum(c, 0.0)
        return method

    def _oxide_on_columns(self) -> np.ndarray:
        """Thickness (nm) of oxide stacked directly above the Si in each column."""
        h = self.grid.spacing_nm
        is_si = self.mat == SI
        si_top = np.where(is_si.any(-1), np.argmax(is_si, axis=-1), self.mat.shape[-1])
        k = np.arange(self.mat.shape[-1])
        above = k < si_top[..., None]
        return ((self.mat == OXIDE) & above).sum(-1) * h

    def _apply_oxidation(self, mat2, tox_old, tox_new, h) -> bool:
        """Convert cells so each column's cell counts follow the continuous thickness.

        Returns True if oxide reached the top of the box (headroom too small).
        """
        hit_top = False
        nz = mat2.shape[1]
        k = np.arange(nz)
        old_si = np.floor(SI_CONSUMED_PER_OXIDE * tox_old / h + 1e-9)
        new_si = np.floor(SI_CONSUMED_PER_OXIDE * tox_new / h + 1e-9)
        old_up = np.floor((1 - SI_CONSUMED_PER_OXIDE) * tox_old / h + 1e-9)
        new_up = np.floor((1 - SI_CONSUMED_PER_OXIDE) * tox_new / h + 1e-9)
        for col in np.nonzero((new_si > old_si) | (new_up > old_up))[0]:
            row = mat2[col]
            si = np.nonzero(row == SI)[0]
            if si.size == 0:
                continue
            for _ in range(int(new_si[col] - old_si[col])):
                top_si = np.nonzero(row == SI)[0]
                if top_si.size:
                    row[top_si[0]] = OXIDE
            for _ in range(int(new_up[col] - old_up[col])):
                solid = np.nonzero(row != GAS)[0]
                if solid.size and solid[0] > 0:
                    row[solid[0] - 1] = OXIDE
                else:
                    hit_top = True
        return hit_top

    # --------------------------------------------------------------- analysis

    def net_doping(self) -> np.ndarray:
        """Donors minus acceptors in cm^-3 (positive = n-type); zero outside Si."""
        net = np.zeros(self.grid.shape)
        for name, c in self.conc.items():
            net += c if get_dopant(name).kind == "n" else -c
        net[self.mat != SI] = 0.0
        return net

    def column(self, pos_nm: list[float] | float | None = None) -> tuple[int, ...]:
        """Lateral index of the column at a position (default: box centre)."""
        lat = self.grid.lateral_nm
        if pos_nm is None:
            pos = [w / 2 for w in lat]
        else:
            pos = [pos_nm] if np.isscalar(pos_nm) else list(pos_nm)
        return tuple(self.grid.lateral_index(p, a) for a, p in enumerate(pos))

    def junctions_along_depth(self, pos_nm=None) -> list[float]:
        """Depths (nm, from the initial Si surface) where net doping changes sign."""
        col = self.column(pos_nm)
        net = self.net_doping()[col]
        si = self.mat[col] == SI
        z = self.grid.z_centers
        out = []
        for i in range(len(net) - 1):
            if si[i] and si[i + 1] and net[i] * net[i + 1] < 0:
                out.append(float(z[i] - net[i] * (z[i + 1] - z[i]) / (net[i + 1] - net[i])))
        return out

    def total_dose(self, dopant: str) -> float:
        """Amount of dopant per unit wafer area (cm^-2), averaged over the box."""
        c = self.conc.get(dopant)
        if c is None:
            return 0.0
        return float(c.sum() * self.grid.cell_volume_cm / self._lateral_area_cm())

    def material_fractions(self) -> dict[str, float]:
        n = self.mat.size
        return {physics.MATERIAL_NAMES[m]: float((self.mat == m).sum() / n)
                for m in physics.MATERIAL_NAMES}

    def summary(self, probe_nm=None) -> dict:
        return {
            "grid": self.grid.describe(),
            "substrate": {"dopant": self.substrate_dopant, "conc": self.substrate_conc},
            "steps": [r.to_dict() for r in self.history],
            "total_seconds": round(sum(r.seconds for r in self.history), 3),
            "dose_cm2": {k: self.total_dose(k) for k in sorted(self.conc)},
            "probe_column_nm": probe_nm if probe_nm is not None else [w / 2 for w in self.grid.lateral_nm],
            "junction_depths_nm": [round(x, 1) for x in self.junctions_along_depth(probe_nm)],
            "oxide_on_probe_column_nm": float(self._oxide_on_columns()[self.column(probe_nm)]),
            "material_fractions": self.material_fractions(),
        }
