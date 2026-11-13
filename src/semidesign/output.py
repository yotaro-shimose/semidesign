"""Write simulation results: arrays (npz), summary (json), cross-section (png)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .materials import COUNT, GAS, NAMES, OXIDE, SI
from .simulation import Simulation


def save_arrays(sim: Simulation, path: Path) -> None:
    g = sim.grid
    arrays = {"material": sim.mat, "net_doping": sim.net_doping(), "z_nm": g.z_centers,
              "interstitial_clusters": sim.defects.clusters}
    for a in range(g.ndim - 1):
        arrays[f"lateral{a}_nm"] = g.lateral_centers(a)
    for k, v in sim.conc.items():
        arrays[f"conc_{k}"] = v
    np.savez_compressed(path, **arrays)


def save_snapshots(sim: Simulation, folder: Path) -> list[str]:
    folder.mkdir(parents=True, exist_ok=True)
    names = []
    for i, s in enumerate(sim.snapshots):
        name = f"step_{i:02d}.npz"
        np.savez_compressed(folder / name, label=np.array(s.label), material=s.mat,
                            interstitial_clusters=s.clusters,
                            **{f"conc_{k}": v for k, v in s.conc.items()})
        names.append(name)
    return names


def _section(sim: Simulation, arr: np.ndarray) -> np.ndarray:
    """2D slice (x, z) of an array; for 3D the plane through the box centre."""
    if arr.ndim == 2:
        return arr
    mid = arr.shape[1] // 2
    return arr[:, mid, :]


def plot_cross_section(sim: Simulation, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    g = sim.grid
    net = _section(sim, sim.net_doping()).T  # rows = depth
    mat = _section(sim, sim.mat).T
    x, z = g.centers(0), g.z_centers
    xe, ze = g.edges[0] / 1000, g.z_edges / 1000

    # signed log scale: +log10 for n-type, -log10 for p-type, |C| < 1e14 -> 0
    with np.errstate(divide="ignore"):
        mag = np.clip(np.log10(np.abs(net)) - 14, 0, 6)
    signed = np.sign(net) * mag
    signed = np.ma.masked_where(mat != SI, signed)

    fig_w = 4 + 3 * g.lateral_nm[0] / max(g.lateral_nm[0], g.depth_nm)
    fig, ax = plt.subplots(figsize=(fig_w + 2, 7))
    # gas, Si, SiO2, poly, nitride, Al2O3, HfO2
    mat_rgb = ListedColormap(["#FFFFFF", "#F1EFE8", "#E6DFC8", "#B4B2A9", "#C9D6C3", "#D8C3D6", "#C3CCDE"])
    ax.pcolormesh(xe, ze, mat, cmap=mat_rgb, vmin=-0.5, vmax=COUNT - 0.5)
    im = ax.pcolormesh(xe, ze, signed, cmap="RdBu_r", vmin=-6, vmax=6)
    ax.set_ylim(ze[-1], ze[0])
    xx, zz = np.meshgrid(x / 1000, z / 1000)
    netm = np.ma.masked_where(mat != SI, net)
    if (netm > 0).any() and (netm < 0).any():
        ax.contour(xx, zz, netm, levels=[0], colors="k", linewidths=0.8, linestyles="--")
    ax.contour(xx, zz, (mat == OXIDE).astype(float), levels=[0.5], colors="#8C7A4A",
               linewidths=0.6)
    ax.set_xlabel("x (μm)")
    ax.set_ylabel("depth (μm)")
    ax.set_title(title, fontsize=10)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_ticks([-6, -3, 0, 3, 6])
    cb.set_ticklabels(["p 1e20", "p 1e17", "<1e14", "n 1e17", "n 1e20"])
    cb.set_label("net doping (cm$^{-3}$)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_outputs(sim: Simulation, out_dir: Path, name: str, probes=None, snapshots: bool = True,
                  plot: bool = True, pixel=None, comparisons=()) -> dict:
    """Write arrays, summary.json, plots, a depth profile per probe and the
    parameters used; with a PixelSpec the pixel evaluation, with Comparisons
    the agreement with measured profiles."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"name": name, **sim.summary(probes)}
    save_arrays(sim, out_dir / "result.npz")
    (out_dir / "parameters.toml").write_text(sim.params.to_toml())
    files = {"arrays": "result.npz", "summary": "summary.json", "parameters": "parameters.toml",
             "profiles": save_profiles(sim, summary["probes"], out_dir / "profiles")}
    if snapshots and sim.snapshots:
        files["snapshots"] = [f"steps/{n}" for n in save_snapshots(sim, out_dir / "steps")]
    if plot:
        plot_cross_section(sim, out_dir / "cross_section.png", name)
        files["cross_section"] = "cross_section.png"
    if comparisons:
        summary["compare"] = [c.evaluate(sim) for c in comparisons]
        if plot:
            files["compare"] = []
            for i, c in enumerate(comparisons, start=1):
                c.plot(sim, out_dir / f"compare_{i}.png")
                files["compare"].append(f"compare_{i}.png")
    if pixel is not None:
        import time
        from .pixel import PixelAnalysis
        t0 = time.perf_counter()
        analysis = PixelAnalysis(sim, pixel)
        summary["pixel"] = analysis.run()
        summary["pixel"]["seconds"] = round(time.perf_counter() - t0, 1)
        np.savez_compressed(out_dir / "pixel.npz", **{
            f"psi_{label.replace(', ', '_').replace(' ', '_')}": pot.psi
            for label, (pot, _) in analysis.states.items()})
        files["pixel_arrays"] = "pixel.npz"
        if plot:
            plot_pixel(analysis, out_dir / "pixel_potential.png", name)
            files["pixel_potential"] = "pixel_potential.png"
    summary["files"] = files
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def save_profiles(sim: Simulation, probes: list[dict], folder: Path) -> list[str]:
    """One CSV per probe column: depth, material and concentrations, like a SIMS export."""
    folder.mkdir(parents=True, exist_ok=True)
    names = []
    for i, probe in enumerate(probes, start=1):
        prof = sim.depth_profile(probe["at_nm"])
        keys = [k for k in prof if k not in ("depth_nm", "material")]
        lines = [f"# column at {probe['at_nm']} nm; depth from the initial front surface; cm^-3",
                 "depth_nm,material," + ",".join(keys)]
        for r in range(prof["depth_nm"].size):
            if prof["material"][r] == GAS:
                continue
            lines.append(f"{prof['depth_nm'][r]:.2f},{NAMES[int(prof['material'][r])]}," +
                         ",".join(f"{prof[k][r]:.4g}" for k in keys))
        name = f"probe_{i}.csv"
        (folder / name).write_text("\n".join(lines) + "\n")
        names.append(f"profiles/{name}")
    return names


def plot_pixel(analysis, path: Path, title: str) -> None:
    """Potential maps (empty PD, TG off and on) with the transfer path, psi along
    it, and psi along the collection path from the bottom of the silicon."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = analysis.grid
    xe, ze = g.edges[0] / 1000, g.z_edges / 1000
    solid = np.flatnonzero((_section_array(analysis.sim.mat) != GAS).any(axis=0))
    top, bottom = ze[solid[0]], ze[solid[-1] + 1]
    states = analysis.states
    fig = plt.figure(figsize=(15, 8.5))
    grid = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.5])
    vmax = max(np.nanmax(pot.psi) for pot, _ in states.values())
    for k, (label, (pot, tp)) in enumerate(states.items()):
        ax = fig.add_subplot(grid[:, k])
        psi = _section_array(pot.psi).T
        im = ax.pcolormesh(xe, ze, np.ma.masked_invalid(psi), cmap="viridis", vmin=-1, vmax=vmax)
        ax.contour(g.centers(0) / 1000, g.z_centers / 1000, np.ma.masked_invalid(psi),
                   levels=np.arange(-1, vmax, 0.5), colors="w", linewidths=0.4)
        ax.plot(tp.points_nm[:, 0] / 1000, tp.points_nm[:, -1] / 1000, "r-", lw=1.2)
        if k == 0:
            cp = analysis.collection_path
            ax.plot(cp.points_nm[:, 0] / 1000, cp.points_nm[:, -1] / 1000, "-", color="orange", lw=1.2)
        ax.set_ylim(bottom, top)
        ax.set_title(f"{label}: psi (V)", fontsize=10)
        ax.set_xlabel("x (μm)")
        ax.set_ylabel("depth (μm)")
        fig.colorbar(im, ax=ax, shrink=0.6)
    ax = fig.add_subplot(grid[0, 2])
    for label, (pot, tp) in states.items():
        ax.plot(tp.distance_nm / 1000, tp.psi, label=label)
        ax.axhline(tp.saddle_v, ls=":", lw=0.8, color=ax.lines[-1].get_color())
    ax.set_xlabel("distance along the transfer path (red), PD → FD (μm)")
    ax.set_ylabel("psi (V)")
    ax.set_title("transfer: electrons collect where psi is high", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax = fig.add_subplot(grid[1, 2])
    cp = analysis.collection_path
    ax.plot(cp.distance_nm / 1000, cp.psi, color="orange")
    ax.axhline(cp.saddle_v, ls=":", lw=0.8, color="orange")
    ax.set_xlabel("distance along the collection path (orange), bottom → PD (μm)")
    ax.set_ylabel("psi (V)")
    ax.set_title("collection, TG off: should only rise toward the PD", fontsize=10)
    ax.grid(alpha=0.3)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _section_array(arr: np.ndarray) -> np.ndarray:
    return arr if arr.ndim == 2 else arr[:, arr.shape[1] // 2, :]
