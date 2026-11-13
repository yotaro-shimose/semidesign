"""Write simulation results: arrays (npz), summary (json), cross-section (png)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .materials import GAS, NITRIDE, OXIDE, SI
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
    mat_rgb = ListedColormap(["#FFFFFF", "#F1EFE8", "#E6DFC8", "#B4B2A9", "#C9D6C3"])
    ax.pcolormesh(xe, ze, mat, cmap=mat_rgb, vmin=GAS - 0.5, vmax=NITRIDE + 0.5)
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


def write_outputs(sim: Simulation, out_dir: Path, name: str, probe_nm=None,
                  snapshots: bool = True, plot: bool = True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"name": name, **sim.summary(probe_nm)}
    save_arrays(sim, out_dir / "result.npz")
    files = {"arrays": "result.npz", "summary": "summary.json"}
    if snapshots and sim.snapshots:
        files["snapshots"] = [f"steps/{n}" for n in save_snapshots(sim, out_dir / "steps")]
    if plot:
        plot_cross_section(sim, out_dir / "cross_section.png", name)
        files["cross_section"] = "cross_section.png"
    summary["files"] = files
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary
