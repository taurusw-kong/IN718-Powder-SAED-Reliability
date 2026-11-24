"""Plots of the main results, made from ../data and from the tables written by diffraction.py and surfaces.py.

    python figures.py
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
RES = os.path.join(HERE, "..", "results")
STREAMS = ["virgin", "reuse", "waste", "filter"]
COLOR = {"virgin": "#2a5caa", "reuse": "#e3772b", "waste": "#a23b72", "filter": "#159a7c"}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})


def read(folder, name):
    with open(os.path.join(folder, name)) as fh:
        return list(csv.DictReader(fh))


def stream_of(pattern):
    return {"V": "virgin", "W": "waste", "F": "filter"}.get(pattern[0], "reuse")


def rings_and_fringes():
    rings = read(DATA, "diffraction_rings.csv")
    fr = read(DATA, "lattice_fringes.csv")
    fig, ax = plt.subplots(figsize=(6.5, 3))
    rng = np.random.default_rng(3)
    for i, s in enumerate(STREAMS):
        y0 = 3 - i
        d = np.array([float(r["d_A"]) for r in fr if r["stream"] == s])
        ax.scatter(d, y0 + 0.1 + rng.uniform(0, 0.3, len(d)), s=3, color=COLOR[s], alpha=0.5, lw=0)
        for r in rings:
            if r["stream"] == s:
                ax.plot([float(r["d_A"])] * 2, [y0 - 0.35, y0 - 0.05], color="k", lw=0.8)
    ax.set_yticks(range(4)[::-1])
    ax.set_yticklabels(STREAMS)
    ax.set_xlim(1.5, 8)
    ax.set_xlabel("Lattice spacing (Å)")
    ax.set_title("Lattice fringes (dots) and diffraction rings (bars)", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "rings_and_fringes.png"))


def decoys():
    top = [r for r in read(RES, "phase_search_top5.csv") if r["rank"] == "1"]
    fig, ax = plt.subplots(figsize=(3.2, 3))
    for r in top:
        c = COLOR[stream_of(r["pattern"])]
        sig = float(r["p_value"]) <= 0.05
        ax.plot(float(r["median_best_decoy"]), float(r["score"]), "o", ms=5, mfc=c if sig else "white", mec=c)
    ax.plot([0.3, 1], [0.3, 1], "k--", lw=0.6)
    ax.set_xlabel("Best decoy score (median of 20 libraries)")
    ax.set_ylabel("Best candidate score")
    ax.set_title("Filled: p ≤ 0.05", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "phase_search_vs_decoys.png"))


def reliability():
    g = [r for r in read(RES, "reliability_grid.csv") if r["tolerance"] == "0.025"]
    ns, cs = [3, 5, 8, 12], ["0.0", "0.01", "0.02", "0.05"]
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.6))
    for ax, key, title in zip(axs, ("exact", "family"), ("Correct structure first (%)", "Correct family first (%)")):
        M = np.array([[100 * float(next(r[key] for r in g if int(r["n_rings"]) == n and r["calibration"] == c)) for n in ns] for c in cs])
        ax.imshow(M, cmap="Blues", vmin=0, vmax=100, aspect="auto")
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", color="white" if M[i, j] > 55 else "black")
        ax.set_xticks(range(4))
        ax.set_xticklabels(ns)
        ax.set_yticks(range(4))
        ax.set_yticklabels(["±0 %", "±1 %", "±2 %", "±5 %"])
        ax.set_xlabel("Rings in pattern")
        ax.set_title(title, fontsize=8)
    axs[0].set_ylabel("Calibration error")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "reliability_map.png"))


def sem_shape():
    parts = [r for r in read(DATA, "sem_particles.csv") if float(r["magnification"]) < 1500]
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.4), sharey=True)
    for ax, q, lab in zip(axs, ("diameter_um", "circularity", "aspect_ratio"), ("Equivalent diameter (µm)", "Circularity", "Aspect ratio")):
        for s in ("waste", "filter"):
            v = np.sort([float(r[q]) for r in parts if r["stream"] == s])
            ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=COLOR[s], label=s)
        ax.set_xlabel(lab)
    axs[0].set_xlim(0, 60)
    axs[2].set_xlim(1, 2.5)
    axs[0].set_ylabel("Cumulative fraction")
    axs[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "sem_shape.png"))


def xps_o1s():
    rows = read(DATA, "xps_spectra.csv")
    shift = {r["stream"]: float(r["binding_energy_shift_eV"]) for r in read(RES, "xps_surface_chemistry.csv")}
    fig, ax = plt.subplots(figsize=(3.2, 3))
    for i, s in enumerate(STREAMS):
        x = np.array([float(r["binding_energy_eV"]) for r in rows if r["stream"] == s and r["region"] == "O 1s"]) + shift[s]
        y = np.array([float(r["counts"]) for r in rows if r["stream"] == s and r["region"] == "O 1s"])
        m = (x >= 526) & (x <= 537)
        y = y[m] - y[m].min()
        ax.plot(x[m], y / y.max() + (3 - i) * 0.9, color=COLOR[s], lw=0.8, label=s)
    ax.set_xlim(537, 526)
    ax.set_yticks([])
    ax.set_xlabel("Binding energy, O 1s (eV)")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "xps_o1s.png"))


if __name__ == "__main__":
    rings_and_fringes()
    decoys()
    reliability()
    sem_shape()
    xps_o1s()
    print("figures written to", os.path.abspath(RES))
