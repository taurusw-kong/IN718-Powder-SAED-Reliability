"""Particles and surfaces: SEM particle shape, whole particles seen in TEM, EDS metal ratios and XPS chemical states.

Reads the CSV files in ../data and writes tables to ../results.

    python surfaces.py
"""
import csv
import os

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import mannwhitneyu

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
RES = os.path.join(HERE, "..", "results")
os.makedirs(RES, exist_ok=True)
STREAMS = ["virgin", "reuse", "waste", "filter"]


def read(name):
    with open(os.path.join(DATA, name)) as fh:
        return list(csv.DictReader(fh))


def write(name, rows):
    with open(os.path.join(RES, name), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def sem():
    parts = [r for r in read("sem_particles.csv") if float(r["magnification"]) < 1500]
    rows, lines = [], []
    for q in ("diameter_um", "circularity", "solidity", "aspect_ratio"):
        w = [float(r[q]) for r in parts if r["stream"] == "waste"]
        f = [float(r[q]) for r in parts if r["stream"] == "filter"]
        med = {}
        for s in ("waste", "filter"):
            per_image = {}
            for r in parts:
                if r["stream"] == s:
                    per_image.setdefault(r["image"], []).append(float(r[q]))
            med[s] = [np.median(v) for v in per_image.values()]
        rows.append(dict(quantity=q, waste_particles=len(w), filter_particles=len(f),
                         waste_median=round(float(np.median(w)), 3), filter_median=round(float(np.median(f)), 3),
                         p_all_particles=float(f"{mannwhitneyu(w, f).pvalue:.2g}"),
                         p_image_medians=float(f"{mannwhitneyu(med['waste'], med['filter']).pvalue:.2g}"),
                         waste_images=len(med["waste"]), filter_images=len(med["filter"])))
    write("sem_shape.csv", rows)
    for r in rows:
        lines.append(f"SEM {r['quantity']}: waste {r['waste_median']}, filter {r['filter_median']}, "
                     f"p (particles) {r['p_all_particles']}, p (image medians) {r['p_image_medians']}")
    return lines


def tem_particles():
    d = [r for r in read("tem_whole_particles.csv") if float(r["arc_visible"]) >= 0.45 and float(r["diameter_um"]) > 10]
    e = read("tem_edge_roughness.csv")
    rows = []
    for s in ("reuse", "waste", "filter"):
        v = [float(r["diameter_um"]) for r in d if r["stream"] == s]
        g = [float(r["rms_roughness_nm"]) for r in e if r["stream"] == s]
        rows.append(dict(stream=s, particles=len(v), diameter_min_um=min(v), diameter_max_um=max(v),
                         edge_arcs=len(g), roughness_min_nm=min(g), roughness_max_nm=max(g)))
    write("tem_whole_particles_summary.csv", rows)
    return [f"TEM whole particles: {sum(r['particles'] for r in rows)} accepted"]


METALS = ["Ni", "Cr", "Fe", "Nb", "Mo", "Ti", "Al", "Ta", "Cu", "Mn", "Co"]
SHOW = ["Ni", "Cr", "Fe", "Nb", "Mo", "Ti", "Al"]


def eds():
    data = read("eds_sum_spectra.csv")
    rows = []
    for s in STREAMS:
        d = {r["element"]: (float(r["wt_pct"]), float(r["sigma"])) for r in data if r["stream"] == s}
        tot = sum(d[e][0] for e in METALS if e in d)
        row = dict(stream=s)
        for e in SHOW:
            row[e] = round(100 * d[e][0] / tot, 2) if e in d else ""
            row[e + "_sigma"] = round(100 * d[e][1] / tot, 2) if e in d else ""
        rows.append(row)
    write("eds_metal_fractions.csv", rows)
    return ["EDS metal fractions written"]


def shirley(x, y, it=60):
    o = np.argsort(x)
    x, y = x[o], y[o]
    lo, hi = y[:6].mean(), y[-6:].mean()
    b = np.linspace(lo, hi, len(y))
    for _ in range(it):
        c = np.cumsum(np.clip(y - b, 0, None))
        if c[-1] <= 0:
            break
        new = lo + (hi - lo) * c / c[-1]
        done = np.max(np.abs(new - b)) < 1e-3 * (abs(hi - lo) + 1)
        b = new
        if done:
            break
    return x, y, b


def gl(x, c, w, m=0.3):
    return (1 - m) * np.exp(-4 * np.log(2) * ((x - c) / w) ** 2) + m / (1 + 4 * ((x - c) / w) ** 2)


FIT = {"O 1s": ((526, 537), [("lattice", 529.8), ("OH", 531.3), ("CO", 532.7)]),
       "Ni 2p": ((849, 868), [("metal", 852.6), ("NiO", 853.9), ("Ni2+", 855.8), ("sat", 861.3)]),
       "Cr 2p": ((571, 581), [("metal", 574.2), ("Cr2O3", 576.3), ("CrOH", 577.3)]),
       "Fe 2p": ((704, 722), [("metal", 707.0), ("Fe2+", 709.6), ("Fe3+", 711.2), ("sat2", 714.8), ("sat3", 719.0)]),
       "Ti 2p": ((452, 462), [("metal", 454.0), ("Ti4+", 458.6)])}
SF = {"C 1s": 1.0, "O 1s": 2.93, "Ti 2p": 5.22, "Cr 2p": 7.69, "Fe 2p": 10.82, "Ni 2p": 14.61, "Mo 3d": 9.5}
AREA = {"Ti 2p": (453, 462), "Cr 2p": (571, 582), "Fe 2p": (704, 717), "Ni 2p": (850, 866), "Mo 3d": (226, 238),
        "O 1s": (526, 537), "C 1s": (281, 292)}


def fit_region(x, y, window, comps):
    a, b = window
    m = (x >= a) & (x <= b)
    xs, ys, bg = shirley(x[m], y[m])
    net = ys - bg
    k = len(comps)

    def model(p):
        return sum(p[1 + 2 * q] * gl(xs, comps[q][1] + p[2 + 2 * q], p[0]) for q in range(k))
    top = max(net.max(), 1)
    p0 = [1.6] + sum(([max(net.max() / 2, 1), 0.0] for _ in comps), [])
    lo = [0.9] + sum(([0, -0.4] for _ in comps), [])
    hi = [2.6] + sum(([top * 3, 0.4] for _ in comps), [])
    p = least_squares(lambda p: model(p) - net, p0, bounds=(lo, hi)).x
    amp = np.array([p[1 + 2 * q] for q in range(k)])
    share = amp / amp.sum() if amp.sum() else amp
    return {c[0]: (float(share[q]), c[1] + float(p[2 + 2 * q])) for q, c in enumerate(comps)}


def xps():
    spec = {}
    for r in read("xps_spectra.csv"):
        spec.setdefault((r["stream"], r["region"]), [[], []])
        spec[(r["stream"], r["region"])][0].append(float(r["binding_energy_eV"]))
        spec[(r["stream"], r["region"])][1].append(float(r["counts"]))
    rows, lines = [], []
    for s in STREAMS:
        R = {reg: (np.array(x), np.array(y)) for (st, reg), (x, y) in spec.items() if st == s}
        x, y = R["C 1s"]
        shift = 284.8 - x[np.argmax(np.convolve(y, np.ones(5) / 5, "same"))]
        area = {}
        for reg, (a, b) in AREA.items():
            xx, yy = R[reg]
            xs, ys, bg = shirley(xx[(xx + shift >= a) & (xx + shift <= b)] + shift, yy[(xx + shift >= a) & (xx + shift <= b)])
            area[reg] = float(np.trapezoid(np.clip(ys - bg, 0, None), xs) / SF[reg])
        fits = {reg: fit_region(R[reg][0] + shift, R[reg][1], w, c) for reg, (w, c) in FIT.items()}
        metals = sum(max(area[e], 0) for e in ("Ti 2p", "Cr 2p", "Fe 2p", "Ni 2p", "Mo 3d"))
        total = metals + area["O 1s"] + area["C 1s"]
        o = fits["O 1s"]
        rows.append(dict(stream=s, binding_energy_shift_eV=round(float(shift), 2),
                         Cr3_percent=round(100 * (1 - fits["Cr 2p"]["metal"][0])),
                         Cr2O3_eV=round(fits["Cr 2p"]["Cr2O3"][1], 1), CrOH3_eV=round(fits["Cr 2p"]["CrOH"][1], 1),
                         Ti4_eV=round(fits["Ti 2p"]["Ti4+"][1], 1),
                         O1s_metal_oxide_percent=round(100 * o["lattice"][0]), O1s_OH_percent=round(100 * o["OH"][0]),
                         O1s_CO_percent=round(100 * o["CO"][0]),
                         Ti_over_Cr=round(max(area["Ti 2p"], 0) / max(area["Cr 2p"], 1e-9), 2),
                         adventitious_C_at_percent=round(100 * area["C 1s"] / total),
                         Ni2_percent=round(100 * (fits["Ni 2p"]["Ni2+"][0] + fits["Ni 2p"]["sat"][0])),
                         O_in_oxide_per_metal=round(area["O 1s"] * (o["lattice"][0] + o["OH"][0]) / metals, 2)))
    write("xps_surface_chemistry.csv", rows)
    for r in rows:
        lines.append(f"XPS {r['stream']}: Cr3+ {r['Cr3_percent']} %, Ti/Cr {r['Ti_over_Cr']}, "
                     f"O 1s oxide/OH/C-O {r['O1s_metal_oxide_percent']}/{r['O1s_OH_percent']}/{r['O1s_CO_percent']} %")
    return lines


if __name__ == "__main__":
    out = sem() + tem_particles() + eds() + xps()
    with open(os.path.join(RES, "summary_surfaces.txt"), "w") as fh:
        fh.write("\n".join(out) + "\n")
    print("\n".join(out))
