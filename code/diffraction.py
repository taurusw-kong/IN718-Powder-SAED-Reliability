"""Diffraction part of the study: ring statistics, lattice fringes, the phase search against decoys,
the recovery tests and the geometric control for crossed fringes.

Reads the CSV files in ../data and writes tables to ../results.

    python diffraction.py

The random seeds are fixed, so the numbers repeat those in the paper. Takes about fifteen minutes on 8 cores,
most of it for the comparison of five scoring rules.
"""
import csv
import itertools
import os
import re
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
RES = os.path.join(HERE, "..", "results")
os.makedirs(RES, exist_ok=True)
STREAMS = ["virgin", "reuse", "waste", "filter"]


FE = {'H': [[0.202, 30.868], [0.244, 8.544], [0.082, 1.273], [0.0, 0.0]],
      'C': [[0.731, 36.995], [1.195, 11.297], [0.456, 2.814], [0.125, 0.346]],
      'N': [[0.572, 28.847], [1.043, 9.054], [0.465, 2.421], [0.131, 0.317]],
      'O': [[0.455, 23.78], [0.917, 7.622], [0.472, 2.144], [0.138, 0.296]],
      'Al': [[2.276, 72.322], [2.428, 19.773], [0.858, 3.08], [0.317, 0.408]],
      'Si': [[2.129, 57.775], [2.533, 16.476], [0.835, 2.88], [0.322, 0.386]],
      'Ti': [[3.565, 81.982], [2.818, 19.049], [1.893, 3.59], [0.483, 0.386]],
      'Cr': [[2.307, 78.405], [2.334, 15.785], [1.823, 3.157], [0.49, 0.364]],
      'Mn': [[2.747, 67.786], [2.456, 15.674], [1.792, 3.0], [0.498, 0.357]],
      'Fe': [[2.544, 64.424], [2.343, 14.88], [1.759, 2.854], [0.506, 0.35]],
      'Co': [[2.367, 61.431], [2.236, 14.18], [1.724, 2.725], [0.515, 0.344]],
      'Ni': [[2.21, 58.727], [2.134, 13.553], [1.689, 2.609], [0.524, 0.339]],
      'Nb': [[4.237, 27.415], [3.105, 5.074], [1.234, 0.593], [0.0, 0.0]],
      'Mo': [[3.12, 72.464], [3.906, 14.642], [2.361, 3.237], [0.85, 0.366]]}

DMIN, TOL = 1.0, 0.025
ALLOY = set("Ni Cr Fe Nb Mo Ti Al C O H N".split())
MAJOR = set("Ni Fe Cr Nb Mo".split())


def read(name):
    with open(os.path.join(DATA, name)) as fh:
        return list(csv.DictReader(fh))


def write(name, rows, cols=None):
    cols = cols or list(rows[0].keys())
    with open(os.path.join(RES, name), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def elements(formula):
    return {a: float(b or 1) for a, b in re.findall(r"([A-Z][a-z]?)([\d.]*)", formula)}


def plausible(formula):
    e = elements(formula)
    if not e or not set(e) <= ALLOY:
        return False
    if "O" in e:
        return True
    metals = {k: v for k, v in e.items() if k not in ("C", "N", "H")}
    return bool(metals) and max(metals, key=metals.get) in MAJOR


def family(formula):
    e = elements(formula)
    m = [k for k in e if k not in ("O", "H", "C", "N")]
    if "O" in e:
        tag = "hydroxide" if "H" in e else "oxide"
        return f"{m[0]} {tag}" if len(m) == 1 else "mixed " + tag
    if "C" in e:
        return "carbide"
    if "N" in e:
        return "nitride"
    return "intermetallic"


def load_structures():
    atoms = {}
    for r in read("candidate_atoms.csv"):
        atoms.setdefault(r["cod_id"], []).append((r["element"], float(r["occupancy"]), float(r["x"]), float(r["y"]), float(r["z"])))
    lib = {}
    for r in read("candidate_structures.csv"):
        if r["cod_id"] not in atoms:
            continue
        lat = np.array([[float(r[f"a{i}{j}"]) for j in "xyz"] for i in "123"])
        lib[r["cod_id"]] = dict(formula=r["formula"], space_group=r["space_group"], mineral=r["mineral"], lattice=lat, atoms=atoms[r["cod_id"]])
    return lib


def rings(st, scale=1.0):
    lat = st["lattice"] * scale
    rec = np.linalg.inv(lat).T
    nmax = np.ceil(np.linalg.norm(lat, axis=1) / DMIN).astype(int)
    hkl = np.array(list(itertools.product(*[range(-n, n + 1) for n in nmax])), float)
    g = np.linalg.norm(hkl @ rec, axis=1)
    keep = (g > 1e-6) & (g <= 1 / DMIN)
    hkl, g = hkl[keep], g[keep]
    d, s = 1 / g, g / 2
    F = np.zeros(len(g), complex)
    for el, occ, x, y, z in st["atoms"]:
        f = sum(a * np.exp(-b * s * s) for a, b in FE[el])
        F += occ * f * np.exp(-0.5 * s * s) * np.exp(2j * np.pi * (hkl @ np.array([x, y, z])))
    I = np.abs(F) ** 2 * d
    o = np.argsort(-d)
    d, I = d[o], I[o]
    D, W = [], []
    for di, ii in zip(d, I):
        if D and abs(D[-1] - di) / di < 0.003:
            W[-1] += ii
        else:
            D.append(di)
            W.append(ii)
    D, W = np.array(D), np.array(W)
    k = W > 1e-6 * W.max()
    D, W = D[k], W[k]
    return D, W / W.max()


FCC = {"lattice": np.eye(3) * 3.60, "atoms": [("Ni", 1.0, 0, 0, 0), ("Ni", 1.0, .5, .5, 0), ("Ni", 1.0, .5, 0, .5), ("Ni", 1.0, 0, .5, .5)]}
MATRIX = rings(FCC)


def score(dm, pm, cand, smin, smax, tol=TOL, rel="candidate"):
    w = np.sqrt(pm)
    w = w / w.sum()
    Dc, Wc = cand
    sel = (Wc >= 0.10) & (Dc >= 1.2) & (Dc <= 7.0)
    if not sel.any():
        return 0.0, 1.0
    S = np.arange(smin, smax + 1e-9, 0.002)
    Dm = MATRIX[0][MATRIX[1] > 0.05]
    mat = (np.abs(Dm[None, :] - dm[:, None]) / dm[:, None] < tol).any(1)
    Dw = Dc[Wc >= 0.03]
    hit = (np.abs(S[:, None, None] * Dw[None, None, :] - dm[None, :, None]) / dm[None, :, None] < tol).any(2)
    C = ((hit | mat[None, :]) * w[None, :]).sum(1)
    Ds = S[:, None, None] * Dc[sel][None, None, :]
    ref = Ds if rel == "candidate" else dm[None, :, None]
    obs = (np.abs(Ds - dm[None, :, None]) / ref < tol).any(1)
    K = (obs * Wc[sel][None, :]).sum(1) / Wc[sel].sum()
    i = int(np.argmax(C * K))
    return float(C[i] * K[i]), float(S[i])


def window(code):
    return (0.88, 1.12) if code == "F" else (0.93, 1.07)


LIB, RINGS, PAT, ORDER = {}, {}, {}, []


def init(lib, rings_, pat, order):
    LIB.update(lib)
    RINGS.update(rings_)
    PAT.update(pat)
    ORDER.extend(order)


def decoy_task(args):
    cid, seed = args
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(20):
        f = rng.uniform(0.75, 0.88) if rng.random() < 0.5 else rng.uniform(1.14, 1.33)
        r = rings(LIB[cid], f)
        out.append([score(*PAT[c], r, *window(c))[0] for c in ORDER])
    return cid, out


def search(dm, pm, smin, smax, tol=TOL, rel="candidate"):
    return sorted(((score(dm, pm, r, smin, smax, tol, rel)[0], k) for k, r in RINGS.items()), reverse=True)


def recovery_task(args):
    cid, seed = args
    rng = np.random.default_rng(seed)
    D, W = RINGS[cid]
    sel = (D >= 1.2) & (D <= 7.0)
    D, W = D[sel], W[sel]
    if len(D) < 3:
        return None
    n = rng.integers(3, 9)
    o = np.argsort(-W)[:n]
    cal = rng.uniform(0.95, 1.05)
    dm = np.concatenate([D[o] * cal * (1 + rng.normal(0, 0.01, len(o))), [2.078, 1.80, 1.273], [rng.uniform(1.3, 6.5)]])
    pm = np.concatenate([W[o] * 10, [5, 2, 1], [2]])
    ranked = [k for _, k in search(dm, pm, 0.93, 1.07)]
    top = ranked[0]
    return dict(cod_id=cid, family=family(LIB[cid]["formula"]), n_rings=int(n), rank=ranked.index(cid) + 1,
                exact=int(top == cid), family_ok=int(family(LIB[top]["formula"]) == family(LIB[cid]["formula"])),
                top_family=family(LIB[top]["formula"]))


def grid_task(args):
    cid, n, c, tol, seed = args
    rng = np.random.default_rng(seed)
    D, W = RINGS[cid]
    sel = (D >= 1.2) & (D <= 7.0)
    D, W = D[sel], W[sel]
    o = np.argsort(-W)[:n]
    cal = rng.uniform(1 - c, 1 + c)
    dm = np.concatenate([D[o] * cal * (1 + rng.normal(0, 0.01, n)), [2.078, 1.80, 1.273], [rng.uniform(1.3, 6.5)]])
    pm = np.concatenate([W[o] * 10, [5, 2, 1], [2]])
    win = c + 0.005
    ranked = [k for _, k in search(dm, pm, 1 - win, 1 + win, tol, rel="measured")]
    top = ranked[0]
    fam = family(LIB[cid]["formula"])
    return dict(cod_id=cid, n_rings=n, calibration=c, tolerance=tol, rank=ranked.index(cid) + 1, exact=int(top == cid),
                family_ok=int(family(LIB[top]["formula"]) == fam), top_family=family(LIB[top]["formula"]))


def fcc_families():
    fam = {}
    for h, k, l in itertools.product(range(-4, 5), repeat=3):
        if (h, k, l) == (0, 0, 0) or len({h % 2, k % 2, l % 2}) != 1:
            continue
        N = h * h + k * k + l * l
        if N <= 24:
            fam.setdefault(N, []).append(np.array([h, k, l], float))
    ang = {}
    for N1 in fam:
        for N2 in fam:
            a = {round(float(np.degrees(np.arccos(min(abs(u @ v) / np.linalg.norm(u) / np.linalg.norm(v), 1)))), 1)
                 for u in fam[N1] for v in fam[N2]}
            ang[(N1, N2)] = np.array(sorted(t for t in a if t > 5))
    return sorted(fam), ang


def fits_cubic(d1, d2, angle, a0, Ns, ANG, tol=0.03):
    t = min(angle, 180 - angle)
    for N1 in Ns:
        for N2 in Ns:
            a1, a2 = d1 * N1 ** .5, d2 * N2 ** .5
            if abs(a1 / a0 - 1) < tol and abs(a2 / a0 - 1) < tol and len(ANG[(N1, N2)]) and np.min(np.abs(ANG[(N1, N2)] - t)) < 3:
                return True
    return False


METHODS = ["CK", "CKsqrt", "POS", "SOFT", "PROFILE"]
DU = 0.001
U0, U1 = np.log(1 / 8.2), np.log(1 / 1.0)
UG = np.arange(U0, U1, DU)
W0, W1 = np.searchsorted(UG, np.log(1 / 7.0)), np.searchsorted(UG, np.log(1 / 1.2))


def profile(D, I, tol):
    sig = tol / 2
    u = np.log(1 / D)
    m = (u > U0 - 5 * sig) & (u < U1 + 5 * sig)
    return (I[m][:, None] * np.exp(-0.5 * ((UG[None, :] - u[m][:, None]) / sig) ** 2)).sum(0)


def prep(cand):
    Dc, Wc = cand
    k = Wc >= 0.01
    return dict(D=Dc, W=Wc, prof={t: profile(Dc[k], Wc[k], t) for t in (0.025, 0.01)})


def all_scores(dm, pm, c, smin, smax, tol, rel):
    Dc, Wc = c["D"], c["W"]
    sel = (Wc >= 0.10) & (Dc >= 1.2) & (Dc <= 7.0)
    if not sel.any():
        return dict.fromkeys(METHODS, 0.0)
    w = np.sqrt(pm)
    w = w / w.sum()
    S = np.arange(smin, smax + 1e-9, 0.002)
    Dm = MATRIX[0][MATRIX[1] > 0.05]
    mat = (np.abs(Dm[None, :] - dm[:, None]) / dm[:, None] < tol).any(1)
    Dw = Dc[Wc >= 0.03]
    err_m = np.abs(S[:, None, None] * Dw[None, None, :] - dm[None, :, None]) / dm[None, :, None]
    Ds = S[:, None, None] * Dc[sel][None, None, :]
    ref = Ds if rel == "candidate" else dm[None, :, None]
    err_c = np.abs(Ds - dm[None, :, None]) / ref
    expl = (err_m < tol).any(2) | mat[None, :]
    obs = (err_c < tol).any(1)
    Ws = Wc[sel]
    C = (expl * w[None, :]).sum(1)
    sig = tol / 2
    gm = np.maximum(np.exp(-0.5 * (err_m.min(2) / sig) ** 2), mat[None, :].astype(float))
    gc = np.exp(-0.5 * (err_c.min(1) / sig) ** 2)
    out = dict(CK=float((C * (obs * Ws).sum(1) / Ws.sum()).max()),
               CKsqrt=float((C * (obs * np.sqrt(Ws)).sum(1) / np.sqrt(Ws).sum()).max()),
               POS=float((expl.mean(1) * obs.mean(1)).max()),
               SOFT=float(((gm * w[None, :]).sum(1) * (gc * Ws[None, :]).sum(1) / Ws.sum()).max()),
               PROFILE=0.0)
    if (~mat).any():
        M = profile(dm[~mat], w[~mat], tol)[W0:W1]
        kmin, kmax = int(np.floor(-np.log(smax) / DU)), int(np.ceil(-np.log(smin) / DU))
        win = np.lib.stride_tricks.sliding_window_view(c["prof"][tol], W1 - W0)[W0 + kmin:W0 + kmax + 1]
        norms = np.sqrt((win * win).sum(1)) * np.linalg.norm(M)
        out["PROFILE"] = float(np.max(np.where(norms > 0, (win @ M) / np.maximum(norms, 1e-300), 0)))
    return out


PREP = {}


def init_scores(lib, prep_, pat, order):
    init(lib, {}, pat, order)
    PREP.update(prep_)


def rank_all(dm, pm, smin, smax, tol, rel, true_id):
    sc = {m: {} for m in METHODS}
    for k in PREP:
        s = all_scores(dm, pm, PREP[k], smin, smax, tol, rel)
        for m in METHODS:
            sc[m][k] = s[m]
    fam = family(LIB[true_id]["formula"])
    res = {}
    for m in METHODS:
        top = max(sc[m].values())
        tied = [k for k, v in sc[m].items() if v >= top - 1e-12]
        res[m + "_exact"] = (1 / len(tied)) if true_id in tied else 0.0
        res[m + "_family"] = float(np.mean([family(LIB[k]["formula"]) == fam for k in tied]))
        res[m + "_rank"] = int(sum(v > sc[m][true_id] + 1e-12 for v in sc[m].values())) + 1
        res[m + "_top"] = family(LIB[max(sc[m], key=sc[m].get)]["formula"])
    return res


def synthetic(cid, n, cal, rng):
    D, W = PREP[cid]["D"], PREP[cid]["W"]
    sel = (D >= 1.2) & (D <= 7.0)
    D, W = D[sel], W[sel]
    o = np.argsort(-W)[:n]
    dm = np.concatenate([D[o] * cal * (1 + rng.normal(0, 0.01, len(o))), [2.078, 1.80, 1.273], [rng.uniform(1.3, 6.5)]])
    return dm, np.concatenate([W[o] * 10, [5, 2, 1], [2]])


def grid_all_task(args):
    cid, n, c, tol, seed = args
    rng = np.random.default_rng(seed)
    dm, pm = synthetic(cid, n, rng.uniform(1 - c, 1 + c), rng)
    return dict(cod_id=cid, n_rings=n, calibration=c, tolerance=tol, **rank_all(dm, pm, 1 - c - 0.005, 1 + c + 0.005, tol, "measured", cid))


def recovery_all_task(args):
    cid, seed = args
    rng = np.random.default_rng(seed)
    if ((PREP[cid]["D"] >= 1.2) & (PREP[cid]["D"] <= 7.0)).sum() < 3:
        return None
    n = int(rng.integers(3, 9))
    dm, pm = synthetic(cid, n, rng.uniform(0.95, 1.05), rng)
    return dict(cod_id=cid, n_rings=n, **rank_all(dm, pm, 0.93, 1.07, 0.025, "candidate", cid))


def real_all_task(cid):
    return cid, {c: all_scores(*PAT[c], PREP[cid], *window(c), 0.025, "candidate") for c in ORDER}


def decoy_all_task(args):
    cid, seed = args
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(20):
        f = rng.uniform(0.75, 0.88) if rng.random() < 0.5 else rng.uniform(1.14, 1.33)
        pr = prep(rings(LIB[cid], f))
        out.append({c: all_scores(*PAT[c], pr, *window(c), 0.025, "candidate") for c in ORDER})
    return cid, out


def compare_scores(lib, rings_lib, PATS, order, ids, out):
    prep_ = {k: prep(rings_lib[k]) for k in ids}
    eligible = lambda n: [k for k in ids if ((rings_lib[k][0] >= 1.2) & (rings_lib[k][0] <= 7.0)).sum() >= n]
    share = lambda rows, m, key: 100 * float(np.mean([r[f"{m}_{key}"] for r in rows]))
    with Pool(initializer=init_scores, initargs=(lib, prep_, PATS, order)) as p:
        rng = np.random.default_rng(101)
        tasks, seed = [], 10_000_000
        for n in (3, 5, 8, 12):
            for c in (0.0, 0.01, 0.02, 0.05):
                for k in rng.choice(eligible(n), size=38, replace=False):
                    tasks.append((k, n, c, 0.025, seed))
                    seed += 1
        selection = p.map(grid_all_task, tasks, chunksize=2)
        rng = np.random.default_rng(7)
        pick = rng.choice(ids, size=min(300, len(ids)), replace=False)
        recovery = [r for r in p.map(recovery_all_task, [(k, i) for i, k in enumerate(pick)], chunksize=2) if r]
        rng = np.random.default_rng(5)
        conds = [(n, c, 0.025) for n in (3, 5, 8, 12) for c in (0.0, 0.01, 0.02, 0.05)] + \
                [(n, c, 0.01) for n in (3, 5, 8, 12) for c in (0.0, 0.01)]
        tasks, seed = [], 0
        for n, c, tol in conds:
            elig = eligible(n)
            for k in rng.choice(elig, size=min(150, len(elig)), replace=False):
                tasks.append((k, n, c, tol, seed))
                seed += 1
        grid = p.map(grid_all_task, tasks, chunksize=4)
        real = dict(p.map(real_all_task, ids, chunksize=8))
        dec = dict(p.map(decoy_all_task, [(k, i) for i, k in enumerate(ids)], chunksize=2))
    sel_best = max(METHODS, key=lambda m: share(selection, m, "exact"))
    chosen = max([m for m in METHODS if share(selection, sel_best, "exact") - share(selection, m, "exact") <= 1.0],
                 key=lambda m: share(selection, m, "family"))
    g25 = [r for r in grid if r["tolerance"] == 0.025]
    rows, top5 = [], []
    for m in METHODS:
        passed = []
        for j, c in enumerate(order):
            sc = sorted(((real[k][c][m], k) for k in ids), reverse=True)
            bd = np.array([max(dec[k][i][c][m] for k in ids) for i in range(20)])
            S, k = sc[0]
            pv = float((bd >= S).mean())
            top5.append(dict(rule=m, pattern=c, cod_id=k, formula=lib[k]["formula"], space_group=lib[k]["space_group"],
                             family=family(lib[k]["formula"]), score=round(S, 4), p_value=round(pv, 2),
                             median_best_decoy=round(float(np.median(bd)), 4)))
            if pv <= 0.05:
                passed.append(c)
        rows.append(dict(rule=m, chosen=int(m == chosen),
                         selection_structure_pct=round(share(selection, m, "exact"), 1), selection_family_pct=round(share(selection, m, "family"), 1),
                         recovery_structure_pct=round(share(recovery, m, "exact"), 1), recovery_family_pct=round(share(recovery, m, "family"), 1),
                         grid_structure_pct=round(share(g25, m, "exact"), 1), grid_family_pct=round(share(g25, m, "family"), 1),
                         measured_p_le_005=len(passed), measured_patterns=" ".join(passed)))
    write("scoring_rules_compared.csv", rows)
    write("phase_search_all_rules.csv", top5)
    grows = []
    for n, c, tol in conds:
        v = [r for r in grid if r["n_rings"] == n and r["calibration"] == c and r["tolerance"] == tol]
        row = dict(n_rings=n, calibration=c, tolerance=tol, patterns=len(v))
        for m in METHODS:
            row[m + "_exact"] = round(float(np.mean([r[m + "_exact"] for r in v])), 3)
            row[m + "_family"] = round(float(np.mean([r[m + "_family"] for r in v])), 3)
        grows.append(row)
    write("reliability_grid_all_rules.csv", grows)
    out.append(f"Scoring rules (grid at 2.5 %: structure / family first, measured patterns with p <= 0.05), chosen: {chosen}")
    out += [f"  {r['rule']:8s} {r['grid_structure_pct']:5.1f} {r['grid_family_pct']:5.1f}  {r['measured_p_le_005']} of {len(order)} {r['measured_patterns']}" for r in rows]


def main():
    out = []


    rings_ = read("diffraction_rings.csv")
    rows = []
    for s in STREAMS:
        v = [float(r["n_eff"]) for r in rings_ if r["stream"] == s and r["n_eff"]]
        rows.append(dict(stream=s, rings=len(v), median_n_eff=round(float(np.median(v)), 1), min_n_eff=min(v), max_n_eff=max(v)))
    write("ring_spottiness_by_stream.csv", rows)
    out.append("Effective diffracting spots per ring (median): " + ", ".join(f"{r['stream']} {r['median_n_eff']}" for r in rows))


    fr = read("lattice_fringes.csv")
    rows = [dict(stream=s, fringes=sum(r["stream"] == s for r in fr), above_4A=sum(r["stream"] == s and float(r["d_A"]) >= 4 for r in fr))
            for s in STREAMS]
    write("lattice_fringes_by_stream.csv", rows)
    out.append(f"Lattice fringes: {len(fr)} in total; above 4 A: " + ", ".join(f"{r['stream']} {r['above_4A']}" for r in rows))
    gs = read("dark_feature_size.csv")
    rows = [dict(stream=s, images=sum(r["stream"] == s for r in gs),
                 median_d50_nm=round(float(np.median([float(r["d50_nm"]) for r in gs if r["stream"] == s])), 1)) for s in STREAMS]
    write("dark_feature_size_by_stream.csv", rows)


    pat = {}
    for r in rings_:
        pat.setdefault(r["pattern"], []).append((float(r["d_A"]), float(r["prominence"])))
    order = list(dict.fromkeys(r["pattern"] for r in rings_))
    PATS = {k: (np.array([a for a, b in v]), np.array([b for a, b in v])) for k, v in pat.items()}


    lib_all = load_structures()
    lib = {k: v for k, v in sorted(lib_all.items()) if plausible(v["formula"]) and len(v["atoms"]) <= 400}
    print(f"structures: {len(lib_all)}, plausible: {len(lib)}", flush=True)
    with Pool(initializer=init, initargs=({}, {}, {}, [])) as p:
        rr = p.map(rings, list(lib.values()))
    rings_lib = {k: r for k, r in zip(lib, rr) if len(r[0])}
    lib = {k: lib[k] for k in rings_lib}
    ids = list(lib)
    out.append(f"Candidate structures: {len(lib_all)} collected, {len(lib)} plausible")

    with Pool(initializer=init, initargs=(lib, rings_lib, PATS, order)) as p:

        real = {c: sorted(((score(*PATS[c], rings_lib[k], *window(c))[0], k) for k in ids), reverse=True) for c in order}
        dec = dict(p.map(decoy_task, [(k, i) for i, k in enumerate(ids)]))
        best_decoy = np.array([dec[k] for k in ids]).max(0)
        rows = []
        for j, c in enumerate(order):
            for rank, (S, k) in enumerate(real[c][:5], 1):
                rows.append(dict(pattern=c, rank=rank, cod_id=k, formula=lib[k]["formula"], space_group=lib[k]["space_group"],
                                 family=family(lib[k]["formula"]), score=round(S, 4),
                                 p_value=round(float((best_decoy[:, j] >= S).mean()), 2),
                                 median_best_decoy=round(float(np.median(best_decoy[:, j])), 3)))
        write("phase_search_top5.csv", rows)
        sig = [r for r in rows if r["rank"] == 1 and r["p_value"] <= 0.05]
        out.append(f"Patterns matched better than decoys (p <= 0.05): {len(sig)} of {len(order)}: "
                   + ", ".join(f"{r['pattern']} {r['formula']} (S={r['score']})" for r in sig))


        rng = np.random.default_rng(7)
        pick = rng.choice(ids, size=min(300, len(ids)), replace=False)
        rec = [r for r in p.map(recovery_task, [(k, i) for i, k in enumerate(pick)]) if r]
        write("recovery_test.csv", rec)
        out.append(f"Recovery test: {len(rec)} synthetic patterns, correct structure first {100 * np.mean([r['exact'] for r in rec]):.1f} %, "
                   f"correct family first {100 * np.mean([r['family_ok'] for r in rec]):.1f} %, "
                   f"top five {100 * np.mean([r['rank'] <= 5 for r in rec]):.1f} %")


        rng = np.random.default_rng(5)
        conds = [(n, c, 0.025) for n in (3, 5, 8, 12) for c in (0.0, 0.01, 0.02, 0.05)] + \
                [(n, c, 0.01) for n in (3, 5, 8, 12) for c in (0.0, 0.01)]
        tasks, seed = [], 0
        for n, c, tol in conds:
            elig = [k for k in ids if ((rings_lib[k][0] >= 1.2) & (rings_lib[k][0] <= 7.0)).sum() >= n]
            for k in rng.choice(elig, size=min(150, len(elig)), replace=False):
                tasks.append((k, n, c, tol, seed))
                seed += 1
        grid = p.map(grid_task, tasks, chunksize=4)
    write("reliability_grid_patterns.csv", grid)
    rows = []
    for n, c, tol in conds:
        v = [r for r in grid if r["n_rings"] == n and r["calibration"] == c and r["tolerance"] == tol]
        rows.append(dict(n_rings=n, calibration=c, tolerance=tol, patterns=len(v), exact=round(float(np.mean([r["exact"] for r in v])), 3),
                         family=round(float(np.mean([r["family_ok"] for r in v])), 3), top5=round(float(np.mean([r["rank"] <= 5 for r in v])), 3)))
    write("reliability_grid.csv", rows)
    out.append(f"Reliability grid: {len(grid)} synthetic patterns (150 per condition)")


    Ns, ANG = fcc_families()
    pairs = read("fringe_pairs.csv")
    lattices = {"fcc matrix": 3.60, "NiO": 4.177, "TiN": 4.24, "FeO": 4.30, "TiC": 4.33, "NbN": 4.39, "NbC": 4.47, "spinel": 8.35}
    rng = np.random.default_rng(1)
    rows = []
    for s in STREAMS:
        C = [(float(r["d1_A"]), float(r["d2_A"]), float(r["angle_deg"])) for r in pairs if r["stream"] == s]
        for name, a0 in lattices.items():
            real = sum(fits_cubic(*c, a0, Ns, ANG) for c in C)
            exchanged = []
            for _ in range(20):
                perm = rng.permutation(len(C))
                n = 0
                for c, j in zip(C, perm):
                    o = C[j]
                    d1, d2 = sorted([c[0], o[1]], reverse=True)
                    n += fits_cubic(d1, d2, o[2], a0, Ns, ANG)
                exchanged.append(n)
            rows.append(dict(stream=s, crystals=len(C), lattice=name, a_A=a0, real_pairs=real,
                             exchanged_mean=round(float(np.mean(exchanged)), 1), exchanged_sd=round(float(np.std(exchanged)), 1)))
    write("crossed_fringes_geometric_control.csv", rows)
    f = [r for r in rows if r["stream"] == "filter" and r["lattice"] in ("FeO", "NbC")]
    out.append("Geometric control, filter: " + ", ".join(f"{r['lattice']} {r['real_pairs']} real vs {r['exchanged_mean']} +- {r['exchanged_sd']} exchanged" for r in f))

    compare_scores(lib, rings_lib, PATS, order, ids, out)

    with open(os.path.join(RES, "summary_diffraction.txt"), "w") as fh:
        fh.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
