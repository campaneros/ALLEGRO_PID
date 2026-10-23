import os
import numpy as np
import uproot

from data import R_IN, M_PI0

# PID_RECO_BASE can point to a local copy of the samples
BASE = os.environ.get("PID_RECO_BASE", "/eos/experiment/drd/drd6/allegro/noble-liquid-cal-rnd/simulation_output/geometry_studies/ALLEGRO_strips/reco")
ENERGIES = [5, 10, 20, 35, 50, 80, 100]
GRID, OFFSET = 0.009817477 / 4, 0.5902785  # theta strip pitch and centre of strip 0 [rad]
DPHI = 2 * np.pi / 1536                    # one electrode (module) in phi
# readout of each design: theta strips merged per layer, module pairs merged per layer (from the k4geo XML)
DESIGNS = {
    "base": ([4, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4], [1] * 11),
    "S3T": ([4, 1, 4, 1, 4, 4, 4, 8, 8, 8, 8], [1] * 7 + [2] * 4),
    "S4T": ([4, 1, 4, 4, 1, 4, 4, 8, 8, 8, 8], [1] * 7 + [2] * 4),
    "S5T": ([4, 1, 4, 4, 4, 1, 4, 8, 8, 8, 8], [1] * 7 + [2] * 4),
    "S4R": ([4, 1, 4, 4, 1, 4, 4, 4], [1] * 8),
}

CL, CC = "AugmentedEMBCaloClusters/AugmentedEMBCaloClusters.", "EMBCaloClusterCells/EMBCaloClusterCells."
CLH, PAR = "_AugmentedEMBCaloClusters_hits/_AugmentedEMBCaloClusters_hits.index", "_MCParticles_parents/_MCParticles_parents.index"
ALL, M = "ECalBarrelModuleThetaMergedPositioned/ECalBarrelModuleThetaMergedPositioned.", "MCParticles/MCParticles."
LK = "CaloHitMCParticleLinks"
BR = [CL + v for v in ("energy", "hits_begin", "hits_end")] + [CLH] \
    + [CC + v for v in ("cellID", "energy", "position.x", "position.y", "position.z")] + [ALL + "cellID"] \
    + [M + v for v in ("momentum.x", "momentum.y", "momentum.z", "parents_begin", "parents_end")] + [PAR] \
    + [f"{LK}/{LK}.weight", f"_{LK}_from/_{LK}_from.index", f"_{LK}_to/_{LK}_to.index"]


def wrap(d):
    return (d + np.pi) % (2 * np.pi) - np.pi


def _event(a, i, is_pi0, E, design):
    # Only reco quantities enter the hits (cells of the leading sliding-window cluster).
    # The truth is used for the label and the training targets frac and entry.
    cl = a[CL + "energy"][i]
    if len(cl) == 0:
        return None
    k = int(np.argmax(cl))
    idx = a[CLH][i][a[CL + "hits_begin"][i][k]:a[CL + "hits_end"][i][k]]
    cid, e = a[CC + "cellID"][i][idx].astype(np.uint64), a[CC + "energy"][i][idx]
    X, Y, Z = (a[CC + "position." + c][i][idx] for c in "xyz")
    ok = e > 0  # the node features use log E
    cid, e, X, Y, Z = cid[ok], e[ok], X[ok], Y[ok], Z[ok]
    rho = np.hypot(X, Y)
    th, ph = np.arctan2(rho, Z), np.arctan2(Y, X)
    lay = ((cid >> np.uint64(11)) & np.uint64(0xFF)).astype(int)

    # local frame: x along theta, y along phi, around the cluster barycentre; theta origin snapped to a strip centre so
    # that strip positions are exact multiples of the strip pitch (shower_shapes bins the strip profile on them)
    th_c = (e * th).sum() / e.sum()
    th0 = OFFSET + GRID * np.round((th_c - OFFSET) / GRID)
    ph0 = np.arctan2((e * np.sin(ph)).sum(), (e * np.cos(ph)).sum())
    x, y, z = R_IN * (th - th0), R_IN * wrap(ph - ph0), rho - R_IN
    nt, nm = (np.array(v)[lay] for v in DESIGNS[design])
    cell = np.stack([R_IN * GRID * nt, R_IN * DPHI * 2 * nm], 1)

    # truth: photon A = most energetic daughter of the pi0 (the photon itself for a gamma); every MC particle is assigned
    # to the daughter it descends from
    p = np.stack([a[M + "momentum." + c][i] for c in "xyz"], 1)
    pb, pe, par = a[M + "parents_begin"][i], a[M + "parents_end"][i], a[PAR][i]
    parent = np.where(pe > pb, par[np.minimum(pb, len(par) - 1)], -1) if len(par) else np.full(len(p), -1)
    if is_pi0:
        dau = np.where(parent == 0)[0]
        if len(dau) < 2:  # pi0 with fewer than two daughters in the MC record: no two-photon truth, event skipped
            return None
        dau = dau[np.argsort(-np.linalg.norm(p[dau], axis=1))]
        A, B = dau[0], dau[1]
    else:
        A = B = 0
    top = np.arange(len(p))
    for _ in range(len(p)):  # climb to the pi0 daughter (or the primary gamma)
        up = (parent[top] > 0) & (top != A) & (top != B)
        if not up.any():
            break
        top = np.where(up, parent[top], top)

    # frac of each cell = share of its link weight that goes to photon A. The links are per sub-cell, so for merged cells
    # this is the plain mean over the sub-cells.
    lcid = a[ALL + "cellID"][i].astype(np.uint64)[a[f"_{LK}_from/_{LK}_from.index"][i]]
    lw, lto = a[f"{LK}/{LK}.weight"][i], a[f"_{LK}_to/_{LK}_to.index"][i]
    u, inv = np.unique(lcid, return_inverse=True)
    wa = np.bincount(inv, lw * (top[lto] == A), len(u)) / np.maximum(np.bincount(inv, lw, len(u)), 1e-12)
    j = np.clip(np.searchsorted(u, cid), 0, len(u) - 1)
    frac = np.where(u[j] == cid, wa[j], 1.0) if is_pi0 else np.ones(len(cid))

    def entry(q):
        return R_IN * (np.arctan2(np.hypot(q[0], q[1]), q[2]) - th0), R_IN * wrap(np.arctan2(q[1], q[0]) - ph0)

    hits = np.stack([x, y, lay, e, nt == 1], 1).astype(np.float32)
    return dict(hits=hits, cell=cell.astype(np.float32), z=z.astype(np.float32), frac=frac.astype(np.float32),
                entry=np.array([entry(p[A]), entry(p[B])], np.float32), y=int(is_pi0), E=float(E),
                m=M_PI0 if is_pi0 else 0.0)


def load(design="base", theta="theta50to130", files=range(8), n_per_point=1000, energies=ENERGIES):
    # n_per_point events for every (particle, energy) point, read in file order from the given file numbers
    out = []
    for E in energies:
        for is_pi0, part in ((0, "gamma"), (1, "pi0")):
            got = 0
            for j in files:
                if got >= n_per_point:
                    break
                a = uproot.open(f"{BASE}/{design}/{theta}/{part}_{E}GeV_{j}_reco.root")["events"].arrays(
                    BR, entry_stop=n_per_point - got, library="np")
                for i in range(len(a[CL + "energy"])):
                    ev = _event(a, i, is_pi0, E, design)
                    if ev is not None:
                        out.append(ev)
                        got += 1
    return out


if __name__ == "__main__":
    evs = load(n_per_point=50, energies=[20])
    g = [e for e in evs if e["y"] == 0]
    p = [e for e in evs if e["y"] == 1]
    assert len(g) == len(p) == 50
    for ev in evs:
        assert ev["hits"][:, 3].min() > 0 and len(ev["hits"]) == len(ev["frac"]) == len(ev["cell"]) == len(ev["z"])
    assert all((e["frac"] == 1).all() for e in g) and np.mean([(e["frac"] < 0.99).any() for e in p]) > 0.9
    eg = np.median([e["hits"][:, 3].sum() for e in g])
    s = np.concatenate([e["hits"][e["hits"][:, 4] == 1, 0] for e in g]) / (R_IN * GRID)
    assert np.allclose(s, np.round(s), atol=1e-2), "strips off the strip grid"
    print("ok", "median gamma cluster E %.2f GeV" % eg, "median hits g/pi0", np.median([len(e["hits"]) for e in g]),
          np.median([len(e["hits"]) for e in p]), "median pi0 opening %.1f mm" % np.median([np.linalg.norm(e["entry"][0] - e["entry"][1]) for e in p]))
