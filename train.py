import argparse
import json
import os
import time

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve

from data import make_dataset
from model import event_to_graph, collate, Pi0GammaNet, GravNetPID, loss_fn, POS_SCALE

def shower_shapes(ev):
    # input variables of the BDT baseline
    x, y, lay, e, strip = ev["hits"].T
    et = e.sum()
    f = []
    f += [e[lay == l].sum() / et for l in range(11)]
    def width(v, w):
        w = w / max(w.sum(), 1e-9)
        m = (w * v).sum()
        return np.sqrt(max((w * (v - m) ** 2).sum(), 0))
    f += [width(x, e), width(y, e)]
    s = strip == 1
    es = e[s].sum()
    if es > 0:
        xs, exs = x[s], e[s]
        f += [width(xs, exs), width(y[s], exs), exs.max() / es]

        # strip profile: second local maximum over the maximum, and energy in 3 strips over 7 strips
        pitch = ev["cell"][s, 0].min()  # strip pitch of this geometry
        ix = np.round(xs / pitch).astype(int)
        prof = np.bincount(ix - ix.min(), exs)
        imax = prof.argmax()
        loc = [i for i in range(1, len(prof) - 1) if prof[i] > prof[i - 1] and prof[i] >= prof[i + 1] and i != imax]
        f += [max(prof[loc]) / prof[imax] if loc else 0.0]
        win = lambda r: prof[max(imax - r, 0):imax + r + 1].sum()
        f += [win(1) / max(win(3), 1e-9)]
    else:
        f += [0, 0, 0, 0, 0]
    f += [np.log(et)]
    return f

def misid_at_eff(score, y, eff=0.95):
    thr = np.quantile(score[y == 0], eff)
    return float((score[y == 1] < thr).mean()) if (y == 1).any() else float("nan")  # pi0 below the gamma threshold: called gamma

def evaluate(score, y, E, bins):
    out = dict(auc=float(roc_auc_score(y, score)), misid95=misid_at_eff(score, y), per_E=[])
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (E >= lo) & (E < hi)
        out["per_E"].append(dict(lo=float(lo), hi=float(hi), E=float(E[m].mean()), misid95=misid_at_eff(score[m], y[m]), n=int(m.sum())))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--strips", default="1", help="parametric sample: comma list of strip layers, e.g. 1 or 1,2")
    ap.add_argument("--design", default=None, help="reco sample of this readout design (base, S3T, S4T, S5T, S4R); "
                    "files 0-7 train, 8 validation (epoch choice, monitor), 9 test. Default: parametric generator")
    ap.add_argument("--theta", default="theta50to130", help="reco sample of the validation / test files: theta50to130 or theta90")
    ap.add_argument("--thetas", default=None, help="comma list of reco samples for the training files 0-7 (default: --theta), "
                    "e.g. theta50to130,theta40to140,theta90")
    ap.add_argument("--arch", default="edgeconv", choices=("edgeconv", "gravnet"), help="Pi0GammaNet (EdgeConv, fixed kNN) or "
                    "GravNetPID (learned neighbourhoods)")
    ap.add_argument("--wd", type=float, default=1e-4, help="AdamW weight decay")
    ap.add_argument("--dropout", type=float, default=0.0, help="dropout before the classifier output")
    ap.add_argument("--smooth", type=float, default=0.0, help="label smoothing of the classification loss")
    ap.add_argument("--flip", type=int, default=0, help="1: random theta mirror (x -> -x around the cluster, the z -> -z symmetry "
                    "of the barrel) of half the training events in every batch; phi is NOT mirrored (inclined electrodes)")
    ap.add_argument("--out", default="run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--monitor-every", type=int, default=5, help="epochs between the monitor plots (out/monitor/ and Comet), 0 = none")
    a = ap.parse_args()
    strips = tuple(int(s) for s in a.strips.split(","))
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    exp = None
    if os.environ.get("COMET_API_KEY"):  # log to Comet when a key is available
        import comet_ml
        exp = comet_ml.start(project_name="pid-allegro", workspace="campaneros")
        exp.set_name(os.path.basename(os.path.normpath(a.out)))
        exp.log_parameters(vars(a))

    t0 = time.time()
    if a.design:
        import reco
        npt = 2 * len(reco.ENERGIES)  # number of (particle, energy) points, same number of events in each
        train = []
        for t in (a.thetas.split(",") if a.thetas else [a.theta]):
            train += reco.load(a.design, t, range(8), -(-a.n // npt))
        val = reco.load(a.design, a.theta, (8,), -(-a.n // 8 // npt))
        test = reco.load(a.design, a.theta, (9,), -(-a.n // 8 // npt))
        bins = np.array(reco.ENERGIES[:1] + [np.sqrt(x * y) for x, y in zip(reco.ENERGIES, reco.ENERGIES[1:])] + [1e9])
        tag = f"{a.design} {a.theta}"
    else:
        train = make_dataset(a.n, seed=a.seed, strip_layers=strips)
        val = make_dataset(a.n // 8, seed=a.seed + 2000, strip_layers=strips)
        test = make_dataset(a.n // 8, seed=a.seed + 1000, strip_layers=strips)
        bins = np.array([5, 10, 20, 35, 50.01])
        tag = f"strips in layers {strips}"
    print(f"loaded {len(train)} training + {len(val)} validation + {len(test)} test events in {time.time()-t0:.0f}s", flush=True)
    lab = lambda evs: (np.array([e["y"] for e in evs]), np.array([e["E"] for e in evs]))
    (yva, Eva), (yte, Ete) = lab(val), lab(test)
    # separation of the two photons at the ECAL face [mm] (0 for a single gamma), and in cells of every layer group (geometry of
    # the layers from the hits: theta and phi cell size of each layer); validation events (the monitor)
    Sva = np.array([np.linalg.norm(e["entry"][0] - e["entry"][1]) for e in val])
    geom = layer_cells(train[:500])
    print("layer groups (layers, theta x phi cell [mm]):", [(g[0], g[1], g[2]) for g in geom])
    Dva = np.array([[max(abs(dx) / wt, abs(dy) / wp) for _, wt, wp in geom] for dx, dy in (e["entry"][0] - e["entry"][1] for e in val)])

    Xtr = np.array([shower_shapes(e) for e in train])
    ytr = np.array([e["y"] for e in train])
    bdt = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05).fit(Xtr, ytr)
    # raw scores (decision_function), not probabilities: no ties from rounding in the ROC
    sb_va, s_bdt = (bdt.decision_function(np.array([shower_shapes(e) for e in evs])) for evs in (val, test))
    res_bdt = evaluate(s_bdt, yte, Ete, bins)
    print("BDT  test auc %.4f  misID@95%% %.4f" % (res_bdt["auc"], res_bdt["misid95"]), flush=True)

    gtr, gva, gte = ([event_to_graph(e) for e in evs] for evs in (train, val, test))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", dev)
    net = (GravNetPID if a.arch == "gravnet" else Pi0GammaNet)(dropout=a.dropout).to(dev)
    print(a.arch, sum(p_.numel() for p_ in net.parameters()), "parameters", flush=True)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=a.wd)
    bs = 64
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=a.epochs * (len(gtr) // bs + 1))
    hist = dict(ep=[], auc=[], misid=[])  # per-epoch values for the monitor plots, plus one list per loss term
    best, best_ep, state, ep0 = np.inf, -1, None, 0
    ck = f"{a.out}/checkpoint.pt"  # written after every epoch, so an interrupted training can resume
    if os.path.exists(ck):
        c_ = torch.load(ck, weights_only=False, map_location=dev)
        net.load_state_dict(c_["net"])
        opt.load_state_dict(c_["opt"])
        sched.load_state_dict(c_["sched"])
        best, best_ep, state, hist, ep0 = c_["best"], c_["best_ep"], c_["state"], c_["hist"], c_["ep"] + 1
        print(f"resumed after epoch {c_['ep']} (best validation loss {best:.4f} at epoch {best_ep})", flush=True)
    for ep in range(ep0, a.epochs):
        net.train()
        perm = torch.randperm(len(gtr))
        acc = {}
        t0 = time.time()
        for i in range(0, len(gtr), bs):
            b = to(collate([gtr[j] for j in perm[i:i + bs]]), dev)
            if a.flip:
                flip_x(b)
            tot, parts = loss_fn(net(b), b, smooth=a.smooth)
            opt.zero_grad()
            tot.backward()
            opt.step()
            sched.step()
            for k, v in parts.items():
                acc[k] = acc.get(k, 0) + v
        nb = len(gtr) // bs + 1
        s, vl = test_pass(net, gva, a.smooth)
        vtot = vl["cls"] + vl["frac"] + 0.1 * vl["pos"]  # same weights as in the training loss
        if vtot < best:  # keep the weights of the best validation epoch
            best, best_ep, state = vtot, ep, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        print(f"ep {ep:2d} " + " ".join(f"{k}={v/nb:.3f}" for k, v in acc.items())
              + f"  val loss={vtot:.4f} auc={roc_auc_score(yva, s):.4f} misID@95%={misid_at_eff(s, yva):.4f}  {time.time()-t0:.0f}s", flush=True)
        hist["ep"].append(ep)
        hist["auc"].append(roc_auc_score(yva, s))
        hist["misid"].append(misid_at_eff(s, yva))
        for k, v in acc.items():
            hist.setdefault(f"loss {k}", []).append(v / nb)
        for k, v in vl.items():
            hist.setdefault(f"val loss {k}", []).append(v)
        if exp:
            # the pos loss is a mean squared distance in units of POS_SCALE: convert it to an rms in mm
            exp.log_metrics({**{f"loss {k}": v / nb for k, v in acc.items()}, **{f"val loss {k}": v for k, v in vl.items()},
                             "val loss": vtot, "val auc": hist["auc"][-1], "val misID@95": hist["misid"][-1],
                             "val misID@90": misid_at_eff(s, yva, 0.90), "val entry rms [mm]": np.sqrt(vl["pos"]) * POS_SCALE,
                             "best epoch": best_ep, "lr": sched.get_last_lr()[0]}, epoch=ep, step=ep)
        if a.monitor_every and ((ep + 1) % a.monitor_every == 0 or ep == a.epochs - 1):
            monitor(hist, s, sb_va, yva, Eva, Sva, Dva, geom, bins, a.out, tag, ep, exp)
        torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), best=best, best_ep=best_ep,
                        state=state, hist=hist, ep=ep), ck + ".tmp")
        os.replace(ck + ".tmp", ck)  # never leave a half-written checkpoint
    net.load_state_dict(state)
    s_gnn = predict(net, gte)
    res_gnn = evaluate(s_gnn, yte, Ete, bins)
    print(f"GNN (best validation epoch {best_ep})  test auc %.4f  misID@95%% %.4f" % (res_gnn["auc"], res_gnn["misid95"]), flush=True)

    torch.save({k: v.cpu() for k, v in net.state_dict().items()}, f"{a.out}/model.pt")  # loadable without a GPU
    json.dump(dict(sample=tag, n_train=len(train), n_val=len(val), n_test=len(test), best_epoch=best_ep, args=vars(a),
                   bdt=res_bdt, gnn=res_gnn), open(f"{a.out}/metrics.json", "w"), indent=1)
    plots(s_bdt, s_gnn, yte, res_bdt, res_gnn, a.out, tag)
    if exp:
        exp.log_metrics({f"{m} test {k}": r[k] for m, r in (("BDT", res_bdt), ("GNN", res_gnn)) for k in ("auc", "misid95")})
        for m, r in (("BDT", res_bdt), ("GNN", res_gnn)):  # mis-ID per energy bin, logged with the mean energy as step
            for p in r["per_E"]:
                exp.log_metric(f"{m} test misID@95 vs E", p["misid95"], step=round(p["E"]))
        for f in ("roc.png", "misid_vs_E.png"):
            exp.log_image(f"{a.out}/{f}")
        exp.log_asset(f"{a.out}/metrics.json")
        from comet_ml.integration.pytorch import log_model
        log_model(exp, net, model_name=type(net).__name__)
        exp.end()

def flip_x(b):
    """theta mirror of half the graphs of a collated batch, in place: x -> -x of the node features, positions and entry points
    (x = R (theta - theta0) around the cluster; the barrel is symmetric under z -> -z, so a mirrored shower is a real one)"""
    f = (torch.rand(b["n_graphs"], device=b["x"].device) < 0.5)[b["batch"]]
    sign = 1 - 2 * f.float()
    for k in ("x", "pos", "entry"):
        b[k] = b[k].clone()
        b[k][:, 0] = b[k][:, 0] * sign

def to(b, dev):
    """a collated batch on the device of the network (n_graphs is an int)"""
    return {k: v.to(dev) if torch.is_tensor(v) else v for k, v in b.items()}

@torch.no_grad()
def predict(net, graphs, bs=256):
    net.eval()
    dev = next(net.parameters()).device
    # logits, not sigmoid(logit): in float32 the sigmoid of a large logit is exactly 1.0 and the ties break the ROC
    return np.concatenate([net(to(collate(graphs[i:i + bs]), dev))["logit"].cpu().numpy() for i in range(0, len(graphs), bs)])

def layer_cells(evs):
    """theta and phi cell size [mm, at the ECAL face] of every layer, from the hits of some events; layers with the same cells
    grouped: [(layers, theta size, phi size), ...] (projective cells: the separation in cells does not change with the radius)"""
    lay = np.concatenate([e["hits"][:, 2] for e in evs]).astype(int)
    c = np.concatenate([e["cell"] for e in evs])
    groups = {}
    for l in np.unique(lay):
        groups.setdefault(tuple(np.round(np.median(c[lay == l], 0), 1)), []).append(int(l))
    return [(ls, wt, wp) for (wt, wp), ls in groups.items()]

def misid_vs(x, s, y, edges, eff=0.95, nmin=20):
    """pi0 mis-ID in bins of x: threshold that keeps eff of ALL the photons, pi0 of each bin; nan below nmin pi0 per bin"""
    thr, p = np.quantile(s[y == 0], eff), y == 1
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = p & (x >= lo) & (x < hi)
        out.append((s[m] < thr).mean() if m.sum() >= nmin else np.nan)
    return np.array(out)

@torch.no_grad()
def test_pass(net, graphs, smooth=0.0, bs=256):
    """logits of the validation events and their loss terms (mean over the batches), for the per-epoch monitoring"""
    net.eval()
    dev = next(net.parameters()).device
    s, acc = [], {}
    for i in range(0, len(graphs), bs):
        b = to(collate(graphs[i:i + bs]), dev)
        out = net(b)
        s.append(out["logit"].cpu().numpy())
        for k, v in loss_fn(out, b, smooth=smooth)[1].items():
            acc[k] = acc.get(k, 0) + v
    nb = -(-len(graphs) // bs)
    return np.concatenate(s), {k: v / nb for k, v in acc.items()}

def monitor(hist, s_gnn, s_bdt, y, E, sep, dcell, geom, bins, out, tag, ep, exp):
    """one figure during the training, saved in out/monitor/ and sent to Comet:
    top: loss terms vs epoch (training solid, validation file 8 dashed); validation AUC and pi0 mis-ID (fake rate) at 95% gamma
         efficiency vs epoch; GNN logit distributions of gamma and pi0 at this epoch
    bottom: ROC, fake rate vs E, fake rate vs the two-photon separation at the ECAL face [mm] (GNN at this epoch vs the BDT)
    right column: fake rate vs the separation in cells of each layer group (max of the theta and phi distances in cells), GNN
    (top) and BDT (bottom)"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 4, figsize=(21, 9))
    ax = ax.ravel()[[0, 1, 2, 4, 5, 6, 3, 7]]  # ax[6] and ax[7] are the right column
    for i, k in enumerate(k[5:] for k in hist if k.startswith("loss")):
        ax[0].plot(hist["ep"], hist[f"loss {k}"], "o-", c=f"C{i}", label=f"{k} training")
        ax[0].plot(hist["ep"], hist[f"val loss {k}"], "s--", c=f"C{i}", mfc="none", label=f"{k} validation")
    ax[0].set(xlabel="epoch", ylabel="loss", yscale="log")
    ax[0].legend(fontsize=8, ncol=2)
    ax[1].plot(hist["ep"], hist["misid"], "o-", label=r"$\pi^0$ mis-ID @ 95% $\gamma$ eff.")
    ax[1].plot(hist["ep"], 1 - np.array(hist["auc"]), "s-", label="1 - AUC")
    ax[1].set(xlabel="epoch", yscale="log")
    ax[1].legend(fontsize=8)
    lo, hi = np.percentile(s_gnn, [0.2, 99.8])
    for c, lab in ((0, r"$\gamma$"), (1, r"$\pi^0$")):
        ax[2].hist(np.clip(s_gnn[y == c], lo, hi), 60, (lo, hi), histtype="step", lw=1.5, label=lab)
    ax[2].axvline(np.quantile(s_gnn[y == 0], 0.95), ls=":", c="gray")
    ax[2].set(xlabel="GNN logit (overflow in the last bins)", ylabel="events", yscale="log")
    ax[2].legend()
    rb, rg = evaluate(s_bdt, y, E, bins), evaluate(s_gnn, y, E, bins)
    # fake rate against the separation: gamma threshold from the whole sample, pi0 in quantiles of the separation
    p = y == 1
    edges = np.quantile(sep[p], np.linspace(0, 1, 9))
    xs = 0.5 * (edges[1:] + edges[:-1])
    ce = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64])  # separation in cells
    for j, (s_, lab) in enumerate(((s_gnn, f"GNN epoch {ep}"), (s_bdt, "BDT"))):
        for g, (ls, wt, wp) in enumerate(geom):
            ax[6 + j].plot(0.5 * (ce[1:] + ce[:-1]), misid_vs(dcell[:, g], s_, y, ce), "o-", ms=3,
                           label=(f"layers {ls[0]}-{ls[-1]}" if len(ls) > 1 and ls == list(range(ls[0], ls[-1] + 1))
                                  else f"layers {ls}") + f": {wt:g} x {wp:g} mm")
        ax[6 + j].set(xlabel="separation in cells of the layer", ylabel=rf"$\pi^0$ mis-ID at 95% $\gamma$ eff., {lab}",
                      xscale="log", yscale="log")
        ax[6 + j].legend(fontsize=7)
    for s_, r, lab in [(s_bdt, rb, "BDT"), (s_gnn, rg, f"GNN epoch {ep}")]:
        fpr, tpr, _ = roc_curve(y, s_)
        ax[3].plot(1 - fpr, 1 - tpr, label=lab)  # pi0 mis-ID = pi0 below the threshold = 1 - tpr
        ax[4].plot([q["E"] for q in r["per_E"]], [q["misid95"] for q in r["per_E"]], "o-", label=lab)
        thr = np.quantile(s_[y == 0], 0.95)
        ax[5].plot(xs, [(s_[p & (sep >= lo) & (sep <= hi)] <= thr).mean() for lo, hi in zip(edges[:-1], edges[1:])], "o-", label=lab)
    ax[3].axvline(0.95, ls=":", c="gray")
    ax[3].set(xlabel=r"$\gamma$ efficiency", ylabel=r"$\pi^0$ mis-ID probability", yscale="log")
    ax[3].legend()
    ax[4].set(xlabel="E [GeV]", ylabel=r"$\pi^0$ mis-ID at 95% $\gamma$ eff.", xscale="log", yscale="log")
    ax[4].legend()
    ax[5].set(xlabel=r"$\gamma\gamma$ separation at the ECAL face [mm]", ylabel=r"$\pi^0$ mis-ID at 95% $\gamma$ eff.",
              xscale="log", yscale="log")
    ax[5].legend()
    for ax_ in fig.axes:  # a log axis without any positive value cannot be drawn, use a linear one
        pos = [np.asarray(l_.get_ydata(), float) for l_ in ax_.get_lines()]
        if ax_.get_yscale() == "log" and not ax_.patches and not any(np.any(np.nan_to_num(p_) > 0) for p_ in pos):
            ax_.set_yscale("linear")
    fig.suptitle(f"{tag}, epoch {ep}")
    fig.tight_layout()
    os.makedirs(f"{out}/monitor", exist_ok=True)
    fig.savefig(f"{out}/monitor/epoch{ep:03d}.png", dpi=110)
    if exp:
        exp.log_figure(figure_name="monitor", figure=fig, step=ep)
    plt.close(fig)

def plots(s_bdt, s_gnn, y, rb, rg, out, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4))
    for s, r, lab in [(s_bdt, rb, "BDT shower shapes"), (s_gnn, rg, "GNN multi-task")]:
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(1 - fpr, 1 - tpr, label=f"{lab} (AUC {r['auc']:.3f})")  # pi0 mis-ID = 1 - tpr
    ax.set_xlabel(r"$\gamma$ efficiency")
    ax.set_ylabel(r"$\pi^0$ mis-ID probability")
    ax.set_yscale("log")
    ax.axvline(0.95, ls=":", c="gray")
    ax.legend()
    ax.set_title(tag)
    fig.tight_layout()
    fig.savefig(f"{out}/roc.png", dpi=130)
    fig, ax = plt.subplots(figsize=(5, 4))
    for r, lab in [(rb, "BDT"), (rg, "GNN")]:
        ax.plot([p["E"] for p in r["per_E"]], [p["misid95"] for p in r["per_E"]], "o-", label=lab)
    ax.set_xlabel("E [GeV]")
    ax.set_ylabel(r"$\pi^0$ mis-ID at 95% $\gamma$ eff.")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title(tag)
    fig.tight_layout()
    fig.savefig(f"{out}/misid_vs_E.png", dpi=130)

if __name__ == "__main__":
    main()
