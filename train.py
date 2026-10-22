"""Train the multi-task GNN, train a shower-shape BDT baseline, compare."""
import argparse, json, os, time
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve

from data import make_dataset, CELL_X
from model import event_to_graph, collate, Pi0GammaNet, loss_fn

def shower_shapes(ev, strip_layers):
    x, y, lay, e, strip = ev["hits"].T
    et = e.sum()
    f = []
    f += [e[lay == l].sum() / et for l in range(11)]
    def width(v, w):
        w = w / max(w.sum(), 1e-9); m = (w * v).sum()
        return np.sqrt(max((w * (v - m) ** 2).sum(), 0))
    f += [width(x, e), width(y, e)]
    s = strip == 1
    es = e[s].sum()
    if es > 0:
        xs, exs = x[s], e[s]
        f += [width(xs, exs), width(y[s], exs), exs.max() / es]

        pitch = CELL_X / 4
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
    return float((score[y == 1] <= thr).mean()) if (y == 1).any() else float("nan")

def evaluate(score, y, E, bins):
    out = dict(auc=float(roc_auc_score(y, score)), misid95=misid_at_eff(score, y), per_E=[])
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (E >= lo) & (E < hi)
        out["per_E"].append(dict(lo=float(lo), hi=float(hi), misid95=misid_at_eff(score[m], y[m]), n=int(m.sum())))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--strips", default="1", help="comma list of strip layers, e.g. 1 or 1,2")
    ap.add_argument("--out", default="run")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    strips = tuple(int(s) for s in a.strips.split(","))
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)

    t0 = time.time()
    train = make_dataset(a.n, seed=a.seed, strip_layers=strips)
    test = make_dataset(a.n // 4, seed=a.seed + 1000, strip_layers=strips)
    print(f"generated {len(train)}+{len(test)} events in {time.time()-t0:.0f}s")
    yte = np.array([e["y"] for e in test]); Ete = np.array([e["E"] for e in test])
    bins = np.array([5, 10, 20, 35, 50.01])

    Xtr = np.array([shower_shapes(e, strips) for e in train]); ytr = np.array([e["y"] for e in train])
    Xte = np.array([shower_shapes(e, strips) for e in test])
    bdt = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05).fit(Xtr, ytr)
    s_bdt = bdt.predict_proba(Xte)[:, 1]
    res_bdt = evaluate(s_bdt, yte, Ete, bins)
    print("BDT  auc %.4f  misID@95%% %.4f" % (res_bdt["auc"], res_bdt["misid95"]))

    gtr = [event_to_graph(e) for e in train]; gte = [event_to_graph(e) for e in test]
    net = Pi0GammaNet()
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=a.epochs * (len(gtr) // 64 + 1))
    bs = 64
    for ep in range(a.epochs):
        net.train(); perm = torch.randperm(len(gtr)); acc = {}
        t0 = time.time()
        for i in range(0, len(gtr), bs):
            b = collate([gtr[j] for j in perm[i:i + bs]])
            tot, parts = loss_fn(net(b), b)
            opt.zero_grad(); tot.backward(); opt.step(); sched.step()
            for k, v in parts.items(): acc[k] = acc.get(k, 0) + v
        nb = len(gtr) // bs + 1
        s = predict(net, gte)
        print(f"ep {ep:2d} " + " ".join(f"{k}={v/nb:.3f}" for k, v in acc.items())
              + f"  test auc={roc_auc_score(yte, s):.4f} misID@95%={misid_at_eff(s, yte):.4f}  {time.time()-t0:.0f}s")
    s_gnn = predict(net, gte)
    res_gnn = evaluate(s_gnn, yte, Ete, bins)
    print("GNN  auc %.4f  misID@95%% %.4f" % (res_gnn["auc"], res_gnn["misid95"]))

    torch.save(net.state_dict(), f"{a.out}/model.pt")
    json.dump(dict(strips=strips, n_train=a.n, bdt=res_bdt, gnn=res_gnn), open(f"{a.out}/metrics.json", "w"), indent=1)
    plots(s_bdt, s_gnn, yte, res_bdt, res_gnn, a.out, strips)

@torch.no_grad()
def predict(net, graphs, bs=256):
    net.eval()
    return np.concatenate([torch.sigmoid(net(collate(graphs[i:i + bs]))["logit"]).numpy() for i in range(0, len(graphs), bs)])

def plots(s_bdt, s_gnn, y, rb, rg, out, strips):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4))
    for s, r, lab in [(s_bdt, rb, "BDT shower shapes"), (s_gnn, rg, "GNN multi-task")]:
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(1 - fpr, tpr, label=f"{lab} (AUC {r['auc']:.3f})")
    ax.set_xlabel(r"$\gamma$ efficiency"); ax.set_ylabel(r"$\pi^0$ mis-ID probability"); ax.set_yscale("log")
    ax.axvline(0.95, ls=":", c="gray"); ax.legend(); ax.set_title(f"strips in layers {strips}")
    fig.tight_layout(); fig.savefig(f"{out}/roc.png", dpi=130)
    fig, ax = plt.subplots(figsize=(5, 4))
    for r, lab in [(rb, "BDT"), (rg, "GNN")]:
        c = [(p["lo"] + p["hi"]) / 2 for p in r["per_E"]]
        ax.plot(c, [p["misid95"] for p in r["per_E"]], "o-", label=lab)
    ax.set_xlabel("E [GeV]"); ax.set_ylabel(r"$\pi^0$ mis-ID at 95% $\gamma$ eff."); ax.set_yscale("log"); ax.legend()
    ax.set_title(f"strips in layers {strips}"); fig.tight_layout(); fig.savefig(f"{out}/misid_vs_E.png", dpi=130)

if __name__ == "__main__":
    main()
