"""Multi-task GNN for gamma / pi0 discrimination on calorimeter hit graphs."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import LAYER_Z, CELL_X, M_PI0

K = 8
POS_SCALE = 50.0

def event_to_graph(ev):
    h = ev["hits"]
    x, y, lay, e, strip = h.T
    z = (LAYER_Z[lay.astype(int)] + LAYER_Z[lay.astype(int) + 1]) / 2
    pos = np.stack([x, y, z], 1) / POS_SCALE
    etot = e.sum()
    feat = np.concatenate([pos, np.log(e)[:, None], strip[:, None], (e / etot)[:, None],
                           (lay / 10.0)[:, None]], 1).astype(np.float32)
    pos_t = torch.tensor(pos, dtype=torch.float32)
    n = len(h)
    k = min(K, n - 1)
    d = torch.cdist(pos_t, pos_t)
    nbr = d.topk(k + 1, largest=False).indices[:, 1:]
    src = torch.arange(n).repeat_interleave(k)
    dst = nbr.reshape(-1)

    dom = (ev["frac"] < 0.5).astype(int)
    entry = ev["entry"][dom] / POS_SCALE
    return dict(x=torch.tensor(feat), pos=pos_t, edge=torch.stack([src, dst]),
                e=torch.tensor(e, dtype=torch.float32),
                frac=torch.tensor(np.maximum(ev["frac"], 1 - ev["frac"]), dtype=torch.float32),
                entry=torch.tensor(entry, dtype=torch.float32),
                y=torch.tensor([ev["y"]], dtype=torch.float32),
                m=torch.tensor([ev["m"] / M_PI0], dtype=torch.float32),
                E=torch.tensor([ev["E"]], dtype=torch.float32))

def collate(graphs):
    out, off = {}, 0
    batch = []
    for i, g in enumerate(graphs):
        n = len(g["x"])
        batch.append(torch.full((n,), i, dtype=torch.long))
        for k, v in g.items():
            out.setdefault(k, []).append(v + off if k == "edge" else v)
        off += n
    out = {k: torch.cat(v, 1 if k == "edge" else 0) for k, v in out.items()}
    out["batch"] = torch.cat(batch)
    out["n_graphs"] = len(graphs)
    return out

def mlp(i, o, h=64):
    return nn.Sequential(nn.Linear(i, h), nn.LeakyReLU(), nn.Linear(h, o), nn.LeakyReLU())

class EdgeConv(nn.Module):

    def __init__(self, din, dout):
        super().__init__()
        self.phi = mlp(2 * din + 3, dout)

    def forward(self, x, pos, edge):
        s, d = edge
        msg = self.phi(torch.cat([x[s], x[d] - x[s], pos[d] - pos[s]], 1))
        out = torch.zeros(len(x), msg.shape[1], device=x.device).index_add_(0, s, msg)
        cnt = torch.zeros(len(x), 1, device=x.device).index_add_(0, s, torch.ones(len(s), 1, device=x.device))
        return out / cnt.clamp(min=1)

class Pi0GammaNet(nn.Module):
    def __init__(self, din=7, h=32, layers=3):
        super().__init__()
        self.enc = mlp(din, h)
        self.convs = nn.ModuleList()
        d = h
        for _ in range(layers):
            self.convs.append(EdgeConv(d, h))
            d += h
        self.node_head = nn.Sequential(mlp(d, h), nn.Linear(h, 3))
        self.graph_head = nn.Sequential(mlp(2 * d, h), nn.Linear(h, 1))

    def forward(self, b):
        x, pos, edge, batch, n = b["x"], b["pos"], b["edge"], b["batch"], b["n_graphs"]
        h = self.enc(x)
        for conv in self.convs:
            h = torch.cat([h, conv(h, pos, edge)], 1)
        node = self.node_head(h)
        frac = 0.5 + 0.5 * torch.sigmoid(node[:, 0])
        entry = node[:, 1:]

        w = b["e"] / torch.zeros(n, device=x.device).index_add_(0, batch, b["e"])[batch]
        mean = torch.zeros(n, h.shape[1], device=x.device).index_add_(0, batch, h * w[:, None])
        mx = torch.full((n, h.shape[1]), -1e9, device=x.device).scatter_reduce_(
            0, batch[:, None].expand_as(h), h, "amax", include_self=False)
        g = self.graph_head(torch.cat([mean, mx], 1))
        return dict(logit=g[:, 0], frac=frac, entry=entry)

def loss_fn(out, b, w_frac=1.0, w_pos=0.1):
    y = b["y"]
    l_cls = F.binary_cross_entropy_with_logits(out["logit"], y)
    pi0_node = y[b["batch"]] > 0.5
    if pi0_node.any():
        we = torch.sqrt(b["e"] * b["frac"])[pi0_node]
        l_frac = (we * (out["frac"] - b["frac"])[pi0_node] ** 2).sum() / we.sum()
        l_pos = ((out["entry"] - b["entry"])[pi0_node] ** 2).sum(1).mean()
    else:
        l_frac = l_pos = torch.zeros((), device=y.device)
    tot = l_cls + w_frac * l_frac + w_pos * l_pos
    return tot, dict(cls=l_cls.item(), frac=l_frac.item(), pos=l_pos.item())

if __name__ == "__main__":
    from data import make_dataset
    gs = [event_to_graph(e) for e in make_dataset(8, seed=3)]
    b = collate(gs)
    net = Pi0GammaNet()
    out = net(b)
    assert out["logit"].shape == (8,) and out["frac"].shape == (len(b["x"]),)
    assert (out["frac"] >= 0.5).all() and (out["frac"] <= 1).all()
    tot, parts = loss_fn(out, b)
    tot.backward()
    assert torch.isfinite(tot) and all(p.grad is not None for p in net.parameters() if p.requires_grad)
    print("ok", {k: round(v, 3) for k, v in parts.items()}, "params", sum(p.numel() for p in net.parameters()))
