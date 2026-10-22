"""Parametric EM-shower generator on an ALLEGRO-like strip ECAL."""
import numpy as np

R_IN = 2160.0
X0 = 14.0
RM = 40.0
EC = 0.010
N_LAYERS = 11
LAYER_X0 = np.array([1.5] + [3.6] * 10)
LAYER_Z = np.concatenate([[0], np.cumsum(LAYER_X0)]) * X0
CELL_X = R_IN * 10e-3
CELL_Y = R_IN * 8.2e-3
NOISE_CELL = 0.004
THRESHOLD = 2.0
M_PI0 = 0.1349768

def pitch_x(layer, strip_layers):
    return CELL_X / 4 if layer in strip_layers else CELL_X

def _shower_quanta(rng, E, x0, y0, tx, ty, n=3000):

    t_conv = rng.exponential(9 / 7)

    tmax = max(np.log(E / EC) - 0.5, 1.0)
    b = 0.5
    a = b * tmax + 1
    a = max(a * rng.normal(1, 0.05), 1.1)
    t = t_conv + rng.gamma(a, 1 / b, n)

    core = rng.random(n) < 0.75
    scale = np.where(core, 0.3, 1.0) * RM * (0.6 + 0.4 * t / tmax)
    r = rng.exponential(scale)
    ang = rng.uniform(0, 2 * np.pi, n)
    z = t * X0
    x = x0 + tx * z + r * np.cos(ang)
    y = y0 + ty * z + r * np.sin(ang)
    w = np.full(n, E / n)
    return x, y, z, w

def _bin(x, y, z, w, src, strip_layers):
    layer = np.searchsorted(LAYER_Z, z, side="right") - 1
    ok = (layer >= 0) & (layer < N_LAYERS)
    x, y, layer, w, src = x[ok], y[ok], layer[ok], w[ok], src[ok]
    px = np.array([pitch_x(l, strip_layers) for l in range(N_LAYERS)])[layer]
    ix = np.floor(x / px).astype(int)
    iy = np.floor(y / CELL_Y).astype(int)
    key = np.stack([layer, ix, iy], 1)
    ukey, inv = np.unique(key, axis=0, return_inverse=True)
    e = np.bincount(inv, w, len(ukey))
    e_a = np.bincount(inv, w * (src == 0), len(ukey))
    frac = np.where(e > 0, e_a / np.maximum(e, 1e-12), 1.0)
    lay = ukey[:, 0]
    pxu = np.array([pitch_x(l, strip_layers) for l in lay])
    cx = (ukey[:, 1] + 0.5) * pxu
    cy = (ukey[:, 2] + 0.5) * CELL_Y
    return cx, cy, lay, e, frac, pxu

def make_event(rng, is_pi0, E, strip_layers=(1,), tilt=0.0):
    x0 = rng.uniform(-CELL_X / 2, CELL_X / 2)
    y0 = rng.uniform(-CELL_Y / 2, CELL_Y / 2)
    tx, ty = np.tan(tilt), 0.0
    if is_pi0:

        c = rng.uniform(-1, 1)
        beta = np.sqrt(1 - (M_PI0 / E) ** 2)
        gam = E / M_PI0
        e1 = gam * M_PI0 / 2 * (1 + beta * c)
        e2 = E - e1

        th1 = np.arctan2(np.sqrt(1 - c * c), gam * (c + beta))
        th2 = np.arctan2(np.sqrt(1 - c * c), gam * (-c + beta))
        az = rng.uniform(0, 2 * np.pi)
        d = [(e1, th1, az), (e2, th2, az + np.pi)]
        parts, entries = [], []
        for src, (e, th, a) in enumerate(d):
            dx, dy = np.tan(th) * np.cos(a), np.tan(th) * np.sin(a)
            ex, ey = x0 + R_IN * dx, y0 + R_IN * dy
            entries.append((ex, ey))
            q = _shower_quanta(rng, e, ex, ey, tx + dx, ty + dy)
            parts.append(q + (np.full(len(q[0]), src),))
        x, y, z, w, src = [np.concatenate([p[i] for p in parts]) for i in range(5)]
        m = M_PI0

        if e2 > e1:
            src = 1 - src
            entries = entries[::-1]
    else:
        x, y, z, w = _shower_quanta(rng, E, x0, y0, tx, ty)
        src = np.zeros(len(x), int)
        entries = [(x0, y0), (x0, y0)]
        m = 0.0
    cx, cy, lay, e, frac, px = _bin(x, y, z, w, src, strip_layers)

    sig = NOISE_CELL * np.sqrt(px / CELL_X)
    e = e + rng.normal(0, sig)
    keep = e > THRESHOLD * sig
    hits = np.stack([cx, cy, lay, e, (px < CELL_X).astype(float)], 1)[keep].astype(np.float32)
    return dict(hits=hits, frac=frac[keep].astype(np.float32), entry=np.array(entries, np.float32),
                y=int(is_pi0), E=float(E), m=float(m))

def make_dataset(n, seed=0, emin=5.0, emax=50.0, strip_layers=(1,), tilt=0.0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        E = np.exp(rng.uniform(np.log(emin), np.log(emax)))
        out.append(make_event(rng, i % 2, E, strip_layers, tilt))
    return out

if __name__ == "__main__":
    rng = np.random.default_rng(1)
    g = make_event(rng, 0, 20.0)
    p = make_event(rng, 1, 20.0)
    assert abs(g["hits"][:, 3].sum() - 20) < 3, "gamma energy not contained"
    assert abs(p["hits"][:, 3].sum() - 20) < 3, "pi0 energy not contained"
    assert np.all(g["frac"] == 1) and (p["frac"] < 1).any(), "fractions wrong"
    assert (g["hits"][:, 4] == 1).any(), "no strip hits"
    print("ok", len(g["hits"]), "gamma hits", len(p["hits"]), "pi0 hits,",
          "pi0 opening", np.linalg.norm(p["entry"][0] - p["entry"][1]).round(1), "mm")
