# π⁰ / γ identification with a multi-task GNN — ALLEGRO ECAL

Three files: `data.py` (event generator), `model.py` (network and loss), `train.py` (training, baseline, metrics, plots). Plain PyTorch + scikit-learn, no torch_geometric, so it runs unchanged on lxplus-gpu.

```
python train.py --n 12000 --epochs 12 --strips 1 --out run_L1
python train.py --n 12000 --epochs 12 --strips 1,2 --out run_L12   # strip-segmentation variant
```

Each run writes `model.pt`, `metrics.json`, `roc.png`, `misid_vs_E.png`.

## 1. Why this design

Reference points (see conversation sources):

- FCC-ee LAr ECAL study: π⁰ mis-ID 10 % at 95 % γ efficiency with a hybrid dense+CNN network. This is the number to beat.
- SOOP (Melennec & Magniette, arXiv:2609.34500): per-hit "dominant energy fraction + entry point" regression disentangles two overlapping photon showers, with static kNN message passing matching object-condensation performance at lower cost. Trained only on photon pairs with 1°–4.5° separation and clustered post hoc with a non-differentiable GMM.
- Transformer clustering (arXiv:2603.18172): a GNN trained only on pairs splits single photons into two clusters.

Consequences:

- Segment-then-classify is avoided: a segmenter that never saw single photons biases the classifier upstream.
- The classifier is the main head, trained end-to-end on γ and π⁰.
- SOOP's per-hit targets are kept as an auxiliary loss, active only on π⁰ events, so the shared encoder is forced to learn two-shower substructure without an external clustering step.
- No mass-regression head: a pure π⁰ sample has a constant mass, nothing to learn. Add it (CMS "domain continuation", arXiv:2204.12313) only for an η/π⁰/ALP mixture.
- Energy range 5–50 GeV, log-uniform: the minimum π⁰ opening angle is 2 m_π/E, i.e. 117 mm at 5 GeV and 12 mm (≈ 2 strips) at 50 GeV at R = 2.16 m. This is the regime below SOOP's 1° floor where strip PID matters.

## 2. `data.py` — parametric shower generator

Stands in for the k4geo full simulation. Output record per event:

| key | shape | content |
|---|---|---|
| `hits` | (N, 5) | x [mm], y [mm], layer, E [GeV], is_strip |
| `frac` | (N,) | fraction of the hit energy from the leading photon (1 for γ) |
| `entry` | (2, 2) | (x, y) entry point on the face of photon A (leading) and B |
| `y` | int | 0 = γ, 1 = π⁰ |
| `E` | float | incident energy [GeV] |
| `m` | float | diphoton mass [GeV], 0 for γ |

Coordinates: x = R·Δθ (strips run along θ), y = R·Δφ, z = depth. Replace `make_event` with a reader producing the same dict and nothing else changes.

Detector constants (module-level, edit there):

| constant | value | source / reason |
|---|---|---|
| `R_IN` | 2160 mm | ALLEGRO ECAL inner radius |
| `N_LAYERS`, `LAYER_X0` | 11 layers, 1.5 + 10×3.6 X₀ | prototype layering, counted from 0 |
| `CELL_X`, `CELL_Y` | 21.6 mm, 17.7 mm | 10 mrad θ, 8.2 mrad φ cells |
| strip pitch | `CELL_X/4` = 5.4 mm | 2.5 mrad θ strips, 4 strips per cell |
| `X0`, `RM`, `EC` | 14 mm, 40 mm, 10 MeV | effective Pb/LAr values; tune on full sim |
| `NOISE_CELL`, `THRESHOLD` | 4 MeV, 2σ | ~½ MIP; strips get noise scaled by √(area) |

`strip_layers` selects which layers carry the ×4 θ segmentation (`--strips`). Channel count is not enforced; the segmentation study must track it separately.

Shower model:

- Conversion depth: exponential, mean 9/7 X₀.
- Longitudinal: Gamma profile (Longo–Sestili), t_max = ln(E/E_c) − 0.5 (clamped ≥ 1 for soft photons), b = 0.5, with a 5 % event-by-event fluctuation of the shape parameter.
- Transverse: 75 % core (0.3 R_M) + 25 % halo (1.0 R_M) exponentials, widening with depth.
- Each shower is 3000 energy quanta histogrammed into cells, giving Poisson-like cell fluctuations.
- π⁰: isotropic decay in the rest frame, boosted; both photons come from the IP, so the opening angle and energy sharing follow the kinematics. Photon A is defined as the more energetic one.
- Impact point uniform within one cell; optional incidence tilt along x.

Self-check: `python data.py`.

## 3. `model.py` — network

Graph construction (`event_to_graph`):

- Node features (7): x, y, z scaled by 50 mm, log E, is_strip, E/E_event, layer/10.
- Edges: kNN with k = 8 in scaled (x, y, z), computed once with `torch.cdist` (events have ~260 hits).
- Targets: `frac` = max(f, 1−f) ∈ [0.5, 1] (SOOP's permutation-safe dominant fraction), `entry` = entry point of the photon dominating each hit.

`collate` concatenates graphs into one disconnected graph with a `batch` index.

Encoder:

- Input MLP → 3 × `EdgeConv` layers: message φ([h_i, h_j − h_i, pos_j − pos_i]), mean aggregation over neighbours, hidden size 32, concatenated skip connections (DenseNet style). 63 k parameters.

Heads:

- `node_head` → dominant fraction (0.5 + 0.5·sigmoid) and entry point (x, y).
- `graph_head` on [energy-weighted mean pooling, max pooling] → γ/π⁰ logit.

Loss (`loss_fn`): BCE on every event + 1.0 × energy-weighted fraction MSE (weights √(E·f), as in SOOP) + 0.1 × entry-point MSE, the latter two only on π⁰ hits.

Self-check: `python model.py`.

## 4. `train.py` — training, baseline, evaluation

- Data: `--n` training events, `n/4` test events, balanced γ/π⁰, independent seeds.
- Baseline: `HistGradientBoostingClassifier` on shower-shape variables (`shower_shapes`): 11 longitudinal energy fractions, energy-weighted widths in x and y (whole shower and strip layer), max-strip share, second-maximum / first-maximum ratio in the 1D strip profile, E(3 strips)/E(7 strips), log E.
- GNN: AdamW, OneCycle LR up to 2e-3, batch 64.
- Figure of merit: `misid_at_eff` = π⁰ efficiency at the threshold that keeps 95 % of photons, overall and in energy bins 5–10–20–35–50 GeV, plus ROC AUC.
- Plots: ROC (log y) and mis-ID vs energy for BDT and GNN.

## 5. Known limits of this version

- Parametric showers: no photon conversion in the tracker, no magnetic field, no φ-dependent sampling fraction, no cross-talk. Treat absolute numbers as pipeline validation only; the BDT/GNN comparison and the energy dependence are the meaningful outputs until the k4geo sample is plugged in.
- CPU-only training here (~17 s per 1000 events per epoch on 2 cores); on lxplus-gpu move tensors with `.to("cuda")` in `train.py`.
- kNN is O(n²) per event; fine at a few hundred hits, replace with a precomputed strip-neighbour table for full-occupancy events.
- No pileup/background overlay.

## 6. Next steps

1. Replace `make_event` with an EDM4hep reader for the ALLEGRO_o1_v03 γ and π⁰ samples (positions from cellID decoding, `frac` from the MC-truth contribution per cell).
2. Run `--strips` variants (1; 1,2; 2) at fixed channel count and plot mis-ID vs energy per configuration.
3. Add a Transformer encoder as a second model under the same heads and compare at the 95 % working point.
4. Fit the shower constants (X₀, R_M, noise) on the full simulation if the parametric generator stays in use for fast scans.
