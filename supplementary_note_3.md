# Supplementary Note 3: Transformer Curvature Profiles — A Third Architecture Type

**Christopher Head**
Navigator's Log R&D — Fresno, California
March 2026

Addendum to: "Surface Geometry of Data Manifolds Predicts Optimal Regularization in Neural Networks" (v3)
Zenodo DOI: 10.5281/zenodo.19298924

---

## Summary

A Tiny Vision Transformer (6 blocks, 128 dim, 811K parameters) trained on CIFAR-10 (80.4% accuracy) reveals a curvature profile that differs fundamentally from both MLPs and CNNs. The CLS token representation shows a drop-rise-peak-decline pattern: curvature drops at block 0, rises through blocks 1-3, peaks at blocks 3-4, then declines through block 5 and the classification head. Meanwhile, the full-sequence representation (all spatial tokens) maintains nearly constant curvature across all six blocks (0.397-0.403). The Transformer does not smooth its manifold like an MLP or expand-then-compress like a CNN. It builds a probe (the CLS token) that extracts information from an unchanged reservoir (the spatial tokens). This asymmetric processing strategy represents a third fundamental architecture type with distinct implications for per-layer regularization.

---

## 1. Experimental Setup

Model: Tiny Vision Transformer. 4x4 patch embedding (64 patches from 32x32 images), 128 embedding dimension, 6 transformer blocks (multi-head self-attention with 4 heads + feedforward network with 256 hidden dim), learnable CLS token, positional embeddings. 811,146 parameters.

Training: CIFAR-10, 30 epochs, AdamW (lr=1e-3, wd=1e-2), cosine learning rate schedule with 2-epoch warmup, gradient clipping at 1.0, batch size 32. Final test accuracy: 80.4%.

Measurement: After training, activations collected on 2,000 training samples (test transform, no augmentation). Curvature measured via local PCA residual variance (k=10 neighbors, 200 sample points per layer) — identical method to all prior experiments. Intrinsic dimension via TwoNN. Two views measured: CLS token only (the single vector the classifier reads) and full sequence (all 65 tokens flattened).

Runtime: 295 minutes on Surface Pro 7 (Intel i5, no GPU).

---

## 2. Results

### 2.1 CLS Token View

| Layer | Curvature | Curv Std | ID | Ambient Dim |
|---|---|---|---|---|
| embedding | 0.346 | 0.025 | 30.9 | 8,320 |
| block_0 | 0.196 | 0.040 | 10.5 | 128 |
| block_1 | 0.254 | 0.029 | 14.0 | 128 |
| block_2 | 0.281 | 0.030 | 15.8 | 128 |
| block_3 | 0.288 | 0.029 | 13.9 | 128 |
| block_4 | 0.288 | 0.030 | 14.7 | 128 |
| block_5 | 0.270 | 0.031 | 13.4 | 128 |
| pre_head | 0.253 | 0.032 | 12.2 | 128 |
| output | 0.106 | 0.029 | 6.7 | 10 |

Profile shape: Drop at block 0 (0.346 to 0.196), rise through blocks 1-3 (0.196 to 0.288), plateau at blocks 3-4 (0.288), decline through block 5 and pre_head (0.288 to 0.253), sharp compression at output (0.106).

### 2.2 Full Sequence View (All Tokens)

| Layer | Curvature | ID | Ambient Dim |
|---|---|---|---|
| block_0 | 0.399 | 37.9 | 8,320 |
| block_1 | 0.400 | 33.2 | 8,320 |
| block_2 | 0.399 | 31.8 | 8,320 |
| block_3 | 0.403 | 31.7 | 8,320 |
| block_4 | 0.403 | 31.8 | 8,320 |
| block_5 | 0.397 | 29.3 | 8,320 |

Curvature: effectively flat (range 0.397-0.403, variation < 2%). The spatial token representations maintain constant curvature across all six transformer blocks. Attention does not smooth the spatial tokens.

ID declines slightly (37.9 to 29.3), indicating mild compression of the full representation even as curvature stays constant.

---

## 3. Three Architecture Types on CIFAR-10

| Architecture | Acc | Profile Shape | Mechanism |
|---|---|---|---|
| MLP (256-128-64-32) | 52.9% | Monotonic decline | Smooths entire manifold |
| CNN (3conv+2fc) | 70.8% | High plateau then drop | Expands then compresses |
| Transformer (6 block ViT) | 80.4% | CLS: drop-rise-peak-decline; Spatial: flat | Builds a probe into unchanged reservoir |

### 3.1 MLP: Smooth the Manifold

Curvature declines gradually from 0.286 to 0.085 across all layers. Every layer reduces curvature. The entire representation gets smoother with depth. The network IS the smoothing process.

Optimal WD: gradual decay matching the gradual curvature decline. On MNIST (easy task), front-loaded decay works. On CIFAR-10 (hard task, under-capacity), uniform WD is better because curvature persists.

### 3.2 CNN: Expand Then Compress

Curvature increases from 0.286 (input) to 0.362 (conv2), then decreases to 0.104 (output). ID follows the same pattern: 22 to 35 to 8. The network first builds a richer feature space (bloom), then selectively compresses it (prune). This is the Ansuini hunchback measured with curvature.

Optimal WD: non-uniform, with the total budget in the Goldilocks zone. Direction is a weak effect — non-uniformity itself drives layer specialization (Supplementary Note 2, 4/5 random seeds beat Fixed).

### 3.3 Transformer: Extract Via Probe

The Transformer does something neither the MLP nor the CNN does. It maintains the source manifold at constant curvature (spatial tokens at 0.40 throughout all blocks) while building a SEPARATE representation (the CLS token) that progressively accumulates information from the source.

The CLS token begins nearly empty after block 0 (curvature 0.196 — the lowest point in the network). Across blocks 1-3, attention pulls information from the spatial tokens into the CLS token, and its curvature rises (0.196 to 0.288). At blocks 3-4, the CLS token reaches maximum information density (peak curvature 0.288). Blocks 5 and pre_head begin compressing the CLS token toward the classification output (curvature drops to 0.253, then 0.106 at the output).

The spatial tokens serve as a stable reservoir. Their curvature doesn't change because attention READS from them without modifying them (in this architecture, the residual connections preserve the spatial token content even after attention mixes information). The CLS token is the probe — it extends into the reservoir, accumulates what it finds, then retracts with the extracted signal.

---

## 4. Implications for Per-Layer Regularization

### 4.1 No Monotonic Schedule Matches the Transformer Profile

The CLS token's drop-rise-peak-decline shape cannot be approximated by exponential decay (front-loaded), linear decay, or reverse (back-loaded). Front-loading would brake hardest at block 0, where the CLS token has its LOWEST curvature and needs the LEAST braking. Reverse would brake hardest at block 5, which is declining but not the peak. The bimodal schedule (light early, peak mid, taper late) is closest but doesn't account for the initial drop at block 0.

### 4.2 The Symmetry-Breaking Finding Applies

Supplementary Note 2 demonstrated that on CIFAR-10 CNN with its non-monotonic profile, random per-layer WD variation (4/5 seeds) beat fixed WD, and the mean random gain exceeded structured schedule gains. The Transformer's even more complex profile makes the case for non-uniform WD stronger — any variation breaks the optimizer's symmetry and forces block-level specialization.

### 4.3 SurfaceGate May Be the Right Tool for Transformers

The main paper showed SurfaceGate (a module that measures curvature at each layer and adjusts WD accordingly) lost to a simple exponential schedule on MLPs — because the MLP curvature profile was too simple to need runtime measurement. On Transformers, the profile is complex, architecture-dependent, and potentially task-dependent (different datasets may shift the peak block). Runtime curvature measurement could add genuine value because the profile shape is not predictable from the architecture alone.

### 4.4 The Full-Sequence Flatness Suggests Attention Is a Leg 2 Operation

The spatial tokens maintain constant curvature across all blocks. Attention circulates information between tokens (Leg 2) but does not dissipate curvature (Leg 3 is not applied to the spatial representation). The dissipation happens only at the CLS token (where the extracted information is progressively compressed) and at the output (where the classification head forces final selection).

In the framework's terms: the transformer maintains the mound (spatial tokens at constant curvature) while building and pruning a probe (CLS token). The mound is not smoothed. The probe is. This is fundamentally different from MLPs (which smooth the mound) and CNNs (which reshape the mound).

---

## 5. Connection to the Three-Level Hierarchy

The three-level hierarchy from Supplementary Note 2 applies to Transformers with one modification:

**Level 1 (capacity):** The Transformer at 80.4% accuracy is firmly in the adequate-capacity regime. Per-layer WD variation should help.

**Level 2 (total budget):** The Goldilocks zone for total WD budget applies. The training used WD=0.01 globally, which achieved 80.4%. The total budget question is the same as for MLPs and CNNs.

**Level 3 (non-uniformity):** Non-uniform WD should help, but the optimal assignment differs from MLPs and CNNs because the curvature profile differs. The CLS token profile suggests: light WD at block 0 (let the CLS token gather freely), moderate-to-heavy WD at blocks 1-4 (prune during accumulation to force selective attention), lighter WD at block 5 (allow final compression without over-pruning). Whether this specific shape outperforms random non-uniform WD is an open question that requires a Thread B-style test on the Transformer.

**New consideration for Transformers:** The asymmetry between CLS and spatial tokens suggests that per-TOKEN-TYPE regularization (different WD for the CLS token's attention weights vs. the spatial tokens' processing) might be more relevant than per-block regularization. This is a dimension of WD scheduling that doesn't exist in MLPs or CNNs.

---

## 6. Summary of All Curvature Profiles

Across all experiments (49+ hours of compute):

| Architecture | Dataset | Profile | Curvature Range | Best WD Strategy |
|---|---|---|---|---|
| MLP | Synthetic | Cliff (zero by L2) | 0.01-0.25 | Exponential decay |
| MLP | MNIST | Gradual decline | 0.054-0.210 | Front-loaded (ExpDecay) |
| MLP | CIFAR-10 | Persistent decline | 0.128-0.286 | Uniform (under-capacity) |
| CNN | MNIST | Nearly flat | 0.129-0.243 | Irrelevant (arch dominates) |
| CNN | CIFAR-10 | Hunchback | 0.104-0.362 | Non-uniform (any pattern) |
| Transformer | CIFAR-10 (CLS) | Drop-rise-peak-decline | 0.106-0.288 | Non-uniform or SurfaceGate |
| Transformer | CIFAR-10 (spatial) | Flat | 0.397-0.403 | N/A (tokens not smoothed) |

The curvature profile is architecture-specific, dataset-specific, and view-specific. No single WD schedule is universally optimal. The three-level hierarchy (capacity, total budget, non-uniformity) provides the general principle. The specific curvature profile determines which form of non-uniformity is most effective — but the symmetry-breaking finding (any non-uniformity helps) provides a robust default when the profile is unknown.

---

## 7. Files

- `tdf_transformer_profiles.py` — Training and curvature measurement script (295 min)
- `tdf_transformer_profiles.json` — Full profiles for CLS and spatial views
- `tdf_transformer_profiles.png` — Visualization comparing all three architectures

All files available at: github.com/NavigatorsLog/tdf-surface-geometry

---

*Christopher Head — Navigator's Log R&D — navigatorslog.netlify.app*
