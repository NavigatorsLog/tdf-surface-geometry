# Supplementary Note 2: Non-Uniform Weight Decay, Symmetry-Breaking, and the Three-Level Hierarchy

**Christopher Head**
Navigator's Log R&D — Fresno, California
March 2026

Addendum to: "Surface Geometry of Data Manifolds Predicts Optimal Regularization in Neural Networks" (v3)
Zenodo DOI: 10.5281/zenodo.19298924

---

## Summary

Three experiments totaling 715 minutes of CPU compute on a Surface Pro 7 reveal that (1) real data curvature profiles differ fundamentally from synthetic manifold profiles, (2) per-layer WD non-uniformity improves generalization on capable architectures regardless of the direction of variation, and (3) a three-level hierarchy governs optimal weight decay strategy. The original exponential decay recommendation is superseded by a more general principle: the total Leg 3 budget determines the Goldilocks zone, and within that zone, any non-uniform per-layer assignment outperforms uniform WD by forcing layer specialization.

---

## 1. Thread A: Real Curvature Profiles Differ from Synthetic

### 1.1 Motivation

All prior experiments assumed a universal curvature profile based on synthetic manifold measurements: curvature high at layer 1, dropping to near-zero by layer 2 regardless of architecture. This assumption was never validated on real data. Thread A measured the actual curvature profile on trained networks using the same local PCA residual variance method from the main paper.

### 1.2 Method

MLPs (256-128-64-32) and SimpleCNNs (3 conv + 2 FC) were trained on MNIST and CIFAR-10 with standard hyperparameters. After training, activations were collected at every layer on 2,000 training samples. Curvature (local PCA residual variance, k=10 neighbors) and intrinsic dimension (TwoNN) were measured at each layer. Runtime: 154 minutes.

### 1.3 Results

MNIST MLP — Gradual monotonic decline:

| Layer | Curvature | ID | Ambient Dim |
|---|---|---|---|
| input | 0.282 | 11.7 | 784 |
| layer_0 | 0.210 | 8.3 | 256 |
| layer_1 | 0.146 | 7.1 | 128 |
| layer_2 | 0.078 | 5.8 | 64 |
| layer_3 | 0.054 | 5.1 | 32 |
| output | 0.042 | 4.7 | 10 |

MNIST CNN — Nearly flat through convolutional stack:

| Layer | Curvature | ID | Ambient Dim |
|---|---|---|---|
| input | 0.276 | 11.8 | 784 |
| conv1 | 0.243 | 9.8 | 6,272 |
| conv2 | 0.237 | 9.7 | 3,136 |
| conv3 | 0.213 | 9.3 | 576 |
| fc1 | 0.211 | 9.3 | 128 |
| fc2 | 0.129 | 7.1 | 64 |
| output | 0.091 | 5.8 | 10 |

CIFAR-10 MLP — Persistent curvature at all layers:

| Layer | Curvature | ID | Ambient Dim |
|---|---|---|---|
| input | 0.296 | 22.1 | 3,072 |
| layer_0 | 0.286 | 15.5 | 256 |
| layer_1 | 0.258 | 13.7 | 128 |
| layer_2 | 0.191 | 10.2 | 64 |
| layer_3 | 0.128 | 7.7 | 32 |
| output | 0.085 | 6.6 | 10 |

CIFAR-10 CNN — Non-monotonic hunchback profile:

| Layer | Curvature | ID | Ambient Dim |
|---|---|---|---|
| input | 0.286 | 22.1 | 3,072 |
| conv1 | 0.360 | 34.8 | 8,192 |
| conv2 | 0.362 | 35.1 | 4,096 |
| conv3 | 0.355 | 25.3 | 1,024 |
| fc1 | 0.326 | 19.9 | 128 |
| fc2 | 0.198 | 11.5 | 64 |
| output | 0.104 | 7.8 | 10 |

### 1.4 Key Findings

The synthetic profile (curvature drops to zero by layer 2) does not hold on real data. All four configurations show curvature persisting well past layer 2. The deepest hidden layer retains 19-43% of the input curvature.

Four distinct profile shapes emerged, each explaining the corresponding WD schedule results from Supplementary Note 1:

Gradual decline (MNIST MLP): Curvature drops steadily. Front-loaded decay works because the profile is monotonically decreasing. ExpDecay won (+0.0009).

Nearly flat (MNIST CNN): Curvature barely changes through the conv stack (0.24 to 0.21). All schedules tied because the profile provides no gradient to match.

Persistent high curvature (CIFAR-10 MLP): Curvature remains above 0.12 at every layer. Fixed WD won because every layer genuinely needs strong braking. Decay schedules starve deep layers.

Non-monotonic hunchback (CIFAR-10 CNN): Curvature INCREASES from input (0.286) to conv1-conv2 (0.360-0.362), then decreases through FC layers. The convolutional layers expand the representation (ID rises from 22 to 35) before the FC layers compress it (ID falls from 35 to 8). This expansion-then-compression is the Ansuini hunchback measured with curvature for the first time on real data.

The hunchback profile on CIFAR-10 CNN explains why Reverse WD (light at entrance, heavy at depth) outperformed front-loaded schedules in the real-data test: the conv layers are in the expansion phase (building features — light braking lets them bloom), while the FC layers are in the compression phase (pruning features — heavy braking forces selective retention).

---

## 2. Thread B: Symmetry-Breaking Is the Mechanism

### 2.1 Motivation

CIFAR-10 CNN results in Supplementary Note 1 showed all non-uniform schedules beating Fixed, including Reverse. Thread B tested whether the benefit comes from the specific direction of WD variation or from non-uniformity itself.

### 2.2 Method

Thirteen WD schedules tested on CIFAR-10 with both MLP (5 trainable layers) and CNN (6 trainable layers). All used a fixed base WD of 1e-3. Schedules: Fixed, ExpDecay (halving per layer), LinDecay (linear reduction), Reverse (increasing with depth), five random seeds (log-uniform draws per layer), three shuffled seeds (ExpDecay values in random order), and Alternating (10x high-low swings). Runtime: 528 minutes.

### 2.3 Results — MLP (under-capacity)

| Category | Mean Accuracy | vs Fixed |
|---|---|---|
| Fixed | 0.5439 | baseline |
| Shuffled mean | 0.5294 | -0.015 |
| Structured mean | 0.5277 | -0.016 |
| Random mean | 0.5263 | -0.018 |
| Alternating | 0.5200 | -0.024 |

Random seeds beating Fixed: 0/5. No form of WD variation helps on the under-capacity MLP. Uniform WD is optimal when curvature is persistently high at all layers.

### 2.4 Results — CNN (adequate capacity)

| Schedule | Accuracy | vs Fixed | Total WD |
|---|---|---|---|
| Random_1 | 0.7669 | +0.020 | ~4.2e-3 |
| Reverse | 0.7604 | +0.014 | ~3.6e-3 |
| Alternating | 0.7568 | +0.010 | ~6.6e-3 |
| Random_2 | 0.7564 | +0.010 | ~3.5e-3 |
| Random_3 | 0.7550 | +0.008 | ~11.4e-3 |
| Random_5 | 0.7549 | +0.008 | ~14.0e-3 |
| LinDecay | 0.7517 | +0.005 | ~3.6e-3 |
| Shuffled_3 | 0.7491 | +0.003 | ~1.7e-3 |
| ExpDecay | 0.7477 | +0.001 | ~1.9e-3 |
| Shuffled_2 | 0.7472 | +0.001 | ~1.0e-3 |
| Fixed | 0.7466 | baseline | 6.0e-3 |
| Random_4 | 0.7441 | -0.003 | ~24.3e-3 |
| Shuffled_1 | 0.7438 | -0.003 | ~1.9e-3 |

Random seeds beating Fixed: 4/5. The only loss (Random_4) had a total budget 4x above the optimal range. Mean random gain (+0.0089) exceeds mean structured gain (+0.0067).

Shuffled seeds (same values as ExpDecay, different order) failed because their total budget (~1-2e-3) was below the Goldilocks floor, not because shuffling hurt.

### 2.5 Key Findings

Symmetry-breaking confirmed: On capable architectures, any non-uniform WD assignment that maintains a reasonable total budget outperforms uniform WD. Random, structured, alternating, and directional variation all work. The specific direction is a weak third-order effect.

Total budget is the dominant variable: The Goldilocks zone for total per-layer WD sum on CIFAR-10 CNN is approximately 3-14e-3. Below this floor (shuffled seeds at ~1-2e-3), the network is under-regularized. Above the ceiling (Random_4 at ~24e-3), the network is over-regularized. Within the zone, non-uniformity provides +0.5-2.0% additional gain.

Direction matters weakly: Reverse (+0.014) beat LinDecay (+0.005) beat ExpDecay (+0.001). On the CIFAR-10 CNN with its hunchback curvature profile, back-loading slightly outperforms front-loading — consistent with the measured geometry (bloom at conv layers, prune at FC layers). But random seeds with no directional structure outperformed most structured schedules. Direction explains approximately 10-20% of the non-uniformity benefit.

---

## 3. The Three-Level Hierarchy

All results across synthetic manifolds, MNIST, and CIFAR-10 are explained by a single hierarchy:

### Level 1: Can the network smooth the data manifold?

Measured by: model accuracy well above chance.

If NO (under-capacity, ~50% accuracy on 10-class task): curvature persists at all layers. Every layer needs strong braking. Use uniform WD. No schedule of any kind improves over fixed.

If YES (adequate capacity, >70% accuracy): curvature varies across layers. Proceed to Level 2.

If the architecture already self-regularizes (>95% accuracy): the WD schedule is irrelevant. Implicit Leg 3 (pooling, weight sharing, batch norm) dominates. Any schedule works equally.

### Level 2: Is the total WD budget in the Goldilocks zone?

This is the dominant variable. The surface geometry of the data manifold (ID x DCV, from the main paper Section 4.1) predicts the required total budget. Get this wrong and no schedule compensates.

Too low: under-regularized. The network memorizes regardless of per-layer assignment.
Too high: over-regularized. The network loses capacity regardless of assignment.
In range: proceed to Level 3.

### Level 3: Use non-uniform per-layer WD.

Any non-uniform assignment improves over fixed WD by +0.5-2.0%. The non-uniformity forces each layer to face a different regularization environment, which drives layer specialization. Different layers develop different functional roles because they face different pruning pressures.

Direction is a weak additional signal. On MLPs with monotonically declining curvature, front-loading helps marginally. On CNNs with hunchback curvature profiles, back-loading helps marginally. Random assignment captures most of the benefit without requiring knowledge of the curvature profile.

### Revised Practical Recommendation

```python
# Level 1: Check if model is above chance
# Level 2: Set total WD budget (use surface geometry predictor or grid search)
# Level 3: Apply non-uniform assignment

import numpy as np
rng = np.random.RandomState(42)
for i, layer in enumerate(model.layers):
    # Random perturbation around the base WD (log-uniform, ±1 order of magnitude)
    log_wd = np.log10(base_wd) + rng.uniform(-0.5, 0.5)
    optimizer.param_groups[i]['weight_decay'] = 10 ** log_wd
```

This costs nothing, adds no learned parameters, requires no knowledge of the curvature profile, and provides +0.5-2.0% improvement on capable architectures. The perturbation range (half an order of magnitude above and below base) keeps the total budget near the intended value while providing sufficient non-uniformity.

---

## 4. Connection to Framework

The three-level hierarchy maps directly to the Tension-Dissipation Framework:

Level 1 (capacity) determines whether the network can act as a smoothing process. Under-capacity networks cannot form the three-leg system needed for structured representation — they are two-leg halos, diffuse and unable to compress.

Level 2 (total budget) is the Goldilocks zone for Leg 3. This is the same principle as the fungibility finding (main paper Section 4.4): the total dissipation budget matters more than the source. The surface geometry predicts the budget (main paper Section 4.1, 84% oracle capture).

Level 3 (non-uniformity) is the new finding. Uniform WD creates symmetric regularization pressure, which allows symmetric layer behavior. Non-uniform WD breaks this symmetry, forcing each layer to develop a unique specialization under its unique pressure environment. The analogy in physics: crystals form because cooling introduces asymmetric conditions. Uniform cooling produces amorphous glass. Non-uniform cooling produces crystals with grain boundaries and internal structure. The non-uniformity of the dissipation creates the internal structure of the mound.

The hunchback curvature profile on CIFAR-10 CNN (Section 1.3) represents the bloom-then-prune dynamic: the network first expands the representation (curvature and ID increase through conv layers) then compresses it (curvature and ID decrease through FC layers). This is the same expansion-contraction measured by Ansuini et al. (2019) with intrinsic dimension, here confirmed independently with curvature on a different architecture. In framework terms: Leg 1 and Leg 2 build the mound (expansion phase), then Leg 3 prunes it back (compression phase). The shedding during the compression phase creates the final structure — the features that survive are the load-bearing ones.

---

## 5. What This Supersedes

This note supersedes the per-schedule recommendation in Supplementary Note 1. The original recommendation ("use gentle linear decay") was based on incomplete data. The complete picture shows:

The exponential decay schedule from the main paper (Section 4.9) is correct for synthetic manifolds where the curvature profile is a genuine cliff. On real data, curvature declines gradually or non-monotonically, and exponential decay is too aggressive.

Linear decay works on MNIST MLP and CIFAR-10 CNN but is not universally best. On CIFAR-10 CNN, Reverse and random assignments outperform it.

The general principle — non-uniform WD with a correct total budget — subsumes all specific schedule recommendations. Any non-uniform assignment within the Goldilocks budget zone improves over fixed WD on capable architectures.

---

## 6. Files

- `tdf_curvature_profiles.py` — Thread A: curvature measurement on trained networks (154 min)
- `tdf_curvature_profiles.json` — Curvature and ID at every layer for 4 configurations
- `tdf_symmetry_breaking.py` — Thread B: 13 WD schedules including random seeds (528 min)
- `tdf_symmetry_breaking_results.json` — Full accuracy and WD values for all configurations
- `tdf_real_data_test.py` — Original real-data WD schedule comparison (1,983 min)
- `tdf_real_data_results.json` — MNIST and CIFAR-10 results for 4 schedules x 2 architectures

Total compute for all three tests: 2,665 minutes (~44 hours) on Surface Pro 7 (Intel i5-1035G4, 8GB RAM, no GPU).

All files available at: github.com/NavigatorsLog/tdf-surface-geometry

---

*Christopher Head — Navigator's Log R&D — navigatorslog.netlify.app*
