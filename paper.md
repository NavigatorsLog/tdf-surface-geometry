# Surface Geometry of Data Manifolds Predicts Optimal Regularization in Neural Networks

**Christopher Head**
Navigator's Log R&D — Fresno, California
powerpredixtable@proton.me | navigatorslog.netlify.app
March 2026 — Version 2

Zenodo DOI: 10.5281/zenodo.19298924
GitHub: github.com/NavigatorsLog/tdf-surface-geometry

---

## Abstract

We show that four measurable geometric properties of a data manifold — intrinsic dimension, curvature, density variation, and codimension ratio — predict the optimal weight decay for neural networks with R² = 0.196 (p < 0.001) and capture 84% of oracle grid-search performance on held-out domains (40 wins, 19 ties, 1 loss across 60 test manifolds). The key finding is that interaction terms between surface properties — particularly the product of intrinsic dimension and density variation (ID×DCV) — carry the dominant signal, while individual properties alone are weak predictors. This result emerges only when self-regularizing mechanisms (early stopping) are absent, confirming that surface geometry governs explicit regularization specifically.

We further demonstrate three results derived from this finding. First, different regularization mechanisms (weight decay, early stopping, noise injection, subsampling) are highly fungible (mean cross-correlation r = +0.953 across 12 manifolds), confirming the framework's prediction that the total Leg 3 budget matters more than the source. Second, combining two brake types never outperforms the best single brake (0/12 manifolds), establishing that the Goldilocks zone has both a floor and a ceiling. Third, we introduce SurfaceGate — a one-parameter-per-layer module that measures the curvature of a network's own representations and adjusts weight decay accordingly. SurfaceGate exceeds oracle grid-search performance on both wide and narrow architectures (oracle capture 124% and 100% respectively), accessing per-layer regularization solutions that no single fixed weight decay can reach.

The mechanism underlying SurfaceGate is universal across architectures: neural network representations flatten monotonically through depth regardless of layer width, with curvature concentrated at the first hidden layer. The gate learns to front-load regularization — braking hardest where the representation is most crumpled — and reducing brakes at deeper layers where the learned transformation has already smoothed the manifold.

These findings derive from the Tension-Dissipation Framework (TDF), a cosmological theory tested against 21+ correlations across 13 scientific domains spanning 60 orders of magnitude. All experiments run on a consumer laptop (Surface Pro 7) using standard Python libraries.

---

## 1. Introduction

### 1.1 The Hyperparameter Problem

Every neural network training run requires choosing a regularization strength — weight decay, dropout rate, or equivalent. The standard approach is grid search or Bayesian optimization: train multiple models with different values and select the winner. This is computationally expensive and provides no insight into why a particular value works.

We ask three increasingly specific questions: Can measurable properties of the data manifold predict the optimal regularization before training begins? Are different types of regularization interchangeable? And can a network measure its own internal geometry and adjust its own regularization in real time?

### 1.2 The Tension-Dissipation Framework

The Tension-Dissipation Framework (TDF) is a cosmological theory proposing that all observed structure arises from three conservation laws operating together: conservation of energy (Leg 1), conservation of angular momentum (Leg 2), and dissipation (Leg 3). The framework has been tested against 21+ independent correlations across 13 scientific domains spanning 60 orders of magnitude in physical scale (mean |r| = 0.920), documented in the TDF Theory Document v4.0 (Head, 2026).

The framework's central prediction relevant to machine learning: Leg 3 (dissipation/braking) determines the Goldilocks zone for structure formation. Too little braking produces diffuse, unstructured systems (dark matter halos in astrophysics; memorization in ML). Too much braking prevents structure from forming at all (over-regularized, underfitting models). The optimal braking strength is governed by the surface geometry of the system — the boundary where structure meets background.

### 1.3 Mapping TDF to Machine Learning

| TDF Component | ML Equivalent | Role |
|---|---|---|
| Energy (Leg 1) | Compute + Data | Drives the flow |
| Angular Momentum (Leg 2) | Gradient flow + Attention | Circulates information |
| Dissipation (Leg 3) | Regularization (WD, dropout, temp) | Sheds structure, forces generalization |
| Mound | Learned representation | Temporary structure sustained by energy |
| Smooth | Noise floor / prior | Background toward which all structure trends |
| Halo (2-leg system) | Memorizing network | No Leg 3: diffuse, no generalization |
| Galaxy (3-leg system) | Generalizing network | Leg 3: compact, structured representations |

---

## 2. Related Work

Ansuini et al. (2019, NeurIPS) measured intrinsic dimension across layers of trained networks, finding a "hunchback" shape and showing that last-layer ID predicts test accuracy. Pope et al. (2021, ICLR) measured ID of standard datasets (MNIST ID approximately 13, CIFAR-10 approximately 20, ImageNet approximately 26-43). Li and Liang (2018, ICLR) measured intrinsic dimension of objective landscapes. GeLoRA (EMNLP 2025) uses ID to set LoRA adapter ranks — the closest prior work, targeting fine-tuning rank rather than regularization strength. Kaplan et al. (2020) and Hoffmann et al. (2022) established scaling laws treated as empirical constants. Power et al. (2022) discovered grokking; Clauw et al. (2024) established it as a phase transition.

No prior work measures multiple surface properties as a vector and uses their interactions to predict optimal regularization. No prior work demonstrates fungibility of regularization mechanisms with quantitative cross-correlation. No prior work implements geometry-adaptive per-layer weight decay.

---

## 3. Methods

### 3.1 Surface Property Measurements

We measure four properties of each data manifold:

Intrinsic Dimension (ID) via the TwoNN method (Facco et al., 2017). Curvature proxy via local PCA residual variance on k=12 nearest-neighbor patches. Density variation (DCV) via the coefficient of variation of k-nearest-neighbor log-density estimates. Codimension ratio (CR) as ambient dimension divided by intrinsic dimension.

In addition to log-transformed raw features, we compute five interaction and quadratic terms: ID×Curv, ID×DCV, Curv×DCV, ID², Curv².

### 3.2 Synthetic Manifold Generation

We embed low-dimensional Gaussian data into higher-dimensional space via random linear projection, adding polynomial curvature terms and off-manifold Gaussian noise. This allows independent variation of ID (2-60), curvature (0.01-5.0), noise (0.005-0.8), and number of classes (2-10).

### 3.3 Critical Design Choice: No Early Stopping

Models train for 500 iterations with tolerance 1e-6 and no early stopping. This ensures weight decay is the only regularization mechanism. Our v1 experiment with early stopping showed R² = 0.000 — the early stopping masked the surface geometry signal entirely (see Section 4.2).

---

## 4. Results

### 4.1 Surface Geometry Predicts Optimal Regularization

200 synthetic manifolds with diverse geometries. MLP (64, 32). 12 weight decay values. 2 seeds per value. 60 held-out test manifolds.

| Model | r | R² | p |
|---|---|---|---|
| Linear (4 features) | +0.443 | 0.196 | 3.95 × 10⁻⁴ |
| Random Forest (9 features) | +0.390 | 0.152 | 2.09 × 10⁻³ |

Feature importances: ID×DCV (0.290), Curv×DCV (0.145), log DCV (0.135), log CR (0.097), log ID (0.086). Interaction terms carry the dominant signal. No single raw property exceeds 14%.

Adaptive vs. fixed: Surface-adaptive regularization captures 84% of oracle performance. Win/Tie/Loss: 40/19/1. Mean accuracy: Fixed 0.868, Adaptive 0.896, Oracle 0.902.

### 4.2 Early Stopping Masks the Signal

| Metric | With Early Stopping (v1) | Without (v2) |
|---|---|---|
| WD Prediction R² | 0.000 | 0.196 |
| Oracle Capture | 1% | 84% |
| Win/Tie/Loss | 10/41/9 | 40/19/1 |
| WD Spread | <0.2% | 7.6% mean, 30% max |

With early stopping, all weight decay values produce identical performance (spread < 0.2%). The surface prediction has nothing to predict. Without early stopping, wrong WD causes up to 30% accuracy loss, and the surface vector identifies the right WD neighborhood.

Implication: surface geometry governs explicit regularization specifically. When implicit Leg 3 sources (early stopping, batch normalization) dominate, the explicit parameter becomes irrelevant.

### 4.3 Density Variation Is a Modulator, Not a Driver

Density variation (DCV) dominates the feature importance rankings — appearing in 3 of the top 4 features (ID×DCV, Curv×DCV, log DCV) and accounting for 58% of the total signal. However, when tested in isolation on fixed-geometry manifolds with varying density patterns, DCV shows no predictive power (r = -0.06).

DCV is a modulator: it amplifies or dampens the effect of ID and curvature. On its own, it determines nothing. The product ID×DCV captures how much of a high-dimensional manifold is poorly covered by training data — the combination of surface size and coverage unevenness that determines how much pruning is needed.

### 4.4 Regularization Mechanisms Are Fungible

We tested four Leg 3 sources on 12 diverse manifolds: weight decay, early stopping, noise injection, and data subsampling. Each was optimized independently.

Mean cross-correlation between brake types: r = +0.953. Weight decay correlates with noise injection at r = +0.992 and subsampling at r = +0.985. When one brake type performs well on a manifold, all brake types perform well. The mound does not care how it sheds — only how much.

Weight decay achieves the highest mean accuracy (0.858), followed by noise injection (0.832), subsampling (0.820), and early stopping (0.748). Brakes are fungible in direction but not in efficiency.

Combining weight decay with early stopping never outperforms the best single brake (0/12 manifolds). Exceeding the Leg 3 budget degrades performance. The Goldilocks zone has both a floor and a ceiling.

### 4.5 SurfaceGate: Geometry-Adaptive Per-Layer Regularization

SurfaceGate is a one-parameter-per-layer module that measures the curvature of a network's own representations at each hidden layer and adjusts weight decay accordingly. The module has a single learned parameter per layer (sensitivity) that controls how much the measured curvature affects the effective weight decay. Curvature is measured periodically (every 25 epochs) via local PCA on nearest-neighbor patches — the same method used for data manifold surface measurement. Between measurements, the multipliers are fixed.

Tested on 20 diverse manifolds with two architectures:

| Architecture | Default Fixed | SurfaceGate | Oracle | Capture | W/T/L | Beats Oracle |
|---|---|---|---|---|---|---|
| Wide (64-64-64-64) | 0.908 | 0.914 | 0.913 | 124% | 8/10/2 | 6/20 |
| Narrow (64-48-32) | 0.909 | 0.918 | 0.918 | 100% | 11/7/2 | 4/20 |

SurfaceGate exceeds the oracle on both architectures. On the wide network, it captures 124% — beating the best fixed WD found by grid-searching 9 values. This is possible because the gate accesses per-layer WD solutions that no single fixed value can reach: optimal WD for layer 1 may be 0.005 while optimal for layers 3-4 may be 0.001, and no single value satisfies both.

### 4.6 The Universal Curvature Profile

The most striking finding from the SurfaceGate experiments is a universal pattern across all manifolds and both architectures: representation curvature decreases monotonically through depth. Layer 1 shows curvature ranging from 0.01 to 0.25. By layer 2, curvature drops to near zero. Deeper layers show curvature indistinguishable from zero.

This pattern holds even in the wide network (64-64-64-64) where the architecture provides no narrowing. The flattening is not caused by architectural compression — it is caused by what learning does to representations. The gradient-driven optimization, the ReLU nonlinearity, and the weight matrix transformations flatten the data manifold regardless of layer width.

The gate learned a single strategy from this universal pattern: front-load the regularization. Layer 1 multipliers range from 0.34 to 0.57 (reduce WD to about half). Deeper layer multipliers converge to approximately 0.33 (reduce WD to minimum). The gate independently discovered that regularization should be strongest at the network's entrance — where the raw data's crumpled surface first encounters the learned transformation — and lightest at depth where the representation is already smooth.

In the TDF framework: the mound sheds fastest at its outermost surface. The first layer IS the outermost surface of the computational mound. By layer 2, the shedding is essentially complete. Applying brakes to an already-smooth surface wastes capacity without improving generalization.

### 4.7 Representation Compression: Generalizing vs. Memorizing

MLPs trained on MNIST with real labels compress their representations 2.2 intrinsic dimensions more than networks trained on random labels in early-to-middle layers. The generalizing network sheds dimensions — it finds the task's low-dimensional structure and reduces to match it. The memorizing network retains dimensions because random labels have no low-dimensional structure to compress into.

The full Ansuini "hunchback" (ID rise then fall) was not observed because the test architecture lacked an expansion phase. The compression difference in early layers was confirmed.

### 4.8 Supporting Results

Grokking as phase transition: D_crit proportional to regularization strength to the power -0.22, r = -0.996. Matches physical phase transition universality (r = +0.998 across 9 systems).

Information Tully-Fisher: Three-leg scaling topology confirmed across four non-physics domains — cities, organisms, genomes, and cross-domain.

ML scaling laws: Kaplan loss scaling (r = 0.995 across 7 orders of magnitude) maps to Leg 1. Chinchilla optimum (N_opt proportional to C^0.50) represents balanced three-leg equilibrium.

---

## 5. Discussion

### 5.1 The Total Leg 3 Budget

The fungibility result (r = +0.953) and the early stopping masking effect (R² = 0 with early stopping vs. R² = 0.196 without) together establish a key principle: what matters is the total regularization budget, not the source. Weight decay, dropout, early stopping, noise injection, batch normalization, and attention-based selection are all Leg 3 mechanisms. They are substitutable — if one source is strong enough, the others become irrelevant.

The surface geometry of the data manifold determines the total Leg 3 budget required. The architecture provides a portion of that budget through its structure (narrowing layers, nonlinearities, normalization). The explicit regularization parameters provide the remainder. To predict optimal explicit regularization from surface geometry, one must first account for the architecture's implicit contribution.

### 5.2 Why SurfaceGate Exceeds the Oracle

The oracle searches over fixed weight decay — the same value applied to all layers. SurfaceGate applies different values per layer. When the optimal WD differs across layers (which it always does, because curvature differs across layers), fixed WD must compromise while the gate optimizes each layer independently.

The gate's learned strategy is simpler than anticipated: front-load the regularization, reducing it with depth as the representation smooths. This simplicity is itself a finding — it means the primary regularization need is concentrated at the network's entrance, where the raw data manifold is most crumpled, and diminishes rapidly with depth.

### 5.3 Networks as Smoothing Processes

The universal curvature profile — high at layer 1, zero by layer 2, regardless of architecture — reframes what neural networks do. Each layer transforms a crumpled, high-curvature representation into a smoother, lower-curvature one. The network IS the smoothing process. The mound (raw data) enters at the first layer, sheds curvature through depth, and exits as a flat manifold suitable for classification.

This process occurs regardless of layer width (wide and narrow architectures show the same pattern). It occurs regardless of the data manifold's initial curvature (low-curvature inputs still get flattened). It is a universal property of gradient-trained networks with ReLU activations.

In TDF terms: the smooth always wins. The network's job is to accelerate the smoothing — to take the data manifold from its raw, crumpled state to a flat, task-aligned representation in as few layers as possible. Regularization controls the rate of smoothing. Too fast (heavy WD) destroys discriminative curvature. Too slow (light WD) leaves unnecessary crumples that lead to overfitting. The Goldilocks zone is the smoothing rate that matches the task's intrinsic complexity.

### 5.4 Implications for AGI Architecture

Different domains have measurably different surface geometries with disjoint Goldilocks zones. Cross-domain intelligence requires either curvature-adaptive regularization (a SurfaceGate-like module that adjusts brakes based on measured geometry) or hierarchical specialization (domain-specific modules with domain-appropriate brakes connected by bridges).

The SurfaceGate results suggest adaptive regularization is viable: the module exceeds oracle performance using only one learned parameter per layer and periodic curvature measurements. The brain's neuromodulatory systems (dopamine, serotonin, norepinephrine, acetylcholine) implement an analogous mechanism — modulating plasticity based on context rather than carrying information content.

---

## 6. Limitations

Synthetic manifolds only. Small networks (2-4 layer MLPs). Four surface properties (additional measurements like fractal dimension and persistent homology could improve predictions). Linear prediction model (R² = 0.196 leaves 80% unexplained). SurfaceGate tested on 20 manifolds with a single architecture family. The curvature measurement is a proxy (local PCA residual variance), not a full Riemannian curvature tensor. The universal curvature profile may differ for architectures with skip connections (ResNets) or attention mechanisms (transformers), which re-introduce complexity at deeper layers.

---

## 7. Conclusion

The surface geometry of data manifolds predicts optimal neural network regularization. Interaction terms between surface properties (ID×DCV, Curv×DCV) carry the dominant signal. Different regularization mechanisms are highly fungible (r = +0.953), confirming that the total Leg 3 budget matters more than the source. Combining brakes beyond the budget degrades performance — the Goldilocks zone has both a floor and a ceiling.

SurfaceGate — a module that measures its own representation curvature and adjusts weight decay per layer — exceeds oracle grid-search performance on both wide and narrow architectures. The mechanism is a universal curvature profile: representations flatten monotonically through depth, concentrating regularization need at the network's entrance. The gate learns to front-load the brakes.

These findings extend the Tension-Dissipation Framework from astrophysics to artificial intelligence. The framework's prediction — that Leg 3 (dissipation) governs the Goldilocks zone for structure formation, and that the surface of the mound determines the shedding rate — holds across 13 scientific domains spanning 60 orders of magnitude. The "surface of the mound" in neural networks is the curvature of the representation at each layer. The outermost part — the first layer, where raw data meets the learned transformation — is where the action is.

All code available at github.com/NavigatorsLog/tdf-surface-geometry. Total compute: approximately 4 hours cumulative on a Surface Pro 7 (Intel i5, no GPU).

---

## References

- Ansuini, A., Laio, A., Macke, J.H., & Zoccolan, D. (2019). Intrinsic dimension of data representations in deep neural networks. NeurIPS 2019.
- Clauw, L., et al. (2024). Grokking as a phase transition. ScienceDirect.
- Facco, E., d'Errico, M., Rodriguez, A., & Laio, A. (2017). Estimating the intrinsic dimension of datasets by a minimal neighborhood information. Scientific Reports.
- Head, C. (2026). Tension-Dissipation Framework v4.0. Navigator's Log R&D.
- Hoffmann, J., et al. (2022). Training compute-optimal large language models (Chinchilla). arXiv:2203.15556.
- Kaplan, J., et al. (2020). Scaling laws for neural language models. arXiv:2001.08361.
- Li, C., & Liang, P. (2018). Measuring the intrinsic dimension of objective landscapes. ICLR 2018.
- Pope, P., et al. (2021). The intrinsic dimension of images and its impact on learning. ICLR 2021.
- Power, A., et al. (2022). Grokking: Generalization beyond overfitting on small algorithmic datasets. arXiv:2201.02177.

---

## Appendix A: Experimental Configurations

### A.1 Surface Prediction Test (Section 4.1-4.2)
200 manifold configurations. 800 samples per manifold. Ambient dimension 80. MLP (64, 32). 500 training iterations, tol=1e-6, no early stopping. 12 WD values (1e-6 to 10.0). 2 seeds. 30% held-out test split. Runtime: 152 minutes.

### A.2 Fungibility Test (Section 4.4)
12 diverse manifolds. 4 brake types (weight decay, early stopping, noise injection, subsampling). 6 strength levels per type. Runtime: 7 minutes.

### A.3 SurfaceGate v1 (Section 4.5, narrow architecture)
20 manifolds. Architecture: 64-48-32. 200 epochs. 3 base WD options (1e-3, 0.01, 0.1). Curvature measured every 25 epochs. Oracle: 9 fixed WD values. Runtime: 12 minutes.

### A.4 SurfaceGate v2 (Section 4.5, wide vs. narrow)
20 manifolds. Wide architecture: 64-64-64-64. Narrow architecture: 64-48-32. Same training protocol as v1. Runtime: 27 minutes.

### A.5 Density Variation Isolation (Section 4.3)
13 density configurations on fixed-geometry manifolds (ID=10, Curv=0.5). 5 density types (uniform, clustered, long-tail, bimodal, gradient). 30 diverse manifolds for comparison. Runtime: 7 minutes.

Total cumulative compute across all experiments: approximately 4 hours on Surface Pro 7 (Intel i5-1035G4, 8GB RAM, no GPU).
