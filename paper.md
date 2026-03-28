# Surface Geometry of Data Manifolds Predicts Optimal Regularization in Neural Networks

**Christopher Head**  
Navigator's Log R&D — Fresno, California  
powerpredixtable@proton.me | navigatorslog.netlify.app  
March 2026

---

## Abstract

We show that four measurable geometric properties of a data manifold — intrinsic dimension, curvature, density variation, and codimension ratio — predict the optimal weight decay for neural networks with R² = 0.196 (p < 0.001) and capture 84% of oracle grid-search performance on held-out domains (40 wins, 19 ties, 1 loss across 60 test manifolds). The key finding is that interaction terms between surface properties — particularly the product of intrinsic dimension and density variation (ID×DCV) — carry the dominant signal, while individual properties alone are weak predictors. This result emerges only when self-regularizing mechanisms (early stopping) are absent, confirming that surface geometry governs explicit regularization specifically. These findings derive from the Tension-Dissipation Framework (TDF), a cosmological theory that identifies three conservation laws (energy, angular momentum, dissipation) as the minimal set producing observed structure at every physical scale. We map these three "legs" to machine learning — compute (Leg 1), gradient flow (Leg 2), and regularization (Leg 3) — and demonstrate that the framework's predictions about the role of dissipation extend from astrophysics to artificial intelligence. All experiments run on a consumer laptop (Surface Pro 7) using standard Python libraries. Code and data are publicly available.

---

## 1. Introduction

### 1.1 The Hyperparameter Problem

Every neural network training run requires choosing a regularization strength — weight decay, dropout rate, or equivalent. The standard approach is grid search or Bayesian optimization: train multiple models with different values and select the winner. This is computationally expensive and provides no insight into *why* a particular value works.

We ask a different question: can measurable properties of the data manifold *predict* the optimal regularization before training begins?

### 1.2 The Tension-Dissipation Framework

The Tension-Dissipation Framework (TDF) is a cosmological theory proposing that all observed structure arises from three conservation laws operating together: conservation of energy (Leg 1), conservation of angular momentum (Leg 2), and dissipation (Leg 3). The framework has been tested against 18+ independent correlations across 12 scientific domains spanning 60 orders of magnitude in physical scale (mean |r| = 0.922), documented in the TDF Theory Document v3.0 (Head, 2026).

The framework's central prediction relevant to machine learning: **Leg 3 (dissipation/braking) determines the Goldilocks zone for structure formation.** Too little braking produces diffuse, unstructured systems (dark matter halos in astrophysics; memorization in ML). Too much braking prevents structure from forming at all (over-regularized, underfitting models). The optimal braking strength is governed by the surface geometry of the system — the boundary where structure meets background.

### 1.3 Mapping TDF to Machine Learning

| TDF Component | ML Equivalent | Role |
|---|---|---|
| Energy (Leg 1) | Compute + Data | Drives the flow |
| Angular Momentum (Leg 2) | Gradient flow + Attention | Circulates information |
| Dissipation (Leg 3) | Regularization (WD, dropout, temp) | Sheds structure, forces generalization |
| Mound | Learned representation | Temporary structure sustained by energy |
| Smooth | Noise floor / prior | Background toward which all structure trends |
| Throat | Loss minimum | Constriction where flow transitions |
| Halo (2-leg system) | Memorizing network | No Leg 3 → diffuse, no generalization |
| Galaxy (3-leg system) | Generalizing network | Leg 3 → compact, structured representations |

The prediction: **the surface geometry of the data manifold — the boundary where learned structure meets noise — determines the optimal strength of Leg 3 (regularization).**

---

## 2. Related Work

**Intrinsic dimension of data representations.** Ansuini et al. (2019, NeurIPS) measured intrinsic dimension (ID) across layers of trained networks, finding a "hunchback" shape where ID rises then falls. The ID of the last hidden layer predicts test accuracy. Networks trained on random labels show flat ID profiles — no compression. Pope et al. (2021, ICLR) measured ID of standard datasets (MNIST ID ≈ 13, CIFAR-10 ID ≈ 20, ImageNet ID ≈ 26–43), showing that lower-ID datasets require fewer samples to learn.

**Intrinsic dimension of objective landscapes.** Li and Liang (2018, ICLR) measured the intrinsic dimension of the solution space, finding that problem difficulty varies across tasks (inverted pendulum is 100× easier than MNIST by this measure).

**Geometry-aware fine-tuning.** GeLoRA (EMNLP 2025) uses intrinsic dimension of representations at each layer to set LoRA adapter ranks. This is the closest prior work to our approach — using manifold geometry to prescribe an architectural choice — but targets fine-tuning rank allocation rather than regularization strength, and uses only ID, not the multi-axis surface vector.

**Scaling laws.** Kaplan et al. (2020) and Hoffmann et al. (2022, "Chinchilla") established power-law relationships between model size, data, compute, and loss. These laws are treated as empirical constants. We connect them to the TDF framework: the Chinchilla optimum (N_opt ∝ C^0.50) represents a balanced three-leg equilibrium where compute (Leg 1) and data (Leg 2) are optimally matched.

**Grokking.** Power et al. (2022) discovered delayed generalization in overparameterized networks trained on algorithmic tasks. Clauw et al. (2024) and subsequent work established grokking as a phase transition. We show that the grokking threshold follows a power law with regularization strength (r = −0.996), matching the universality class of physical phase transitions (r = +0.998 for critical temperature vs. interaction energy across 9 physical systems).

**Gap in the literature.** No prior work measures *multiple* surface properties (ID, curvature, density variation, codimension) as a vector and uses their *interactions* to predict optimal regularization.

---

## 3. Methods

### 3.1 Surface Property Measurements

We measure four properties of each data manifold:

**Intrinsic Dimension (ID).** Estimated via the TwoNN method (Facco et al., 2017), which uses only the ratio of first and second nearest-neighbor distances. Asymptotically correct for non-uniform distributions.

**Curvature proxy.** For each point, we compute local PCA on k=12 nearest neighbors and measure the fraction of variance not captured by the first 5 principal components. Higher residual variance indicates the manifold bends out of its local tangent plane — higher curvature.

**Density variation (DCV).** The coefficient of variation of k-nearest-neighbor log-density estimates. Higher DCV indicates more non-uniform coverage of the manifold.

**Codimension ratio (CR).** The ratio of ambient dimension to intrinsic dimension. Higher CR indicates more "room" for the manifold to twist in its embedding space.

### 3.2 Feature Engineering

In addition to the four raw features (in log space), we compute five interaction and quadratic terms: ID×Curv, ID×DCV, Curv×DCV, ID², Curv². The rationale: the TDF framework predicts that the shedding rate depends on how surface properties *combine*, not on any single axis.

### 3.3 Synthetic Manifold Generation

We generate synthetic manifolds with controlled properties by embedding low-dimensional Gaussian data into higher-dimensional space via random linear projection, adding polynomial curvature terms, and injecting off-manifold Gaussian noise. Classification labels are assigned by partitioning the manifold along its first coordinate. This allows independent variation of ID (2–60), curvature (0.01–5.0), noise (0.005–0.8), and number of classes (2–10).

### 3.4 Optimal Weight Decay Search

For each manifold, we train an MLP classifier (hidden layers: 64, 32) with each of 12 weight decay values (1e-6 to 10.0) across 2 random seeds. We average test accuracy across seeds and select the WD with highest mean accuracy as the oracle optimal.

**Critical design choice:** No early stopping. Models train for 500 iterations with tolerance 1e-6. This ensures weight decay is the *only* regularization mechanism, isolating the effect we aim to predict.

### 3.5 Prediction Models

**Linear regression** on the 4 raw log-space features plus intercept.

**Random Forest** (100 trees, max depth 8) on all 9 features (4 raw + 5 interactions).

Both are evaluated on a held-out 30% test split (60 manifolds).

### 3.6 Adaptive vs. Fixed Evaluation

On the held-out manifolds, we compare three strategies: (1) best fixed WD (single value used for all domains), (2) surface-adaptive WD (predicted by the trained RF from surface measurements), (3) oracle (per-domain grid search).

---

## 4. Results

### 4.1 Weight Decay Sensitivity

Without early stopping, weight decay has substantial impact on performance:

- Mean spread (best WD accuracy − worst WD accuracy): **7.6%**
- Maximum spread: **30.0%**
- Configurations with >5% spread: **108/200 (54%)**
- Configurations with >10% spread: **55/200 (28%)**

For comparison, with early stopping (v1), all fixed WD values produced nearly identical accuracy (spread < 0.2%), confirming that early stopping masked the WD signal entirely.

### 4.2 WD Prediction Quality

| Model | r | R² | p |
|---|---|---|---|
| Linear (4 features) | +0.443 | 0.196 | 3.95 × 10⁻⁴ |
| Random Forest (9 features) | +0.390 | 0.152 | 2.09 × 10⁻³ |

The linear model outperforms the Random Forest, indicating the relationship is approximately linear in log space. Both are statistically significant (p < 0.005).

### 4.3 Feature Importances

RF feature importances reveal that **interaction terms dominate**:

| Feature | Importance |
|---|---|
| **ID × DCV** | **0.290** |
| Curv × DCV | 0.145 |
| log DCV | 0.135 |
| log CR | 0.097 |
| log ID | 0.086 |
| ID × Curv | 0.078 |
| ID² | 0.067 |
| log Curv | 0.062 |
| Curv² | 0.040 |

The product of intrinsic dimension and density variation (ID×DCV) alone accounts for 29% of the RF's predictive power. Density variation appears in 3 of the top 4 features. No single raw property exceeds 14% importance.

This interaction finding is **robust across sample sizes**: Curv×DCV and ID×DCV were the top features in the 20-configuration pilot, the 60-configuration intermediate test, and the 200-configuration final test.

### 4.4 Adaptive vs. Fixed Performance

| Strategy | Mean Accuracy |
|---|---|
| Fixed WD = 1e-4 | 0.828 |
| Fixed WD = 1e-3 | 0.829 |
| Fixed WD = 1e-2 | 0.829 |
| Fixed WD = 1e-1 | 0.834 |
| Fixed WD = 1.0 | 0.868 (best fixed) |
| **Surface-Adaptive (RF)** | **0.896** |
| Oracle (grid search) | 0.902 |

**Oracle capture: 84%.** The surface-adaptive method closes 84% of the gap between the best fixed WD and the per-domain oracle.

**Win/Tie/Loss: 40/19/1.** The surface-adaptive method outperforms the best fixed WD on 40 out of 60 held-out domains, ties on 19, and loses on only 1.

**Tracks oracle: r = 0.967.** The adaptive method's per-domain accuracy correlates almost perfectly with the oracle's accuracy, indicating it successfully identifies which domains need different brakes.

### 4.5 Representation Compression (Partial Hunchback Replication)

We trained MLPs on MNIST with real labels and random labels and measured intrinsic dimension at each layer:

| Layer | Real Labels (ID) | Random Labels (ID) | Difference |
|---|---|---|---|
| Input | 10.6 | 10.6 | 0.0 |
| Layer 1 (ReLU) | 8.1 | 9.7 | −1.6 |
| Layer 3 (ReLU) | 6.8 | 9.0 | −2.2 |
| Layer 5 (ReLU) | 6.2 | 5.8 | +0.4 |
| Layer 7 (ReLU) | 5.8 | 5.0 | +0.7 |
| Output | 5.3 | 4.8 | +0.5 |

The generalizing network compresses representations **2.2 dimensions more** than the memorizing network in early-to-middle layers (layers 1–3). This difference disappears in later layers where architectural bottlenecks (decreasing layer width) force compression regardless of task structure. The Ansuini "hunchback" shape (ID rise then fall) was not observed because the architecture lacked an expansion phase — all layers are narrower than the input. A deeper network with at least one layer wider than the input would be required for full replication.

### 4.6 Supporting Results

**Grokking as phase transition.** The critical data fraction for grokking follows a power law with regularization strength: D_crit ∝ λ^(−0.22), r = −0.996. This matches the universality class of physical phase transitions (T_c vs. interaction energy, r = +0.998 across 9 systems spanning 10 orders of magnitude in temperature).

**Information Tully-Fisher.** The three-leg topology confirmed across four non-physics domains: cities (Bettencourt scaling β ≈ 0.83 sublinear infrastructure, β ≈ 1.15 superlinear social output), organisms (Kleiber's law BMR ∝ M^0.75), genomes (transcription rate ∝ Genome^−0.17, r = −0.957), and cross-domain (exponent tracks interaction dimensionality, r = +0.70).

**ML scaling laws.** Kaplan's loss ∝ N^−0.076 across 7 orders of magnitude maps to Leg 1 (compute/parameters). Chinchilla's N_opt ∝ C^0.50 is the balanced three-leg equilibrium. Double descent is bistability at the capacity threshold.

---

## 5. Discussion

### 5.1 Why Interactions Dominate

The finding that ID×DCV is the strongest predictor has a physical interpretation within TDF. A high-dimensional manifold with uniform data coverage is navigable — every region is well-sampled, so the model can learn global structure with moderate brakes. A high-dimensional manifold with *non-uniform* coverage creates a problem: some regions are over-sampled (inviting memorization), others are under-sampled (requiring generalization from few examples). The *product* of dimension and density variation captures this interaction — it measures how much of the surface is poorly covered, which determines how aggressively the model needs to prune.

### 5.2 Why Early Stopping Masked the Signal

Early stopping acts as a powerful implicit Leg 3. It monitors validation loss and halts training when overfitting begins, regardless of the explicit weight decay value. With early stopping active, the model self-regulates: low WD runs train longer (more iterations before overfitting), high WD runs train shorter (regularization prevents overfitting but also slows learning). The net effect is that all WD values produce similar final performance — the early stopping compensates.

This has a practical implication: **surface geometry predicts optimal *explicit* regularization only when the architecture does not already self-regulate.** For architectures with strong implicit regularization (early stopping, batch normalization, dropout, attention-based selection), the surface prediction's value is reduced because the implicit brakes dominate.

### 5.3 The 84% Oracle Capture

The surface-adaptive method captures 84% of the oracle's improvement over fixed WD, using only four pre-training measurements and a linear model. The remaining 16% likely resides in surface properties not captured by our four measurements (fractal dimension of decision boundaries, geodesic connectivity, symmetry group structure, local-global curvature mismatch) or in nonlinearities beyond the linear model's capacity.

### 5.4 Implications for AGI Architecture

The TDF framework predicts that cross-domain generalization requires domain-specific Leg 3 (regularization). Different data domains have different surface geometries: modular arithmetic has ID ≈ 2 with optimal WD ≈ 1.0; natural images have ID ≈ 20–35 with optimal WD ≈ 10⁻⁴; language has estimated ID ≈ 200.

An AGI system operating across all these domains simultaneously cannot satisfy all Goldilocks zones with a single fixed regularization. Two architectural solutions exist:

1. **Curvature-adaptive regularization:** A module that measures the surface geometry of the current representation at each layer and adjusts weight decay accordingly. This is analogous to the brain's neuromodulatory systems (dopamine, serotonin), which modulate plasticity based on context rather than carrying information content.

2. **Hierarchical specialization:** Domain-specific modules with domain-appropriate brakes, connected by bridges that transfer information without transferring regularization parameters. This is the brain's architecture: different cortical areas have different pruning schedules, connected by white matter tracts.

The surface geometry test provides a quantitative framework for deciding between these approaches: if the curvature-adaptive module captures >80% of oracle performance across diverse domains (as it does in our synthetic tests), adaptive regularization may be sufficient. If performance plateaus well below oracle, hierarchical specialization is required.

---

## 6. Limitations

**Synthetic manifolds only.** The 84% oracle capture was demonstrated on synthetic manifolds with controlled properties. Real-world datasets have additional confounds (optimizer choice, architecture-specific implicit regularization, representation learning dynamics) that may reduce the surface prediction's effectiveness.

**Small networks.** All experiments used 2-layer MLPs. The relationship between surface geometry and optimal regularization may differ for deep networks, convolutional architectures, or transformers where implicit regularization is stronger.

**Four surface properties.** We measure a subset of the manifold's geometric properties. Additional measurements (fractal dimension, persistent homology, Riemannian curvature tensor) could improve prediction quality but at higher computational cost.

**Linear prediction model.** The R² = 0.196 from the linear model leaves 80% of variance unexplained. A nonlinear predictor with more training data could potentially capture more signal, but risks overfitting.

---

## 7. Conclusion

The surface geometry of data manifolds predicts optimal neural network regularization. The prediction requires multiple geometric properties measured together — intrinsic dimension alone is insufficient (R² ≈ 0 in isolation). The interaction between intrinsic dimension and density variation is the dominant signal. When self-regularizing mechanisms are absent, the surface-adaptive approach captures 84% of oracle performance with 40 wins and 1 loss across 60 held-out domains.

These findings emerge from applying the Tension-Dissipation Framework — a cosmological theory tested across 12 scientific domains — to machine learning. The framework's prediction that dissipation (Leg 3) governs the Goldilocks zone for structure formation extends from galaxy evolution to neural network training. The "surface of the mound" — the boundary where structure meets noise — determines how much a system needs to forget in order to generalize.

All code is available at github.com/navigatorslog/tdf-surface-geometry. Total compute: 2.5 hours on a Surface Pro 7 (Intel i5, no GPU).

---

## References

- Ansuini, A., Laio, A., Macke, J.H., & Zoccolan, D. (2019). Intrinsic dimension of data representations in deep neural networks. NeurIPS 2019.
- Clauw, L., et al. (2024). Grokking as a phase transition. ScienceDirect.
- Facco, E., d'Errico, M., Rodriguez, A., & Laio, A. (2017). Estimating the intrinsic dimension of datasets by a minimal neighborhood information. Scientific Reports.
- Head, C. (2026). Tension-Dissipation Framework v3.0. Navigator's Log R&D. navigatorslog.netlify.app.
- Hoffmann, J., et al. (2022). Training compute-optimal large language models (Chinchilla). arXiv:2203.15556.
- Kaplan, J., et al. (2020). Scaling laws for neural language models. arXiv:2001.08361.
- Li, C., & Liang, P. (2018). Measuring the intrinsic dimension of objective landscapes. ICLR 2018.
- Pope, P., et al. (2021). The intrinsic dimension of images and its impact on learning. ICLR 2021.
- Power, A., et al. (2022). Grokking: Generalization beyond overfitting on small algorithmic datasets. arXiv:2201.02177.

---

## Appendix A: Experimental Configuration

| Parameter | Value |
|---|---|
| Manifold configurations | 200 |
| Samples per manifold | 800 |
| Ambient dimension | 80 |
| MLP architecture | 64, 32 (no early stopping) |
| Training iterations | 500 (tol = 1e-6) |
| WD grid | 1e-6, 1e-5, 1e-4, 1e-3, 5e-3, 0.01, 0.05, 0.1, 0.3, 1.0, 3.0, 10.0 |
| Seeds per WD | 2 |
| Test split | 30% (60 held-out manifolds) |
| Surface measurement samples | 300 |
| Total runtime | 152.5 minutes (Surface Pro 7, Intel i5) |
