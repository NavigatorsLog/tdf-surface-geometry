# TDF Surface Geometry: Predicting Optimal Regularization from Data Manifold Properties

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Surface geometry of data manifolds predicts optimal neural network regularization.**

Four measurable properties of your data — intrinsic dimension, curvature, density variation, and codimension ratio — predict the optimal weight decay with 84% oracle capture and a 40/19/1 win/tie/loss record across 60 held-out domains.

📄 **Paper:** [Surface Geometry of Data Manifolds Predicts Optimal Regularization](paper.md)  
🔗 **Navigator's Log R&D:** [navigatorslog.netlify.app](https://navigatorslog.netlify.app)  
📧 **Contact:** powerpredixtable@proton.me

---

## Key Results

| Metric | With Early Stopping (v1) | Without Early Stopping (v2) |
|---|---|---|
| WD Prediction R² | 0.000 | **0.196** (p < 0.001) |
| Oracle Capture | 1% | **84%** |
| Win/Tie/Loss | 10/41/9 | **40/19/1** |
| Mean Gain over Fixed | +0.0% | **+3.3%** |
| Tracks Oracle | r = 0.951 | **r = 0.967** |

**Top feature: ID × Density CV (importance = 0.290).** Interaction terms between surface properties dominate — no single property exceeds 14% importance.

---

## Quick Start

### Requirements

```bash
pip install numpy scikit-learn scipy matplotlib
```

Optional (for hunchback replication and LLM temperature tests):
```bash
pip install torch torchvision anthropic
```

### Run the Core Test (~2.5 hours on a laptop)

```bash
python tdf_surface_v2_no_earlystop.py
```

This generates 200 synthetic manifolds, measures surface properties, finds optimal weight decay via grid search, trains a Random Forest predictor, and tests adaptive vs. fixed regularization on 60 held-out domains.

**Outputs:**
- `tdf_surface_v2_results.json` — Raw results
- `tdf_surface_v2_test.png` — 6-panel visualization

### Run Supporting Tests

```bash
# Test 1: Surface measurements on real datasets (MNIST, CIFAR-10, CIFAR-100)
python tdf_local_tests.py --test 1    # ~15 min, downloads datasets

# Test 2: Representation compression (generalizing vs memorizing networks)
python tdf_local_tests.py --test 2    # ~20 min

# Test 3: LLM temperature as domain-specific Leg 3
export ANTHROPIC_API_KEY="your-key"   # Linux/Mac
# $env:ANTHROPIC_API_KEY = "your-key" # Windows PowerShell
python tdf_local_tests.py --test 3    # ~5 min, ~$0.05 API cost
```

### Quick Validation (~20 min)

For a faster run to validate the methodology, edit `tdf_surface_v2_no_earlystop.py` and change `N_CONFIGS = 200` to `N_CONFIGS = 60`.

---

## Repository Contents

```
├── README.md                         # This file
├── paper.md                          # Full technical paper
├── tdf_surface_v2_no_earlystop.py    # Core test: 200 manifolds, no early stopping
├── tdf_surface_full_test.py          # Comparison test: with early stopping (R²=0)
├── tdf_local_tests.py                # Supporting tests (real datasets, hunchback, temperature)
├── results/
│   ├── tdf_surface_v2_results.json   # Core test results (84% oracle capture)
│   ├── tdf_surface_results.json      # Early stopping results (1% oracle capture)
│   ├── test2_hunchback.json          # Compression test results
│   └── *.png                         # Visualizations
└── tdf_theory_document_v3.pdf        # TDF framework (cosmological theory)
```

---

## How It Works

### The Tension-Dissipation Framework (TDF)

TDF is a cosmological theory that identifies three conservation laws producing structure at every physical scale:

1. **Energy (Leg 1):** Drives the flow. In ML: compute + data.
2. **Angular Momentum (Leg 2):** Creates circulation. In ML: gradient flow + attention.
3. **Dissipation (Leg 3):** Sheds energy, forces structure. In ML: regularization.

The framework predicts that **the surface geometry of the data manifold determines the Goldilocks zone for Leg 3.** Too little regularization → memorization (a "halo" — diffuse, unstructured, like dark matter without dissipation). Too much → underfitting (over-compressed, like a collapsed star). The optimal regularization produces a "galaxy" — compact, structured, generalizing.

### Surface Measurements

| Property | What It Measures | How |
|---|---|---|
| Intrinsic Dimension (ID) | How many degrees of freedom the data has | TwoNN (Facco et al. 2017) |
| Curvature | How crumpled the manifold is | Local PCA residual variance |
| Density CV (DCV) | How unevenly data covers the manifold | KNN density coefficient of variation |
| Codimension Ratio (CR) | How much room the manifold has to twist | Ambient dim / Intrinsic dim |

### Why Interactions Matter

Individual surface properties are weak predictors (each < 14% importance). Their **products** are strong:

- **ID × DCV (29%):** A high-dimensional manifold with non-uniform coverage creates unseen regions the model must generalize across. The product measures this exposure.
- **Curv × DCV (15%):** A crumpled manifold with non-uniform coverage creates many poorly-sampled folds. More aggressive regularization forces the model past local memorization toward global structure.

### Why Early Stopping Masks the Signal

Early stopping is a powerful implicit Leg 3. It monitors validation loss and halts training when overfitting begins, regardless of explicit weight decay. Result: all WD values produce similar accuracy. The surface prediction has nothing to predict.

Remove early stopping → WD becomes the only brake → wrong WD causes catastrophic overfitting (up to 30% accuracy loss) → the surface prediction becomes valuable.

**Implication:** Surface geometry predicts optimal *explicit* regularization specifically. For architectures with strong implicit regularization, the prediction's value is reduced.

---

## Connection to AGI

Different domains have measurably different surface geometries. Modular arithmetic (ID ≈ 2) needs WD ≈ 1.0. Natural images (ID ≈ 20–35) need WD ≈ 10⁻⁴. Language (estimated ID ≈ 200) has different requirements entirely. A single fixed regularization cannot satisfy all Goldilocks zones simultaneously.

This suggests two paths to cross-domain intelligence:

1. **Curvature-adaptive regularization:** A module that measures representation geometry at each layer and adjusts regularization in real time. Analogous to the brain's neuromodulatory systems.

2. **Hierarchical specialization:** Domain-specific modules with domain-appropriate brakes, connected by bridges. Analogous to the brain's cortical organization.

See [paper.md](paper.md) Section 5.4 for detailed discussion.

---

## Reproducing Our Results

The core result (84% oracle capture) should reproduce within ±5% due to random seed variation. The key qualitative findings — interaction terms dominating, early stopping masking the signal, adaptive beating fixed — should reproduce exactly.

Expected runtimes (consumer laptop, no GPU):

| Test | Runtime |
|---|---|
| Core test (200 configs, no early stopping) | ~2.5 hours |
| Core test (60 configs, quick validation) | ~45 min |
| Real dataset surfaces (MNIST, CIFAR-10, CIFAR-100) | ~15 min |
| Hunchback replication (MNIST real vs random labels) | ~20 min |
| LLM temperature sweep (needs API key) | ~5 min |

---

## Citation

```
@article{head2026surface,
  title={Surface Geometry of Data Manifolds Predicts Optimal Regularization in Neural Networks},
  author={Head, Christopher},
  year={2026},
  journal={Navigator's Log R\&D},
  url={https://navigatorslog.netlify.app}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

The Tension-Dissipation Framework (TDF) theory document is © Christopher Head 2026. The code and experimental methodology are freely available under MIT.

---

**Navigator's Log R&D** — Fresno, California  
*The smooth always wins. But the mounds are beautiful while they last.*
