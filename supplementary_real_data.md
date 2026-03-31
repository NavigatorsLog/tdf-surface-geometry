# Supplementary Note: Real-Data Validation Results

**Christopher Head**
Navigator's Log R&D — Fresno, California
March 2026

Addendum to: "Surface Geometry of Data Manifolds Predicts Optimal Regularization in Neural Networks" (v3)
Zenodo DOI: 10.5281/zenodo.19298924

---

## Summary

The exponential WD decay schedule reported in the main paper (Section 4.9) was tested on two real datasets (MNIST, CIFAR-10) with two architectures (MLP, SimpleCNN). The results partially confirm and partially narrow the original synthetic-manifold findings. Per-layer WD variation improves performance when the architecture has sufficient capacity to smooth the data manifold. The specific decay direction (front-loaded vs. back-loaded) matters less on real data than on synthetic manifolds. On under-capacity architectures, uniform WD is optimal. All claims below are supported by the accompanying data files.

---

## 1. Experimental Setup

Datasets: MNIST (60,000 train / 10,000 test) and CIFAR-10 (50,000 train / 10,000 test).

Architectures: MLP (256-128-64-32, 4 hidden layers) and SimpleCNN (3 convolutional layers with max-pooling + 2 fully connected layers).

Schedules tested: Fixed (uniform WD), Exponential Decay (WD_i = base × 0.5^i), Linear Decay (WD decreasing linearly from full to 20% across layers), Reverse (WD increasing with depth — control).

Each schedule was tested with 5 base WD values (1e-5, 1e-4, 1e-3, 0.01, 0.1). Best base reported per schedule. MLP trained for 30 epochs, CNN for 20 epochs. Adam optimizer. No early stopping.

Total compute: 1,983 minutes (~33 hours) on Surface Pro 7 (Intel i5, no GPU).

---

## 2. Results

### 2.1 MNIST

| Schedule | MLP Accuracy | CNN Accuracy |
|---|---|---|
| Fixed | 0.9827 | 0.9932 |
| ExpDecay | **0.9836** (+0.0009) | 0.9929 (-0.0003) |
| LinDecay | 0.9834 (+0.0007) | 0.9933 (+0.0001) |
| Reverse | 0.9821 (-0.0006) | 0.9931 (-0.0001) |

MNIST MLP: Front-loaded schedules (ExpDecay, LinDecay) beat Fixed. Reverse is worst. Direction matters weakly — the ordering matches the synthetic result but margins are small (+0.09% absolute at a 98%+ ceiling).

MNIST CNN: All schedules within 0.04% of each other. The CNN's implicit regularization (pooling, weight sharing) dominates explicit WD. The schedule is irrelevant when the architecture provides sufficient Leg 3.

### 2.2 CIFAR-10

| Schedule | MLP Accuracy | CNN Accuracy |
|---|---|---|
| Fixed | **0.5454** | 0.7587 |
| ExpDecay | 0.5403 (-0.0051) | 0.7600 (+0.0013) |
| LinDecay | 0.5422 (-0.0032) | **0.7654** (+0.0067) |
| Reverse | 0.5450 (-0.0004) | 0.7613 (+0.0026) |

CIFAR-10 MLP: Fixed WD wins. All decay schedules lose. The 4-layer MLP lacks capacity to flatten CIFAR-10's high-dimensional manifold (ID approximately 20-24). Curvature persists through all layers, and every layer needs strong braking. Reducing WD at depth starves layers that still need it.

CIFAR-10 CNN: Per-layer variation helps. LinDecay wins by +0.67%, the largest gain in the test suite. However, Reverse also beats Fixed (+0.26%), indicating that the benefit comes from per-layer WD variation itself rather than from front-loading specifically.

---

## 3. Interpretation

### 3.1 What Holds

The total Leg 3 budget principle is confirmed on real data. When the architecture provides sufficient implicit regularization (CNN on MNIST), explicit WD is irrelevant regardless of schedule. This is the same masking effect reported in the main paper Section 4.2 (early stopping masked the surface signal).

Per-layer WD variation improves performance on capable architectures applied to hard tasks (CIFAR-10 CNN: +0.67% for LinDecay). Breaking the symmetry of uniform WD allows different layers to receive different regularization budgets, which the data suggests is beneficial when the architecture has enough capacity to process the task.

### 3.2 What Narrows

The exponential decay schedule is not universally optimal on real data. On synthetic manifolds, ExpDecay won 7/20 manifolds and was the clear best strategy. On real data, LinDecay outperforms ExpDecay on the hardest meaningful test (CIFAR-10 CNN). The gentler linear schedule preserves more regularization at depth, which real manifolds appear to need more than synthetic ones.

Directionality is weaker on real data than on synthetic manifolds. On synthetic manifolds, Reverse won 0/20 — a clean negative control. On real data (CIFAR-10 CNN), Reverse beats Fixed by +0.26%. The curvature profile measured in the main paper (curvature drops monotonically through depth) is a real phenomenon, but it does not translate into a strong directional advantage for WD scheduling on real datasets. The primary mechanism appears to be symmetry-breaking (any per-layer variation helps) rather than curvature-matching (front-loading specifically helps).

Under-capacity architectures want uniform braking. When the network cannot flatten the manifold (CIFAR-10 MLP at 54% accuracy), curvature persists at all layers and every layer needs strong regularization. Decay schedules create a catch-22: the base WD must be high enough for deep layers but this over-regularizes early layers, or vice versa. The schedule works only when the architecture has sufficient capacity for the curvature profile to emerge.

### 3.3 Revised Recommendation

For practitioners: Per-layer WD variation is a safe, zero-cost modification that improves or matches fixed WD when the architecture has capacity (model accuracy significantly above chance). Linear decay (gradual reduction across layers) is preferable to exponential decay on hard tasks. On easy tasks or strong architectures, the schedule has negligible effect — which means it can be applied universally without risk of harm. The worst-case penalty observed across all tests was -0.51% (CIFAR-10 MLP ExpDecay), which occurred in an under-capacity regime.

Revised one-line recommendation:

```python
# Gentle linear decay — safe default for any architecture
for i, layer in enumerate(model.layers):
    decay_factor = 1.0 - 0.6 * (i / (len(model.layers) - 1))
    optimizer.param_groups[i]['weight_decay'] = base_wd * decay_factor
```

---

## 4. Capacity Condition

The results suggest a capacity condition for per-layer WD scheduling: the schedule improves performance when the network has sufficient capacity to flatten the data manifold within its depth. A practical proxy: if the model achieves well above chance accuracy (indicating the architecture can represent the task), per-layer variation helps. If the model is near chance or struggling, uniform WD is preferable.

| Condition | Model Accuracy | Schedule Effect | Recommendation |
|---|---|---|---|
| Under-capacity | Near chance (~50% on 10-class) | Decay hurts | Use uniform WD |
| Adequate capacity | Well above chance (>70%) | Decay helps (+0.1-0.7%) | Use linear decay |
| Over-capacity / easy task | Near ceiling (>95%) | No effect | Schedule irrelevant |

This maps directly to the framework's prediction: the curvature profile (and thus the optimal schedule) depends on the network's ability to smooth the manifold. When the network can smooth it, curvature drops with depth and decay helps. When it cannot, curvature persists everywhere and uniform braking is correct.

---

## 5. Additional Result: Temperature as Domain-Specific Leg 3

Claude Sonnet was tested at 5 temperatures (0.0, 0.3, 0.5, 0.7, 1.0) across 5 domains (math, code, factual, creative, reasoning). Lexical diversity was used as a proxy for output variation.

| Domain | Most Sensitive | Least Sensitive | Diversity Range |
|---|---|---|---|
| Creative | 0.071 (best at T=1.0) | — | Highest |
| Math | 0.050 (best at T=0.7) | — | Moderate |
| Factual | 0.034 (best at T=0.5) | — | Moderate |
| Reasoning | 0.028 (best at T=1.0) | — | Low |
| Code | 0.011 (best at T=0.0) | — | Lowest |

Math and code produce nearly identical output from T=0.0 to T=0.5 — the training regularization dominates the inference temperature for constrained domains. Creative writing is the most temperature-sensitive, consistent with a higher-dimensional output manifold where inference-time Leg 3 (temperature) has room to explore. This supports the framework's prediction that domain-specific Goldilocks zones exist for regularization at inference time, not just during training.

---

## 6. Files

- `tdf_real_data_test.py` — Test script (MNIST + CIFAR-10, MLP + CNN, 4 schedules)
- `tdf_real_data_results.json` — Full results with all accuracies and base WDs
- `tdf_real_data_test.png` — Visualization
- `test3_temperature.json` — Temperature sweep results

All files available at: github.com/NavigatorsLog/tdf-surface-geometry

---

*Christopher Head — Navigator's Log R&D — navigatorslog.netlify.app*
