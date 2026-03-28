#!/usr/bin/env python3
"""
TDF Surface Geometry Test — Full Run
======================================
Navigator's Log R&D | March 2026
Christopher Head | navigatorslog.netlify.app

Run this on any machine with Python 3.8+ and:
  pip install numpy scikit-learn scipy matplotlib

Estimated runtime on Surface Pro 7: 30-45 minutes
No GPU needed. All CPU. All sklearn.

What it does:
1. Generates 200 synthetic manifolds with diverse surface geometries
2. Measures 4 surface properties on each (ID, curvature, density, codimension)
3. Finds optimal weight decay for each via grid search
4. Trains a Random Forest to predict optimal WD from surface properties
5. Tests whether surface-adaptive regularization beats fixed regularization
6. Generates a publication-quality visualization

The prediction: surface-adaptive brakes beat fixed brakes,
especially on domains whose geometry is far from average.
"""

import numpy as np
import time
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from scipy.stats import pearsonr, linregress
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# CONFIGURATION — adjust these if you want faster/slower runs
# ================================================================
N_CONFIGS = 200       # Number of manifold configurations (200 = full, 80 = quick)
N_SAMPLES = 800       # Points per manifold (800 = full, 400 = quick)
AMBIENT_DIM = 80      # Embedding dimension
N_SURFACE_SAMPLES = 300  # Points used for surface measurement
MLP_MAX_ITER = 200    # Training iterations per model
RF_TREES = 100        # Random Forest size
WD_GRID = [1e-6, 1e-5, 1e-4, 1e-3, 5e-3, 0.01, 0.05, 0.1, 0.3, 1.0, 3.0, 10.0]
SEEDS_PER_WD = 2      # Train each WD this many times (reduces noise)
OUTPUT_DIR = "."       # Where to save outputs (change to your preferred path)

np.random.seed(42)

# ================================================================
# MANIFOLD GENERATOR
# ================================================================
def make_manifold(n, id_dim, ambient, n_classes, curvature=0.0, noise=0.0, seed=0):
    """Generate data on a manifold with controlled surface properties."""
    rng = np.random.RandomState(seed)
    Z = rng.randn(n, id_dim)
    proj = Z[:, 0]
    labels = np.digitize(proj, np.linspace(proj.min(), proj.max(), 
                          min(n_classes, id_dim) + 1)[1:-1])
    A = rng.randn(id_dim, ambient) / np.sqrt(id_dim)
    X = Z @ A
    if curvature > 0:
        for i in range(min(id_dim, 3)):
            X += curvature * 0.1 * np.outer(Z[:, i]**2, A[i % id_dim])
    if noise > 0:
        X += noise * rng.randn(n, ambient)
    return X, labels

# ================================================================
# SURFACE MEASUREMENT
# ================================================================
def measure_surface(X, n_sample=None):
    """Measure 4 surface properties: ID, curvature, density CV, codimension ratio."""
    if n_sample is None:
        n_sample = N_SURFACE_SAMPLES
    n_s = min(n_sample, len(X))
    ambient = X.shape[1]
    
    # 1. Intrinsic Dimension (TwoNN estimator, Facco et al. 2017)
    nn = NearestNeighbors(n_neighbors=3).fit(X)
    dist, _ = nn.kneighbors(X[:n_s])
    r1, r2 = dist[:, 1], dist[:, 2]
    mask = (r1 > 1e-10) & (r2 > 1e-10)
    mu = r2[mask] / r1[mask]
    id_est = len(mu) / max(1, np.sum(np.log(mu)))
    
    # 2. Curvature proxy (residual variance beyond first few local PCs)
    nn2 = NearestNeighbors(n_neighbors=12).fit(X)
    _, indices = nn2.kneighbors(X[:min(n_s, 200)])
    curvatures = []
    for i in range(min(n_s, 200)):
        local = X[indices[i, 1:]] - X[indices[i, 1:]].mean(0)
        try:
            _, s, _ = np.linalg.svd(local, full_matrices=False)
            s = s[s > 1e-10]
            if len(s) > 3:
                cumvar = np.cumsum(s**2) / np.sum(s**2)
                curvatures.append(1.0 - cumvar[min(4, len(cumvar) - 1)])
        except:
            pass
    curv_est = np.mean(curvatures) if curvatures else 1e-6
    
    # 3. Density variation (coefficient of variation of KNN density)
    nn3 = NearestNeighbors(n_neighbors=8).fit(X)
    dist3, _ = nn3.kneighbors(X[:n_s])
    r_k = dist3[:, -1]
    r_k = r_k[r_k > 1e-10]
    log_density = -np.log(r_k)
    dens_cv = np.std(log_density) / (np.abs(np.mean(log_density)) + 1e-10)
    
    # 4. Codimension ratio
    codim_ratio = ambient / max(1, id_est)
    
    return {
        'id': id_est,
        'curvature': curv_est,
        'density_cv': dens_cv,
        'codim_ratio': codim_ratio,
    }

# ================================================================
# OPTIMAL WD FINDER
# ================================================================
def find_optimal_wd(X, y, wd_grid, n_seeds=None):
    """Find optimal weight decay via grid search with optional multi-seed averaging."""
    if n_seeds is None:
        n_seeds = SEEDS_PER_WD
    scores = {wd: [] for wd in wd_grid}
    
    for seed in range(n_seeds):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.25, random_state=seed)
        mu = X_tr.mean(0)
        sd = X_tr.std(0) + 1e-8
        X_tr = (X_tr - mu) / sd
        X_te = (X_te - mu) / sd
        
        for wd in wd_grid:
            clf = MLPClassifier(
                hidden_layer_sizes=(32,),
                alpha=wd,
                max_iter=MLP_MAX_ITER,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=seed
            )
            try:
                clf.fit(X_tr, y_tr)
                scores[wd].append(clf.score(X_te, y_te))
            except:
                scores[wd].append(0)
    
    means = {wd: np.mean(s) for wd, s in scores.items()}
    best_wd = max(means, key=means.get)
    return best_wd, means[best_wd], means

def eval_single_wd(X, y, wd):
    """Train and evaluate with a specific WD."""
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
    mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-8
    X_tr, X_te = (X_tr - mu) / sd, (X_te - mu) / sd
    clf = MLPClassifier(hidden_layer_sizes=(32,), alpha=wd, max_iter=MLP_MAX_ITER,
                       early_stopping=True, validation_fraction=0.15, random_state=42)
    clf.fit(X_tr, y_tr)
    return clf.score(X_te, y_te)

# ================================================================
# MAIN EXPERIMENT
# ================================================================
def main():
    start_time = time.time()
    
    print("=" * 70)
    print("  TDF SURFACE GEOMETRY TEST — FULL RUN")
    print(f"  {N_CONFIGS} manifolds × {len(WD_GRID)} weight decays × {SEEDS_PER_WD} seeds")
    print(f"  Estimated time: {N_CONFIGS * len(WD_GRID) * SEEDS_PER_WD * 0.8 / 60:.0f} minutes")
    print("=" * 70)
    
    # Generate random manifold configurations
    print(f"\n  Generating {N_CONFIGS} manifold configurations...")
    configs = []
    for i in range(N_CONFIGS):
        rng = np.random.RandomState(i + 1000)
        id_dim = int(np.exp(rng.uniform(np.log(2), np.log(60))))
        curvature = np.exp(rng.uniform(np.log(0.01), np.log(5.0)))
        noise = np.exp(rng.uniform(np.log(0.005), np.log(0.8)))
        n_classes = rng.choice([2, 3, 5, 8, 10])
        configs.append({
            'id_dim': id_dim,
            'curvature': curvature,
            'noise': noise,
            'n_classes': n_classes,
        })
    
    # Phase 1: Measure surfaces and find optimal WD
    print("\n  Phase 1: Measuring surfaces and finding optimal WD...")
    all_features = []
    all_targets = []
    all_records = []
    
    for i, cfg in enumerate(configs):
        if i % 20 == 0:
            elapsed = time.time() - start_time
            eta = (elapsed / max(1, i)) * (N_CONFIGS - i) / 60 if i > 0 else 0
            print(f"    [{i:3d}/{N_CONFIGS}]  Elapsed: {elapsed/60:.1f}min  ETA: {eta:.1f}min")
        
        X, y = make_manifold(
            N_SAMPLES, cfg['id_dim'], AMBIENT_DIM, cfg['n_classes'],
            curvature=cfg['curvature'], noise=cfg['noise'], seed=5000 + i
        )
        
        # Measure surface
        surface = measure_surface(X)
        
        # Find optimal WD
        opt_wd, best_acc, all_scores = find_optimal_wd(X, y, WD_GRID)
        
        # Build feature vector (raw + interactions)
        lid = np.log10(max(1, surface['id']))
        lcv = np.log10(max(1e-6, surface['curvature']))
        ldcv = np.log10(max(1e-6, surface['density_cv']))
        lcr = np.log10(max(0.1, surface['codim_ratio']))
        
        features = [
            lid, lcv, ldcv, lcr,           # Raw features
            lid * lcv,                       # ID × Curvature interaction
            lid * ldcv,                      # ID × Density interaction
            lcv * ldcv,                      # Curvature × Density interaction
            lid ** 2,                        # ID squared
            lcv ** 2,                        # Curvature squared
        ]
        
        all_features.append(features)
        all_targets.append(np.log10(max(1e-7, opt_wd)))
        all_records.append({
            **cfg, **surface,
            'opt_wd': opt_wd, 'best_acc': best_acc,
        })
    
    X_feat = np.array(all_features)
    y_target = np.array(all_targets)
    
    phase1_time = time.time() - start_time
    print(f"\n  Phase 1 complete: {phase1_time/60:.1f} minutes")
    
    # Phase 2: Train prediction models
    print("\n  Phase 2: Training WD prediction models...")
    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X_feat, y_target, range(N_CONFIGS), test_size=0.3, random_state=42
    )
    
    # Linear (4 features only)
    from numpy.linalg import lstsq
    X_lin_tr = np.column_stack([X_tr[:, :4], np.ones(len(X_tr))])
    X_lin_te = np.column_stack([X_te[:, :4], np.ones(len(X_te))])
    coeffs_lin, _, _, _ = lstsq(X_lin_tr, y_tr, rcond=None)
    pred_lin = X_lin_te @ coeffs_lin
    r_lin, p_lin = pearsonr(pred_lin, y_te)
    
    # Linear (all 9 features)
    X_full_tr = np.column_stack([X_tr, np.ones(len(X_tr))])
    X_full_te = np.column_stack([X_te, np.ones(len(X_te))])
    coeffs_full, _, _, _ = lstsq(X_full_tr, y_tr, rcond=None)
    pred_full = X_full_te @ coeffs_full
    r_full, p_full = pearsonr(pred_full, y_te)
    
    # Random Forest (all 9 features)
    rf = RandomForestRegressor(
        n_estimators=RF_TREES, max_depth=8, random_state=42, n_jobs=-1
    )
    rf.fit(X_tr, y_tr)
    pred_rf = rf.predict(X_te)
    r_rf, p_rf = pearsonr(pred_rf, y_te)
    
    feat_names = ['log_ID', 'log_Curv', 'log_DCV', 'log_CR',
                  'ID×Curv', 'ID×DCV', 'Curv×DCV', 'ID²', 'Curv²']
    
    print(f"\n  WD PREDICTION QUALITY (on {len(y_te)} held-out manifolds):")
    print(f"  {'Model':<35s} {'r':>8s} {'R²':>8s} {'p':>10s}")
    print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*10}")
    print(f"  {'Linear (4 raw features)':<35s} {r_lin:>+8.3f} {r_lin**2:>8.3f} {p_lin:>10.2e}")
    print(f"  {'Linear (9 feat + interactions)':<35s} {r_full:>+8.3f} {r_full**2:>8.3f} {p_full:>10.2e}")
    print(f"  {'Random Forest (9 features)':<35s} {r_rf:>+8.3f} {r_rf**2:>8.3f} {p_rf:>10.2e}")
    
    print(f"\n  RF Feature Importances:")
    for name, imp in sorted(zip(feat_names, rf.feature_importances_), key=lambda x: -x[1]):
        bar = '█' * int(imp * 50)
        print(f"    {name:<12s}: {imp:.3f} {bar}")
    
    # Phase 3: Test adaptive vs fixed
    print(f"\n  Phase 3: Testing adaptive vs fixed on {len(idx_te)} held-out domains...")
    
    fixed_wds_test = [1e-4, 1e-3, 0.01, 0.1, 1.0]
    adaptive_accs = []
    fixed_accs = {wd: [] for wd in fixed_wds_test}
    oracle_accs = []
    test_records = []
    
    for j, i in enumerate(idx_te):
        if j % 10 == 0:
            print(f"    [{j}/{len(idx_te)}]")
        
        cfg = configs[i]
        X, y = make_manifold(
            N_SAMPLES, cfg['id_dim'], AMBIENT_DIM, cfg['n_classes'],
            curvature=cfg['curvature'], noise=cfg['noise'], seed=5000 + i
        )
        
        # Predict WD from surface
        pred_log_wd = rf.predict(X_feat[i:i+1])[0]
        pred_wd = 10 ** np.clip(pred_log_wd, -6, 2)
        
        # Evaluate adaptive
        adapt_acc = eval_single_wd(X, y, pred_wd)
        adaptive_accs.append(adapt_acc)
        
        # Fixed baselines
        for fwd in fixed_wds_test:
            fixed_accs[fwd].append(eval_single_wd(X, y, fwd))
        
        # Oracle
        oracle_accs.append(all_records[i]['best_acc'])
        
        test_records.append({
            'idx': i,
            'pred_wd': pred_wd,
            'oracle_wd': all_records[i]['opt_wd'],
            'adapt_acc': adapt_acc,
            'oracle_acc': all_records[i]['best_acc'],
        })
    
    # Aggregate
    adapt_mean = np.mean(adaptive_accs)
    oracle_mean = np.mean(oracle_accs)
    best_fixed_wd = max(fixed_wds_test, key=lambda w: np.mean(fixed_accs[w]))
    best_fixed_mean = np.mean(fixed_accs[best_fixed_wd])
    
    adapt_arr = np.array(adaptive_accs)
    fixed_arr = np.array(fixed_accs[best_fixed_wd])
    oracle_arr = np.array(oracle_accs)
    gains = adapt_arr - fixed_arr
    
    wins = np.sum(gains > 0.005)
    ties = np.sum(np.abs(gains) <= 0.005)
    losses = np.sum(gains < -0.005)
    
    oracle_gain = oracle_mean - best_fixed_mean
    adapt_gain = adapt_mean - best_fixed_mean
    capture = (adapt_gain / oracle_gain * 100) if oracle_gain > 0 else 0
    
    r_track, _ = pearsonr(adapt_arr, oracle_arr)
    
    total_time = time.time() - start_time
    
    # Print results
    print(f"\n" + "=" * 70)
    print(f"  RESULTS ({len(idx_te)} held-out domains)")
    print(f"  Total runtime: {total_time/60:.1f} minutes")
    print("=" * 70)
    
    print(f"\n  Strategy comparison:")
    print(f"  {'Strategy':<35s} {'Mean Acc':>10s}")
    print(f"  {'-'*35} {'-'*10}")
    for fwd in sorted(fixed_wds_test):
        m = np.mean(fixed_accs[fwd])
        tag = " ← best" if fwd == best_fixed_wd else ""
        print(f"  Fixed WD = {fwd:<10.1e}                {m:>10.4f}{tag}")
    print(f"  {'Surface-Adaptive (RF)':<35s} {adapt_mean:>10.4f}")
    print(f"  {'Oracle (grid search)':<35s} {oracle_mean:>10.4f}")
    
    print(f"\n  Adaptive vs Best Fixed (WD={best_fixed_wd:.0e}):")
    print(f"    Win/Tie/Loss:    {wins}/{ties}/{losses} out of {len(idx_te)}")
    print(f"    Mean gain:       {adapt_gain:+.4f} ({adapt_gain/best_fixed_mean*100:+.1f}%)")
    print(f"    Oracle capture:  {capture:.0f}%")
    print(f"    Tracks oracle:   r = {r_track:.3f}")
    
    # Top wins and losses
    sorted_gains = sorted(enumerate(gains), key=lambda x: -x[1])
    print(f"\n  Top 5 wins for adaptive:")
    for j, g in sorted_gains[:5]:
        i = idx_te[j]
        rec = all_records[i]
        print(f"    ID={rec['id']:.0f} Curv={rec['curvature']:.3f}: Δ={g:+.3f}")
    
    print(f"\n  Top 3 losses:")
    for j, g in sorted_gains[-3:]:
        i = idx_te[j]
        rec = all_records[i]
        print(f"    ID={rec['id']:.0f} Curv={rec['curvature']:.3f}: Δ={g:+.3f}")
    
    # ================================================================
    # VISUALIZATION
    # ================================================================
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(21, 12))
        fig.suptitle(
            f"Surface Geometry Predicts the Goldilocks Zone\n"
            f"{N_CONFIGS} manifolds · RF predictor · Oracle capture: {capture:.0f}%",
            fontsize=14, fontweight='bold', y=0.98, color='white'
        )
        fig.patch.set_facecolor('#08080f')
        
        # P1: Predicted vs actual WD
        ax = axes[0, 0]
        ax.scatter(y_te, pred_rf, c='#4ecdc4', s=30, alpha=0.6, 
                   edgecolors='white', linewidths=0.3)
        lims = [min(y_te.min(), pred_rf.min())-0.5, max(y_te.max(), pred_rf.max())+0.5]
        ax.plot(lims, lims, '--', color='white', alpha=0.2)
        ax.set_xlabel('Actual log(Opt WD)', fontsize=9, color='#a0b0c0')
        ax.set_ylabel('Predicted log(Opt WD)', fontsize=9, color='#a0b0c0')
        ax.set_title(f'WD Prediction: r={r_rf:+.3f}, R²={r_rf**2:.3f}', 
                     fontsize=10, color='#4ecdc4')
        ax.set_facecolor('#08080f'); ax.tick_params(colors='#606080')
        
        # P2: Feature importance
        ax = axes[0, 1]
        si = np.argsort(rf.feature_importances_)
        cols = ['#4ecdc4' if rf.feature_importances_[i] > 0.08 else '#606080' for i in si]
        ax.barh(range(9), rf.feature_importances_[si], color=cols, 
                edgecolor='white', linewidth=0.3)
        ax.set_yticks(range(9))
        ax.set_yticklabels([feat_names[i] for i in si], fontsize=7, color='#a0b0c0')
        ax.set_xlabel('Importance', fontsize=9, color='#a0b0c0')
        ax.set_title('Surface Properties That Matter', fontsize=10, color='white')
        ax.set_facecolor('#08080f'); ax.tick_params(colors='#606080')
        
        # P3: Model comparison
        ax = axes[0, 2]
        models = ['Linear\n(4 feat)', 'Linear\n+interact', 'Random\nForest']
        r2s = [r_lin**2, r_full**2, r_rf**2]
        colors_m = ['#e74c3c', '#e8a87c', '#4ecdc4']
        bars = ax.bar(range(3), r2s, color=colors_m, edgecolor='white', linewidth=0.5)
        for k, r2 in enumerate(r2s):
            ax.text(k, r2+0.01, f'R²={r2:.3f}', ha='center', fontsize=9, color='#a0b0c0')
        ax.set_xticks(range(3))
        ax.set_xticklabels(models, fontsize=8, color='#a0b0c0')
        ax.set_ylabel('R²', fontsize=9, color='#a0b0c0')
        ax.set_title('Prediction Model Comparison', fontsize=10, color='white')
        ax.set_ylim(0, max(r2s) * 1.4)
        ax.set_facecolor('#08080f'); ax.tick_params(colors='#606080')
        
        # P4: Adaptive vs Fixed scatter
        ax = axes[1, 0]
        ax.scatter(fixed_arr, adapt_arr, c='#4ecdc4', s=30, alpha=0.6,
                   edgecolors='white', linewidths=0.3)
        lims2 = [min(min(fixed_arr), min(adapt_arr))-0.05, 
                 max(max(fixed_arr), max(adapt_arr))+0.05]
        ax.plot(lims2, lims2, '--', color='white', alpha=0.2)
        above = np.sum(adapt_arr > fixed_arr)
        ax.set_xlabel(f'Fixed (WD={best_fixed_wd:.0e})', fontsize=9, color='#a0b0c0')
        ax.set_ylabel('Adaptive', fontsize=9, color='#a0b0c0')
        ax.set_title(f'{above}/{len(adapt_arr)} above diagonal', fontsize=10, color='#4ecdc4')
        ax.set_facecolor('#08080f'); ax.tick_params(colors='#606080')
        
        # P5: Gain distribution
        ax = axes[1, 1]
        ax.hist(gains, bins=20, color='#4ecdc4', edgecolor='white', linewidth=0.3, alpha=0.8)
        ax.axvline(0, color='#e74c3c', linestyle='--', alpha=0.5)
        ax.axvline(np.mean(gains), color='#f1c40f', linestyle='-', linewidth=2,
                   label=f'Mean: {np.mean(gains):+.3f}')
        ax.set_xlabel('Accuracy Gain (Adaptive - Fixed)', fontsize=9, color='#a0b0c0')
        ax.set_ylabel('Count', fontsize=9, color='#a0b0c0')
        ax.set_title(f'W:{wins} T:{ties} L:{losses}', fontsize=10, color='white')
        ax.legend(fontsize=7, facecolor='#08080f', edgecolor='#303040', labelcolor='#a0b0c0')
        ax.set_facecolor('#08080f'); ax.tick_params(colors='#606080')
        
        # P6: Strategy summary
        ax = axes[1, 2]
        strats = ['Best\nFixed', 'Surface\nAdaptive', 'Oracle']
        vals = [best_fixed_mean, adapt_mean, oracle_mean]
        colors_s = ['#e74c3c', '#4ecdc4', '#f1c40f']
        bars = ax.bar(range(3), vals, color=colors_s, edgecolor='white', 
                      linewidth=0.5, alpha=0.8)
        for k, v in enumerate(vals):
            ax.text(k, v+0.004, f'{v:.3f}', ha='center', fontsize=11, 
                    color='#a0b0c0', fontweight='bold')
        ax.set_xticks(range(3))
        ax.set_xticklabels(strats, fontsize=9, color='#a0b0c0')
        ax.set_ylabel('Mean Accuracy', fontsize=9, color='#a0b0c0')
        ax.set_title(f'Oracle Capture: {capture:.0f}%', fontsize=11,
                     color='#f1c40f' if capture > 35 else '#4ecdc4')
        ax.set_ylim(min(vals) - 0.08, max(vals) + 0.04)
        ax.set_facecolor('#08080f'); ax.tick_params(colors='#606080')
        
        plt.tight_layout(rect=[0, 0.02, 1, 0.93])
        fig.text(0.5, 0.005, 
                 "Christopher Head · Navigator's Log R&D · navigatorslog.netlify.app",
                 ha='center', fontsize=8, color='#404060')
        
        outpath = f'{OUTPUT_DIR}/tdf_surface_full_test.png'
        plt.savefig(outpath, dpi=180, bbox_inches='tight', facecolor='#08080f')
        plt.close()
        print(f"\n  Visualization saved to {outpath}")
        
    except ImportError:
        print("\n  matplotlib not available — skipping visualization")
    
    # Save raw results
    import json
    results_out = {
        'n_configs': N_CONFIGS,
        'n_held_out': len(idx_te),
        'runtime_minutes': total_time / 60,
        'prediction_r2': {
            'linear_4feat': r_lin**2,
            'linear_9feat': r_full**2,
            'random_forest': r_rf**2,
        },
        'feature_importances': dict(zip(feat_names, rf.feature_importances_.tolist())),
        'adaptive_vs_fixed': {
            'best_fixed_wd': best_fixed_wd,
            'best_fixed_mean': best_fixed_mean,
            'adaptive_mean': adapt_mean,
            'oracle_mean': oracle_mean,
            'oracle_capture_pct': capture,
            'wins': int(wins),
            'ties': int(ties),
            'losses': int(losses),
            'mean_gain': float(adapt_gain),
            'tracks_oracle_r': float(r_track),
        },
    }
    
    json_path = f'{OUTPUT_DIR}/tdf_surface_results.json'
    with open(json_path, 'w') as f:
        json.dump(results_out, f, indent=2)
    print(f"  Raw results saved to {json_path}")
    
    print(f"\n  Total runtime: {total_time/60:.1f} minutes")
    print("  Done.")

if __name__ == '__main__':
    main()
