#!/usr/bin/env python3
"""
TDF Surface Test v2 — NO EARLY STOPPING
==========================================
Navigator's Log R&D | March 2026

The first test showed R²=0 because early stopping self-regulated
the model, making explicit WD irrelevant. This version removes
early stopping entirely. The model trains to convergence (or
overfitting). In this regime, WD is the ONLY brake.

If the surface predicts optimal WD here, the insight is real but
only applies when the architecture doesn't self-regulate.

Estimated runtime on Surface Pro 7: ~25-35 minutes
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

N_CONFIGS = 200
N_SAMPLES = 800
AMBIENT_DIM = 80
WD_GRID = [1e-6, 1e-5, 1e-4, 1e-3, 5e-3, 0.01, 0.05, 0.1, 0.3, 1.0, 3.0, 10.0]
OUTPUT_DIR = "."
np.random.seed(42)

def make_manifold(n, id_dim, ambient, n_classes, curvature=0.0, noise=0.0, seed=0):
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

def measure_surface(X, n_sample=300):
    n_s = min(n_sample, len(X))
    ambient = X.shape[1]
    nn = NearestNeighbors(n_neighbors=3).fit(X)
    dist, _ = nn.kneighbors(X[:n_s])
    r1, r2 = dist[:, 1], dist[:, 2]
    mask = (r1 > 1e-10) & (r2 > 1e-10)
    mu = r2[mask] / r1[mask]
    id_est = len(mu) / max(1, np.sum(np.log(mu)))
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
        except: pass
    curv_est = np.mean(curvatures) if curvatures else 1e-6
    nn3 = NearestNeighbors(n_neighbors=8).fit(X)
    dist3, _ = nn3.kneighbors(X[:n_s])
    r_k = dist3[:, -1]; r_k = r_k[r_k > 1e-10]
    log_density = -np.log(r_k)
    dens_cv = np.std(log_density) / (np.abs(np.mean(log_density)) + 1e-10)
    codim_ratio = ambient / max(1, id_est)
    return id_est, curv_est, dens_cv, codim_ratio

def find_optimal_wd_no_earlystop(X, y, wd_grid):
    """
    KEY CHANGE: No early stopping. Train to full convergence.
    WD is the ONLY regularization. If WD is wrong, the model
    overfits catastrophically or underfits severely.
    """
    best_wd, best_acc = wd_grid[0], 0
    all_scores = {}

    for seed in range(2):
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=seed)
        mu = X_tr.mean(0); sd = X_tr.std(0) + 1e-8
        X_tr = (X_tr - mu) / sd; X_te = (X_te - mu) / sd

        for wd in wd_grid:
            clf = MLPClassifier(
                hidden_layer_sizes=(64, 32),  # Bigger network = more capacity to overfit
                alpha=wd,
                max_iter=500,                 # Train longer
                early_stopping=False,         # NO EARLY STOPPING — this is the key change
                random_state=seed,
                tol=1e-6,                     # Very tight tolerance — don't stop early
            )
            try:
                clf.fit(X_tr, y_tr)
                acc = clf.score(X_te, y_te)
            except:
                acc = 0
            if wd not in all_scores:
                all_scores[wd] = []
            all_scores[wd].append(acc)

    means = {wd: np.mean(s) for wd, s in all_scores.items()}
    best_wd = max(means, key=means.get)
    return best_wd, means[best_wd], means

def eval_single_wd(X, y, wd):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
    mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-8
    X_tr, X_te = (X_tr - mu) / sd, (X_te - mu) / sd
    clf = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        alpha=wd,
        max_iter=500,
        early_stopping=False,
        random_state=42,
        tol=1e-6,
    )
    clf.fit(X_tr, y_tr)
    return clf.score(X_te, y_te)

def main():
    start_time = time.time()

    print("=" * 70)
    print("  TDF SURFACE TEST v2 — NO EARLY STOPPING")
    print("  WD is the ONLY brake. Wrong WD = catastrophic overfitting.")
    print(f"  {N_CONFIGS} manifolds × {len(WD_GRID)} weight decays × 2 seeds")
    print("=" * 70)

    # Generate configs
    print(f"\n  Generating {N_CONFIGS} manifold configurations...")
    configs = []
    for i in range(N_CONFIGS):
        rng = np.random.RandomState(i + 1000)
        configs.append({
            'id_dim': int(np.exp(rng.uniform(np.log(2), np.log(60)))),
            'curvature': float(np.exp(rng.uniform(np.log(0.01), np.log(5.0)))),
            'noise': float(np.exp(rng.uniform(np.log(0.005), np.log(0.8)))),
            'n_classes': int(rng.choice([2, 3, 5, 8, 10])),
        })

    # Phase 1: Measure + find optimal WD
    print("\n  Phase 1: Measuring surfaces and finding optimal WD...")
    print("  (No early stopping — models train to convergence)")
    all_features = []
    all_targets = []
    all_records = []

    for i, cfg in enumerate(configs):
        if i % 20 == 0:
            elapsed = time.time() - start_time
            eta = (elapsed / max(1, i)) * (N_CONFIGS - i) / 60 if i > 0 else 0
            print(f"    [{i:3d}/{N_CONFIGS}]  Elapsed: {elapsed/60:.1f}min  ETA: {eta:.1f}min")

        X, y = make_manifold(N_SAMPLES, cfg['id_dim'], AMBIENT_DIM, cfg['n_classes'],
                            curvature=cfg['curvature'], noise=cfg['noise'], seed=5000 + i)
        surface = measure_surface(X)
        opt_wd, best_acc, all_scores = find_optimal_wd_no_earlystop(X, y, WD_GRID)

        lid = np.log10(max(1, surface[0]))
        lcv = np.log10(max(1e-6, surface[1]))
        ldcv = np.log10(max(1e-6, surface[2]))
        lcr = np.log10(max(0.1, surface[3]))

        features = [lid, lcv, ldcv, lcr, lid*lcv, lid*ldcv, lcv*ldcv, lid**2, lcv**2]
        all_features.append(features)
        all_targets.append(np.log10(max(1e-7, opt_wd)))
        all_records.append({
            'id_dim': cfg['id_dim'], 'curvature': cfg['curvature'],
            'noise': cfg['noise'], 'n_classes': cfg['n_classes'],
            'id': float(surface[0]), 'curv': float(surface[1]),
            'dcv': float(surface[2]), 'cr': float(surface[3]),
            'opt_wd': float(opt_wd), 'best_acc': float(best_acc),
            'wd_spread': float(max(all_scores.values()) - min(all_scores.values())),
        })

    X_feat = np.array(all_features)
    y_target = np.array(all_targets)

    # Check: does WD actually matter now?
    spreads = [r['wd_spread'] for r in all_records]
    print(f"\n  Phase 1 complete: {(time.time()-start_time)/60:.1f} minutes")
    print(f"\n  WD SENSITIVITY CHECK:")
    print(f"    Mean spread (best WD acc - worst WD acc): {np.mean(spreads):.4f}")
    print(f"    Max spread: {np.max(spreads):.4f}")
    print(f"    Configs where spread > 5%: {np.sum(np.array(spreads)>0.05)}/{N_CONFIGS}")
    print(f"    Configs where spread > 10%: {np.sum(np.array(spreads)>0.10)}/{N_CONFIGS}")

    if np.mean(spreads) < 0.02:
        print(f"\n  WARNING: WD spread is still small ({np.mean(spreads):.4f}).")
        print(f"  The architecture may still be self-regularizing even without early stopping.")

    # Phase 2: Train predictors
    print("\n  Phase 2: Training WD prediction models...")
    X_tr, X_te, y_tr, y_te = train_test_split(X_feat, y_target, test_size=0.3, random_state=42)
    idx_tr, idx_te = train_test_split(range(N_CONFIGS), test_size=0.3, random_state=42)

    from numpy.linalg import lstsq
    X_lin_tr = np.column_stack([X_tr[:, :4], np.ones(len(X_tr))])
    X_lin_te = np.column_stack([X_te[:, :4], np.ones(len(X_te))])
    coeffs_lin, _, _, _ = lstsq(X_lin_tr, y_tr, rcond=None)
    pred_lin = X_lin_te @ coeffs_lin
    r_lin, p_lin = pearsonr(pred_lin, y_te)

    rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    pred_rf = rf.predict(X_te)
    r_rf, p_rf = pearsonr(pred_rf, y_te)

    feat_names = ['log_ID', 'log_Curv', 'log_DCV', 'log_CR',
                  'ID×Curv', 'ID×DCV', 'Curv×DCV', 'ID²', 'Curv²']

    print(f"\n  WD PREDICTION QUALITY (on {len(y_te)} held-out manifolds):")
    print(f"  {'Model':<35s} {'r':>8s} {'R²':>8s} {'p':>10s}")
    print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*10}")
    print(f"  {'Linear (4 raw features)':<35s} {r_lin:>+8.3f} {r_lin**2:>8.3f} {p_lin:>10.2e}")
    print(f"  {'Random Forest (9 features)':<35s} {r_rf:>+8.3f} {r_rf**2:>8.3f} {p_rf:>10.2e}")

    print(f"\n  RF Feature Importances:")
    for name, imp in sorted(zip(feat_names, rf.feature_importances_), key=lambda x: -x[1]):
        bar = '█' * int(imp * 50)
        print(f"    {name:<12s}: {imp:.3f} {bar}")

    # Phase 3: Test adaptive vs fixed
    print(f"\n  Phase 3: Testing adaptive vs fixed on {len(idx_te)} held-out domains...")
    fixed_wds_test = [1e-4, 1e-3, 0.01, 0.1, 1.0]
    adaptive_accs, oracle_accs = [], []
    fixed_accs = {wd: [] for wd in fixed_wds_test}

    for j, i in enumerate(idx_te):
        if j % 10 == 0: print(f"    [{j}/{len(idx_te)}]")
        cfg = configs[i]
        X, y = make_manifold(N_SAMPLES, cfg['id_dim'], AMBIENT_DIM, cfg['n_classes'],
                            curvature=cfg['curvature'], noise=cfg['noise'], seed=5000 + i)
        pred_log_wd = rf.predict(X_feat[i:i+1])[0]
        pred_wd = 10 ** np.clip(pred_log_wd, -6, 2)
        adaptive_accs.append(eval_single_wd(X, y, pred_wd))
        for fwd in fixed_wds_test:
            fixed_accs[fwd].append(eval_single_wd(X, y, fwd))
        oracle_accs.append(all_records[i]['best_acc'])

    adapt_mean = np.mean(adaptive_accs)
    oracle_mean = np.mean(oracle_accs)
    best_fixed_wd = max(fixed_wds_test, key=lambda w: np.mean(fixed_accs[w]))
    best_fixed_mean = np.mean(fixed_accs[best_fixed_wd])
    adapt_arr = np.array(adaptive_accs)
    fixed_arr = np.array(fixed_accs[best_fixed_wd])
    oracle_arr = np.array(oracle_accs)
    gains = adapt_arr - fixed_arr
    wins = int(np.sum(gains > 0.005))
    ties = int(np.sum(np.abs(gains) <= 0.005))
    losses = int(np.sum(gains < -0.005))
    oracle_gain = oracle_mean - best_fixed_mean
    adapt_gain = adapt_mean - best_fixed_mean
    capture = (adapt_gain / oracle_gain * 100) if oracle_gain > 0.001 else 0
    r_track, _ = pearsonr(adapt_arr, oracle_arr)

    total_time = time.time() - start_time

    print(f"\n" + "=" * 70)
    print(f"  RESULTS — NO EARLY STOPPING ({len(idx_te)} held-out)")
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
    print(f"    Mean gain:       {adapt_gain:+.4f} ({adapt_gain/max(0.001,best_fixed_mean)*100:+.1f}%)")
    print(f"    Oracle capture:  {capture:.0f}%")
    print(f"    Tracks oracle:   r = {r_track:.3f}")

    print(f"\n  WD SENSITIVITY (does removing early stopping make WD matter?):")
    print(f"    Mean spread: {np.mean(spreads):.4f}")
    print(f"    Oracle - Best Fixed gap: {oracle_gain:.4f}")
    print(f"    (Compare to v1 gap of 0.018 with early stopping)")

    # Save results
    import json
    results_out = {
        'version': 'v2_no_early_stopping',
        'n_configs': N_CONFIGS,
        'runtime_minutes': total_time / 60,
        'wd_sensitivity': {
            'mean_spread': float(np.mean(spreads)),
            'max_spread': float(np.max(spreads)),
            'configs_gt_5pct': int(np.sum(np.array(spreads) > 0.05)),
            'configs_gt_10pct': int(np.sum(np.array(spreads) > 0.10)),
        },
        'prediction_r2': {
            'linear_4feat': float(r_lin**2),
            'random_forest': float(r_rf**2),
        },
        'feature_importances': {n: float(v) for n, v in zip(feat_names, rf.feature_importances_)},
        'adaptive_vs_fixed': {
            'best_fixed_wd': float(best_fixed_wd),
            'best_fixed_mean': float(best_fixed_mean),
            'adaptive_mean': float(adapt_mean),
            'oracle_mean': float(oracle_mean),
            'oracle_capture_pct': float(capture),
            'wins': wins, 'ties': ties, 'losses': losses,
            'mean_gain': float(adapt_gain),
            'tracks_oracle_r': float(r_track),
        },
    }
    with open(f'{OUTPUT_DIR}/tdf_surface_v2_results.json', 'w') as f:
        json.dump(results_out, f, indent=2)

    # Visualization
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(21, 12))
        fig.suptitle("Surface Test v2: No Early Stopping — WD Is the Only Brake",
                     fontsize=14, fontweight='bold', y=0.98, color='white')
        fig.patch.set_facecolor('#08080f')

        # P1: WD spread distribution
        ax = axes[0, 0]
        ax.hist(spreads, bins=25, color='#f1c40f', edgecolor='white', linewidth=0.3, alpha=0.8)
        ax.axvline(0.05, color='#e74c3c', linestyle='--', alpha=0.5, label='5% threshold')
        ax.set_xlabel('WD Spread (best - worst acc)', fontsize=9, color='#a0b0c0')
        ax.set_ylabel('Count', fontsize=9, color='#a0b0c0')
        ax.set_title(f'Does WD Matter Now?\n{np.sum(np.array(spreads)>0.05)} configs with >5% spread',
                     fontsize=10, color='#f1c40f')
        ax.legend(fontsize=7, facecolor='#08080f', edgecolor='#303040', labelcolor='#a0b0c0')
        ax.set_facecolor('#08080f'); ax.tick_params(colors='#606080')

        # P2: Predicted vs actual WD
        ax = axes[0, 1]
        ax.scatter(y_te, pred_rf, c='#4ecdc4', s=30, alpha=0.6, edgecolors='white', linewidths=0.3)
        lims = [min(y_te.min(), pred_rf.min())-0.5, max(y_te.max(), pred_rf.max())+0.5]
        ax.plot(lims, lims, '--', color='white', alpha=0.2)
        ax.set_xlabel('Actual log(Opt WD)', fontsize=9, color='#a0b0c0')
        ax.set_ylabel('Predicted', fontsize=9, color='#a0b0c0')
        ax.set_title(f'WD Prediction: r={r_rf:+.3f}, R²={r_rf**2:.3f}', fontsize=10, color='#4ecdc4')
        ax.set_facecolor('#08080f'); ax.tick_params(colors='#606080')

        # P3: Feature importances
        ax = axes[0, 2]
        si = np.argsort(rf.feature_importances_)
        cols = ['#4ecdc4' if rf.feature_importances_[i] > 0.08 else '#606080' for i in si]
        ax.barh(range(9), rf.feature_importances_[si], color=cols, edgecolor='white', linewidth=0.3)
        ax.set_yticks(range(9))
        ax.set_yticklabels([feat_names[i] for i in si], fontsize=7, color='#a0b0c0')
        ax.set_title('Feature Importances', fontsize=10, color='white')
        ax.set_facecolor('#08080f'); ax.tick_params(colors='#606080')

        # P4: Adaptive vs Fixed scatter
        ax = axes[1, 0]
        ax.scatter(fixed_arr, adapt_arr, c='#4ecdc4', s=30, alpha=0.6, edgecolors='white', linewidths=0.3)
        lims2 = [min(min(fixed_arr), min(adapt_arr))-0.05, max(max(fixed_arr), max(adapt_arr))+0.05]
        ax.plot(lims2, lims2, '--', color='white', alpha=0.2)
        above = np.sum(adapt_arr > fixed_arr)
        ax.set_xlabel(f'Fixed (WD={best_fixed_wd:.0e})', fontsize=9, color='#a0b0c0')
        ax.set_ylabel('Adaptive', fontsize=9, color='#a0b0c0')
        ax.set_title(f'{above}/{len(adapt_arr)} above diagonal', fontsize=10, color='#4ecdc4')
        ax.set_facecolor('#08080f'); ax.tick_params(colors='#606080')

        # P5: Gain histogram
        ax = axes[1, 1]
        ax.hist(gains, bins=20, color='#4ecdc4', edgecolor='white', linewidth=0.3, alpha=0.8)
        ax.axvline(0, color='#e74c3c', linestyle='--', alpha=0.5)
        ax.axvline(np.mean(gains), color='#f1c40f', linestyle='-', linewidth=2,
                   label=f'Mean: {np.mean(gains):+.3f}')
        ax.set_xlabel('Gain (Adaptive - Fixed)', fontsize=9, color='#a0b0c0')
        ax.set_title(f'W:{wins} T:{ties} L:{losses}', fontsize=10, color='white')
        ax.legend(fontsize=7, facecolor='#08080f', edgecolor='#303040', labelcolor='#a0b0c0')
        ax.set_facecolor('#08080f'); ax.tick_params(colors='#606080')

        # P6: Summary comparison (v1 vs v2)
        ax = axes[1, 2]
        strats = ['Best\nFixed', 'Adaptive', 'Oracle']
        vals = [best_fixed_mean, adapt_mean, oracle_mean]
        colors_s = ['#e74c3c', '#4ecdc4', '#f1c40f']
        bars = ax.bar(range(3), vals, color=colors_s, edgecolor='white', linewidth=0.5, alpha=0.8)
        for k, v in enumerate(vals):
            ax.text(k, v + 0.004, f'{v:.3f}', ha='center', fontsize=11,
                    color='#a0b0c0', fontweight='bold')
        ax.set_xticks(range(3))
        ax.set_xticklabels(strats, fontsize=9, color='#a0b0c0')
        ax.set_ylabel('Mean Accuracy', fontsize=9, color='#a0b0c0')
        ax.set_title(f'Oracle Capture: {capture:.0f}%\n(v1 was 1%)',
                     fontsize=11, color='#f1c40f' if capture > 10 else '#4ecdc4')
        ax.set_ylim(min(vals) - 0.08, max(vals) + 0.04)
        ax.set_facecolor('#08080f'); ax.tick_params(colors='#606080')

        plt.tight_layout(rect=[0, 0.02, 1, 0.93])
        fig.text(0.5, 0.005, "Christopher Head · Navigator's Log R&D · navigatorslog.netlify.app",
                 ha='center', fontsize=8, color='#404060')
        plt.savefig(f'{OUTPUT_DIR}/tdf_surface_v2_test.png', dpi=180, bbox_inches='tight', facecolor='#08080f')
        plt.close()
        print(f"\n  Visualization saved to tdf_surface_v2_test.png")
    except ImportError:
        print("\n  matplotlib not available — skipping visualization")

    print(f"  Results saved to tdf_surface_v2_results.json")
    print(f"\n  Total runtime: {total_time/60:.1f} minutes")
    print("  Done.")

if __name__ == '__main__':
    main()
