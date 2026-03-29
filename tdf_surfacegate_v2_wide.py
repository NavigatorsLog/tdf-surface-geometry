#!/usr/bin/env python3
"""
TDF SurfaceGate v2 — Wide Network (No Architectural Compression)
===================================================================
Navigator's Log R&D | March 2026

v1 showed SurfaceGate beat the oracle (204% capture) but the
mechanism was simpler than expected: the gate mainly learned to
scale WD down globally because the narrowing architecture (64→48→32)
was already compressing the representation.

This test removes the architectural compression. All hidden layers
are the SAME WIDTH (64-64-64-64). The architecture provides NO
implicit Leg 3 through narrowing. If the gate still helps — and
especially if it now differentiates between layers — the per-layer
adaptation is real, not just a learned global scaler.

Also tests a DEEPER network (4 hidden layers instead of 3) to give
the gate more surface to adapt to.

Requirements: pip install torch numpy scikit-learn scipy matplotlib
Runtime: ~15-25 minutes on Surface Pro 7
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from scipy.stats import pearsonr
import time
import json
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = "."
np.random.seed(42)
torch.manual_seed(42)

def make_manifold(n, id_dim, ambient, n_classes, curvature=0.5, noise=0.1, seed=0):
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
    return X.astype(np.float32), labels.astype(np.int64)

def measure_layer_surface(activations, n_sample=200):
    X = activations.detach().cpu().numpy()
    if len(X) < 20 or X.shape[1] < 3:
        return 0.5
    n_s = min(n_sample, len(X))
    X_sub = X[:n_s]
    try:
        nn = NearestNeighbors(n_neighbors=min(10, n_s-1)).fit(X_sub)
        _, indices = nn.kneighbors(X_sub[:min(100, n_s)])
        curvatures = []
        for i in range(min(100, n_s)):
            local = X_sub[indices[i, 1:]] - X_sub[indices[i, 1:]].mean(0)
            try:
                _, s, _ = np.linalg.svd(local, full_matrices=False)
                s = s[s > 1e-10]
                if len(s) > 2:
                    cumvar = np.cumsum(s**2) / np.sum(s**2)
                    curvatures.append(1.0 - cumvar[min(3, len(cumvar)-1)])
            except: pass
        if curvatures:
            return float(np.mean(curvatures))
    except: pass
    return 0.5

class FixedWDModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, n_classes):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class SurfaceGateModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, n_classes, base_wd=0.01):
        super().__init__()
        self.base_wd = base_wd
        self.layers = nn.ModuleList()
        self.activations_list = []
        prev = input_dim
        for h in hidden_dims:
            self.layers.append(nn.Linear(prev, h))
            prev = h
        self.output_layer = nn.Linear(prev, n_classes)
        self.gate_sensitivity = nn.ParameterList([
            nn.Parameter(torch.tensor(1.0)) for _ in hidden_dims
        ])
        self.layer_curvatures = [0.5] * len(hidden_dims)
        self.layer_wd_multipliers = [1.0] * len(hidden_dims)
    
    def forward(self, x):
        self.activations_list = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = torch.relu(x)
            self.activations_list.append(x)
        return self.output_layer(x)
    
    def measure_surfaces(self):
        for i, act in enumerate(self.activations_list):
            curv = measure_layer_surface(act)
            self.layer_curvatures[i] = curv
            sensitivity = torch.sigmoid(self.gate_sensitivity[i]).item()
            self.layer_wd_multipliers[i] = float(
                np.exp(curv * sensitivity * 3) / np.exp(0.5 * sensitivity * 3)
            )
    
    def get_adaptive_wd_loss(self):
        wd_loss = torch.tensor(0.0)
        for i, layer in enumerate(self.layers):
            multiplier = self.layer_wd_multipliers[i]
            effective_wd = self.base_wd * multiplier
            wd_loss = wd_loss + effective_wd * torch.sum(layer.weight ** 2)
        wd_loss = wd_loss + self.base_wd * torch.sum(self.output_layer.weight ** 2)
        return wd_loss

def train_fixed_wd(X_tr, y_tr, X_te, y_te, hidden_dims, n_classes, wd, epochs=200):
    model = FixedWDModel(X_tr.shape[1], hidden_dims, n_classes)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()
    dataset = TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr))
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    for epoch in range(epochs):
        model.train()
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        pred = model(torch.FloatTensor(X_te)).argmax(dim=1).numpy()
    return float(np.mean(pred == y_te))

def train_surface_gate(X_tr, y_tr, X_te, y_te, hidden_dims, n_classes,
                       base_wd=0.01, epochs=200, measure_every=20):
    model = SurfaceGateModel(X_tr.shape[1], hidden_dims, n_classes, base_wd=base_wd)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
    criterion = nn.CrossEntropyLoss()
    dataset = TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr))
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    for epoch in range(epochs):
        model.train()
        for bx, by in loader:
            optimizer.zero_grad()
            cls_loss = criterion(model(bx), by)
            wd_loss = model.get_adaptive_wd_loss()
            (cls_loss + wd_loss).backward()
            optimizer.step()
        if (epoch + 1) % measure_every == 0:
            model.eval()
            with torch.no_grad():
                idx = np.random.choice(len(X_tr), min(300, len(X_tr)), replace=False)
                _ = model(torch.FloatTensor(X_tr[idx]))
            model.measure_surfaces()
            model.train()
    
    model.eval()
    with torch.no_grad():
        pred = model(torch.FloatTensor(X_te)).argmax(dim=1).numpy()
    return float(np.mean(pred == y_te)), model.layer_curvatures, model.layer_wd_multipliers

def main():
    start = time.time()
    
    print("=" * 70)
    print("  SURFACEGATE v2 — WIDE NETWORK (no architectural compression)")
    print("  All hidden layers same width: 64-64-64-64")
    print("  The architecture provides NO narrowing. The gate must do the work.")
    print("=" * 70)
    
    # KEY CHANGE: all layers same width, 4 layers deep
    WIDE_DIMS = [64, 64, 64, 64]
    # Comparison: the original narrowing architecture
    NARROW_DIMS = [64, 48, 32]
    
    EPOCHS = 200
    N_MANIFOLDS = 20
    BASE_WDS = [1e-3, 0.01, 0.1]
    
    print(f"\n  Generating {N_MANIFOLDS} diverse manifolds...")
    manifolds = []
    for i in range(N_MANIFOLDS):
        rng = np.random.RandomState(i + 3000)
        id_dim = int(np.exp(rng.uniform(np.log(3), np.log(40))))
        curvature = float(np.exp(rng.uniform(np.log(0.1), np.log(3.0))))
        noise = float(np.exp(rng.uniform(np.log(0.01), np.log(0.5))))
        n_classes = int(rng.choice([2, 3, 5, 8]))
        X, y = make_manifold(800, id_dim, 60, n_classes,
                            curvature=curvature, noise=noise, seed=4000+i)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
        mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-8
        X_tr, X_te = (X_tr - mu) / sd, (X_te - mu) / sd
        manifolds.append({
            'X_tr': X_tr, 'X_te': X_te, 'y_tr': y_tr, 'y_te': y_te,
            'n_classes': n_classes, 'id_dim': id_dim,
            'curvature': curvature, 'desc': f"ID={id_dim} Cv={curvature:.1f}",
        })
    
    # ================================================================
    # RUN ON BOTH ARCHITECTURES
    # ================================================================
    for arch_name, hidden_dims in [("WIDE (64-64-64-64)", WIDE_DIMS), 
                                     ("NARROW (64-48-32)", NARROW_DIMS)]:
        print(f"\n{'='*70}")
        print(f"  ARCHITECTURE: {arch_name}")
        print(f"{'='*70}")
        
        print(f"\n  {'Manifold':<16s} | {'Fixed':>7s} {'Gate':>7s} {'Oracle':>7s} | {'S>F':>4s} {'S>O':>4s}")
        print(f"  {'-'*16}-+-{'-'*7}-{'-'*7}-{'-'*7}-+-{'-'*4}-{'-'*4}")
        
        results = []
        for mi, mf in enumerate(manifolds):
            if mi % 5 == 0:
                el = time.time() - start
                print(f"  ... [{mi}/{N_MANIFOLDS}] {el/60:.1f}min elapsed")
            
            X_tr, X_te = mf['X_tr'], mf['X_te']
            y_tr, y_te = mf['y_tr'], mf['y_te']
            nc = mf['n_classes']
            
            # Oracle
            best_oracle, best_oracle_wd = 0, 0.01
            for wd in [1e-5, 1e-4, 1e-3, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
                a = train_fixed_wd(X_tr, y_tr, X_te, y_te, hidden_dims, nc, wd, EPOCHS)
                if a > best_oracle: best_oracle = a; best_oracle_wd = wd
            
            # Default fixed
            default_acc = train_fixed_wd(X_tr, y_tr, X_te, y_te, hidden_dims, nc, 0.01, EPOCHS)
            
            # SurfaceGate
            best_sg, best_sg_curvs, best_sg_mults = 0, [], []
            for bwd in BASE_WDS:
                a, curvs, mults = train_surface_gate(
                    X_tr, y_tr, X_te, y_te, hidden_dims, nc,
                    base_wd=bwd, epochs=EPOCHS, measure_every=25)
                if a > best_sg:
                    best_sg = a; best_sg_curvs = curvs; best_sg_mults = mults
            
            sf = "✓" if best_sg > default_acc + 0.005 else " "
            so = "✓" if best_sg > best_oracle + 0.005 else " "
            
            results.append({
                'desc': mf['desc'], 'id_dim': mf['id_dim'],
                'default': float(default_acc), 'oracle': float(best_oracle),
                'gate': float(best_sg), 'curvatures': [float(c) for c in best_sg_curvs],
                'multipliers': [float(m) for m in best_sg_mults],
            })
            
            print(f"  {mf['desc']:<16s} | {default_acc:>7.3f} {best_sg:>7.3f} {best_oracle:>7.3f} | {sf:>4s} {so:>4s}")
        
        # Aggregate
        da = np.array([r['default'] for r in results])
        sa = np.array([r['gate'] for r in results])
        oa = np.array([r['oracle'] for r in results])
        gains = sa - da
        wins = int(np.sum(gains > 0.005))
        ties = int(np.sum(np.abs(gains) <= 0.005))
        losses = int(np.sum(gains < -0.005))
        beats_oracle = int(np.sum(sa > oa + 0.005))
        
        og = oa.mean() - da.mean()
        sg = sa.mean() - da.mean()
        cap = (sg / og * 100) if og > 0.001 else 0
        r_track, _ = pearsonr(sa, oa)
        
        print(f"\n  {arch_name} SUMMARY:")
        print(f"    Default Fixed:  {da.mean():.4f}")
        print(f"    SurfaceGate:    {sa.mean():.4f}")
        print(f"    Oracle:         {oa.mean():.4f}")
        print(f"    W/T/L vs fixed: {wins}/{ties}/{losses}")
        print(f"    Beats oracle:   {beats_oracle}/{N_MANIFOLDS}")
        print(f"    Oracle capture: {cap:.0f}%")
        print(f"    Tracks oracle:  r = {r_track:.3f}")
        print(f"    Mean gain:      {sg:+.4f} ({sg/da.mean()*100:+.1f}%)")
        
        # Curvature profiles
        print(f"\n    Curvature profiles (does curvature VARY across layers?):")
        curv_ranges = []
        for r in results:
            if r['curvatures']:
                cr = max(r['curvatures']) - min(r['curvatures'])
                curv_ranges.append(cr)
                print(f"      {r['desc']:<14s}: {' → '.join(f'{c:.3f}' for c in r['curvatures'])}"
                      f"  range={cr:.3f}"
                      f"  mults={' → '.join(f'{m:.2f}' for m in r['multipliers'])}")
        
        if len(curv_ranges) >= 5:
            gains_list = [r['gate'] - r['default'] for r in results if r['curvatures']]
            rc, pc = pearsonr(curv_ranges[:len(gains_list)], gains_list)
            print(f"\n    Curvature range → gain: r = {rc:+.3f}, p = {pc:.4f}")
        
        # Save
        output = {
            'architecture': arch_name,
            'hidden_dims': hidden_dims,
            'default_mean': float(da.mean()),
            'gate_mean': float(sa.mean()),
            'oracle_mean': float(oa.mean()),
            'oracle_capture': float(cap),
            'wins': wins, 'ties': ties, 'losses': losses,
            'beats_oracle': beats_oracle,
            'per_manifold': results,
        }
        tag = 'wide' if 'WIDE' in arch_name else 'narrow'
        with open(f'{OUTPUT_DIR}/tdf_surfacegate_v2_{tag}.json', 'w') as f:
            json.dump(output, f, indent=2)
    
    # ================================================================
    # VISUALIZATION
    # ================================================================
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # Load both result sets
        with open(f'{OUTPUT_DIR}/tdf_surfacegate_v2_wide.json') as f:
            wide_res = json.load(f)
        with open(f'{OUTPUT_DIR}/tdf_surfacegate_v2_narrow.json') as f:
            narrow_res = json.load(f)
        
        fig, axes = plt.subplots(2, 3, figsize=(21, 12))
        fig.suptitle("SurfaceGate v2: Wide vs Narrow Architecture\n"
                     "Does the gate adapt when the architecture doesn't compress?",
                     fontsize=14, fontweight='bold', y=0.98, color='white')
        fig.patch.set_facecolor('#08080f')
        
        for col, (res, label, color) in enumerate([
            (wide_res, 'Wide (64-64-64-64)', '#4ecdc4'),
            (narrow_res, 'Narrow (64-48-32)', '#f1c40f')]):
            
            da = np.array([r['default'] for r in res['per_manifold']])
            sa = np.array([r['gate'] for r in res['per_manifold']])
            oa = np.array([r['oracle'] for r in res['per_manifold']])
            gains = sa - da
            
            # Row 1: scatter
            ax = axes[0, col]
            ax.scatter(da, sa, c=color, s=80, edgecolors='white', linewidths=1)
            lm = [min(min(da),min(sa))-0.05, max(max(da),max(sa))+0.05]
            ax.plot(lm, lm, '--', color='white', alpha=0.2)
            above = np.sum(sa > da)
            ax.set_xlabel('Fixed WD', fontsize=9, color='#a0b0c0')
            ax.set_ylabel('SurfaceGate', fontsize=9, color='#a0b0c0')
            ax.set_title(f'{label}\n{above}/{len(da)} above, capture={res["oracle_capture"]:.0f}%',
                        fontsize=10, color=color)
            ax.set_facecolor('#08080f'); ax.tick_params(colors='#606080')
            
            # Row 2: curvature profiles
            ax = axes[1, col]
            for r in res['per_manifold']:
                if r['curvatures']:
                    c2 = color if r['gate'] > r['default'] else '#e74c3c'
                    ax.plot(range(len(r['curvatures'])), r['curvatures'],
                           'o-', color=c2, alpha=0.5, markersize=4)
            ax.set_xlabel('Layer', fontsize=9, color='#a0b0c0')
            ax.set_ylabel('Curvature', fontsize=9, color='#a0b0c0')
            ax.set_title(f'Curvature Across Layers ({label})', fontsize=10, color='white')
            ax.set_facecolor('#08080f'); ax.tick_params(colors='#606080')
        
        # P3 (top right): comparison summary
        ax = axes[0, 2]
        x = np.arange(3)
        w = 0.3
        wide_vals = [wide_res['default_mean'], wide_res['gate_mean'], wide_res['oracle_mean']]
        narr_vals = [narrow_res['default_mean'], narrow_res['gate_mean'], narrow_res['oracle_mean']]
        ax.bar(x-w/2, wide_vals, w, color='#4ecdc4', edgecolor='white', linewidth=0.5, alpha=0.8, label='Wide')
        ax.bar(x+w/2, narr_vals, w, color='#f1c40f', edgecolor='white', linewidth=0.5, alpha=0.8, label='Narrow')
        for i, (wv, nv) in enumerate(zip(wide_vals, narr_vals)):
            ax.text(i-w/2, wv+0.003, f'{wv:.3f}', ha='center', fontsize=7, color='#4ecdc4')
            ax.text(i+w/2, nv+0.003, f'{nv:.3f}', ha='center', fontsize=7, color='#f1c40f')
        ax.set_xticks(x)
        ax.set_xticklabels(['Fixed', 'Gate', 'Oracle'], fontsize=9, color='#a0b0c0')
        ax.set_ylabel('Mean Accuracy', fontsize=9, color='#a0b0c0')
        ax.set_title('Wide vs Narrow Comparison', fontsize=10, color='white')
        ax.legend(fontsize=7, facecolor='#08080f', edgecolor='#303040', labelcolor='#a0b0c0')
        ymin = min(min(wide_vals), min(narr_vals)) - 0.05
        ymax = max(max(wide_vals), max(narr_vals)) + 0.03
        ax.set_ylim(ymin, ymax)
        ax.set_facecolor('#08080f'); ax.tick_params(colors='#606080')
        
        # P6 (bottom right): multiplier comparison
        ax = axes[1, 2]
        # Average multiplier per layer position for wide vs narrow
        wide_mults_by_layer = [[] for _ in range(4)]
        narrow_mults_by_layer = [[] for _ in range(3)]
        for r in wide_res['per_manifold']:
            for j, m in enumerate(r['multipliers']):
                wide_mults_by_layer[j].append(m)
        for r in narrow_res['per_manifold']:
            for j, m in enumerate(r['multipliers']):
                narrow_mults_by_layer[j].append(m)
        
        wide_means = [np.mean(l) if l else 0 for l in wide_mults_by_layer]
        narrow_means = [np.mean(l) if l else 0 for l in narrow_mults_by_layer]
        
        ax.plot(range(4), wide_means, 'o-', color='#4ecdc4', linewidth=2, markersize=8, label='Wide')
        ax.plot(range(3), narrow_means, 's-', color='#f1c40f', linewidth=2, markersize=8, label='Narrow')
        ax.axhline(1.0, color='white', linestyle=':', alpha=0.2)
        ax.set_xlabel('Layer', fontsize=9, color='#a0b0c0')
        ax.set_ylabel('Mean WD Multiplier', fontsize=9, color='#a0b0c0')
        ax.set_title('How the Gate Adjusts Brakes\n(>1=heavier, <1=lighter than fixed)', fontsize=10, color='white')
        ax.legend(fontsize=7, facecolor='#08080f', edgecolor='#303040', labelcolor='#a0b0c0')
        ax.set_facecolor('#08080f'); ax.tick_params(colors='#606080')
        
        plt.tight_layout(rect=[0, 0.02, 1, 0.93])
        fig.text(0.5, 0.005, "Christopher Head · Navigator's Log R&D · navigatorslog.netlify.app",
                 ha='center', fontsize=8, color='#404060')
        plt.savefig(f'{OUTPUT_DIR}/tdf_surfacegate_v2.png', dpi=180,
                    bbox_inches='tight', facecolor='#08080f')
        plt.close()
        print(f"\n  Visualization saved to tdf_surfacegate_v2.png")
    except Exception as e:
        print(f"\n  Visualization error: {e}")
    
    total = time.time() - start
    print(f"\n  Total runtime: {total/60:.1f} minutes")
    print("  Done.")

if __name__ == '__main__':
    main()
