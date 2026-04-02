#!/usr/bin/env python3
"""
MULTI-SEED VALIDATION: CIFAR-10 CNN
=====================================
Navigator's Log R&D | April 2026

ADDRESS THE BIGGEST VULNERABILITY: Every result so far is a single
training run. This script runs the 4 most important WD schedules
on CIFAR-10 CNN with 5 different training seeds each, producing
error bars that either confirm or kill the findings.

THE FOUR SCHEDULES THAT MATTER:
  Fixed:    The baseline. Everything is measured against this.
  LinDecay: The best structured schedule (+0.67% in single run).
  Reverse:  The surprise winner (+1.38% in symmetry-breaking test).
  Random:   The symmetry-breaking proof (4/5 seeds beat Fixed).

IF THE FINDINGS ARE REAL:
  - LinDecay mean > Fixed mean, with non-overlapping std
  - Reverse mean > Fixed mean
  - Random mean > Fixed mean
  - The ordering should be roughly consistent across seeds

IF THE FINDINGS WERE NOISE:
  - All four schedules overlap within standard deviation
  - The ordering changes with each seed
  - Fixed wins on some seeds, loses on others randomly

This is the test that turns observations into results.

Requirements: pip install torch torchvision numpy matplotlib
Runtime: ~20 runs x ~30min each = ~10 hours on Surface Pro 7
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import time
import json
import gc
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = "."

# ================================================================
# MODEL (identical to Thread B CNN)
# ================================================================
class SimpleCNN(nn.Module):
    def __init__(self, n_channels=3, n_classes=10):
        super().__init__()
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(n_channels, 32, 3, padding=1),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.Conv2d(64, 64, 3, padding=1),
        ])
        self.pool = nn.MaxPool2d(2, 2)
        self.fc_layers = nn.ModuleList([
            nn.Linear(64 * 4 * 4, 128),
            nn.Linear(128, 64),
        ])
        self.output = nn.Linear(64, n_classes)
        self._fc_init = False
    
    def forward(self, x):
        for conv in self.conv_layers:
            x = self.pool(torch.relu(conv(x)))
        x = x.view(x.size(0), -1)
        if not self._fc_init:
            self.fc_layers[0] = nn.Linear(x.size(1), 128).to(x.device)
            self._fc_init = True
        for fc in self.fc_layers:
            x = torch.relu(fc(x))
        return self.output(x)

# ================================================================
# TRAINING WITH PER-LAYER WD
# ================================================================
def get_layer_modules(model):
    layers = []
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            layers.append(module)
    return layers

def train_with_layer_wds(train_loader, test_loader, layer_wds, 
                          epochs=15, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = SimpleCNN()
    # Initialize fc layer
    model(torch.randn(1, 3, 32, 32))
    
    layers = get_layer_modules(model)
    param_groups = []
    for i, layer in enumerate(layers):
        wd = layer_wds.get(i, 1e-3)
        param_groups.append({'params': layer.parameters(), 'weight_decay': wd, 'lr': lr})
    
    optimizer = optim.Adam(param_groups)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        for bx, by in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
    
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for bx, by in test_loader:
            pred = model(bx).argmax(dim=1)
            correct += (pred == by).sum().item()
            total += len(by)
    
    acc = correct / total
    del model, optimizer, criterion, param_groups
    gc.collect()
    return acc

# ================================================================
# WD SCHEDULES
# ================================================================
BASE_WD = 1e-3

def schedule_fixed(n_layers):
    return {i: BASE_WD for i in range(n_layers)}

def schedule_lin_decay(n_layers):
    return {i: BASE_WD * (1.0 - i * 0.8 / max(1, n_layers - 1)) for i in range(n_layers)}

def schedule_reverse(n_layers):
    return {i: BASE_WD * (0.2 + i * 0.8 / max(1, n_layers - 1)) for i in range(n_layers)}

def schedule_random(n_layers, seed=0):
    rng = np.random.RandomState(seed)
    log_base = np.log10(BASE_WD)
    log_wds = rng.uniform(log_base - 1, log_base + 1, n_layers)
    return {i: float(10 ** log_wds[i]) for i in range(n_layers)}

# ================================================================
# MAIN
# ================================================================
def main():
    start = time.time()
    
    print("=" * 70)
    print("  MULTI-SEED VALIDATION: CIFAR-10 CNN")
    print("  Do the findings reproduce across training seeds?")
    print("=" * 70)
    
    BATCH_SIZE = 128
    EPOCHS = 15
    N_LAYERS = 6  # 3 conv + 2 fc + 1 output
    
    # 5 training seeds
    TRAIN_SEEDS = [42, 123, 456, 789, 1337]
    
    # 4 schedules + 3 random WD seeds = 7 configs x 5 training seeds = 35 runs
    # Random uses WD seed 1 (the best from Thread B), seed 2, and seed 5
    SCHEDULES = {
        'Fixed': lambda: schedule_fixed(N_LAYERS),
        'LinDecay': lambda: schedule_lin_decay(N_LAYERS),
        'Reverse': lambda: schedule_reverse(N_LAYERS),
        'Random_1': lambda: schedule_random(N_LAYERS, seed=1),
        'Random_2': lambda: schedule_random(N_LAYERS, seed=2),
        'Random_5': lambda: schedule_random(N_LAYERS, seed=5),
    }
    
    print(f"\n  Loading CIFAR-10...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    train_ds = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                             download=True, transform=transform)
    test_ds = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=0)
    
    print(f"  {len(SCHEDULES)} schedules x {len(TRAIN_SEEDS)} seeds = {len(SCHEDULES)*len(TRAIN_SEEDS)} training runs")
    print(f"  Epochs: {EPOCHS}, Batch: {BATCH_SIZE}, Base WD: {BASE_WD}")
    
    # Results: schedule -> list of accuracies across seeds
    all_results = {name: [] for name in SCHEDULES}
    detailed = {}
    
    total_runs = len(SCHEDULES) * len(TRAIN_SEEDS)
    run_count = 0
    
    for seed in TRAIN_SEEDS:
        print(f"\n  {'='*60}")
        print(f"  TRAINING SEED: {seed}")
        print(f"  {'='*60}")
        print(f"  {'Schedule':<15s} {'Accuracy':>10s} {'Run':>8s}")
        print(f"  {'-'*15} {'-'*10} {'-'*8}")
        
        for sched_name, sched_fn in SCHEDULES.items():
            layer_wds = sched_fn()
            
            acc = train_with_layer_wds(train_loader, test_loader, layer_wds,
                                        epochs=EPOCHS, lr=1e-3, seed=seed)
            
            all_results[sched_name].append(acc)
            detailed[f"{sched_name}_seed{seed}"] = float(acc)
            
            run_count += 1
            print(f"  {sched_name:<15s} {acc:>10.4f} {run_count:>5d}/{total_runs}")
            
            # Incremental save
            save_data = {
                'runtime_so_far': (time.time()-start)/60,
                'completed_runs': run_count,
                'total_runs': total_runs,
                'seeds': TRAIN_SEEDS,
                'detailed': detailed,
                'summary': {}
            }
            for name, accs in all_results.items():
                if len(accs) > 0:
                    save_data['summary'][name] = {
                        'mean': float(np.mean(accs)),
                        'std': float(np.std(accs)) if len(accs) > 1 else 0.0,
                        'min': float(np.min(accs)),
                        'max': float(np.max(accs)),
                        'n': len(accs),
                        'values': [float(a) for a in accs],
                    }
            with open(f'{OUTPUT_DIR}/tdf_multiseed_validation.json', 'w') as f:
                json.dump(save_data, f, indent=2)
    
    # ================================================================
    # FINAL ANALYSIS
    # ================================================================
    total_time = time.time() - start
    
    print(f"\n{'='*70}")
    print(f"  MULTI-SEED VALIDATION RESULTS")
    print(f"  {total_runs} runs, {total_time/60:.1f} minutes")
    print(f"{'='*70}")
    
    print(f"\n  {'Schedule':<15s} {'Mean':>10s} {'Std':>10s} {'Min':>10s} {'Max':>10s} {'vs Fixed':>10s}")
    print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    
    fixed_mean = np.mean(all_results['Fixed'])
    fixed_std = np.std(all_results['Fixed'])
    
    for name in SCHEDULES:
        accs = all_results[name]
        mean = np.mean(accs)
        std = np.std(accs)
        delta = mean - fixed_mean
        print(f"  {name:<15s} {mean:>10.4f} {std:>10.4f} {np.min(accs):>10.4f} {np.max(accs):>10.4f} {delta:>+10.4f}")
    
    # Statistical test: does each schedule reliably beat Fixed?
    print(f"\n  Per-seed comparison (each schedule vs Fixed on same seed):")
    print(f"  {'Schedule':<15s} {'Wins':>6s} {'Ties':>6s} {'Losses':>6s} {'Mean Delta':>12s}")
    print(f"  {'-'*15} {'-'*6} {'-'*6} {'-'*6} {'-'*12}")
    
    for name in SCHEDULES:
        if name == 'Fixed':
            continue
        wins, ties, losses = 0, 0, 0
        deltas = []
        for i in range(len(TRAIN_SEEDS)):
            d = all_results[name][i] - all_results['Fixed'][i]
            deltas.append(d)
            if d > 0.001:
                wins += 1
            elif d < -0.001:
                losses += 1
            else:
                ties += 1
        mean_delta = np.mean(deltas)
        print(f"  {name:<15s} {wins:>6d} {ties:>6d} {losses:>6d} {mean_delta:>+12.4f}")
    
    # Overlap test
    print(f"\n  Overlap test (do confidence intervals separate?):")
    for name in SCHEDULES:
        if name == 'Fixed':
            continue
        sched_mean = np.mean(all_results[name])
        sched_std = np.std(all_results[name])
        separated = (sched_mean - sched_std) > (fixed_mean + fixed_std)
        overlaps = not separated
        direction = "ABOVE" if sched_mean > fixed_mean else "BELOW"
        print(f"  {name:<15s}: {direction} Fixed, {'OVERLAPS' if overlaps else 'SEPARATED'} "
              f"({sched_mean:.4f}+/-{sched_std:.4f} vs {fixed_mean:.4f}+/-{fixed_std:.4f})")
    
    # Verdict
    print(f"\n  VERDICT:")
    non_uniform_wins = 0
    non_uniform_total = 0
    for name in SCHEDULES:
        if name == 'Fixed':
            continue
        for i in range(len(TRAIN_SEEDS)):
            non_uniform_total += 1
            if all_results[name][i] > all_results['Fixed'][i] + 0.001:
                non_uniform_wins += 1
    
    win_rate = non_uniform_wins / non_uniform_total if non_uniform_total > 0 else 0
    print(f"  Non-uniform schedules beat Fixed: {non_uniform_wins}/{non_uniform_total} ({win_rate:.0%})")
    
    if win_rate > 0.7:
        print(f"  >>> FINDING CONFIRMED: Non-uniform WD reliably beats Fixed across seeds")
    elif win_rate > 0.5:
        print(f"  >>> FINDING SUPPORTED: Non-uniform WD beats Fixed more often than not")
    elif win_rate > 0.3:
        print(f"  >>> FINDING WEAK: Non-uniform WD shows inconsistent advantage")
    else:
        print(f"  >>> FINDING REJECTED: Non-uniform WD does not reliably beat Fixed")
    
    # ================================================================
    # VISUALIZATION
    # ================================================================
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle("Multi-Seed Validation: CIFAR-10 CNN\n"
                     "Do the findings reproduce across training seeds?",
                     fontsize=14, fontweight='bold', y=0.98, color='white')
        fig.patch.set_facecolor('#08080f')
        
        # Left: box plot
        ax = axes[0]
        names = list(SCHEDULES.keys())
        data = [all_results[n] for n in names]
        colors = ['#e74c3c', '#f1c40f', '#606080', '#9b59b6', '#9b59b6', '#9b59b6']
        
        bp = ax.boxplot(data, labels=names, patch_artist=True, 
                       medianprops=dict(color='white', linewidth=2))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax.axhline(fixed_mean, color='#e74c3c', linestyle='--', alpha=0.4)
        ax.set_ylabel('Test Accuracy', fontsize=10, color='#a0b0c0')
        ax.set_title('Distribution Across 5 Seeds', fontsize=10, color='white')
        ax.set_xticklabels(names, fontsize=7, color='#a0b0c0', rotation=30)
        ax.set_facecolor('#08080f'); ax.tick_params(colors='#606080')
        
        # Right: paired deltas per seed
        ax = axes[1]
        x = range(len(TRAIN_SEEDS))
        for name in SCHEDULES:
            if name == 'Fixed':
                continue
            deltas = [all_results[name][i] - all_results['Fixed'][i] for i in range(len(TRAIN_SEEDS))]
            color = {'LinDecay': '#f1c40f', 'Reverse': '#606080', 
                     'Random_1': '#9b59b6', 'Random_2': '#c39bd3', 'Random_5': '#7d3c98'}[name]
            ax.plot(x, deltas, 'o-', color=color, label=name, markersize=6, linewidth=1.5)
        
        ax.axhline(0, color='#e74c3c', linestyle='--', alpha=0.4, label='Fixed baseline')
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in TRAIN_SEEDS], fontsize=8, color='#a0b0c0')
        ax.set_xlabel('Training Seed', fontsize=9, color='#a0b0c0')
        ax.set_ylabel('Delta vs Fixed', fontsize=9, color='#a0b0c0')
        ax.set_title('Per-Seed Advantage Over Fixed', fontsize=10, color='white')
        ax.legend(fontsize=7, facecolor='#08080f', edgecolor='#303040', labelcolor='#a0b0c0')
        ax.set_facecolor('#08080f'); ax.tick_params(colors='#606080')
        
        plt.tight_layout(rect=[0, 0.02, 1, 0.92])
        fig.text(0.5, 0.005, "Christopher Head · Navigator's Log R&D · navigatorslog.netlify.app",
                 ha='center', fontsize=8, color='#404060')
        plt.savefig(f'{OUTPUT_DIR}/tdf_multiseed_validation.png', dpi=180,
                    bbox_inches='tight', facecolor='#08080f')
        plt.close()
        print(f"\n  Visualization saved to tdf_multiseed_validation.png")
    except Exception as e:
        print(f"\n  Visualization error: {e}")
    
    # Final save
    save_data['runtime_minutes'] = total_time / 60
    save_data['completed'] = True
    with open(f'{OUTPUT_DIR}/tdf_multiseed_validation.json', 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\n  Results saved to tdf_multiseed_validation.json")
    print(f"  Total runtime: {total_time/60:.1f} minutes")
    print("  Done.")

if __name__ == '__main__':
    main()
