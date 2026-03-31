#!/usr/bin/env python3
"""
THREAD B: Does Random Per-Layer WD Variation Beat Fixed WD?
==============================================================
Navigator's Log R&D | March 2026

CIFAR-10 CNN showed ALL non-uniform schedules beat Fixed — including
Reverse. This suggests the mechanism isn't front-loading or matching
a curvature profile. It might be symmetry-breaking: giving different
layers different WD values forces the optimizer to allocate capacity
differently, which helps regardless of direction.

THE TEST: Compare Fixed WD against RANDOM per-layer WD assignments.
Each layer gets a different WD drawn from a distribution around the
base value. If random variation beats fixed as reliably as structured
schedules do, the finding simplifies to: "non-uniform WD helps."

Tests on CIFAR-10 CNN (the configuration where all schedules beat Fixed)
and CIFAR-10 MLP (the configuration where Fixed won).

Requirements: pip install torch torchvision numpy matplotlib
Runtime: ~60-90 minutes on Surface Pro 7
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
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = "."
torch.manual_seed(42)
np.random.seed(42)

# ================================================================
# MODELS (same as real data test)
# ================================================================
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, n_classes):
        super().__init__()
        self.layers = nn.ModuleList()
        prev = input_dim
        for h in hidden_dims:
            self.layers.append(nn.Linear(prev, h))
            prev = h
        self.output = nn.Linear(prev, n_classes)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.output(x)

class SimpleCNN(nn.Module):
    def __init__(self, n_channels, n_classes):
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
    """Extract all layers with trainable weights."""
    layers = []
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            layers.append(module)
    return layers

def train_with_layer_wds(model, train_loader, test_loader, layer_wds, epochs=20, lr=1e-3):
    """Train with specific WD per layer."""
    layers = get_layer_modules(model)
    param_groups = []
    for i, layer in enumerate(layers):
        wd = layer_wds.get(i, 1e-4)
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
    return correct / total

# ================================================================
# WD SCHEDULE GENERATORS
# ================================================================
def schedule_fixed(base_wd, n_layers):
    return {i: base_wd for i in range(n_layers)}

def schedule_exp_decay(base_wd, n_layers):
    return {i: base_wd * (0.5 ** i) for i in range(n_layers)}

def schedule_lin_decay(base_wd, n_layers):
    return {i: base_wd * (1.0 - i * 0.8 / max(1, n_layers - 1)) for i in range(n_layers)}

def schedule_reverse(base_wd, n_layers):
    return {i: base_wd * (0.2 + i * 0.8 / max(1, n_layers - 1)) for i in range(n_layers)}

def schedule_random(base_wd, n_layers, seed=0):
    """Random WD per layer. Each layer gets a different value drawn
    from a log-uniform distribution around the base."""
    rng = np.random.RandomState(seed)
    log_base = np.log10(base_wd)
    # Draw from [base/10, base*10] in log space
    log_wds = rng.uniform(log_base - 1, log_base + 1, n_layers)
    return {i: float(10 ** log_wds[i]) for i in range(n_layers)}

def schedule_shuffled(base_wd, n_layers, seed=0):
    """Take the exponential decay schedule and SHUFFLE it randomly.
    Same set of WD values, random assignment to layers.
    If this matches ExpDecay, the specific assignment doesn't matter.
    If it matches Fixed, the exponential VALUES matter but not the order."""
    rng = np.random.RandomState(seed)
    exp_wds = [base_wd * (0.5 ** i) for i in range(n_layers)]
    rng.shuffle(exp_wds)
    return {i: exp_wds[i] for i in range(n_layers)}

def schedule_alternating(base_wd, n_layers):
    """Alternating high/low. No directional trend.
    Tests whether variation pattern matters."""
    return {i: base_wd * (2.0 if i % 2 == 0 else 0.2) for i in range(n_layers)}

# ================================================================
# MAIN
# ================================================================
def main():
    start = time.time()
    
    print("=" * 70)
    print("  THREAD B: Does Random WD Variation Beat Fixed WD?")
    print("  Testing the symmetry-breaking hypothesis")
    print("=" * 70)
    
    BATCH_SIZE = 128
    EPOCHS_MLP = 25
    EPOCHS_CNN = 15
    
    # Only test on CIFAR-10 (where we saw the interesting results)
    print("\n  Loading CIFAR-10...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    train_ds = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_ds = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    
    # Schedules to test
    # We use the best base WD from the real data test:
    # CIFAR-10 MLP: 1e-3 (Fixed won)
    # CIFAR-10 CNN: 1e-3 (all schedules used this)
    BASE_WD = 1e-3
    
    schedules = {
        'Fixed': lambda n: schedule_fixed(BASE_WD, n),
        'ExpDecay': lambda n: schedule_exp_decay(BASE_WD, n),
        'LinDecay': lambda n: schedule_lin_decay(BASE_WD, n),
        'Reverse': lambda n: schedule_reverse(BASE_WD, n),
        'Random_1': lambda n: schedule_random(BASE_WD, n, seed=1),
        'Random_2': lambda n: schedule_random(BASE_WD, n, seed=2),
        'Random_3': lambda n: schedule_random(BASE_WD, n, seed=3),
        'Random_4': lambda n: schedule_random(BASE_WD, n, seed=4),
        'Random_5': lambda n: schedule_random(BASE_WD, n, seed=5),
        'Shuffled_1': lambda n: schedule_shuffled(BASE_WD, n, seed=1),
        'Shuffled_2': lambda n: schedule_shuffled(BASE_WD, n, seed=2),
        'Shuffled_3': lambda n: schedule_shuffled(BASE_WD, n, seed=3),
        'Alternating': lambda n: schedule_alternating(BASE_WD, n),
    }
    
    all_results = {}
    
    for model_name, ModelClass, model_args, epochs in [
        ('MLP', MLP, (3072, [256, 128, 64, 32], 10), EPOCHS_MLP),
        ('CNN', SimpleCNN, (3, 10), EPOCHS_CNN),
    ]:
        print(f"\n{'='*70}")
        print(f"  MODEL: {model_name}")
        print(f"{'='*70}")
        
        # Count layers
        dummy_model = ModelClass(*model_args)
        if model_name == 'CNN':
            dummy_model(torch.randn(1, 3, 32, 32))  # init fc layer
        n_layers = len(get_layer_modules(dummy_model))
        del dummy_model
        
        print(f"  {n_layers} trainable layers, base WD = {BASE_WD}")
        print(f"\n  {'Schedule':<15s} {'Accuracy':>10s} {'vs Fixed':>10s} {'WD values':>40s}")
        print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*40}")
        
        results = {}
        fixed_acc = None
        
        for sched_name, sched_fn in schedules.items():
            layer_wds = sched_fn(n_layers)
            
            model = ModelClass(*model_args)
            if model_name == 'CNN':
                model(torch.randn(1, 3, 32, 32))  # init fc layer
            
            acc = train_with_layer_wds(model, train_loader, test_loader, 
                                       layer_wds, epochs=epochs)
            
            if sched_name == 'Fixed':
                fixed_acc = acc
            
            delta = acc - fixed_acc if fixed_acc is not None else 0
            wd_str = ', '.join(f'{layer_wds[i]:.1e}' for i in range(min(5, n_layers)))
            if n_layers > 5:
                wd_str += '...'
            
            results[sched_name] = {
                'accuracy': float(acc),
                'delta': float(delta),
                'layer_wds': {str(k): float(v) for k, v in layer_wds.items()},
            }
            
            marker = " <<<" if abs(delta) > 0.005 and delta > 0 else ""
            print(f"  {sched_name:<15s} {acc:>10.4f} {delta:>+10.4f} {wd_str:>40s}{marker}")
        
        all_results[model_name] = results
        
        # Analysis
        print(f"\n  ANALYSIS for {model_name}:")
        
        structured = ['ExpDecay', 'LinDecay', 'Reverse']
        random_names = [n for n in results if n.startswith('Random_')]
        shuffled_names = [n for n in results if n.startswith('Shuffled_')]
        
        struct_accs = [results[n]['accuracy'] for n in structured]
        random_accs = [results[n]['accuracy'] for n in random_names]
        shuffled_accs = [results[n]['accuracy'] for n in shuffled_names]
        alt_acc = results['Alternating']['accuracy']
        
        print(f"    Fixed:              {fixed_acc:.4f}")
        print(f"    Structured mean:    {np.mean(struct_accs):.4f} ({np.mean(struct_accs)-fixed_acc:+.4f})")
        print(f"    Random mean:        {np.mean(random_accs):.4f} ({np.mean(random_accs)-fixed_acc:+.4f})")
        print(f"    Shuffled mean:      {np.mean(shuffled_accs):.4f} ({np.mean(shuffled_accs)-fixed_acc:+.4f})")
        print(f"    Alternating:        {alt_acc:.4f} ({alt_acc-fixed_acc:+.4f})")
        
        # Count how many random seeds beat Fixed
        random_wins = sum(1 for a in random_accs if a > fixed_acc + 0.001)
        print(f"\n    Random seeds beating Fixed: {random_wins}/{len(random_accs)}")
        
        if np.mean(random_accs) > fixed_acc + 0.002:
            print(f"    >>> SYMMETRY-BREAKING CONFIRMED: Random variation helps!")
            print(f"    >>> Direction doesn't matter. Non-uniformity is the mechanism.")
        elif np.mean(struct_accs) > fixed_acc + 0.002 and np.mean(random_accs) <= fixed_acc + 0.002:
            print(f"    >>> STRUCTURE MATTERS: Only structured schedules help.")
            print(f"    >>> The curvature-matching story holds — direction matters.")
        else:
            print(f"    >>> NO CLEAR SIGNAL: Neither random nor structured beats Fixed reliably.")
    
    # ================================================================
    # OVERALL VERDICT
    # ================================================================
    total_time = time.time() - start
    
    print(f"\n{'='*70}")
    print(f"  OVERALL VERDICT")
    print(f"  Runtime: {total_time/60:.1f} minutes")
    print(f"{'='*70}")
    
    for model_name, results in all_results.items():
        fixed_acc = results['Fixed']['accuracy']
        random_accs = [results[n]['accuracy'] for n in results if n.startswith('Random_')]
        struct_accs = [results[n]['accuracy'] for n in ['ExpDecay', 'LinDecay', 'Reverse']]
        
        print(f"\n  {model_name}:")
        print(f"    Fixed:      {fixed_acc:.4f}")
        print(f"    Structured: {np.mean(struct_accs):.4f} (mean of Exp/Lin/Rev)")
        print(f"    Random:     {np.mean(random_accs):.4f} (mean of 5 seeds)")
        
        if np.mean(random_accs) > fixed_acc + 0.002:
            print(f"    FINDING: Symmetry-breaking is the mechanism (random helps)")
        elif np.mean(struct_accs) > fixed_acc + 0.002:
            print(f"    FINDING: Structure matters (only directional schedules help)")
        else:
            print(f"    FINDING: WD schedule doesn't matter for this config")
    
    # Save
    output = {
        'runtime_minutes': total_time / 60,
        'base_wd': BASE_WD,
        'results': all_results,
    }
    with open(f'{OUTPUT_DIR}/tdf_symmetry_breaking_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    # ================================================================
    # VISUALIZATION
    # ================================================================
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle("Thread B: Is Non-Uniformity the Mechanism?\n"
                     "Does RANDOM per-layer WD beat Fixed?",
                     fontsize=14, fontweight='bold', y=0.98, color='white')
        fig.patch.set_facecolor('#08080f')
        
        for ax_idx, (model_name, results) in enumerate(all_results.items()):
            ax = axes[ax_idx]
            
            names = list(results.keys())
            accs = [results[n]['accuracy'] for n in names]
            fixed_acc = results['Fixed']['accuracy']
            
            colors = []
            for n in names:
                if n == 'Fixed': colors.append('#e74c3c')
                elif n.startswith('Random'): colors.append('#9b59b6')
                elif n.startswith('Shuffled'): colors.append('#e8a87c')
                elif n == 'Alternating': colors.append('#2ecc71')
                elif n == 'ExpDecay': colors.append('#4ecdc4')
                elif n == 'LinDecay': colors.append('#f1c40f')
                elif n == 'Reverse': colors.append('#606080')
                else: colors.append('#aaa')
            
            bars = ax.bar(range(len(names)), accs, color=colors, 
                         edgecolor='white', linewidth=0.3, alpha=0.8)
            ax.axhline(fixed_acc, color='#e74c3c', linestyle='--', alpha=0.4, label='Fixed baseline')
            
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, fontsize=6, color='#a0b0c0', rotation=45, ha='right')
            ax.set_ylabel('Test Accuracy', fontsize=9, color='#a0b0c0')
            ax.set_title(f'CIFAR-10 {model_name}', fontsize=11, color='white')
            ax.set_ylim(min(accs) - 0.01, max(accs) + 0.01)
            ax.legend(fontsize=7, facecolor='#08080f', edgecolor='#303040', labelcolor='#a0b0c0')
            ax.set_facecolor('#08080f'); ax.tick_params(colors='#606080')
            
            for i, a in enumerate(accs):
                ax.text(i, a + 0.001, f'{a:.4f}', ha='center', fontsize=5.5, color='#a0b0c0', rotation=90)
        
        plt.tight_layout(rect=[0, 0.02, 1, 0.92])
        fig.text(0.5, 0.005, "Christopher Head · Navigator's Log R&D · navigatorslog.netlify.app",
                 ha='center', fontsize=8, color='#404060')
        plt.savefig(f'{OUTPUT_DIR}/tdf_symmetry_breaking.png', dpi=180,
                    bbox_inches='tight', facecolor='#08080f')
        plt.close()
        print(f"\n  Visualization saved to tdf_symmetry_breaking.png")
    except Exception as e:
        print(f"\n  Visualization error: {e}")
    
    print(f"  Results saved to tdf_symmetry_breaking_results.json")
    print(f"\n  Total runtime: {total_time/60:.1f} minutes")
    print("  Done.")

if __name__ == '__main__':
    main()
