#!/usr/bin/env python3
"""
TDF REAL DATA TEST: Does Exponential WD Decay Work on Real Datasets?
======================================================================
Navigator's Log R&D | March 2026

The simple rule test showed exponential WD decay (halving WD at each
layer) beats fixed WD, grid search, AND SurfaceGate on synthetic
manifolds. But does it work on REAL data?

Tests on MNIST and CIFAR-10 with both MLP and simple CNN architectures.
If it works here, the finding generalizes and the one-line code change
is immediately adoptable.

Requirements: pip install torch torchvision numpy matplotlib
Runtime: ~20-40 minutes on Surface Pro 7
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
# MODELS
# ================================================================
class MLP(nn.Module):
    """Simple MLP — same family as our synthetic tests."""
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
    """Simple CNN — tests whether the finding extends beyond MLPs."""
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(n_channels, 32, 3, padding=1),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.Conv2d(64, 64, 3, padding=1),
        ])
        self.pool = nn.MaxPool2d(2, 2)
        # After 3 pools on 32x32: 4x4x64 = 1024 (CIFAR)
        # After 3 pools on 28x28: 3x3x64 = 576 (MNIST)
        self.fc_layers = nn.ModuleList([
            nn.Linear(64 * 4 * 4, 128),  # adjusted per dataset below
            nn.Linear(128, 64),
        ])
        self.output = nn.Linear(64, n_classes)
        self._fc_input_size = None
    
    def forward(self, x):
        for conv in self.conv_layers:
            x = self.pool(torch.relu(conv(x)))
        x = x.view(x.size(0), -1)
        if self._fc_input_size is None:
            self._fc_input_size = x.size(1)
            self.fc_layers[0] = nn.Linear(self._fc_input_size, 128).to(x.device)
        for fc in self.fc_layers:
            x = torch.relu(fc(x))
        return self.output(x)

# ================================================================
# TRAINING FUNCTION WITH PER-LAYER WD
# ================================================================
def train_model(model, train_loader, test_loader, layer_wds, epochs=30, lr=1e-3):
    """
    Train with per-layer weight decay.
    layer_wds: dict mapping parameter group index to WD value.
    """
    # Build parameter groups
    param_groups = []
    
    # Collect all named layers that have weight parameters
    layers_with_params = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            layers_with_params.append((name, module))
    
    for i, (name, module) in enumerate(layers_with_params):
        wd = layer_wds.get(i, layer_wds.get('default', 1e-4))
        param_groups.append({
            'params': module.parameters(),
            'weight_decay': wd,
            'lr': lr,
        })
    
    optimizer = optim.Adam(param_groups)
    criterion = nn.CrossEntropyLoss()
    
    best_test_acc = 0
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
        
        # Test accuracy
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                pred = model(batch_x).argmax(dim=1)
                correct += (pred == batch_y).sum().item()
                total += len(batch_y)
        test_acc = correct / total
        if test_acc > best_test_acc:
            best_test_acc = test_acc
    
    return best_test_acc

# ================================================================
# WD SCHEDULE GENERATORS
# ================================================================
def make_fixed_wds(base_wd, n_layers):
    return {i: base_wd for i in range(n_layers)}

def make_exp_decay_wds(base_wd, n_layers):
    return {i: base_wd * (0.5 ** i) for i in range(n_layers)}

def make_linear_decay_wds(base_wd, n_layers):
    return {i: base_wd * (1.0 - i * 0.8 / max(1, n_layers - 1)) for i in range(n_layers)}

def make_reverse_wds(base_wd, n_layers):
    return {i: base_wd * (0.2 + i * 0.8 / max(1, n_layers - 1)) for i in range(n_layers)}

# ================================================================
# MAIN EXPERIMENT
# ================================================================
def main():
    start = time.time()
    
    print("=" * 70)
    print("  REAL DATA TEST: Does Exponential WD Decay Work on Real Datasets?")
    print("=" * 70)
    
    EPOCHS_MLP = 30
    EPOCHS_CNN = 20
    BATCH_SIZE = 128
    BASE_WDS = [1e-5, 1e-4, 1e-3, 0.01, 0.1]
    
    # Load datasets
    print("\n  Loading datasets...")
    
    transform_mnist = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    transform_cifar = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    datasets = {}
    
    try:
        mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
        mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)
        datasets['MNIST'] = {
            'train': DataLoader(mnist_train, batch_size=BATCH_SIZE, shuffle=True),
            'test': DataLoader(mnist_test, batch_size=BATCH_SIZE),
            'input_dim': 784, 'n_channels': 1, 'n_classes': 10,
            'img_size': 28,
        }
        print(f"    MNIST: {len(mnist_train)} train, {len(mnist_test)} test")
    except Exception as e:
        print(f"    MNIST failed: {e}")
    
    try:
        cifar_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_cifar)
        cifar_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_cifar)
        datasets['CIFAR-10'] = {
            'train': DataLoader(cifar_train, batch_size=BATCH_SIZE, shuffle=True),
            'test': DataLoader(cifar_test, batch_size=BATCH_SIZE),
            'input_dim': 3072, 'n_channels': 3, 'n_classes': 10,
            'img_size': 32,
        }
        print(f"    CIFAR-10: {len(cifar_train)} train, {len(cifar_test)} test")
    except Exception as e:
        print(f"    CIFAR-10 failed: {e}")
    
    if not datasets:
        print("  No datasets available. Exiting.")
        return
    
    all_results = {}
    
    for ds_name, ds in datasets.items():
        print(f"\n{'='*70}")
        print(f"  DATASET: {ds_name}")
        print(f"{'='*70}")
        
        all_results[ds_name] = {}
        
        # ============================================================
        # TEST 1: MLP
        # ============================================================
        hidden_dims = [256, 128, 64, 32]
        n_layers = len(hidden_dims) + 1  # hidden + output
        
        print(f"\n  MODEL: MLP {hidden_dims}")
        print(f"  {'Schedule':<15s} {'Base WD':>10s} {'Test Acc':>10s}")
        print(f"  {'-'*15} {'-'*10} {'-'*10}")
        
        mlp_results = {}
        schedules = {
            'Fixed': make_fixed_wds,
            'ExpDecay': make_exp_decay_wds,
            'LinDecay': make_linear_decay_wds,
            'Reverse': make_reverse_wds,
        }
        
        for sched_name, sched_fn in schedules.items():
            best_acc = 0
            best_bwd = 0
            for bwd in BASE_WDS:
                model = MLP(ds['input_dim'], hidden_dims, ds['n_classes'])
                layer_wds = sched_fn(bwd, n_layers)
                acc = train_model(model, ds['train'], ds['test'], layer_wds, 
                                 epochs=EPOCHS_MLP)
                if acc > best_acc:
                    best_acc = acc
                    best_bwd = bwd
            
            mlp_results[sched_name] = {'acc': best_acc, 'best_bwd': best_bwd}
            print(f"  {sched_name:<15s} {best_bwd:>10.1e} {best_acc:>10.4f}")
        
        # Oracle: best single fixed WD from full grid
        oracle_acc = mlp_results['Fixed']['acc']
        
        all_results[ds_name]['MLP'] = mlp_results
        
        # Summary
        print(f"\n  MLP Summary:")
        best_sched = max(mlp_results, key=lambda k: mlp_results[k]['acc'])
        for name in ['Fixed', 'ExpDecay', 'LinDecay', 'Reverse']:
            acc = mlp_results[name]['acc']
            delta = acc - mlp_results['Fixed']['acc']
            marker = " ← BEST" if name == best_sched else ""
            print(f"    {name:<15s}: {acc:.4f} ({delta:+.4f} vs Fixed){marker}")
        
        # ============================================================
        # TEST 2: CNN (if applicable)
        # ============================================================
        if ds['img_size'] >= 28:
            print(f"\n  MODEL: SimpleCNN (3 conv + 2 fc)")
            print(f"  {'Schedule':<15s} {'Base WD':>10s} {'Test Acc':>10s}")
            print(f"  {'-'*15} {'-'*10} {'-'*10}")
            
            cnn_results = {}
            n_cnn_layers = 6  # 3 conv + 2 fc + 1 output
            
            for sched_name, sched_fn in schedules.items():
                best_acc = 0
                best_bwd = 0
                for bwd in BASE_WDS:
                    model = SimpleCNN(ds['n_channels'], ds['n_classes'])
                    # Do a dummy forward pass to initialize fc layer size
                    dummy = torch.randn(1, ds['n_channels'], ds['img_size'], ds['img_size'])
                    model(dummy)
                    
                    layer_wds = sched_fn(bwd, n_cnn_layers)
                    acc = train_model(model, ds['train'], ds['test'], layer_wds,
                                    epochs=EPOCHS_CNN)
                    if acc > best_acc:
                        best_acc = acc
                        best_bwd = bwd
                
                cnn_results[sched_name] = {'acc': best_acc, 'best_bwd': best_bwd}
                print(f"  {sched_name:<15s} {best_bwd:>10.1e} {best_acc:>10.4f}")
            
            all_results[ds_name]['CNN'] = cnn_results
            
            # Summary
            print(f"\n  CNN Summary:")
            best_sched_cnn = max(cnn_results, key=lambda k: cnn_results[k]['acc'])
            for name in ['Fixed', 'ExpDecay', 'LinDecay', 'Reverse']:
                acc = cnn_results[name]['acc']
                delta = acc - cnn_results['Fixed']['acc']
                marker = " ← BEST" if name == best_sched_cnn else ""
                print(f"    {name:<15s}: {acc:.4f} ({delta:+.4f} vs Fixed){marker}")
    
    # ================================================================
    # OVERALL VERDICT
    # ================================================================
    total_time = time.time() - start
    
    print(f"\n{'='*70}")
    print(f"  OVERALL VERDICT")
    print(f"  Runtime: {total_time/60:.1f} minutes")
    print(f"{'='*70}")
    
    exp_wins = 0
    total_tests = 0
    
    for ds_name, ds_results in all_results.items():
        for model_name, model_results in ds_results.items():
            total_tests += 1
            exp_acc = model_results.get('ExpDecay', {}).get('acc', 0)
            fixed_acc = model_results.get('Fixed', {}).get('acc', 0)
            rev_acc = model_results.get('Reverse', {}).get('acc', 0)
            
            exp_beats_fixed = exp_acc > fixed_acc + 0.001
            rev_loses = rev_acc < fixed_acc - 0.001
            
            if exp_beats_fixed:
                exp_wins += 1
            
            print(f"\n  {ds_name} / {model_name}:")
            print(f"    Fixed:    {fixed_acc:.4f}")
            print(f"    ExpDecay: {exp_acc:.4f} ({'✓ BEATS' if exp_beats_fixed else '~ ties'} fixed)")
            print(f"    Reverse:  {rev_acc:.4f} ({'✓ LOSES' if rev_loses else '~ ties'} to fixed)")
            print(f"    Direction matters: {'YES' if rev_loses else 'NO'}")
    
    print(f"\n  ExpDecay beats Fixed: {exp_wins}/{total_tests} dataset/model combinations")
    
    if exp_wins == total_tests:
        print(f"\n  ✓✓ CONFIRMED: Exponential WD decay works on ALL real datasets tested.")
        print(f"  The one-line code change generalizes from synthetic to real data.")
    elif exp_wins > 0:
        print(f"\n  ~ PARTIAL: ExpDecay helps on some but not all configurations.")
        print(f"  The finding may be architecture or dataset dependent.")
    else:
        print(f"\n  ✗ NOT CONFIRMED: ExpDecay does not help on real datasets.")
        print(f"  The finding is specific to synthetic manifolds.")
    
    # Save results
    output = {
        'runtime_minutes': total_time / 60,
        'results': {},
    }
    for ds_name, ds_results in all_results.items():
        output['results'][ds_name] = {}
        for model_name, model_results in ds_results.items():
            output['results'][ds_name][model_name] = {
                name: {'acc': float(r['acc']), 'best_bwd': float(r['best_bwd'])}
                for name, r in model_results.items()
            }
    
    with open(f'{OUTPUT_DIR}/tdf_real_data_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to tdf_real_data_results.json")
    
    # ================================================================
    # VISUALIZATION
    # ================================================================
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        n_plots = sum(len(v) for v in all_results.values())
        fig, axes = plt.subplots(1, max(n_plots, 2), figsize=(7*max(n_plots,2), 6))
        if n_plots == 1:
            axes = [axes]
        fig.suptitle("Does Exponential WD Decay Work on Real Data?",
                     fontsize=14, fontweight='bold', y=0.98, color='white')
        fig.patch.set_facecolor('#08080f')
        
        colors = {'Fixed': '#e74c3c', 'ExpDecay': '#4ecdc4', 
                  'LinDecay': '#f1c40f', 'Reverse': '#606080'}
        
        plot_idx = 0
        for ds_name, ds_results in all_results.items():
            for model_name, model_results in ds_results.items():
                ax = axes[plot_idx] if n_plots > 1 else axes[0]
                names = list(model_results.keys())
                accs = [model_results[n]['acc'] for n in names]
                cols = [colors.get(n, '#aaa') for n in names]
                
                bars = ax.bar(range(len(names)), accs, color=cols,
                             edgecolor='white', linewidth=0.5, alpha=0.8)
                for i, a in enumerate(accs):
                    ax.text(i, a + 0.002, f'{a:.4f}', ha='center', fontsize=8, color='#a0b0c0')
                
                ax.set_xticks(range(len(names)))
                ax.set_xticklabels(names, fontsize=8, color='#a0b0c0', rotation=30)
                ax.set_ylabel('Test Accuracy', fontsize=9, color='#a0b0c0')
                
                best = max(names, key=lambda n: model_results[n]['acc'])
                ax.set_title(f'{ds_name} / {model_name}\nBest: {best}',
                            fontsize=10, color='#4ecdc4' if best == 'ExpDecay' else 'white')
                ax.set_ylim(min(accs) - 0.02, max(accs) + 0.02)
                ax.set_facecolor('#08080f'); ax.tick_params(colors='#606080')
                plot_idx += 1
        
        plt.tight_layout(rect=[0, 0.02, 1, 0.93])
        fig.text(0.5, 0.005, "Christopher Head · Navigator's Log R&D · navigatorslog.netlify.app",
                 ha='center', fontsize=8, color='#404060')
        plt.savefig(f'{OUTPUT_DIR}/tdf_real_data_test.png', dpi=180,
                    bbox_inches='tight', facecolor='#08080f')
        plt.close()
        print(f"  Visualization saved to tdf_real_data_test.png")
    except Exception as e:
        print(f"  Visualization error: {e}")
    
    print(f"\n  Total runtime: {total_time/60:.1f} minutes")
    print("  Done.")

if __name__ == '__main__':
    main()
