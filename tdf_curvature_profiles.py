#!/usr/bin/env python3
"""
THREAD A: Measure the Actual Curvature Profile on Real Data
==============================================================
Navigator's Log R&D | March 2026

We ASSUMED curvature drops monotonically based on synthetic manifolds.
We never MEASURED it on real data. This script does.

Train MLP and CNN on MNIST and CIFAR-10, hook activations at every
layer, measure curvature with the same local-PCA method used in
all prior experiments.

THE QUESTION: Does curvature really drop to zero by layer 2 on
real data? Or does it persist deeper — explaining why Reverse
worked on CIFAR-10 CNN?

Requirements: pip install torch torchvision numpy scikit-learn matplotlib
Runtime: ~10-15 minutes on Surface Pro 7
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from sklearn.neighbors import NearestNeighbors
import time
import json
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = "."
torch.manual_seed(42)
np.random.seed(42)

# ================================================================
# CURVATURE MEASUREMENT (same method as all prior experiments)
# ================================================================
def measure_curvature(X, n_sample=300, k=10):
    """
    Local PCA residual variance on k-nearest-neighbor patches.
    Same method used in SurfaceGate and all synthetic tests.
    Higher value = more crumpled surface.
    """
    if len(X) < 20 or X.shape[1] < 3:
        return 0.0, 0.0  # curvature, std
    n_s = min(n_sample, len(X))
    X_sub = X[:n_s]
    try:
        nn_model = NearestNeighbors(n_neighbors=min(k+1, n_s-1)).fit(X_sub)
        _, indices = nn_model.kneighbors(X_sub[:min(200, n_s)])
        curvatures = []
        for i in range(min(200, n_s)):
            local = X_sub[indices[i, 1:]] - X_sub[indices[i, 1:]].mean(0)
            try:
                _, s, _ = np.linalg.svd(local, full_matrices=False)
                s = s[s > 1e-10]
                if len(s) > 2:
                    cumvar = np.cumsum(s**2) / np.sum(s**2)
                    curvatures.append(1.0 - cumvar[min(3, len(cumvar)-1)])
            except:
                pass
        if curvatures:
            return float(np.mean(curvatures)), float(np.std(curvatures))
    except:
        pass
    return 0.0, 0.0

def measure_id_twonm(X, n_sample=300):
    """TwoNN intrinsic dimension estimate."""
    n_s = min(n_sample, len(X))
    nn_model = NearestNeighbors(n_neighbors=3).fit(X[:n_s])
    d, _ = nn_model.kneighbors(X[:n_s])
    mask = (d[:,1] > 1e-10) & (d[:,2] > 1e-10)
    mu = d[:,2][mask] / d[:,1][mask]
    if len(mu) < 10:
        return 0.0
    return float(len(mu) / max(1, np.sum(np.log(mu))))

# ================================================================
# MODELS WITH ACTIVATION HOOKS
# ================================================================
class HookedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, n_classes):
        super().__init__()
        self.layers = nn.ModuleList()
        prev = input_dim
        for h in hidden_dims:
            self.layers.append(nn.Linear(prev, h))
            prev = h
        self.output = nn.Linear(prev, n_classes)
        self.activations = {}
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        self.activations['input'] = x.detach()
        for i, layer in enumerate(self.layers):
            x = torch.relu(layer(x))
            self.activations[f'layer_{i}'] = x.detach()
        out = self.output(x)
        self.activations['output'] = out.detach()
        return out

class HookedCNN(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = None  # initialized on first forward
        self.fc2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, n_classes)
        self.activations = {}
        self._fc1_init = False
    
    def forward(self, x):
        self.activations['input'] = x.detach().view(x.size(0), -1)
        
        x = self.pool(torch.relu(self.conv1(x)))
        self.activations['conv1'] = x.detach().view(x.size(0), -1)
        
        x = self.pool(torch.relu(self.conv2(x)))
        self.activations['conv2'] = x.detach().view(x.size(0), -1)
        
        x = self.pool(torch.relu(self.conv3(x)))
        self.activations['conv3'] = x.detach().view(x.size(0), -1)
        
        x = x.view(x.size(0), -1)
        if not self._fc1_init:
            self.fc1 = nn.Linear(x.size(1), 128).to(x.device)
            self._fc1_init = True
        
        x = torch.relu(self.fc1(x))
        self.activations['fc1'] = x.detach()
        
        x = torch.relu(self.fc2(x))
        self.activations['fc2'] = x.detach()
        
        out = self.output(x)
        self.activations['output'] = out.detach()
        return out

# ================================================================
# TRAIN AND MEASURE
# ================================================================
def train_and_measure(model, train_loader, test_loader, epochs, lr=1e-3, wd=1e-3):
    """Train the model, then measure curvature at every layer."""
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        for bx, by in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
    
    # Evaluate
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for bx, by in test_loader:
            pred = model(bx).argmax(dim=1)
            correct += (pred == by).sum().item()
            total += len(by)
    test_acc = correct / total
    
    # Collect activations on a large batch
    model.eval()
    all_activations = {}
    n_collected = 0
    max_samples = 2000
    
    with torch.no_grad():
        for bx, by in train_loader:
            _ = model(bx)
            for name, act in model.activations.items():
                if name not in all_activations:
                    all_activations[name] = []
                all_activations[name].append(act.numpy())
            n_collected += len(bx)
            if n_collected >= max_samples:
                break
    
    # Concatenate
    for name in all_activations:
        all_activations[name] = np.concatenate(all_activations[name], axis=0)[:max_samples]
    
    # Measure curvature and ID at every layer
    profiles = {}
    for name, acts in all_activations.items():
        curv_mean, curv_std = measure_curvature(acts)
        id_est = measure_id_twonm(acts)
        profiles[name] = {
            'curvature_mean': curv_mean,
            'curvature_std': curv_std,
            'intrinsic_dim': id_est,
            'ambient_dim': acts.shape[1],
            'n_samples': acts.shape[0],
        }
    
    return test_acc, profiles

# ================================================================
# MAIN
# ================================================================
def main():
    start = time.time()
    
    print("=" * 70)
    print("  THREAD A: Measure Actual Curvature Profiles on Real Data")
    print("  Does curvature really drop to zero by layer 2?")
    print("=" * 70)
    
    BATCH_SIZE = 128
    
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
        }
        print(f"    MNIST loaded")
    except Exception as e:
        print(f"    MNIST failed: {e}")
    
    try:
        cifar_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_cifar)
        cifar_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_cifar)
        datasets['CIFAR-10'] = {
            'train': DataLoader(cifar_train, batch_size=BATCH_SIZE, shuffle=True),
            'test': DataLoader(cifar_test, batch_size=BATCH_SIZE),
            'input_dim': 3072, 'n_channels': 3, 'n_classes': 10,
        }
        print(f"    CIFAR-10 loaded")
    except Exception as e:
        print(f"    CIFAR-10 failed: {e}")
    
    all_results = {}
    
    for ds_name, ds in datasets.items():
        print(f"\n{'='*70}")
        print(f"  DATASET: {ds_name}")
        print(f"{'='*70}")
        all_results[ds_name] = {}
        
        # MLP
        hidden_dims = [256, 128, 64, 32]
        print(f"\n  Training MLP {hidden_dims}...")
        model = HookedMLP(ds['input_dim'], hidden_dims, ds['n_classes'])
        acc, profiles = train_and_measure(model, ds['train'], ds['test'], 
                                          epochs=20, wd=1e-3)
        all_results[ds_name]['MLP'] = {'accuracy': acc, 'profiles': profiles}
        
        print(f"  MLP accuracy: {acc:.4f}")
        print(f"  {'Layer':<12s} {'Curvature':>10s} {'Curv Std':>10s} {'ID':>8s} {'Ambient':>8s}")
        print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
        for name in sorted(profiles.keys(), key=lambda x: list(profiles.keys()).index(x)):
            p = profiles[name]
            print(f"  {name:<12s} {p['curvature_mean']:>10.4f} {p['curvature_std']:>10.4f} "
                  f"{p['intrinsic_dim']:>8.1f} {p['ambient_dim']:>8d}")
        
        # CNN
        print(f"\n  Training SimpleCNN...")
        model = HookedCNN(ds['n_channels'], ds['n_classes'])
        acc, profiles = train_and_measure(model, ds['train'], ds['test'],
                                          epochs=15, wd=1e-3)
        all_results[ds_name]['CNN'] = {'accuracy': acc, 'profiles': profiles}
        
        print(f"  CNN accuracy: {acc:.4f}")
        print(f"  {'Layer':<12s} {'Curvature':>10s} {'Curv Std':>10s} {'ID':>8s} {'Ambient':>8s}")
        print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
        for name in sorted(profiles.keys(), key=lambda x: list(profiles.keys()).index(x)):
            p = profiles[name]
            print(f"  {name:<12s} {p['curvature_mean']:>10.4f} {p['curvature_std']:>10.4f} "
                  f"{p['intrinsic_dim']:>8.1f} {p['ambient_dim']:>8d}")
    
    # ================================================================
    # COMPARISON: Does the profile match synthetic results?
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  COMPARISON: Real vs Synthetic Curvature Profiles")
    print(f"{'='*70}")
    print(f"\n  Synthetic finding (from SurfaceGate tests):")
    print(f"    Layer 1: curvature 0.01-0.25")
    print(f"    Layer 2: curvature near zero")
    print(f"    Layer 3+: curvature zero")
    print(f"    Pattern: monotonic decrease, flat by layer 2")
    
    for ds_name, ds_results in all_results.items():
        for model_name, model_results in ds_results.items():
            profiles = model_results['profiles']
            # Extract hidden layer curvatures (skip input and output)
            hidden_curvs = []
            hidden_names = []
            for name, p in profiles.items():
                if name not in ('input', 'output'):
                    hidden_curvs.append(p['curvature_mean'])
                    hidden_names.append(name)
            
            if len(hidden_curvs) >= 2:
                monotonic = all(hidden_curvs[i] >= hidden_curvs[i+1] - 0.01 
                               for i in range(len(hidden_curvs)-1))
                flat_by_2 = hidden_curvs[1] < 0.02 if len(hidden_curvs) > 1 else False
                curv_range = max(hidden_curvs) - min(hidden_curvs)
                
                print(f"\n  {ds_name} / {model_name}:")
                print(f"    Curvatures: {' -> '.join(f'{c:.4f}' for c in hidden_curvs)}")
                print(f"    Monotonic decrease: {'YES' if monotonic else 'NO'}")
                print(f"    Flat by layer 2:    {'YES' if flat_by_2 else 'NO'}")
                print(f"    Curvature range:    {curv_range:.4f}")
                
                if not monotonic:
                    peak_idx = np.argmax(hidden_curvs)
                    print(f"    PEAK at: {hidden_names[peak_idx]} (curvature {hidden_curvs[peak_idx]:.4f})")
                    print(f"    >>> NON-MONOTONIC: curvature profile differs from synthetic!")
    
    # Save
    output = {
        'runtime_minutes': (time.time() - start) / 60,
        'results': {}
    }
    for ds_name, ds_results in all_results.items():
        output['results'][ds_name] = {}
        for model_name, model_results in ds_results.items():
            output['results'][ds_name][model_name] = {
                'accuracy': model_results['accuracy'],
                'profiles': {name: {k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v 
                                    for k, v in p.items()} 
                            for name, p in model_results['profiles'].items()}
            }
    
    with open(f'{OUTPUT_DIR}/tdf_curvature_profiles.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    # ================================================================
    # VISUALIZATION
    # ================================================================
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        n_configs = sum(len(v) for v in all_results.values())
        fig, axes = plt.subplots(2, n_configs, figsize=(7*n_configs, 10))
        if n_configs == 1:
            axes = axes.reshape(2, 1)
        fig.suptitle("Thread A: Actual Curvature Profiles on Real Data\n"
                     "Does curvature really drop to zero by layer 2?",
                     fontsize=14, fontweight='bold', y=0.98, color='white')
        fig.patch.set_facecolor('#08080f')
        
        plot_idx = 0
        for ds_name, ds_results in all_results.items():
            for model_name, model_results in ds_results.items():
                profiles = model_results['profiles']
                hidden_names = [n for n in profiles if n not in ('input', 'output')]
                hidden_curvs = [profiles[n]['curvature_mean'] for n in hidden_names]
                hidden_curvs_std = [profiles[n]['curvature_std'] for n in hidden_names]
                hidden_ids = [profiles[n]['intrinsic_dim'] for n in hidden_names]
                
                # Top: curvature profile
                ax = axes[0, plot_idx]
                x = range(len(hidden_names))
                ax.errorbar(x, hidden_curvs, yerr=hidden_curvs_std, 
                           fmt='o-', color='#4ecdc4', linewidth=2, markersize=8,
                           capsize=5, ecolor='#4ecdc4', alpha=0.7)
                ax.set_xticks(x)
                ax.set_xticklabels(hidden_names, fontsize=7, color='#a0b0c0', rotation=30)
                ax.set_ylabel('Curvature', fontsize=9, color='#a0b0c0')
                ax.set_title(f'{ds_name} / {model_name}\nAcc: {model_results["accuracy"]:.4f}',
                            fontsize=10, color='white')
                ax.set_facecolor('#08080f'); ax.tick_params(colors='#606080')
                
                # Bottom: ID profile
                ax = axes[1, plot_idx]
                ax.plot(x, hidden_ids, 'o-', color='#f1c40f', linewidth=2, markersize=8)
                ax.set_xticks(x)
                ax.set_xticklabels(hidden_names, fontsize=7, color='#a0b0c0', rotation=30)
                ax.set_ylabel('Intrinsic Dimension', fontsize=9, color='#a0b0c0')
                ax.set_title('ID Profile', fontsize=10, color='#f1c40f')
                ax.set_facecolor('#08080f'); ax.tick_params(colors='#606080')
                
                plot_idx += 1
        
        plt.tight_layout(rect=[0, 0.02, 1, 0.93])
        fig.text(0.5, 0.005, "Christopher Head · Navigator's Log R&D · navigatorslog.netlify.app",
                 ha='center', fontsize=8, color='#404060')
        plt.savefig(f'{OUTPUT_DIR}/tdf_curvature_profiles.png', dpi=180,
                    bbox_inches='tight', facecolor='#08080f')
        plt.close()
        print(f"\n  Visualization saved to tdf_curvature_profiles.png")
    except Exception as e:
        print(f"\n  Visualization error: {e}")
    
    print(f"\n  Results saved to tdf_curvature_profiles.json")
    print(f"  Total runtime: {(time.time()-start)/60:.1f} minutes")
    print("  Done.")

if __name__ == '__main__':
    main()
