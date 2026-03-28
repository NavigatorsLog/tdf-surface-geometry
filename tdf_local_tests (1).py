#!/usr/bin/env python3
"""
TDF LOCAL TEST SUITE
=====================
Navigator's Log R&D | March 2026
Run on Surface Pro 7 alongside tdf_surface_full_test.py

Three tests that require data downloads or API access:

TEST 1: Real Dataset Surface Geometry (~15 min)
  Downloads MNIST, CIFAR-10, CIFAR-100 via torchvision
  Measures surface properties on REAL data
  Compares to synthetic manifold results

TEST 2: Ansuini Hunchback Replication (~20 min)
  Trains networks on MNIST (real labels vs random labels)
  Measures intrinsic dimension at every layer
  Confirms: generalizing networks compress, memorizing don't

TEST 3: LLM Temperature as Leg 3 (~5 min, needs API key)
  Sends the same prompts at different temperatures
  Measures quality across domains (math, code, creative)
  Tests: optimal temperature differs by domain

Requirements:
  pip install numpy scikit-learn scipy matplotlib torch torchvision

Optional for Test 3:
  pip install anthropic
  Set ANTHROPIC_API_KEY environment variable

Usage:
  python tdf_local_tests.py              # Run all tests
  python tdf_local_tests.py --test 1     # Run specific test
  python tdf_local_tests.py --test 2
  python tdf_local_tests.py --test 3
"""

import numpy as np
import time
import sys
import os
import json
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = "."

# ================================================================
# TEST 1: Real Dataset Surface Geometry
# ================================================================
def test1_real_surfaces():
    """Measure surface properties on MNIST, CIFAR-10, CIFAR-100
    and compare to the synthetic manifold predictions."""
    
    print("=" * 70)
    print("  TEST 1: Surface Geometry of Real Datasets")
    print("=" * 70)
    
    import torch
    import torchvision
    import torchvision.transforms as transforms
    from sklearn.neighbors import NearestNeighbors
    from scipy.stats import pearsonr
    
    def measure_surface(X, n_sample=500):
        n_s = min(n_sample, len(X))
        ambient = X.shape[1]
        
        # ID (TwoNN)
        nn = NearestNeighbors(n_neighbors=3).fit(X)
        dist, _ = nn.kneighbors(X[:n_s])
        r1, r2 = dist[:, 1], dist[:, 2]
        mask = (r1 > 1e-10) & (r2 > 1e-10)
        mu = r2[mask] / r1[mask]
        id_est = len(mu) / max(1, np.sum(np.log(mu)))
        
        # Curvature
        nn2 = NearestNeighbors(n_neighbors=15).fit(X)
        _, indices = nn2.kneighbors(X[:min(n_s, 300)])
        curvatures = []
        for i in range(min(n_s, 300)):
            local = X[indices[i, 1:]] - X[indices[i, 1:]].mean(0)
            try:
                _, s, _ = np.linalg.svd(local, full_matrices=False)
                s = s[s > 1e-10]
                if len(s) > 4:
                    cumvar = np.cumsum(s**2) / np.sum(s**2)
                    curvatures.append(1.0 - cumvar[min(5, len(cumvar)-1)])
            except: pass
        curv_est = np.mean(curvatures) if curvatures else 0
        
        # Density CV
        nn3 = NearestNeighbors(n_neighbors=10).fit(X)
        dist3, _ = nn3.kneighbors(X[:n_s])
        r_k = dist3[:, -1]; r_k = r_k[r_k > 1e-10]
        log_d = -np.log(r_k)
        dens_cv = np.std(log_d) / (np.abs(np.mean(log_d)) + 1e-10)
        
        codim = ambient / max(1, id_est)
        return id_est, curv_est, dens_cv, codim
    
    # Known optimal WDs for these datasets (from established literature)
    datasets_info = {
        'MNIST': {
            'loader': lambda: torchvision.datasets.MNIST(root='/tmp/data', train=True, download=True),
            'reshape': lambda d: d.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0,
            'opt_wd': 1e-4,  # Standard for MNIST
            'published_id': 13,  # Pope et al. 2021
        },
        'CIFAR-10': {
            'loader': lambda: torchvision.datasets.CIFAR10(root='/tmp/data', train=True, download=True),
            'reshape': lambda d: np.array(d.data).reshape(-1, 3072).astype(np.float32) / 255.0,
            'opt_wd': 5e-4,  # He et al. 2016
            'published_id': 20,  # Pope et al. 2021 (17-24 range)
        },
        'CIFAR-100': {
            'loader': lambda: torchvision.datasets.CIFAR100(root='/tmp/data', train=True, download=True),
            'reshape': lambda d: np.array(d.data).reshape(-1, 3072).astype(np.float32) / 255.0,
            'opt_wd': 5e-4,
            'published_id': 30,  # Estimated higher than CIFAR-10
        },
    }
    
    results = {}
    for name, info in datasets_info.items():
        print(f"\n  Loading {name}...")
        try:
            dataset = info['loader']()
            data = info['reshape'](dataset)
            
            # Subsample for speed
            np.random.seed(42)
            idx = np.random.choice(len(data), 3000, replace=False)
            data_sub = data[idx]
            
            print(f"    Shape: {data_sub.shape}")
            print(f"    Measuring surface properties...")
            
            id_m, curv_m, dcv, cr = measure_surface(data_sub)
            
            results[name] = {
                'id_measured': id_m,
                'id_published': info['published_id'],
                'curvature': curv_m,
                'density_cv': dcv,
                'codim_ratio': cr,
                'opt_wd': info['opt_wd'],
                'ambient_dim': data_sub.shape[1],
            }
            
            print(f"    ID (measured):  {id_m:.1f} (published: {info['published_id']})")
            print(f"    Curvature:      {curv_m:.4f}")
            print(f"    Density CV:     {dcv:.3f}")
            print(f"    Codim ratio:    {cr:.1f}")
            print(f"    Known opt WD:   {info['opt_wd']:.1e}")
            
        except Exception as e:
            print(f"    ERROR: {e}")
    
    # Add modular arithmetic for comparison
    print(f"\n  Generating modular arithmetic (mod 97)...")
    p = 97
    mod_data = []
    for a in range(p):
        for b in range(p):
            vec = np.zeros(2*p, dtype=np.float32)
            vec[a] = 1; vec[p+b] = 1
            mod_data.append(vec)
    mod_data = np.array(mod_data)
    idx = np.random.choice(len(mod_data), 3000, replace=False)
    id_m, curv_m, dcv, cr = measure_surface(mod_data[idx])
    results['Modular Arith'] = {
        'id_measured': id_m, 'id_published': 2,
        'curvature': curv_m, 'density_cv': dcv,
        'codim_ratio': cr, 'opt_wd': 1.0,
        'ambient_dim': 2*p,
    }
    print(f"    ID: {id_m:.1f}, Curv: {curv_m:.4f}, DCV: {dcv:.3f}")
    
    # Summary
    print(f"\n  {'Dataset':<15s} {'ID':>6s} {'Curv':>8s} {'DCV':>8s} {'CodR':>6s} {'OptWD':>8s}")
    print(f"  {'-'*15} {'-'*6} {'-'*8} {'-'*8} {'-'*6} {'-'*8}")
    for name, r in results.items():
        print(f"  {name:<15s} {r['id_measured']:>6.1f} {r['curvature']:>8.4f} "
              f"{r['density_cv']:>8.3f} {r['codim_ratio']:>6.1f} {r['opt_wd']:>8.1e}")
    
    # Save
    with open(f'{OUTPUT_DIR}/test1_real_surfaces.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to test1_real_surfaces.json")
    
    return results


# ================================================================
# TEST 2: Ansuini Hunchback — ID Across Layers
# ================================================================
def test2_hunchback():
    """Train networks on MNIST with real labels vs random labels.
    Measure intrinsic dimension at every layer.
    
    Prediction: real-label network shows hunchback (ID up then down).
    Random-label network shows flat ID (memorization, no compression).
    """
    
    print("\n" + "=" * 70)
    print("  TEST 2: Ansuini Hunchback (ID Across Layers)")
    print("=" * 70)
    
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.neighbors import NearestNeighbors
    
    # Load MNIST
    print("\n  Loading MNIST...")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = torchvision.datasets.MNIST(root='/tmp/data', train=True, download=True, transform=transform)
    test_ds = torchvision.datasets.MNIST(root='/tmp/data', train=False, download=True, transform=transform)
    
    # Subsample for speed
    train_data = train_ds.data[:5000].float().reshape(-1, 784) / 255.0
    train_labels_real = train_ds.targets[:5000]
    train_labels_random = torch.randint(0, 10, (5000,))
    
    test_data = test_ds.data[:1000].float().reshape(-1, 784) / 255.0
    test_labels = test_ds.targets[:1000]
    
    # Define a simple MLP with hooks to extract intermediate representations
    class MLPWithHooks(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(784, 256), nn.ReLU(),   # Layer 0-1
                nn.Linear(256, 128), nn.ReLU(),   # Layer 2-3
                nn.Linear(128, 64),  nn.ReLU(),   # Layer 4-5
                nn.Linear(64, 32),   nn.ReLU(),   # Layer 6-7
                nn.Linear(32, 10),                 # Layer 8
            )
            self.activations = {}
            
        def forward(self, x):
            self.activations['input'] = x.detach()
            for i, layer in enumerate(self.layers):
                x = layer(x)
                self.activations[f'layer_{i}'] = x.detach()
            return x
    
    def measure_id_twonn(X):
        if len(X) < 10: return 0
        X_np = X.numpy() if hasattr(X, 'numpy') else X
        nn = NearestNeighbors(n_neighbors=3).fit(X_np)
        dist, _ = nn.kneighbors(X_np[:min(500, len(X_np))])
        r1, r2 = dist[:, 1], dist[:, 2]
        mask = (r1 > 1e-10) & (r2 > 1e-10)
        if mask.sum() < 3: return 0
        mu = r2[mask] / r1[mask]
        return len(mu) / max(1, np.sum(np.log(mu)))
    
    def train_and_measure(model, data, labels, name, epochs=30):
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        dataset = TensorDataset(data, labels)
        loader = DataLoader(dataset, batch_size=256, shuffle=True)
        
        print(f"\n  Training '{name}' for {epochs} epochs...")
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                out = model(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                model.eval()
                with torch.no_grad():
                    out = model(test_data)
                    pred = out.argmax(dim=1)
                    acc = (pred == test_labels).float().mean().item()
                    print(f"    Epoch {epoch+1}: loss={total_loss/len(loader):.4f}, test_acc={acc:.3f}")
        
        # Measure ID at each layer
        print(f"  Measuring ID at each layer...")
        model.eval()
        with torch.no_grad():
            _ = model(data[:1000])
        
        layer_ids = {}
        for key in sorted(model.activations.keys()):
            act = model.activations[key]
            if act.dim() > 1 and act.shape[1] > 2:
                id_val = measure_id_twonn(act[:1000])
                layer_ids[key] = id_val
                print(f"    {key:<12s}: ID = {id_val:.1f} (dim={act.shape[1]})")
        
        return layer_ids
    
    # Train with real labels
    model_real = MLPWithHooks()
    ids_real = train_and_measure(model_real, train_data, train_labels_real, "real labels")
    
    # Train with random labels (memorization)
    model_random = MLPWithHooks()
    ids_random = train_and_measure(model_random, train_data, train_labels_random, "random labels")
    
    # Summary
    print(f"\n  HUNCHBACK COMPARISON:")
    print(f"  {'Layer':<12s} {'Real Labels':>12s} {'Random Labels':>14s} {'Difference':>12s}")
    print(f"  {'-'*12} {'-'*12} {'-'*14} {'-'*12}")
    for key in sorted(ids_real.keys()):
        r = ids_real.get(key, 0)
        m = ids_random.get(key, 0)
        diff = r - m
        print(f"  {key:<12s} {r:>12.1f} {m:>14.1f} {diff:>+12.1f}")
    
    # Does real show hunchback? (increase then decrease)
    real_vals = [ids_real[k] for k in sorted(ids_real.keys())]
    if len(real_vals) >= 4:
        peak_idx = np.argmax(real_vals)
        has_hunchback = 0 < peak_idx < len(real_vals) - 1
        print(f"\n  Hunchback shape (peak in middle)? {'YES' if has_hunchback else 'NO'}")
        print(f"  Peak at: {sorted(ids_real.keys())[peak_idx]} (ID={real_vals[peak_idx]:.1f})")
    
    # Is random flatter?
    rand_vals = [ids_random[k] for k in sorted(ids_random.keys())]
    real_range = max(real_vals) - min(real_vals)
    rand_range = max(rand_vals) - min(rand_vals) if rand_vals else 0
    print(f"  ID range (real labels):   {real_range:.1f}")
    print(f"  ID range (random labels): {rand_range:.1f}")
    print(f"  Random is {'FLATTER' if rand_range < real_range else 'NOT flatter'}")
    
    results = {
        'ids_real': ids_real,
        'ids_random': ids_random,
        'real_range': real_range,
        'random_range': rand_range,
    }
    
    with open(f'{OUTPUT_DIR}/test2_hunchback.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to test2_hunchback.json")
    
    return results


# ================================================================
# TEST 3: LLM Temperature as Leg 3
# ================================================================
def test3_temperature():
    """Test whether optimal temperature differs by domain.
    
    Temperature controls the softmax sharpness — it IS regularization.
    Low temp = sharp distribution = heavy brakes (deterministic)
    High temp = flat distribution = light brakes (creative/random)
    
    Prediction: math/code want low temp, creative wants higher temp.
    """
    
    print("\n" + "=" * 70)
    print("  TEST 3: Temperature as Leg 3 (LLM Domain Test)")
    print("=" * 70)
    
    api_key = os.environ.get('ANTHROPIC_API_KEY', '')
    if not api_key:
        print("\n  No ANTHROPIC_API_KEY found. Skipping API test.")
        print("  To run: export ANTHROPIC_API_KEY='your-key-here'")
        print("  Then re-run: python tdf_local_tests.py --test 3")
        return None
    
    try:
        from anthropic import Anthropic
    except ImportError:
        print("\n  anthropic package not installed.")
        print("  Run: pip install anthropic")
        return None
    
    client = Anthropic(api_key=api_key)
    
    # Test prompts across domains
    prompts = {
        'math': "What is the derivative of f(x) = x^3 * sin(x)? Show your work step by step.",
        'code': "Write a Python function that finds the longest palindromic substring in a string. Include comments.",
        'factual': "Explain the three main types of plate boundaries in geology and give one example of each.",
        'creative': "Write a short paragraph describing a sunrise from the perspective of the last person on Earth.",
        'reasoning': "A bat and a ball cost $1.10 together. The bat costs $1.00 more than the ball. How much does the ball cost? Explain your reasoning carefully.",
    }
    
    temperatures = [0.0, 0.3, 0.5, 0.7, 1.0]
    
    print(f"\n  Testing {len(prompts)} domains × {len(temperatures)} temperatures...")
    print(f"  ({len(prompts) * len(temperatures)} API calls total)\n")
    
    results = {}
    for domain, prompt in prompts.items():
        results[domain] = {}
        print(f"  Domain: {domain}")
        
        for temp in temperatures:
            try:
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=500,
                    temperature=temp,
                    messages=[{"role": "user", "content": prompt}]
                )
                text = response.content[0].text
                
                # Simple quality metrics
                word_count = len(text.split())
                # Unique words ratio (lexical diversity)
                words = text.lower().split()
                unique_ratio = len(set(words)) / max(1, len(words))
                
                results[domain][str(temp)] = {
                    'word_count': word_count,
                    'unique_ratio': unique_ratio,
                    'text_preview': text[:100] + '...',
                }
                
                print(f"    T={temp}: {word_count} words, diversity={unique_ratio:.3f}")
                
            except Exception as e:
                print(f"    T={temp}: ERROR - {e}")
                results[domain][str(temp)] = {'error': str(e)}
        
        print()
    
    # Analysis: which temperature is "best" for each domain?
    # (using lexical diversity as proxy — more principled scoring would need human eval)
    print(f"  ANALYSIS:")
    print(f"  {'Domain':<12s} {'Best Temp':>10s} {'Worst Temp':>10s} {'Range':>8s}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*8}")
    
    for domain in prompts:
        if domain not in results: continue
        scores = {}
        for temp_str, data in results[domain].items():
            if 'error' not in data:
                # Combine word count and diversity
                scores[float(temp_str)] = data['unique_ratio']
        
        if scores:
            best_temp = max(scores, key=scores.get)
            worst_temp = min(scores, key=scores.get)
            score_range = max(scores.values()) - min(scores.values())
            print(f"  {domain:<12s} {best_temp:>10.1f} {worst_temp:>10.1f} {score_range:>8.3f}")
    
    with open(f'{OUTPUT_DIR}/test3_temperature.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to test3_temperature.json")
    
    return results


# ================================================================
# MAIN
# ================================================================
def main():
    start = time.time()
    
    # Parse arguments
    test_num = None
    if '--test' in sys.argv:
        idx = sys.argv.index('--test')
        if idx + 1 < len(sys.argv):
            test_num = int(sys.argv[idx + 1])
    
    print("=" * 70)
    print("  TDF LOCAL TEST SUITE")
    print("  Navigator's Log R&D | March 2026")
    print("=" * 70)
    
    if test_num is None or test_num == 1:
        try:
            test1_real_surfaces()
        except Exception as e:
            print(f"\n  TEST 1 FAILED: {e}")
    
    if test_num is None or test_num == 2:
        try:
            test2_hunchback()
        except Exception as e:
            print(f"\n  TEST 2 FAILED: {e}")
    
    if test_num is None or test_num == 3:
        try:
            test3_temperature()
        except Exception as e:
            print(f"\n  TEST 3 FAILED: {e}")
    
    total = time.time() - start
    print(f"\n  Total runtime: {total/60:.1f} minutes")
    print("  All output files saved to current directory.")
    print("  Bring the JSON files back to your next Claude session.")

if __name__ == '__main__':
    main()
