#!/usr/bin/env python3
"""
TRANSFORMER CURVATURE PROFILES
================================
Navigator's Log R&D | March 2026

THE QUESTION: What does the curvature profile look like inside
a Transformer? We measured MLPs (gradual decline) and CNNs
(hunchback — curvature INCREASES then decreases). Transformers
have attention, which re-introduces information at every block.
The profile might be:
  - Monotonic decline (like MLPs)
  - Hunchback (like CNNs)
  - Something new (oscillating? flat? rising?)

If attention re-crumples the manifold at every block, the profile
should show curvature persisting or oscillating rather than
monotonically declining. That would explain why front-loaded
WD schedules don't generalize — and why SurfaceGate (runtime
measurement) might be the right tool for Transformers.

Uses a tiny Vision Transformer (ViT) on CIFAR-10.
Same curvature measurement method as all prior experiments.
Direct comparison to MLP and CNN profiles from Thread A.

Requirements: pip install torch torchvision numpy scikit-learn matplotlib
Runtime: ~30-60 minutes on Surface Pro 7 (CPU only)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from sklearn.neighbors import NearestNeighbors
import math
import time
import json
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = "."
torch.manual_seed(42)
np.random.seed(42)

# ================================================================
# CURVATURE & ID MEASUREMENT (identical to Thread A)
# ================================================================
def measure_curvature(X, n_sample=300, k=10):
    if len(X) < 20 or X.shape[1] < 3:
        return 0.0, 0.0
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

def measure_id(X, n_sample=300):
    n_s = min(n_sample, len(X))
    nn_model = NearestNeighbors(n_neighbors=3).fit(X[:n_s])
    d, _ = nn_model.kneighbors(X[:n_s])
    mask = (d[:,1] > 1e-10) & (d[:,2] > 1e-10)
    mu = d[:,2][mask] / d[:,1][mask]
    if len(mu) < 10:
        return 0.0
    return float(len(mu) / max(1, np.sum(np.log(mu))))

# ================================================================
# TINY VISION TRANSFORMER
# ================================================================
class PatchEmbedding(nn.Module):
    """Split image into patches and embed them."""
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=128):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, 
                             kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        # x: (B, C, H, W) -> (B, n_patches, embed_dim)
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, n_patches, embed_dim)
        return x

class TransformerBlock(nn.Module):
    """Standard transformer block: attention + FFN + residuals + layer norm."""
    def __init__(self, embed_dim=128, n_heads=4, ff_dim=256, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, 
                                          dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        # Self-attention with residual
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        # FFN with residual
        x = x + self.ff(self.norm2(x))
        return x

class TinyViT(nn.Module):
    """
    Tiny Vision Transformer for CIFAR-10.
    Small enough for CPU training, deep enough to show curvature dynamics.
    
    Architecture:
      - 4x4 patch embedding (32x32 -> 64 patches of dim 128)
      - Learnable [CLS] token
      - Positional embedding
      - 6 transformer blocks (attention + FFN each)
      - Classification head on [CLS] token
    
    6 blocks gives us 6 measurement points to see how
    curvature evolves through the transformer.
    """
    def __init__(self, img_size=32, patch_size=4, in_channels=3, 
                 n_classes=10, embed_dim=128, n_blocks=6, n_heads=4, 
                 ff_dim=256, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches
        
        # Learnable CLS token and positional embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1, embed_dim) * 0.02)
        self.embed_dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, ff_dim, dropout)
            for _ in range(n_blocks)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)
        
        # Storage for activations (populated during forward ONLY when measuring)
        self.activations = {}
        self.collecting = False  # OFF during training, ON during measurement
    
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)
        
        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, n_patches+1, embed_dim)
        x = x + self.pos_embed
        x = self.embed_dropout(x)
        
        if self.collecting:
            self.activations['embedding'] = x.detach().reshape(B, -1)
        
        # Transformer blocks
        for i, block in enumerate(self.blocks):
            x = block(x)
            if self.collecting:
                self.activations[f'block_{i}_cls'] = x[:, 0, :].detach()
                self.activations[f'block_{i}_full'] = x.detach().reshape(B, -1)
        
        # Classification on CLS token
        cls_out = self.norm(x[:, 0])
        if self.collecting:
            self.activations['pre_head'] = cls_out.detach()
        
        logits = self.head(cls_out)
        if self.collecting:
            self.activations['output'] = logits.detach()
        
        return logits

# ================================================================
# TRAINING
# ================================================================
def train_model(model, train_loader, test_loader, epochs=30, lr=1e-3, wd=1e-3):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()
    
    # Learning rate warmup + cosine decay
    total_steps = epochs * len(train_loader)
    warmup_steps = 2 * len(train_loader)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for bx, by in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for bx, by in test_loader:
                    pred = model(bx).argmax(dim=1)
                    correct += (pred == by).sum().item()
                    total += len(by)
            print(f"    Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}, "
                  f"test_acc={correct/total:.4f}")
    
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for bx, by in test_loader:
            pred = model(bx).argmax(dim=1)
            correct += (pred == by).sum().item()
            total += len(by)
    return correct / total

# ================================================================
# MAIN
# ================================================================
def main():
    start = time.time()
    
    print("=" * 70)
    print("  TRANSFORMER CURVATURE PROFILES")
    print("  What does the manifold look like inside a Transformer?")
    print("=" * 70)
    
    BATCH_SIZE = 32
    EPOCHS = 30
    
    # Load CIFAR-10
    print("\n  Loading CIFAR-10...")
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    train_ds = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                             download=True, transform=transform_train)
    test_ds = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform_test)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=0)
    
    # ============================================================
    # CONFIGURATION 1: Tiny ViT (6 blocks, 128 dim)
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  TINY ViT: 6 blocks, 128 embed dim, 4 heads, 4x4 patches")
    print(f"  Training for {EPOCHS} epochs...")
    print(f"{'='*70}")
    
    model = TinyViT(
        img_size=32, patch_size=4, in_channels=3, n_classes=10,
        embed_dim=128, n_blocks=6, n_heads=4, ff_dim=256, dropout=0.1
    )
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    
    test_acc = train_model(model, train_loader, test_loader, 
                           epochs=EPOCHS, lr=1e-3, wd=1e-2)
    print(f"\n  Final test accuracy: {test_acc:.4f}")
    
    # ============================================================
    # COLLECT ACTIVATIONS AND MEASURE CURVATURE
    # ============================================================
    print(f"\n  Collecting activations on 2000 samples...")
    
    model.eval()
    model.collecting = True  # NOW start storing activations
    all_activations = {}
    n_collected = 0
    max_samples = 2000
    
    # Use test transform for consistent measurements
    measure_loader = DataLoader(
        torchvision.datasets.CIFAR10(root='./data', train=True, 
                                      download=False, transform=transform_test),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    
    with torch.no_grad():
        for bx, by in measure_loader:
            _ = model(bx)
            for name, act in model.activations.items():
                if name not in all_activations:
                    all_activations[name] = []
                all_activations[name].append(act.numpy())
            n_collected += len(bx)
            if n_collected >= max_samples:
                break
    
    for name in all_activations:
        all_activations[name] = np.concatenate(all_activations[name], axis=0)[:max_samples]
    
    # ============================================================
    # MEASURE CURVATURE AND ID AT EVERY POINT
    # ============================================================
    print(f"\n  Measuring curvature and ID at every layer...")
    print(f"  {'Layer':<20s} {'Curvature':>10s} {'Curv Std':>10s} {'ID':>8s} {'Ambient':>8s}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
    
    profiles = {}
    
    # Measure in order: embedding, then blocks (CLS token view), then pre_head, output
    layer_order = ['embedding']
    for i in range(6):
        layer_order.append(f'block_{i}_cls')
    layer_order.extend(['pre_head', 'output'])
    
    for name in layer_order:
        if name not in all_activations:
            continue
        acts = all_activations[name]
        curv_mean, curv_std = measure_curvature(acts)
        id_est = measure_id(acts)
        
        profiles[name] = {
            'curvature_mean': curv_mean,
            'curvature_std': curv_std,
            'intrinsic_dim': id_est,
            'ambient_dim': acts.shape[1],
            'n_samples': acts.shape[0],
        }
        
        print(f"  {name:<20s} {curv_mean:>10.4f} {curv_std:>10.4f} "
              f"{id_est:>8.1f} {acts.shape[1]:>8d}")
    
    # Also measure full-sequence view for comparison
    print(f"\n  Full-sequence view (all tokens flattened):")
    print(f"  {'Layer':<20s} {'Curvature':>10s} {'ID':>8s} {'Ambient':>8s}")
    print(f"  {'-'*20} {'-'*10} {'-'*8} {'-'*8}")
    
    full_profiles = {}
    for i in range(6):
        name = f'block_{i}_full'
        if name not in all_activations:
            continue
        acts = all_activations[name]
        curv_mean, curv_std = measure_curvature(acts, n_sample=200)
        id_est = measure_id(acts, n_sample=200)
        full_profiles[name] = {
            'curvature_mean': curv_mean,
            'curvature_std': curv_std,
            'intrinsic_dim': id_est,
            'ambient_dim': acts.shape[1],
        }
        print(f"  {name:<20s} {curv_mean:>10.4f} {id_est:>8.1f} {acts.shape[1]:>8d}")
    
    # ============================================================
    # COMPARISON TO MLP AND CNN PROFILES
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  COMPARISON: Transformer vs MLP vs CNN on CIFAR-10")
    print(f"{'='*70}")
    
    # Extract hidden layer curvatures (skip input-like and output)
    transformer_curvs = []
    transformer_names = []
    for name in layer_order:
        if name in profiles and name not in ('embedding', 'output'):
            transformer_curvs.append(profiles[name]['curvature_mean'])
            transformer_names.append(name)
    
    print(f"\n  Transformer (CLS token view):")
    print(f"    {' -> '.join(f'{c:.3f}' for c in transformer_curvs)}")
    
    # Check monotonicity
    if len(transformer_curvs) >= 2:
        monotonic_dec = all(transformer_curvs[i] >= transformer_curvs[i+1] - 0.01 
                           for i in range(len(transformer_curvs)-1))
        monotonic_inc = all(transformer_curvs[i] <= transformer_curvs[i+1] + 0.01 
                           for i in range(len(transformer_curvs)-1))
        has_hunchback = any(transformer_curvs[i] < transformer_curvs[i+1] - 0.01
                           for i in range(len(transformer_curvs)-1))
        
        peak_idx = np.argmax(transformer_curvs)
        trough_idx = np.argmin(transformer_curvs)
        curv_range = max(transformer_curvs) - min(transformer_curvs)
        
        print(f"\n  Profile analysis:")
        print(f"    Monotonic decrease:  {'YES' if monotonic_dec else 'NO'}")
        print(f"    Monotonic increase:  {'YES' if monotonic_inc else 'NO'}")
        print(f"    Has hunchback:       {'YES' if has_hunchback else 'NO'}")
        print(f"    Peak at:             {transformer_names[peak_idx]} ({transformer_curvs[peak_idx]:.4f})")
        print(f"    Trough at:           {transformer_names[trough_idx]} ({transformer_curvs[trough_idx]:.4f})")
        print(f"    Curvature range:     {curv_range:.4f}")
        
        # Compare to known profiles
        print(f"\n  Known profiles from Thread A:")
        print(f"    MLP on CIFAR-10:   0.286 -> 0.258 -> 0.191 -> 0.128  (gradual decline)")
        print(f"    CNN on CIFAR-10:   0.360 -> 0.362 -> 0.355 -> 0.326 -> 0.198  (hunchback)")
        
        if monotonic_dec:
            print(f"\n  VERDICT: Transformer shows MONOTONIC DECLINE (like MLP)")
            print(f"  Attention does NOT re-crumple the manifold.")
            print(f"  Front-loaded WD schedules should work.")
        elif has_hunchback:
            peak_name = transformer_names[peak_idx]
            print(f"\n  VERDICT: Transformer shows HUNCHBACK (like CNN)")
            print(f"  Curvature peaks at {peak_name}")
            print(f"  Attention re-crumples during the expansion phase.")
            print(f"  Bimodal WD schedule may be optimal.")
        else:
            print(f"\n  VERDICT: Transformer shows a NOVEL profile")
            print(f"  Neither monotonic nor simple hunchback.")
            print(f"  SurfaceGate (runtime measurement) may be needed.")
    
    # ============================================================
    # SAVE RESULTS
    # ============================================================
    results = {
        'runtime_minutes': (time.time() - start) / 60,
        'model': 'TinyViT',
        'config': {
            'embed_dim': 128, 'n_blocks': 6, 'n_heads': 4,
            'ff_dim': 256, 'patch_size': 4, 'dropout': 0.1,
            'n_params': n_params, 'epochs': EPOCHS,
        },
        'test_accuracy': test_acc,
        'profiles_cls': {name: {k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v 
                                for k, v in p.items()} 
                        for name, p in profiles.items()},
        'profiles_full': {name: {k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v 
                                 for k, v in p.items()} 
                         for name, p in full_profiles.items()},
    }
    
    with open(f'{OUTPUT_DIR}/tdf_transformer_profiles.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # ============================================================
    # VISUALIZATION
    # ============================================================
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(21, 7))
        fig.suptitle("Transformer Curvature Profile on CIFAR-10\n"
                     "Does attention re-crumple the manifold?",
                     fontsize=14, fontweight='bold', y=0.98, color='white')
        fig.patch.set_facecolor('#08080f')
        
        # P1: Transformer curvature profile (CLS token)
        ax = axes[0]
        x = range(len(transformer_curvs))
        ax.plot(x, transformer_curvs, 'o-', color='#e8a87c', linewidth=2.5, 
                markersize=8, label='Transformer (CLS)')
        ax.set_xticks(x)
        ax.set_xticklabels(transformer_names, fontsize=5, color='#a0b0c0', rotation=45, ha='right')
        ax.set_ylabel('Curvature', fontsize=9, color='#a0b0c0')
        ax.set_title(f'Transformer Profile\nAcc: {test_acc:.4f}', fontsize=10, color='#e8a87c')
        ax.legend(fontsize=7, facecolor='#08080f', edgecolor='#303040', labelcolor='#a0b0c0')
        ax.set_facecolor('#08080f'); ax.tick_params(colors='#606080')
        
        # P2: ID profile
        ax = axes[1]
        transformer_ids = [profiles[name]['intrinsic_dim'] for name in transformer_names]
        ax.plot(x, transformer_ids, 'o-', color='#f1c40f', linewidth=2.5, markersize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(transformer_names, fontsize=5, color='#a0b0c0', rotation=45, ha='right')
        ax.set_ylabel('Intrinsic Dimension', fontsize=9, color='#a0b0c0')
        ax.set_title('ID Profile', fontsize=10, color='#f1c40f')
        ax.set_facecolor('#08080f'); ax.tick_params(colors='#606080')
        
        # P3: Comparison — all three architectures
        ax = axes[2]
        # MLP profile (from Thread A)
        mlp_curvs = [0.286, 0.258, 0.191, 0.128, 0.085]
        mlp_x = np.linspace(0, 1, len(mlp_curvs))
        ax.plot(mlp_x, mlp_curvs, 's-', color='#4ecdc4', linewidth=2, markersize=7,
                label='MLP (gradual decline)', alpha=0.8)
        
        # CNN profile (from Thread A)
        cnn_curvs = [0.360, 0.362, 0.355, 0.326, 0.198, 0.104]
        cnn_x = np.linspace(0, 1, len(cnn_curvs))
        ax.plot(cnn_x, cnn_curvs, 'D-', color='#e74c3c', linewidth=2, markersize=7,
                label='CNN (hunchback)', alpha=0.8)
        
        # Transformer profile
        trans_x = np.linspace(0, 1, len(transformer_curvs))
        ax.plot(trans_x, transformer_curvs, 'o-', color='#e8a87c', linewidth=2.5, markersize=8,
                label='Transformer')
        
        ax.set_xlabel('Relative Depth (0=input, 1=output)', fontsize=9, color='#a0b0c0')
        ax.set_ylabel('Curvature', fontsize=9, color='#a0b0c0')
        ax.set_title('Three Architectures on CIFAR-10', fontsize=10, color='white')
        ax.legend(fontsize=7, facecolor='#08080f', edgecolor='#303040', labelcolor='#a0b0c0')
        ax.set_facecolor('#08080f'); ax.tick_params(colors='#606080')
        
        plt.tight_layout(rect=[0, 0.02, 1, 0.93])
        fig.text(0.5, 0.005, "Christopher Head · Navigator's Log R&D · navigatorslog.netlify.app",
                 ha='center', fontsize=8, color='#404060')
        plt.savefig(f'{OUTPUT_DIR}/tdf_transformer_profiles.png', dpi=180,
                    bbox_inches='tight', facecolor='#08080f')
        plt.close()
        print(f"\n  Visualization saved to tdf_transformer_profiles.png")
    except Exception as e:
        print(f"\n  Visualization error: {e}")
    
    print(f"\n  Results saved to tdf_transformer_profiles.json")
    print(f"  Total runtime: {(time.time()-start)/60:.1f} minutes")
    print("  Done.")

if __name__ == '__main__':
    main()
