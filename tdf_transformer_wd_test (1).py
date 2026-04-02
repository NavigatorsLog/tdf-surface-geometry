#!/usr/bin/env python3
"""
TRANSFORMER WD SYMMETRY-BREAKING TEST
========================================
Navigator's Log R&D | March 2026

Does non-uniform WD help on Transformers too?

We confirmed symmetry-breaking on CNN (4/5 random seeds beat Fixed).
The Transformer has a MORE complex curvature profile (drop-rise-peak-decline
on CLS, flat on spatial tokens). If non-uniformity helps on a simple
hunchback (CNN), it should help even more on a complex profile (Transformer).

OR: the Transformer's attention mechanism already provides sufficient
implicit regularization (like CNN on MNIST), making the WD schedule
irrelevant. The spatial tokens staying flat across blocks might indicate
that attention self-regulates.

13 schedules on CIFAR-10 with the same Tiny ViT from the profiles test.

Requirements: pip install torch torchvision numpy matplotlib
Runtime: several hours on Surface Pro 7 (CPU only)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import math
import time
import json
import gc
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = "."

# ================================================================
# TINY VISION TRANSFORMER (identical to profiles test)
# ================================================================
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=128):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, 
                             kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class TransformerBlock(nn.Module):
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
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x

class TinyViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, 
                 n_classes=10, embed_dim=128, n_blocks=6, n_heads=4, 
                 ff_dim=256, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1, embed_dim) * 0.02)
        self.embed_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, ff_dim, dropout)
            for _ in range(n_blocks)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)
    
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        x = self.embed_dropout(x)
        for block in self.blocks:
            x = block(x)
        cls_out = self.norm(x[:, 0])
        return self.head(cls_out)

# ================================================================
# TRAINING WITH PER-BLOCK WD
# ================================================================
def get_block_param_groups(model):
    """
    Create parameter groups for per-block WD control.
    Groups: patch_embed, cls+pos, block_0...block_5, norm+head
    """
    groups = []
    
    # Patch embedding
    groups.append({
        'name': 'patch_embed',
        'params': list(model.patch_embed.parameters()),
    })
    
    # CLS token + positional embedding
    groups.append({
        'name': 'cls_pos',
        'params': [model.cls_token, model.pos_embed],
    })
    
    # Each transformer block
    for i, block in enumerate(model.blocks):
        groups.append({
            'name': f'block_{i}',
            'params': list(block.parameters()),
        })
    
    # Final norm + classification head
    groups.append({
        'name': 'head',
        'params': list(model.norm.parameters()) + list(model.head.parameters()),
    })
    
    return groups

def train_with_block_wds(train_loader, test_loader, block_wds, 
                          epochs=30, lr=1e-3, seed=42):
    """Train a fresh TinyViT with specific WD per block group."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = TinyViT()
    groups = get_block_param_groups(model)
    
    # Assign WD to each group
    param_groups = []
    for i, g in enumerate(groups):
        wd = block_wds.get(i, 1e-2)
        param_groups.append({
            'params': g['params'],
            'weight_decay': wd,
            'lr': lr,
        })
    
    optimizer = optim.AdamW(param_groups)
    criterion = nn.CrossEntropyLoss()
    
    # LR schedule: warmup + cosine
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
        for bx, by in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
    
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for bx, by in test_loader:
            pred = model(bx).argmax(dim=1)
            correct += (pred == by).sum().item()
            total += len(by)
    acc = correct / total
    
    # Explicit cleanup to prevent OOM across 13 sequential runs
    del model, optimizer, scheduler, criterion, param_groups, groups
    gc.collect()
    
    return acc

# ================================================================
# WD SCHEDULE GENERATORS (8 groups: patch, cls_pos, 6 blocks, head)
# ================================================================
N_GROUPS = 8

def schedule_fixed(base_wd):
    return {i: base_wd for i in range(N_GROUPS)}

def schedule_exp_decay(base_wd):
    return {i: base_wd * (0.5 ** i) for i in range(N_GROUPS)}

def schedule_lin_decay(base_wd):
    return {i: base_wd * (1.0 - i * 0.8 / max(1, N_GROUPS - 1)) for i in range(N_GROUPS)}

def schedule_reverse(base_wd):
    return {i: base_wd * (0.2 + i * 0.8 / max(1, N_GROUPS - 1)) for i in range(N_GROUPS)}

def schedule_random(base_wd, seed=0):
    rng = np.random.RandomState(seed)
    log_base = np.log10(base_wd)
    log_wds = rng.uniform(log_base - 1, log_base + 1, N_GROUPS)
    return {i: float(10 ** log_wds[i]) for i in range(N_GROUPS)}

def schedule_shuffled(base_wd, seed=0):
    rng = np.random.RandomState(seed)
    exp_wds = [base_wd * (0.5 ** i) for i in range(N_GROUPS)]
    rng.shuffle(exp_wds)
    return {i: exp_wds[i] for i in range(N_GROUPS)}

def schedule_alternating(base_wd):
    return {i: base_wd * (2.0 if i % 2 == 0 else 0.2) for i in range(N_GROUPS)}

# ================================================================
# MAIN
# ================================================================
def main():
    start = time.time()
    
    print("=" * 70)
    print("  TRANSFORMER WD SYMMETRY-BREAKING TEST")
    print("  Does non-uniform WD help on Transformers?")
    print("=" * 70)
    
    BATCH_SIZE = 16
    EPOCHS = 30
    BASE_WD = 1e-2  # Same as the profiles test
    
    print(f"\n  Loading CIFAR-10...")
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
    
    # Group names for display
    group_names = ['patch', 'cls_pos', 'blk0', 'blk1', 'blk2', 'blk3', 'blk4', 'blk5', 'head']
    
    schedules = {
        'Fixed': lambda: schedule_fixed(BASE_WD),
        'ExpDecay': lambda: schedule_exp_decay(BASE_WD),
        'LinDecay': lambda: schedule_lin_decay(BASE_WD),
        'Reverse': lambda: schedule_reverse(BASE_WD),
        'Random_1': lambda: schedule_random(BASE_WD, seed=1),
        'Random_2': lambda: schedule_random(BASE_WD, seed=2),
        'Random_3': lambda: schedule_random(BASE_WD, seed=3),
        'Random_4': lambda: schedule_random(BASE_WD, seed=4),
        'Random_5': lambda: schedule_random(BASE_WD, seed=5),
        'Shuffled_1': lambda: schedule_shuffled(BASE_WD, seed=1),
        'Shuffled_2': lambda: schedule_shuffled(BASE_WD, seed=2),
        'Shuffled_3': lambda: schedule_shuffled(BASE_WD, seed=3),
        'Alternating': lambda: schedule_alternating(BASE_WD),
    }
    
    print(f"\n  TinyViT: 6 blocks, 128 dim, {EPOCHS} epochs, base WD = {BASE_WD}")
    print(f"  {N_GROUPS} parameter groups: {', '.join(group_names[:N_GROUPS])}")
    print(f"\n  {'Schedule':<15s} {'Accuracy':>10s} {'vs Fixed':>10s} {'WD values (first 6 groups)':>45s}")
    print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*45}")
    
    results = {}
    fixed_acc = None
    
    for sched_name, sched_fn in schedules.items():
        block_wds = sched_fn()
        
        t0 = time.time()
        acc = train_with_block_wds(train_loader, test_loader, block_wds,
                                    epochs=EPOCHS, lr=1e-3)
        elapsed = (time.time() - t0) / 60
        
        if sched_name == 'Fixed':
            fixed_acc = acc
        
        delta = acc - fixed_acc if fixed_acc is not None else 0
        wd_str = ', '.join(f'{block_wds[i]:.1e}' for i in range(min(6, N_GROUPS)))
        if N_GROUPS > 6:
            wd_str += '...'
        
        total_wd = sum(block_wds.values())
        results[sched_name] = {
            'accuracy': float(acc),
            'delta': float(delta),
            'total_wd': float(total_wd),
            'block_wds': {str(k): float(v) for k, v in block_wds.items()},
            'elapsed_min': float(elapsed),
        }
        
        marker = " <<<" if delta > 0.005 else ""
        print(f"  {sched_name:<15s} {acc:>10.4f} {delta:>+10.4f} {wd_str:>45s}{marker}")
        
        # Save incrementally so completed results survive crashes
        with open(f'{OUTPUT_DIR}/tdf_transformer_wd_test.json', 'w') as f:
            json.dump({'runtime_so_far': (time.time()-start)/60, 'base_wd': BASE_WD,
                       'results': results}, f, indent=2)
    
    # ================================================================
    # ANALYSIS
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  ANALYSIS")
    print(f"{'='*70}")
    
    structured = ['ExpDecay', 'LinDecay', 'Reverse']
    random_names = [n for n in results if n.startswith('Random_')]
    shuffled_names = [n for n in results if n.startswith('Shuffled_')]
    
    struct_accs = [results[n]['accuracy'] for n in structured]
    random_accs = [results[n]['accuracy'] for n in random_names]
    shuffled_accs = [results[n]['accuracy'] for n in shuffled_names]
    alt_acc = results['Alternating']['accuracy']
    
    print(f"\n  Fixed:              {fixed_acc:.4f}")
    print(f"  Structured mean:    {np.mean(struct_accs):.4f} ({np.mean(struct_accs)-fixed_acc:+.4f})")
    print(f"  Random mean:        {np.mean(random_accs):.4f} ({np.mean(random_accs)-fixed_acc:+.4f})")
    print(f"  Shuffled mean:      {np.mean(shuffled_accs):.4f} ({np.mean(shuffled_accs)-fixed_acc:+.4f})")
    print(f"  Alternating:        {alt_acc:.4f} ({alt_acc-fixed_acc:+.4f})")
    
    random_wins = sum(1 for a in random_accs if a > fixed_acc + 0.001)
    print(f"\n  Random seeds beating Fixed: {random_wins}/{len(random_accs)}")
    
    # Compare to CNN results
    print(f"\n  CNN Thread B comparison:")
    print(f"    CNN: Random mean +0.0089, Structured mean +0.0067, 4/5 random won")
    
    if np.mean(random_accs) > fixed_acc + 0.003:
        print(f"\n  >>> SYMMETRY-BREAKING CONFIRMED ON TRANSFORMERS")
        print(f"  >>> Non-uniformity helps on all three architecture families")
    elif np.mean(struct_accs) > fixed_acc + 0.003:
        print(f"\n  >>> STRUCTURE MATTERS: Only directional schedules help on Transformers")
    else:
        print(f"\n  >>> ATTENTION SELF-REGULATES: WD schedule irrelevant on Transformers")
        print(f"  >>> (Same pattern as CNN on MNIST — architecture provides implicit Leg 3)")
    
    # ================================================================
    # SAVE
    # ================================================================
    total_time = time.time() - start
    
    output = {
        'runtime_minutes': total_time / 60,
        'base_wd': BASE_WD,
        'model': 'TinyViT',
        'config': {'embed_dim': 128, 'n_blocks': 6, 'n_heads': 4,
                   'ff_dim': 256, 'patch_size': 4, 'dropout': 0.1,
                   'epochs': EPOCHS, 'batch_size': BATCH_SIZE},
        'results': results,
    }
    
    with open(f'{OUTPUT_DIR}/tdf_transformer_wd_test.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    # ================================================================
    # VISUALIZATION
    # ================================================================
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(1, 1, figsize=(14, 7))
        fig.suptitle("Transformer WD Symmetry-Breaking Test\n"
                     "Does non-uniform WD help on Transformers?",
                     fontsize=14, fontweight='bold', y=0.98, color='white')
        fig.patch.set_facecolor('#08080f')
        
        names = list(results.keys())
        accs = [results[n]['accuracy'] for n in names]
        
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
        ax.set_xticklabels(names, fontsize=7, color='#a0b0c0', rotation=45, ha='right')
        ax.set_ylabel('Test Accuracy', fontsize=10, color='#a0b0c0')
        ax.set_title(f'TinyViT on CIFAR-10 (base WD={BASE_WD})', fontsize=11, color='white')
        ax.set_ylim(min(accs) - 0.01, max(accs) + 0.01)
        ax.legend(fontsize=8, facecolor='#08080f', edgecolor='#303040', labelcolor='#a0b0c0')
        ax.set_facecolor('#08080f'); ax.tick_params(colors='#606080')
        
        for i, a in enumerate(accs):
            ax.text(i, a + 0.001, f'{a:.4f}', ha='center', fontsize=6, color='#a0b0c0', rotation=90)
        
        plt.tight_layout(rect=[0, 0.02, 1, 0.92])
        fig.text(0.5, 0.005, "Christopher Head · Navigator's Log R&D · navigatorslog.netlify.app",
                 ha='center', fontsize=8, color='#404060')
        plt.savefig(f'{OUTPUT_DIR}/tdf_transformer_wd_test.png', dpi=180,
                    bbox_inches='tight', facecolor='#08080f')
        plt.close()
        print(f"\n  Visualization saved to tdf_transformer_wd_test.png")
    except Exception as e:
        print(f"\n  Visualization error: {e}")
    
    print(f"\n  Results saved to tdf_transformer_wd_test.json")
    print(f"  Total runtime: {total_time/60:.1f} minutes")
    print("  Done.")

if __name__ == '__main__':
    main()
