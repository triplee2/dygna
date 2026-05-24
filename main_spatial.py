"""
Training loop for spatial consistency classification.

Phase 1 of spatial reasoning:
  teach the model to detect geometric contradictions
  in spatial region graphs.

Trains for 50 epochs on 50 environments (200 samples with augmentation).
Validates on a held-out split of environments.
Saves best model by validation accuracy.

Expected behavior if the model is learning:
  - Epoch 1:  accuracy ~50% (random)
  - Epoch 10: accuracy > 60%
  - Epoch 30: accuracy > 75%
  - Epoch 50: accuracy > 85%

If accuracy stays at 50% after 10 epochs, the model is not learning.
That means either the corruption is too subtle, the model is too small,
or there is a bug in the gradient flow.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np

sys.path.insert(0, '/workspaces/dygna')
from spatial_dataset import SpatialConsistencyDataset, collate_consistency_batch
from spatial_model import SpatialConsistencyClassifier
from utils.utils import set_seed

# Config
ENV_DIR     = '/workspaces/dygna/environments'
SAVE_DIR    = '/workspaces/dygna/spatial_checkpoints'
EPOCHS      = 50
BATCH_SIZE  = 8
LR          = 1e-3
LATENT_SIZE = 64
MLP_LAYERS  = 2
SEED        = 42
VAL_SPLIT   = 0.2  # 20% of environments for validation

os.makedirs(SAVE_DIR, exist_ok=True)
set_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Dataset
print("\nLoading dataset...")
full_dataset = SpatialConsistencyDataset(
    env_dir=ENV_DIR,
    augmentations=4
)

# Split by graph index to avoid data leakage
# (same environment should not appear in both train and val)
n_graphs = len(full_dataset.graphs)
n_val_graphs = max(1, int(n_graphs * VAL_SPLIT))
n_train_graphs = n_graphs - n_val_graphs

print(f"Train graphs: {n_train_graphs}, Val graphs: {n_val_graphs}")

# Create split datasets manually
class SplitDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, graph_indices, augmentations):
        self.base = base_dataset
        self.graph_indices = graph_indices
        self.augmentations = augmentations

    def __len__(self):
        return len(self.graph_indices) * self.augmentations

    def __getitem__(self, idx):
        graph_idx = self.graph_indices[idx % len(self.graph_indices)]
        from spatial_dataset import corrupt_graph
        graph = self.base.graphs[graph_idx]
        corrupted = corrupt_graph(graph)
        return {'consistent': graph, 'corrupted': corrupted}

all_indices = list(range(n_graphs))
train_indices = all_indices[:n_train_graphs]
val_indices = all_indices[n_train_graphs:]

train_dataset = SplitDataset(full_dataset, train_indices, augmentations=4)
val_dataset   = SplitDataset(full_dataset, val_indices,   augmentations=4)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE,
    shuffle=True, collate_fn=collate_consistency_batch
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE,
    shuffle=False, collate_fn=collate_consistency_batch
)

print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

# Model
model = SpatialConsistencyClassifier(
    latent_size=LATENT_SIZE,
    mlp_layers=MLP_LAYERS
).to(device)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model parameters: {n_params:,}")

optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=10
)
criterion = nn.BCELoss()

best_val_acc = 0.0
best_model_path = None

print(f"\nTraining for {EPOCHS} epochs...")
print(f"{'Epoch':>6} {'Train Loss':>12} {'Train Acc':>10} {'Val Acc':>10} {'Best':>8}")
print("-" * 52)

for epoch in range(1, EPOCHS + 1):
    # Training
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for consistent_batch, corrupted_batch in train_loader:
        consistent_batch = consistent_batch.to(device)
        corrupted_batch  = corrupted_batch.to(device)

        optimizer.zero_grad()

        # Forward pass on both graphs
        score_consistent = model(consistent_batch)  # should be near 0
        score_corrupted  = model(corrupted_batch)   # should be near 1

        # Labels
        labels_consistent = torch.zeros_like(score_consistent)
        labels_corrupted  = torch.ones_like(score_corrupted)

        # Combined loss
        loss = (
            criterion(score_consistent, labels_consistent) +
            criterion(score_corrupted,  labels_corrupted)
        ) / 2

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item()

        # Accuracy
        pred_consistent = (score_consistent < 0.5).float()
        pred_corrupted  = (score_corrupted  > 0.5).float()
        batch_size = score_consistent.shape[0]
        train_correct += pred_consistent.sum().item() + pred_corrupted.sum().item()
        train_total   += 2 * batch_size

    train_loss /= len(train_loader)
    train_acc   = train_correct / train_total

    # Validation
    model.eval()
    val_correct = 0
    val_total   = 0

    with torch.no_grad():
        for consistent_batch, corrupted_batch in val_loader:
            consistent_batch = consistent_batch.to(device)
            corrupted_batch  = corrupted_batch.to(device)

            score_consistent = model(consistent_batch)
            score_corrupted  = model(corrupted_batch)

            pred_consistent = (score_consistent < 0.5).float()
            pred_corrupted  = (score_corrupted  > 0.5).float()

            batch_size = score_consistent.shape[0]
            val_correct += pred_consistent.sum().item() + pred_corrupted.sum().item()
            val_total   += 2 * batch_size

    val_acc = val_correct / val_total
    scheduler.step(val_acc)

    # Save best model
    is_best = val_acc > best_val_acc
    if is_best:
        best_val_acc = val_acc
        if best_model_path and os.path.exists(best_model_path):
            os.remove(best_model_path)
        best_model_path = os.path.join(
            SAVE_DIR, f'best_epoch{epoch:03d}_acc{val_acc:.3f}.pth'
        )
        torch.save(model.state_dict(), best_model_path)

    marker = ' <-- best' if is_best else ''
    print(f"{epoch:>6} {train_loss:>12.4f} {train_acc:>10.3f} "
          f"{val_acc:>10.3f}{marker}")

print(f"\nTraining complete.")
print(f"Best val accuracy: {best_val_acc:.3f}")
print(f"Best model saved: {best_model_path}")

# Final assessment
print("\nAssessment:")
if best_val_acc > 0.85:
    print("STRONG: model learned geometric consistency well.")
    print("Ready for Phase 2: counterfactual simulation.")
elif best_val_acc > 0.70:
    print("MODERATE: model is learning but not saturating.")
    print("Try more epochs or larger latent size.")
elif best_val_acc > 0.55:
    print("WEAK: model is barely above chance.")
    print("Check corruption magnitude and model architecture.")
else:
    print("FAILED: model is at chance — not learning.")
    print("Debug gradient flow and loss values.")
