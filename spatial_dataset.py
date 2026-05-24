"""
SpatialConsistencyDataset

For each environment generates pairs of graphs:
  - consistent graph: built directly from mesh geometry
  - corrupted graph: one region's boundary vectors rotated by 90-180 degrees

The model must classify which graph in each pair is corrupted.

This is Phase 1 of spatial reasoning training:
  teach the model what geometric consistency means
  before teaching it to reason about spatial changes.

Labels:
  0 = consistent (real geometry)
  1 = corrupted (injected contradiction)
"""

import os
import sys
import torch
import numpy as np
import random
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch

sys.path.insert(0, '/workspaces/dygna')
from build_region_graph import build_spatial_graph


def rotate_vectors_random(vectors, min_angle=90, max_angle=180):
    """
    Rotates a set of 3D vectors by a random angle around a random axis.
    Used to inject geometric contradictions into boundary vectors.

    min_angle=90 ensures the corruption is large enough to be detectable.
    max_angle=180 is the maximum possible rotation.

    Returns rotated vectors of same shape.
    """
    angle = np.radians(random.uniform(min_angle, max_angle))

    # Random unit axis
    axis = np.random.randn(3)
    axis = axis / np.linalg.norm(axis)

    # Rodrigues rotation formula
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    ax, ay, az = axis

    R = np.array([
        [cos_a + ax*ax*(1-cos_a),    ax*ay*(1-cos_a) - az*sin_a, ax*az*(1-cos_a) + ay*sin_a],
        [ay*ax*(1-cos_a) + az*sin_a, cos_a + ay*ay*(1-cos_a),    ay*az*(1-cos_a) - ax*sin_a],
        [az*ax*(1-cos_a) - ay*sin_a, az*ay*(1-cos_a) + ax*sin_a, cos_a + az*az*(1-cos_a)   ]
    ], dtype=np.float32)

    rotated = (R @ vectors.T).T
    return torch.tensor(rotated, dtype=torch.float32)


def corrupt_graph(graph):
    """
    Injects a geometric contradiction into a spatial graph.

    Selects one random region (node) and rotates all boundary vectors
    on edges involving that region by 90-180 degrees.

    The corruption is detectable because the rotated boundary vectors
    are inconsistent with the actual geometric positions of the regions.

    Returns a new corrupted graph. Original graph is not modified.
    """
    num_nodes = graph.num_nodes
    assert num_nodes >= 2, "Graph must have at least 2 nodes to corrupt"

    # Select a random region to corrupt
    corrupt_node = random.randint(0, num_nodes - 1)

    senders, receivers = graph.edge_index
    boundary_vecs = graph.boundary_vecs.clone()

    # Find all edges involving the corrupted node
    corrupt_mask = (senders == corrupt_node) | (receivers == corrupt_node)
    corrupt_indices = corrupt_mask.nonzero(as_tuple=True)[0]

    assert len(corrupt_indices) > 0, (
        f"Node {corrupt_node} has no edges — cannot corrupt. "
        f"Graph has {num_nodes} nodes and {graph.edge_index.shape[1]} edges."
    )

    # Rotate boundary vectors for corrupted edges
    vecs_to_corrupt = boundary_vecs[corrupt_indices].numpy()
    rotated = rotate_vectors_random(vecs_to_corrupt)
    boundary_vecs[corrupt_indices] = rotated

    corrupted = Data(
        pos=graph.pos.clone(),
        normals=graph.normals.clone(),
        boundary_vecs=boundary_vecs,
        edge_index=graph.edge_index.clone(),
        num_nodes=num_nodes
    )

    return corrupted


class SpatialConsistencyDataset(Dataset):
    """
    Dataset of (consistent_graph, corrupted_graph) pairs.

    Each environment produces one pair per call.
    With 50 environments and augmentation, we generate
    multiple corruptions per environment to increase dataset size.

    Args:
        env_dir: directory containing .obj environment files
        augmentations: number of corrupted versions per environment
        normal_y_threshold: floor extraction threshold
        min_faces: minimum region size filter
    """
    def __init__(self, env_dir, augmentations=4,
                 normal_y_threshold=0.5, min_faces=1):
        self.env_dir = env_dir
        self.augmentations = augmentations
        self.normal_y_threshold = normal_y_threshold
        self.min_faces = min_faces

        # Load all valid graphs upfront
        print(f"Loading graphs from {env_dir}...")
        self.graphs = []
        files = sorted([f for f in os.listdir(env_dir) if f.endswith('.obj')])

        failed = []
        for fname in files:
            path = os.path.join(env_dir, fname)
            try:
                graph = build_spatial_graph(
                    path,
                    normal_y_threshold=normal_y_threshold,
                    min_faces=min_faces
                )
                # Only keep graphs with at least 2 nodes and 2 edges
                if graph.num_nodes >= 2 and graph.edge_index.shape[1] >= 2:
                    self.graphs.append(graph)
            except Exception as e:
                failed.append((fname, str(e)))

        assert len(self.graphs) > 0, (
            f"No valid graphs loaded from {env_dir}. "
            f"Failed: {failed[:3]}"
        )

        print(f"Loaded {len(self.graphs)} valid graphs "
              f"({len(failed)} failed)")
        print(f"Dataset size: {len(self)} samples "
              f"({augmentations} augmentations per graph)")

    def __len__(self):
        return len(self.graphs) * self.augmentations

    def __getitem__(self, idx):
        graph_idx = idx % len(self.graphs)
        graph = self.graphs[graph_idx]

        # Generate one corrupted version
        corrupted = corrupt_graph(graph)

        # Return as a dict — we handle batching manually
        # label 0 = consistent, label 1 = corrupted
        return {
            'consistent': graph,
            'corrupted': corrupted,
        }


def collate_consistency_batch(batch):
    """
    Custom collate for consistency classification.
    Returns two batched graphs and their labels.
    """
    consistent_list = [item['consistent'] for item in batch]
    corrupted_list = [item['corrupted'] for item in batch]

    consistent_batch = Batch.from_data_list(consistent_list)
    corrupted_batch = Batch.from_data_list(corrupted_list)

    return consistent_batch, corrupted_batch


if __name__ == "__main__":
    print("Testing SpatialConsistencyDataset...")
    print()

    dataset = SpatialConsistencyDataset(
        env_dir='/workspaces/dygna/environments',
        augmentations=4
    )

    print(f"\nDataset length: {len(dataset)}")

    # Test first three samples
    for i in range(3):
        sample = dataset[i]
        g = sample['consistent']
        c = sample['corrupted']

        print(f"\nSample {i}:")
        print(f"  Consistent graph: {g.num_nodes} nodes, "
              f"{g.edge_index.shape[1]} edges")
        print(f"  Corrupted graph:  {c.num_nodes} nodes, "
              f"{c.edge_index.shape[1]} edges")

        # Verify corruption actually changed something
        boundary_diff = (g.boundary_vecs - c.boundary_vecs).abs().max().item()
        print(f"  Max boundary vector change: {boundary_diff:.4f} "
              f"(should be > 0)")
        assert boundary_diff > 0, "Corruption did not change boundary vectors"

        # Verify positions and normals are unchanged
        pos_diff = (g.pos - c.pos).abs().max().item()
        norm_diff = (g.normals - c.normals).abs().max().item()
        assert pos_diff == 0, f"Corruption changed positions: {pos_diff}"
        assert norm_diff == 0, f"Corruption changed normals: {norm_diff}"
        print(f"  Positions unchanged: YES")
        print(f"  Normals unchanged: YES")

    # Test collate
    from torch.utils.data import DataLoader
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_consistency_batch
    )

    consistent_batch, corrupted_batch = next(iter(loader))
    print(f"\nBatch test:")
    print(f"  Consistent batch nodes: {consistent_batch.num_nodes}")
    print(f"  Corrupted batch nodes:  {corrupted_batch.num_nodes}")
    print(f"  Consistent batch edges: {consistent_batch.edge_index.shape[1]}")
    print(f"  Corrupted batch edges:  {corrupted_batch.edge_index.shape[1]}")

    print(f"\nSpatialConsistencyDataset verified.")
    print(f"Ready to build the training loop.")
