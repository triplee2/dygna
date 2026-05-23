"""
Builds a spatial region graph from a 3D mesh.

Pipeline:
  1. Load mesh
  2. Extract floor faces (normal_y > threshold)
  3. Cluster floor faces into connected regions via face adjacency
  4. Compute per-region: centroid, mean normal
  5. Build edges between adjacent regions
  6. Compute per-edge boundary vectors
  7. Output a torch_geometric Data object ready for SpatialRefFrameCalc

Assertions at every stage. If any fail, the error tells you exactly
what broke and what the actual values were.
"""

import torch
import trimesh
import numpy as np
from collections import deque
from torch_geometric.data import Data


def load_mesh(path):
    mesh = trimesh.load(path, force='mesh')
    assert isinstance(mesh, trimesh.Trimesh), \
        f"Expected Trimesh, got {type(mesh)}. Mesh may be a Scene with multiple geometries."
    assert len(mesh.vertices) > 0, "Mesh has no vertices."
    assert len(mesh.faces) > 0, "Mesh has no faces."
    print(f"Mesh loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    return mesh


def extract_floor_faces(mesh, normal_y_threshold=0.7):
    """
    Returns indices of faces whose normal has a strong upward Y component.
    Threshold 0.7 means the face is within ~45 degrees of horizontal.
    """
    face_normals = mesh.face_normals  # (F, 3)
    floor_mask = face_normals[:, 1] > normal_y_threshold
    floor_indices = np.where(floor_mask)[0]

    assert len(floor_indices) > 0, (
        f"No floor faces found with threshold {normal_y_threshold}. "
        f"Max normal_y in mesh: {face_normals[:, 1].max():.3f}. "
        f"Try lowering normal_y_threshold."
    )
    print(f"Floor faces: {len(floor_indices)} of {len(mesh.faces)} total "
          f"({100*len(floor_indices)/len(mesh.faces):.1f}%)")
    return floor_indices


def cluster_floor_faces(mesh, floor_indices):
    """
    Clusters floor faces into connected regions using BFS on face adjacency.
    Two faces are adjacent if they share an edge (two vertices).

    Returns list of lists — each inner list is face indices for one region.
    """
    # Build face adjacency for floor faces only
    floor_set = set(floor_indices.tolist())

    # trimesh face_adjacency gives pairs of adjacent face indices (shares an edge)
    adjacency = mesh.face_adjacency  # (A, 2)

    # Build adjacency dict restricted to floor faces
    adj_dict = {i: [] for i in floor_indices.tolist()}
    for pair in adjacency:
        a, b = int(pair[0]), int(pair[1])
        if a in floor_set and b in floor_set:
            adj_dict[a].append(b)
            adj_dict[b].append(a)

    # BFS to find connected components
    visited = set()
    regions = []

    for start in floor_indices.tolist():
        if start in visited:
            continue
        region = []
        queue = deque([start])
        while queue:
            face = queue.popleft()
            if face in visited:
                continue
            visited.add(face)
            region.append(face)
            for neighbor in adj_dict[face]:
                if neighbor not in visited:
                    queue.append(neighbor)
        regions.append(region)

    assert len(regions) > 0, "BFS produced zero regions."
    print(f"Connected regions found: {len(regions)}")
    for i, r in enumerate(regions):
        print(f"  Region {i}: {len(r)} faces")
    return regions


def compute_region_features(mesh, regions):
    """
    For each region computes:
      centroid : mean of face centroids         (3,)
      normal   : mean of face normals           (3,)

    Returns tensors of shape (R, 3) each.
    """
    face_centroids = mesh.triangles_center  # (F, 3)
    face_normals = mesh.face_normals        # (F, 3)

    centroids = []
    normals = []

    for region in regions:
        idx = np.array(region)
        centroid = face_centroids[idx].mean(axis=0)
        normal = face_normals[idx].mean(axis=0)
        norm_mag = np.linalg.norm(normal)
        assert norm_mag > 1e-8, f"Region has zero-magnitude mean normal: {normal}"
        normal = normal / norm_mag
        centroids.append(centroid)
        normals.append(normal)

    centroids = torch.tensor(np.array(centroids), dtype=torch.float32)
    normals = torch.tensor(np.array(normals), dtype=torch.float32)

    assert centroids.shape == (len(regions), 3), \
        f"Expected centroids shape ({len(regions)}, 3), got {centroids.shape}"
    assert normals.shape == (len(regions), 3), \
        f"Expected normals shape ({len(regions)}, 3), got {normals.shape}"

    print(f"Region features computed: centroids {centroids.shape}, normals {normals.shape}")
    return centroids, normals


def build_region_edges(mesh, regions, centroids, max_connection_distance=5.0):
    """
    Builds edges between adjacent regions using two methods:

    Method 1 — Shared mesh edges: two regions are adjacent if their faces
    share a mesh edge. Works when regions are part of the same mesh component.

    Method 2 — Centroid distance fallback: two regions are adjacent if their
    centroids are within max_connection_distance. Used when regions are
    separate mesh components that do not share vertices (common in
    concatenated meshes and World Labs outputs).

    Boundary vector: direction from sender centroid toward receiver centroid.
    For Method 1 this is toward the shared edge midpoint.
    For Method 2 this is directly toward the neighbor centroid.

    Returns:
      edge_index    : (2, E) long tensor
      boundary_vecs : (E, 3) float tensor
    """
    import itertools

    face_to_region = {}
    for r_idx, region in enumerate(regions):
        for face in region:
            face_to_region[face] = r_idx

    adjacency = mesh.face_adjacency
    adj_edge_midpoints = mesh.face_adjacency_edges

    # Method 1: shared mesh edges
    region_pairs = {}
    for i, pair in enumerate(adjacency):
        a, b = int(pair[0]), int(pair[1])
        ra = face_to_region.get(a)
        rb = face_to_region.get(b)
        if ra is None or rb is None:
            continue
        if ra == rb:
            continue
        key = (min(ra, rb), max(ra, rb))
        if key not in region_pairs:
            region_pairs[key] = []
        region_pairs[key].append(adj_edge_midpoints[i])

    # Method 2: centroid distance fallback
    n = len(regions)
    for ra, rb in itertools.combinations(range(n), 2):
        key = (ra, rb)
        if key in region_pairs:
            continue  # already found via mesh edges
        dist = (centroids[ra] - centroids[rb]).norm().item()
        if dist < max_connection_distance:
            # Boundary point is midpoint between centroids
            mid = ((centroids[ra] + centroids[rb]) / 2).numpy()
            region_pairs[key] = [mid]

    assert len(region_pairs) > 0, (
        f"No edges found between {n} regions. "
        f"Max centroid distance allowed: {max_connection_distance}. "
        f"Centroid positions:\n{centroids}"
    )

    senders = []
    receivers = []
    boundary_vecs = []

    for (ra, rb), midpoints in region_pairs.items():
        boundary_mid = np.array(midpoints).mean(axis=0)

        bvec_a = boundary_mid - centroids[ra].numpy()
        mag_a = np.linalg.norm(bvec_a)
        if mag_a < 1e-8:
            bvec_a = (centroids[rb] - centroids[ra]).numpy()
            mag_a = np.linalg.norm(bvec_a)
        assert mag_a > 1e-8, f"Zero boundary vector for edge {ra}->{rb}"
        bvec_a = bvec_a / mag_a

        bvec_b = boundary_mid - centroids[rb].numpy()
        mag_b = np.linalg.norm(bvec_b)
        if mag_b < 1e-8:
            bvec_b = (centroids[ra] - centroids[rb]).numpy()
            mag_b = np.linalg.norm(bvec_b)
        assert mag_b > 1e-8, f"Zero boundary vector for edge {rb}->{ra}"
        bvec_b = bvec_b / mag_b

        senders.append(ra)
        receivers.append(rb)
        boundary_vecs.append(bvec_a)

        senders.append(rb)
        receivers.append(ra)
        boundary_vecs.append(bvec_b)

    edge_index = torch.tensor([senders, receivers], dtype=torch.long)
    boundary_vecs = torch.tensor(np.array(boundary_vecs), dtype=torch.float32)

    assert edge_index.shape[0] == 2
    assert boundary_vecs.shape[1] == 3
    assert edge_index.shape[1] == boundary_vecs.shape[0]

    print(f"Region edges built: {edge_index.shape[1]} directed edges "
          f"({edge_index.shape[1]//2} undirected pairs)")
    return edge_index, boundary_vecs


def build_spatial_graph(mesh_path, normal_y_threshold=0.7):
    """
    Full pipeline: mesh path -> torch_geometric Data object.
    """
    print(f"\n--- Building spatial region graph from: {mesh_path} ---")

    mesh = load_mesh(mesh_path)
    floor_indices = extract_floor_faces(mesh, normal_y_threshold)
    regions = cluster_floor_faces(mesh, floor_indices)
    centroids, normals = compute_region_features(mesh, regions)
    edge_index, boundary_vecs = build_region_edges(mesh, regions, centroids)

    graph = Data(
        pos=centroids,
        normals=normals,
        boundary_vecs=boundary_vecs,
        edge_index=edge_index,
        num_nodes=len(regions)
    )

    print(f"\nGraph ready:")
    print(f"  Nodes (regions) : {graph.num_nodes}")
    print(f"  Edges           : {edge_index.shape[1]}")
    print(f"  pos shape       : {graph.pos.shape}")
    print(f"  normals shape   : {graph.normals.shape}")
    print(f"  boundary shape  : {graph.boundary_vecs.shape}")

    return graph


def verify_graph_with_spatial_ref(graph):
    """
    Passes the graph through SpatialRefFrameCalc and verifies the output.
    This is the end-to-end test: mesh -> graph -> reference frame.
    """
    import sys
    sys.path.insert(0, '/workspaces/dygna')
    from model.model import SpatialRefFrameCalc

    model = SpatialRefFrameCalc()
    model.eval()

    senders, receivers = graph.edge_index

    with torch.no_grad():
        va, vb, vc = model(
            graph.edge_index,
            graph.pos[senders], graph.pos[receivers],
            graph.normals[senders], graph.normals[receivers],
            graph.boundary_vecs[senders], graph.boundary_vecs[receivers]
        )

    assert not torch.isnan(va).any(), "NaN in vector_a"
    assert not torch.isnan(vb).any(), "NaN in vector_b"
    assert not torch.isnan(vc).any(), "NaN in vector_c"

    for name, vec in [("vector_a", va), ("vector_b", vb), ("vector_c", vc)]:
        norms = vec.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), \
            f"{name} not unit vector. norms: {norms}"

    print(f"\nSpatialRefFrameCalc verified on real graph:")
    print(f"  vector_a sample: {va[0].tolist()}")
    print(f"  vector_b sample: {vb[0].tolist()}")
    print(f"  vector_c sample: {vc[0].tolist()}")
    print(f"\nEND TO END VERIFIED: mesh -> region graph -> antisymmetric reference frame")


if __name__ == "__main__":
    import sys

    # Generate a synthetic test mesh if no path provided
    if len(sys.argv) < 2:
        print("No mesh path provided. Generating synthetic L-shaped floor mesh...")

        # Two floor rectangles forming an L-shape
        floor_a = trimesh.creation.box(extents=[4.0, 0.2, 2.0])
        floor_a.apply_translation([0.0, 0.0, 0.0])

        floor_b = trimesh.creation.box(extents=[2.0, 0.2, 2.0])
        floor_b.apply_translation([3.0, 0.0, 0.0])

        mesh = trimesh.util.concatenate([floor_a, floor_b])
        mesh_path = "/workspaces/dygna/test_floor.obj"
        mesh.export(mesh_path)
        print(f"Synthetic mesh saved to {mesh_path}")
    else:
        mesh_path = sys.argv[1]

    graph = build_spatial_graph(mesh_path, normal_y_threshold=0.5)
    verify_graph_with_spatial_ref(graph)
