"""
Patches build_region_edges in build_region_graph.py to use
centroid distance for adjacency instead of mesh edge sharing.

This handles the common case where floor regions are separate
mesh components that do not share vertices.
"""

old_func = '''def build_region_edges(mesh, regions, centroids):
    """
    Builds edges between adjacent regions.
    Two regions are adjacent if they share at least one pair of adjacent faces
    (one face from each region that share a mesh edge).

    For each edge computes boundary vector:
      direction from sender centroid toward the shared boundary midpoint.

    Returns:
      edge_index    : (2, E) long tensor
      boundary_vecs : (E, 3) float tensor
    """
    # Map face index to region index
    face_to_region = {}
    for r_idx, region in enumerate(regions):
        for face in region:
            face_to_region[face] = r_idx

    adjacency = mesh.face_adjacency          # (A, 2)
    adj_centroids = mesh.face_adjacency_edges  # (A, 3) midpoint of shared edge

    # Find region pairs that are adjacent
    region_pairs = {}  # (r_a, r_b) -> list of boundary midpoints

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
        region_pairs[key].append(adj_centroids[i])

    assert len(region_pairs) > 0, (
        "No edges found between regions. "
        "All floor faces may be in one region, or no regions are adjacent."
    )

    senders = []
    receivers = []
    boundary_vecs = []

    for (ra, rb), midpoints in region_pairs.items():
        boundary_mid = np.array(midpoints).mean(axis=0)

        # Edge ra -> rb
        bvec_a = boundary_mid - centroids[ra].numpy()
        mag_a = np.linalg.norm(bvec_a)
        assert mag_a > 1e-8, f"Zero boundary vector for edge {ra}->{rb}"
        bvec_a = bvec_a / mag_a

        # Edge rb -> ra (reverse)
        bvec_b = boundary_mid - centroids[rb].numpy()
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

    assert edge_index.shape[0] == 2, f"edge_index must have shape (2, E), got {edge_index.shape}"
    assert boundary_vecs.shape[1] == 3, f"boundary_vecs must have shape (E, 3), got {boundary_vecs.shape}"
    assert edge_index.shape[1] == boundary_vecs.shape[0], \\
        f"edge_index and boundary_vecs must have same number of edges. " \\
        f"Got {edge_index.shape[1]} and {boundary_vecs.shape[0]}"

    print(f"Region edges built: {edge_index.shape[1]} directed edges "
          f"({edge_index.shape[1]//2} undirected pairs)")
    return edge_index, boundary_vecs'''

new_func = '''def build_region_edges(mesh, regions, centroids, max_connection_distance=5.0):
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
        f"Centroid positions:\\n{centroids}"
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
    return edge_index, boundary_vecs'''

with open('/workspaces/dygna/build_region_graph.py', 'r') as f:
    content = f.read()

assert old_func in content, "Could not find the old function — paste the error exactly as shown"
content = content.replace(old_func, new_func)

with open('/workspaces/dygna/build_region_graph.py', 'w') as f:
    f.write(content)

print("Patch applied.")
