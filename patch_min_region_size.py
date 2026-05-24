old = '''    assert len(regions) > 0, "BFS produced zero regions."
    print(f"Connected regions found: {len(regions)}")
    for i, r in enumerate(regions):
        print(f"  Region {i}: {len(r)} faces")
    return regions'''

new = '''    assert len(regions) > 0, "BFS produced zero regions."

    # Filter out noise regions — tiny disconnected fragments from furniture,
    # mesh artifacts, and surface details that are not real spatial regions.
    # min_faces=20 removes single triangles and small clusters while keeping
    # genuine room-scale floor areas.
    all_count = len(regions)
    regions = [r for r in regions if len(r) >= min_faces]

    assert len(regions) > 0, (
        f"All {all_count} regions were smaller than min_faces={min_faces}. "
        f"Try lowering min_faces."
    )

    print(f"Connected regions found: {all_count} total, "
          f"{len(regions)} after min_faces={min_faces} filter")
    for i, r in enumerate(regions):
        print(f"  Region {i}: {len(r)} faces")
    return regions'''

old_sig = '''def cluster_floor_faces(mesh, floor_indices):'''
new_sig = '''def cluster_floor_faces(mesh, floor_indices, min_faces=20):'''

with open('/workspaces/dygna/build_region_graph.py', 'r') as f:
    content = f.read()

assert old_sig in content, "Could not find cluster_floor_faces signature"
assert old in content, "Could not find assert block in cluster_floor_faces"

content = content.replace(old_sig, new_sig)
content = content.replace(old, new)

with open('/workspaces/dygna/build_region_graph.py', 'w') as f:
    f.write(content)

print("Patch applied.")
