old = '''    # Filter out noise regions — tiny disconnected fragments from furniture,
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

new = '''    all_count = len(regions)

    # Adaptive min_faces: if caller passed min_faces=1 (default for synthetic
    # meshes), use 1. Otherwise use the passed value.
    # For real meshes (World Labs GLB), pass min_faces=20 explicitly.
    # For synthetic meshes, min_faces=1 keeps all regions.
    if min_faces > 1:
        regions = [r for r in regions if len(r) >= min_faces]
    else:
        # For synthetic meshes: filter only truly isolated single faces
        # that have no adjacent floor faces (pure noise triangles).
        # Keep anything with >= 1 face since synthetic rooms have 2 faces each.
        regions = [r for r in regions if len(r) >= 1]

    assert len(regions) > 0, (
        f"All {all_count} regions were smaller than min_faces={min_faces}. "
        f"Try lowering min_faces."
    )

    print(f"Connected regions found: {all_count} total, "
          f"{len(regions)} after min_faces={min_faces} filter")
    for i, r in enumerate(regions):
        print(f"  Region {i}: {len(r)} faces")
    return regions'''

with open('/workspaces/dygna/build_region_graph.py', 'r') as f:
    content = f.read()

assert old in content, "Could not find target block"
content = content.replace(old, new)

# Also update build_spatial_graph to accept and pass min_faces
old_sig = '''def build_spatial_graph(mesh_path, normal_y_threshold=0.7):'''
new_sig = '''def build_spatial_graph(mesh_path, normal_y_threshold=0.7, min_faces=1):'''

old_call = '''    regions = cluster_floor_faces(mesh, floor_indices)'''
new_call = '''    regions = cluster_floor_faces(mesh, floor_indices, min_faces=min_faces)'''

assert old_sig in content, "Could not find build_spatial_graph signature"
assert old_call in content, "Could not find cluster_floor_faces call"

content = content.replace(old_sig, new_sig)
content = content.replace(old_call, new_call)

with open('/workspaces/dygna/build_region_graph.py', 'w') as f:
    f.write(content)

print("Patch applied.")
