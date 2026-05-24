old_func = '''def extract_floor_faces(mesh, normal_y_threshold=0.7):
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
    return floor_indices'''

new_func = '''def extract_floor_faces(mesh, normal_y_threshold=0.5, height_tolerance=0.5):
    """
    Returns indices of faces that are genuinely floor faces.

    Two filters applied in sequence:
      1. Normal filter: face must point upward (normal_y > threshold)
      2. Height filter: face centroid must be within height_tolerance
         of the lowest upward-facing face. Removes wall tops and
         ceiling faces that also point upward.

    height_tolerance=0.5 handles slightly uneven floors while
    rejecting wall tops which are typically 2+ meters above floor level.
    """
    face_normals = mesh.face_normals
    face_centroids = mesh.triangles_center

    # Filter 1: upward facing
    upward_mask = face_normals[:, 1] > normal_y_threshold
    up_indices = np.where(upward_mask)[0]

    assert len(up_indices) > 0, (
        f"No upward-facing faces found with threshold {normal_y_threshold}. "
        f"Max normal_y in mesh: {face_normals[:, 1].max():.3f}. "
        f"Try lowering normal_y_threshold."
    )

    # Filter 2: height — keep only faces near the lowest floor level
    up_centroids = face_centroids[up_indices]
    min_y = up_centroids[:, 1].min()
    height_mask = up_centroids[:, 1] <= min_y + height_tolerance
    floor_indices = up_indices[height_mask]

    assert len(floor_indices) > 0, (
        f"Height filter removed all faces. "
        f"Min Y: {min_y:.3f}, tolerance: {height_tolerance}. "
        f"Try increasing height_tolerance."
    )

    print(f"Floor faces: {len(floor_indices)} of {len(mesh.faces)} total "
          f"({100*len(floor_indices)/len(mesh.faces):.1f}%) "
          f"[upward filter: {len(up_indices)}, height filter: {len(floor_indices)}]")
    return floor_indices'''

with open('/workspaces/dygna/build_region_graph.py', 'r') as f:
    content = f.read()

assert old_func in content, "Could not find old extract_floor_faces — check for whitespace differences"
content = content.replace(old_func, new_func)

with open('/workspaces/dygna/build_region_graph.py', 'w') as f:
    f.write(content)

print("Patch applied.")
