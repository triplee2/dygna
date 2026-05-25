with open('/workspaces/dygna/model/model.py', 'r') as f:
    content = f.read()

new_class = '''

class SpatialRefFrameCalc(nn.Module):
    """
    Computes antisymmetric local reference frame for spatial region graphs.

    Adapts RefFrameCalc for spatial reasoning by replacing velocity inputs
    with surface normals and boundary vectors. Gram-Schmidt block is
    identical to RefFrameCalc — only inputs change.

    Inputs per edge:
      senders_pos / receivers_pos        : region centroids       (E, 3)
      senders_normal / receivers_normal  : mean surface normal    (E, 3)
      senders_boundary / receivers_boundary : centroid-to-boundary vector (E, 3)

    Antisymmetry guarantee:
      Reversing an edge (swap sender/receiver) negates vector_a.
      This is the same guarantee RefFrameCalc provides for physics.
    """
    def __init__(self):
        super(SpatialRefFrameCalc, self).__init__()
        self.eps = 1e-8

    def forward(self, edge_index,
                senders_pos, receivers_pos,
                senders_normal, receivers_normal,
                senders_boundary, receivers_boundary):

        # vector_a: normalized edge direction (antisymmetric by construction)
        rel_pos = receivers_pos - senders_pos
        dist = rel_pos.norm(dim=1, keepdim=True).clamp(min=self.eps)
        vector_a = rel_pos / dist

        def normalize(tensor):
            return tensor / tensor.norm(dim=1, keepdim=True).clamp(min=self.eps)

        # Normal-based terms — replace vel and prev_vel
        diff_normal = receivers_normal - senders_normal   # antisymmetric
        sum_normal  = senders_normal  + receivers_normal  # symmetric

        # Boundary-based terms — replace omega
        diff_boundary = receivers_boundary - senders_boundary  # antisymmetric
        sum_boundary  = senders_boundary  + receivers_boundary  # symmetric

        # Four basis contributors — same structure as RefFrameCalc
        b_i   = normalize(torch.cross(diff_normal,   vector_a, dim=1))
        b_ii  = normalize(sum_normal)
        b_iii = normalize(torch.cross(diff_boundary, vector_a, dim=1))
        b_iv  = normalize(sum_boundary)

        b = b_i + b_ii + b_iii + b_iv

        # Gram-Schmidt — identical to RefFrameCalc, do not modify
        b_prl_dot = (b * vector_a).sum(dim=1, keepdim=True)
        b_prl     = b_prl_dot * vector_a
        b_prp     = b - b_prl

        vector_b = normalize(torch.cross(b_prp, vector_a, dim=1))
        vector_c = normalize(torch.cross(b_prl, vector_b, dim=1))

        return vector_a, vector_b, vector_c

'''

# Insert after RefFrameCalc, before NodeEncoder
insert_marker = 'class NodeEncoder(nn.Module):'
assert insert_marker in content, "Could not find NodeEncoder class — model.py structure may have changed"

content = content.replace(insert_marker, new_class + insert_marker)

with open('/workspaces/dygna/model/model.py', 'w') as f:
    f.write(content)

print("Done. SpatialRefFrameCalc inserted before NodeEncoder.")
