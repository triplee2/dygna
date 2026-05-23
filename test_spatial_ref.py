"""
Tests SpatialRefFrameCalc with three spatial regions forming an L-shape.

Three tests:
  1. No NaN in output
  2. All output vectors are unit vectors (norm = 1 within 1e-5)
  3. Antisymmetry: reversing edge 0->1 to 1->0 negates vector_a

This tests the architectural property before any training data exists.
If all three pass, SpatialRefFrameCalc is mathematically correct.
If any fail, paste the full assertion error before touching anything else.
"""
import torch
import sys
sys.path.insert(0, '/workspaces/dygna')
from model.model import SpatialRefFrameCalc

def test_spatial_ref_frame():
    model = SpatialRefFrameCalc()
    model.eval()

    # Three spatial regions — simple L-shape
    # Region 0: hallway at origin
    # Region 1: room A to the right
    # Region 2: room B above
    pos = torch.tensor([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
    ])

    # All floor regions — normals point up
    normals = torch.tensor([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ])

    # Boundary vectors: direction from region centroid toward shared edge
    boundaries = torch.tensor([
        [1.0, 0.0, 0.0],   # region 0 boundary toward region 1
        [-1.0, 0.0, 0.0],  # region 1 boundary toward region 0
        [0.0, -1.0, 0.0],  # region 2 boundary toward region 0
    ])

    # Two edges: 0->1 and 0->2
    edge_index = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)
    senders, receivers = edge_index

    with torch.no_grad():
        va, vb, vc = model(
            edge_index,
            pos[senders], pos[receivers],
            normals[senders], normals[receivers],
            boundaries[senders], boundaries[receivers]
        )

    # TEST 1: No NaN
    assert not torch.isnan(va).any(), f"FAIL NaN in vector_a:\n{va}"
    assert not torch.isnan(vb).any(), f"FAIL NaN in vector_b:\n{vb}"
    assert not torch.isnan(vc).any(), f"FAIL NaN in vector_c:\n{vc}"
    print("PASS test 1: no NaN in any output vector")

    # TEST 2: Unit vectors
    for name, vec in [("vector_a", va), ("vector_b", vb), ("vector_c", vc)]:
        norms = vec.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
            f"FAIL {name} is not unit vector\nnorms: {norms}"
    print("PASS test 2: all vectors are unit vectors")
    print(f"  vector_a norms: {va.norm(dim=1).tolist()}")
    print(f"  vector_b norms: {vb.norm(dim=1).tolist()}")
    print(f"  vector_c norms: {vc.norm(dim=1).tolist()}")

    # TEST 3: Antisymmetry on edge 0->1
    edge_rev = torch.tensor([[1], [0]], dtype=torch.long)
    senders_rev, receivers_rev = edge_rev

    with torch.no_grad():
        va_rev, vb_rev, vc_rev = model(
            edge_rev,
            pos[senders_rev], pos[receivers_rev],
            normals[senders_rev], normals[receivers_rev],
            boundaries[senders_rev], boundaries[receivers_rev]
        )

    va_fwd = va[0]
    va_bwd = va_rev[0]

    assert torch.allclose(va_fwd, -va_bwd, atol=1e-5), \
        f"FAIL antisymmetry broken\n  edge 0->1: {va_fwd.tolist()}\n  edge 1->0: {va_bwd.tolist()}\n  sum (should be zero): {(va_fwd + va_bwd).tolist()}"

    print("PASS test 3: antisymmetry holds — vector_a flips when edge is reversed")
    print(f"  edge 0->1 vector_a: {va_fwd.tolist()}")
    print(f"  edge 1->0 vector_a: {va_bwd.tolist()}")

    print("\nALL THREE TESTS PASSED")
    print("SpatialRefFrameCalc is mathematically correct.")
    print("Next step: build the spatial region graph from a mesh.")

if __name__ == "__main__":
    test_spatial_ref_frame()
