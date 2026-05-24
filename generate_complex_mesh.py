"""
Generates a complex synthetic indoor environment for stress testing
the region graph builder before we run it on real World Labs meshes.

Layout (top view):
  [Room A] -- [Corridor] -- [Room B]
                  |
              [Room C]

Four floor regions at slightly different heights to test normal threshold.
Walls included so the mesh has non-floor geometry to filter out.
"""
import trimesh
import numpy as np

def make_floor(x, z, width, depth, height=0.0):
    floor = trimesh.creation.box(extents=[width, 0.1, depth])
    floor.apply_translation([x, height, z])
    return floor

def make_wall(x, z, width, height_wall, depth):
    wall = trimesh.creation.box(extents=[width, height_wall, depth])
    wall.apply_translation([x, height_wall/2, z])
    return wall

# Four floor regions
room_a    = make_floor(x=0.0,  z=0.0,  width=4.0, depth=4.0, height=0.0)
corridor  = make_floor(x=5.5,  z=0.0,  width=3.0, depth=2.0, height=0.0)
room_b    = make_floor(x=10.0, z=0.0,  width=4.0, depth=4.0, height=0.0)
room_c    = make_floor(x=5.5,  z=4.0,  width=3.0, depth=3.0, height=0.0)

# Some walls (non-floor geometry — should be filtered out)
wall_1 = make_wall(x=0.0,  z=2.5,  width=4.0, height_wall=2.5, depth=0.2)
wall_2 = make_wall(x=10.0, z=2.5,  width=4.0, height_wall=2.5, depth=0.2)
wall_3 = make_wall(x=3.0,  z=0.0,  width=0.2, height_wall=2.5, depth=4.0)

mesh = trimesh.util.concatenate([
    room_a, corridor, room_b, room_c,
    wall_1, wall_2, wall_3
])

path = '/workspaces/dygna/complex_floor.obj'
mesh.export(path)

print(f"Complex mesh saved: {path}")
print(f"Vertices: {len(mesh.vertices)}")
print(f"Faces: {len(mesh.faces)}")
print(f"Expected floor regions: 4 (room_a, corridor, room_b, room_c)")
print(f"Expected non-floor faces: walls should be filtered out")
