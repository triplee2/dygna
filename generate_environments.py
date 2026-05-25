"""
Generates a dataset of synthetic indoor environments for spatial reasoning training.

Each environment is a random arrangement of rooms connected by corridors.
Saved as OBJ files in /workspaces/dygna/environments/

Layout types generated:
  - Linear: Room - Corridor - Room - Corridor - Room
  - L-shape: Rooms arranged in an L
  - T-shape: Three rooms meeting at a junction
  - Grid: 2x2 room arrangement
  - Random: 3-6 rooms with random positions and connections

Target: 50 environments minimum for training validation.
"""

import trimesh
import numpy as np
import os
import json
import random

OUTPUT_DIR = "/workspaces/dygna/environments"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def make_floor(cx, cz, width, depth, y=0.0):
    """Create a floor tile at center (cx, y, cz) with given width and depth."""
    floor = trimesh.creation.box(extents=[width, 0.1, depth])
    floor.apply_translation([cx, y + 0.05, cz])
    return floor

def make_wall(cx, cy, cz, width, height, depth):
    """Create a wall box."""
    wall = trimesh.creation.box(extents=[width, height, depth])
    wall.apply_translation([cx, cy, cz])
    return wall

def add_walls_around_floor(cx, cz, width, depth, wall_height=2.5):
    """Add four walls around a floor region."""
    walls = []
    hw, hd = width/2, depth/2
    # North, South, East, West walls
    walls.append(make_wall(cx, wall_height/2, cz+hd, width, wall_height, 0.2))
    walls.append(make_wall(cx, wall_height/2, cz-hd, width, wall_height, 0.2))
    walls.append(make_wall(cx+hw, wall_height/2, cz, 0.2, wall_height, depth))
    walls.append(make_wall(cx-hw, wall_height/2, cz, 0.2, wall_height, depth))
    return walls

def generate_linear(n_rooms=3, room_size=4.0, corridor_width=1.5, corridor_length=2.0):
    """Linear arrangement: R - C - R - C - R"""
    geometries = []
    rooms = []
    x = 0.0
    for i in range(n_rooms):
        floor = make_floor(x, 0.0, room_size, room_size)
        geometries.append(floor)
        geometries.extend(add_walls_around_floor(x, 0.0, room_size, room_size))
        rooms.append({'cx': x, 'cz': 0.0, 'type': 'room'})
        if i < n_rooms - 1:
            cx = x + room_size/2 + corridor_length/2
            floor_c = make_floor(cx, 0.0, corridor_length, corridor_width)
            geometries.append(floor_c)
            rooms.append({'cx': cx, 'cz': 0.0, 'type': 'corridor'})
            x += room_size + corridor_length
    return geometries, rooms

def generate_l_shape(room_size=4.0, corridor_width=1.5, corridor_length=2.0):
    """L-shaped: two rooms horizontal, one room vertical."""
    geometries = []
    rooms = []
    positions = [
        (0.0, 0.0),
        (room_size + corridor_length, 0.0),
        (0.0, room_size + corridor_length),
    ]
    corridors = [
        (room_size/2 + corridor_length/2, 0.0, corridor_length, corridor_width),
        (0.0, room_size/2 + corridor_length/2, corridor_width, corridor_length),
    ]
    for cx, cz in positions:
        floor = make_floor(cx, cz, room_size, room_size)
        geometries.append(floor)
        geometries.extend(add_walls_around_floor(cx, cz, room_size, room_size))
        rooms.append({'cx': cx, 'cz': cz, 'type': 'room'})
    for cx, cz, w, d in corridors:
        floor = make_floor(cx, cz, w, d)
        geometries.append(floor)
        rooms.append({'cx': cx, 'cz': cz, 'type': 'corridor'})
    return geometries, rooms

def generate_t_shape(room_size=4.0, corridor_width=1.5, corridor_length=2.0):
    """T-shaped: three rooms meeting at a junction."""
    geometries = []
    rooms = []
    positions = [
        (0.0, 0.0),
        (room_size + corridor_length, 0.0),
        (room_size/2 + corridor_length/2, room_size + corridor_length),
    ]
    corridors = [
        (room_size/2 + corridor_length/2, 0.0, corridor_length, corridor_width),
        (room_size/2 + corridor_length/2,
         room_size/2 + corridor_length/2, corridor_width, corridor_length),
    ]
    for cx, cz in positions:
        floor = make_floor(cx, cz, room_size, room_size)
        geometries.append(floor)
        geometries.extend(add_walls_around_floor(cx, cz, room_size, room_size))
        rooms.append({'cx': cx, 'cz': cz, 'type': 'room'})
    for cx, cz, w, d in corridors:
        floor = make_floor(cx, cz, w, d)
        geometries.append(floor)
        rooms.append({'cx': cx, 'cz': cz, 'type': 'corridor'})
    return geometries, rooms

def generate_grid(room_size=3.0, corridor_width=1.2, corridor_length=1.5):
    """2x2 grid of rooms."""
    geometries = []
    rooms = []
    gap = room_size + corridor_length
    for row in range(2):
        for col in range(2):
            cx = col * gap
            cz = row * gap
            floor = make_floor(cx, cz, room_size, room_size)
            geometries.append(floor)
            geometries.extend(add_walls_around_floor(cx, cz, room_size, room_size))
            rooms.append({'cx': cx, 'cz': cz, 'type': 'room'})
            if col < 1:
                floor_c = make_floor(cx + room_size/2 + corridor_length/2,
                                     cz, corridor_length, corridor_width)
                geometries.append(floor_c)
                rooms.append({'cx': cx + room_size/2 + corridor_length/2,
                              'cz': cz, 'type': 'corridor'})
            if row < 1:
                floor_c = make_floor(cx, cz + room_size/2 + corridor_length/2,
                                     corridor_width, corridor_length)
                geometries.append(floor_c)
                rooms.append({'cx': cx, 'cz': cz + room_size/2 + corridor_length/2,
                              'type': 'corridor'})
    return geometries, rooms

def generate_random(n_rooms=None, seed=None):
    """Random arrangement of 3-6 rooms with random sizes."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    if n_rooms is None:
        n_rooms = random.randint(3, 6)

    geometries = []
    rooms = []
    placed = []

    for i in range(n_rooms):
        w = random.uniform(2.5, 5.0)
        d = random.uniform(2.5, 5.0)

        if not placed:
            cx, cz = 0.0, 0.0
        else:
            ref = random.choice(placed)
            side = random.choice(['right', 'top', 'left', 'bottom'])
            gap = random.uniform(1.0, 2.5)
            if side == 'right':
                cx = ref['cx'] + ref['w']/2 + gap + w/2
                cz = ref['cz'] + random.uniform(-1.0, 1.0)
            elif side == 'left':
                cx = ref['cx'] - ref['w']/2 - gap - w/2
                cz = ref['cz'] + random.uniform(-1.0, 1.0)
            elif side == 'top':
                cx = ref['cx'] + random.uniform(-1.0, 1.0)
                cz = ref['cz'] + ref['d']/2 + gap + d/2
            else:
                cx = ref['cx'] + random.uniform(-1.0, 1.0)
                cz = ref['cz'] - ref['d']/2 - gap - d/2

            # Add corridor between ref and new room
            mid_cx = (ref['cx'] + cx) / 2
            mid_cz = (ref['cz'] + cz) / 2
            corr_w = min(abs(cx - ref['cx']), 1.5)
            corr_d = min(abs(cz - ref['cz']), 1.5)
            if corr_w < 0.3:
                corr_w = 1.2
            if corr_d < 0.3:
                corr_d = 1.2
            floor_c = make_floor(mid_cx, mid_cz, corr_w, corr_d)
            geometries.append(floor_c)
            rooms.append({'cx': mid_cx, 'cz': mid_cz, 'type': 'corridor'})

        floor = make_floor(cx, cz, w, d)
        geometries.append(floor)
        geometries.extend(add_walls_around_floor(cx, cz, w, d))
        rooms.append({'cx': cx, 'cz': cz, 'type': 'room'})
        placed.append({'cx': cx, 'cz': cz, 'w': w, 'd': d})

    return geometries, rooms

GENERATORS = [
    ('linear_3', lambda: generate_linear(n_rooms=3)),
    ('linear_4', lambda: generate_linear(n_rooms=4)),
    ('linear_5', lambda: generate_linear(n_rooms=5)),
    ('l_shape',  lambda: generate_l_shape()),
    ('t_shape',  lambda: generate_t_shape()),
    ('grid_2x2', lambda: generate_grid()),
]

# Add 44 random environments to reach 50 total
for i in range(494):
    GENERATORS.append((f'random_{i:03d}', lambda i=i: generate_random(seed=i)))

print(f"Generating {len(GENERATORS)} environments...")
success = 0
failed = 0

for name, gen_fn in GENERATORS:
    try:
        geometries, rooms = gen_fn()
        assert len(geometries) > 0, "No geometries generated"

        mesh = trimesh.util.concatenate(geometries)
        assert len(mesh.vertices) > 0, "Empty mesh"

        path = os.path.join(OUTPUT_DIR, f"{name}.obj")
        mesh.export(path)

        meta = {'name': name, 'rooms': rooms, 'n_geometries': len(geometries)}
        with open(os.path.join(OUTPUT_DIR, f"{name}.json"), 'w') as f:
            json.dump(meta, f)

        success += 1
    except Exception as e:
        print(f"  FAILED {name}: {e}")
        failed += 1

print(f"\nDone. {success} environments generated, {failed} failed.")
print(f"Saved to: {OUTPUT_DIR}")
print(f"Files: {len(os.listdir(OUTPUT_DIR))}")
