"""
Fetches a real 3D mesh from the World Labs Marble API.

Pipeline:
  1. POST to worlds:generate with a text prompt
  2. Poll operation until done (world takes ~5 minutes)
  3. Extract collider_mesh_url from response
  4. Download the GLB file
  5. Load it with trimesh and verify it is a valid mesh
  6. Run it through build_region_graph pipeline

Usage:
  python3 fetch_world_mesh.py
"""

import os
import time
import requests
import trimesh
import sys

API_KEY = os.environ.get("WORLD_LABS_API_KEY", "")
BASE_URL = "https://api.worldlabs.ai/marble/v1"
HEADERS = {
    "WLT-Api-Key": API_KEY,
    "Content-Type": "application/json"
}

assert API_KEY, "WORLD_LABS_API_KEY environment variable not set. Run: export WORLD_LABS_API_KEY=your_key"
assert len(API_KEY) > 10, "API key looks too short — check it is set correctly"


def generate_world(prompt, model="marble-1.1"):
    print(f"Requesting world generation...")
    print(f"  Prompt: {prompt}")
    print(f"  Model: {model}")

    payload = {
        "display_name": "LSM Training Environment",
        "model": model,
        "world_prompt": {
            "type": "text",
            "text_prompt": prompt
        }
    }

    r = requests.post(f"{BASE_URL}/worlds:generate", json=payload, headers=HEADERS, timeout=30)

    assert r.status_code == 200, (
        f"Generation request failed.\n"
        f"Status: {r.status_code}\n"
        f"Response: {r.text[:500]}"
    )

    data = r.json()
    operation_id = data.get("operation_id")
    assert operation_id, f"No operation_id in response: {data}"

    print(f"Operation started: {operation_id}")
    return operation_id


def poll_operation(operation_id, max_wait_minutes=10):
    print(f"Polling operation (world generation takes ~5 minutes)...")
    url = f"{BASE_URL}/operations/{operation_id}"
    deadline = time.time() + max_wait_minutes * 60
    attempt = 0

    while time.time() < deadline:
        attempt += 1
        r = requests.get(url, headers=HEADERS, timeout=30)

        assert r.status_code == 200, (
            f"Poll request failed.\n"
            f"Status: {r.status_code}\n"
            f"Response: {r.text[:500]}"
        )

        data = r.json()
        done = data.get("done", False)
        error = data.get("error")
        metadata = data.get("metadata", {})
        progress = metadata.get("progress", {}) if metadata else {}
        status = progress.get("status", "UNKNOWN")
        description = progress.get("description", "")

        print(f"  [{attempt:03d}] {status}: {description}")

        assert error is None, f"World generation failed with error: {error}"

        if done:
            response = data.get("response")
            assert response, f"Operation done but no response field: {data}"
            print(f"World generation complete.")
            return response

        time.sleep(15)

    raise TimeoutError(f"World generation did not complete within {max_wait_minutes} minutes")


def extract_mesh_url(world_response):
    assets = world_response.get("assets", {})
    mesh = assets.get("mesh", {})
    url = mesh.get("collider_mesh_url")

    assert url, (
        f"No collider_mesh_url in response assets.\n"
        f"Assets keys: {list(assets.keys())}\n"
        f"Mesh keys: {list(mesh.keys())}"
    )

    world_id = world_response.get("id", "unknown")
    print(f"World ID: {world_id}")
    print(f"Mesh URL: {url[:80]}...")
    return url, world_id


def download_mesh(url, output_path):
    print(f"Downloading mesh to {output_path}...")
    r = requests.get(url, timeout=120, stream=True)

    assert r.status_code == 200, (
        f"Mesh download failed.\n"
        f"Status: {r.status_code}"
    )

    with open(output_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    assert size_mb > 0.01, f"Downloaded file is too small ({size_mb:.3f} MB) — likely empty"
    print(f"Downloaded: {size_mb:.2f} MB")
    return output_path


def load_and_verify_mesh(path):
    print(f"Loading mesh from {path}...")
    scene = trimesh.load(path)

    # GLB files are usually scenes with multiple geometries
    if isinstance(scene, trimesh.Scene):
        geometries = list(scene.geometry.values())
        print(f"Scene with {len(geometries)} geometries")
        assert len(geometries) > 0, "Scene has no geometries"

        # Concatenate all geometries into one mesh
        mesh = trimesh.util.concatenate(geometries)
        print(f"Concatenated mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    else:
        mesh = scene
        print(f"Single mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    assert len(mesh.vertices) > 100, f"Mesh has too few vertices: {len(mesh.vertices)}"
    assert len(mesh.faces) > 100, f"Mesh has too few faces: {len(mesh.faces)}"

    return mesh


if __name__ == "__main__":
    # Indoor environment — good for floor extraction
    PROMPT = "A modern apartment interior with a living room, hallway, and kitchen"

    # Step 1: Generate
    operation_id = generate_world(PROMPT)

    # Step 2: Poll
    world_response = poll_operation(operation_id)

    # Step 3: Extract URL
    mesh_url, world_id = extract_mesh_url(world_response)

    # Step 4: Download
    output_path = f"/workspaces/dygna/world_{world_id[:8]}.glb"
    download_mesh(mesh_url, output_path)

    # Step 5: Load and verify
    mesh = load_and_verify_mesh(output_path)

    # Step 6: Run through region graph pipeline
    print(f"\nRunning region graph pipeline on real World Labs mesh...")
    sys.path.insert(0, '/workspaces/dygna')
    from build_region_graph import build_spatial_graph, verify_graph_with_spatial_ref

    graph = build_spatial_graph(output_path)
    verify_graph_with_spatial_ref(graph)

    print(f"\nFull pipeline verified on real World Labs mesh.")
    print(f"Mesh saved at: {output_path}")
