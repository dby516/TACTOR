"""
Step 2: Simulate PC(t+1) — Redundant Contact

This script samples 20 NEW points from the original full point cloud that are
very CLOSE to points in PC(t), simulating redundant contact with minimal exploration gain.

These are appended to PC(t) from step1, then downsampled back to 256 points
using Farthest Point Sampling (FPS).

Input:  data/shapenetcorev2_PC_2048/raw/pc_*.ply     (raw full PC)
        data/shapenetcorev2_PC_2048/step1/pc_*_0.ply (PC(t) from step1)
Output: data/shapenetcorev2_PC_2048/step2/pc_*_1.ply
"""

import numpy as np
import torch
import os
from glob import glob

def load_ply(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    start = lines.index("end_header\n") + 1
    data = np.array([[float(v) for v in line.strip().split()] for line in lines[start:]])
    return data

def save_ply(filename, points):
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")

def sample_redundant_from_raw(full_pc, base_pc, k):
    # base_pc_np: [256, 3], full_pc_np: [2048, 3]
    from sklearn.neighbors import KDTree

    tree = KDTree(base_pc)
    dist, ind = tree.query(full_pc, k=1)
    mask = dist.squeeze() < 0.05  # "redundant" if within 5cm

    close_candidates = full_pc[mask]
    base_set = {tuple(p) for p in np.round(base_pc, 4)}

    # Only choose points that are not already in pc_prev
    new_close = [p for p in close_candidates if tuple(np.round(p, 4)) not in base_set]

    if len(new_close) < k:
        # fallback: allow some overlap
        new_close = close_candidates

    selected = np.array(new_close)[np.random.choice(len(new_close), size=k, replace=False)]
    return selected


def farthest_point_sampling(points, k):
    # Ensure on CPU for NumPy ops
    points = points.detach().cpu().numpy()

    sampled = [points[np.random.randint(len(points))]]
    distances = np.linalg.norm(points - sampled[0], axis=1)
    for _ in range(1, k):
        farthest_idx = np.argmax(distances)
        sampled.append(points[farthest_idx])
        distances = np.minimum(distances, np.linalg.norm(points - points[farthest_idx], axis=1))

    return torch.tensor(np.stack(sampled), dtype=torch.float32).cuda()


# Paths
raw_pc_dir = "data/shapenetcorev2_PC_2048/raw"
step1_dir = "data/shapenetcorev2_PC_2048/step1"
output_dir = "data/shapenetcorev2_PC_2048/step2"
os.makedirs(output_dir, exist_ok=True)

step1_files = sorted(glob(os.path.join(step1_dir, "pc_*_0.ply")))

for step1_path in step1_files:
    name_base = os.path.basename(step1_path).replace("_0.ply", "")
    raw_path = os.path.join(raw_pc_dir, name_base + ".ply")

    base_pc = load_ply(step1_path)       # PC(t): 256 points
    full_pc = load_ply(raw_path)         # raw full PC: 2048 points

    redundant_pts = sample_redundant_from_raw(full_pc, base_pc, k=20)
    merged = np.concatenate([base_pc, redundant_pts], axis=0)  # [276, 3]

    final_pc = farthest_point_sampling(merged, 256)
    save_ply(os.path.join(output_dir, f"{name_base}_1.ply"), final_pc)
    print(f"Saved: {name_base}_1.ply")
