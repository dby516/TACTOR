"""
Step 3: Simulate PC(t+1) — Exploratory Contact

This script simulates good exploratory behavior by generating 20 new points that are:
- Far from the current PC(t)
- Close to each other (to simulate a local patch of contact)

These points are appended to PC(t), and the final PC(t+1) is downsampled to 256 points
using Farthest Point Sampling (FPS).

Purpose: Evaluate whether the reward function encourages novel but physically plausible
surface exploration that increases total shape coverage.

Input:  data/shapenetcorev2_PC_2048/raw/pc_*.ply (raw full 2048-point PC)
        data/shapenetcorev2_PC_2048/step1/pc_*_0.ply (PC(t) from step1)
Output: data/shapenetcorev2_PC_2048/step3/pc_*_1.ply (exploratory PC(t+1))
"""

import numpy as np
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

def sample_exploratory(full_pc, base_pc, k):
    # Compute distances from full_pc to base_pc
    dists = np.min(np.linalg.norm(full_pc[:, None] - base_pc[None], axis=2), axis=1)
    far_candidates = full_pc[dists > np.percentile(dists, 75)]  # choose far quartile
    anchor = far_candidates[np.random.randint(len(far_candidates))]

    # Sample patch around anchor
    noise = np.random.normal(scale=0.01, size=(k, 3))  # slightly looser than redundant
    patch = anchor + noise
    return patch

def farthest_point_sampling(points, k):
    sampled = [points[np.random.randint(len(points))]]
    distances = np.linalg.norm(points - sampled[0], axis=1)
    for _ in range(1, k):
        farthest_idx = np.argmax(distances)
        sampled.append(points[farthest_idx])
        distances = np.minimum(distances, np.linalg.norm(points - points[farthest_idx], axis=1))
    return np.stack(sampled, axis=0)

# Paths
raw_pc_dir = "data/shapenetcorev2_PC_2048/raw"
step1_dir = "data/shapenetcorev2_PC_2048/step1"
output_dir = "data/shapenetcorev2_PC_2048/step3"
os.makedirs(output_dir, exist_ok=True)

step1_files = sorted(glob(os.path.join(step1_dir, "pc_*_0.ply")))

for step1_path in step1_files:
    name_base = os.path.basename(step1_path).replace("_0.ply", "")
    raw_path = os.path.join(raw_pc_dir, name_base + ".ply")

    base_pc = load_ply(step1_path)        # PC(t): 256 points
    full_pc = load_ply(raw_path)          # original full PC: 2048 points

    exploratory_patch = sample_exploratory(full_pc, base_pc, k=20)
    merged = np.concatenate([base_pc, exploratory_patch], axis=0)  # [276, 3]

    final_pc = farthest_point_sampling(merged, 256)
    save_ply(os.path.join(output_dir, f"{name_base}_1.ply"), final_pc)
    print(f"Saved: {name_base}_1.ply")
