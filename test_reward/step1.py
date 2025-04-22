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

def sample_nearest(points, k):
    anchor_idx = np.random.randint(len(points))
    anchor = points[anchor_idx]
    dists = np.linalg.norm(points - anchor, axis=1)
    idx = np.argsort(dists)[:k]
    return points[idx]

# Paths
input_dir = "data/shapenetcorev2_PC_2048"
output_dir = "data/shapenetcorev2_PC_2048/step1"
os.makedirs(output_dir, exist_ok=True)

ply_files = sorted(glob(os.path.join(input_dir, "pc_*.ply")))

for ply_path in ply_files:
    points = load_ply(ply_path)
    if len(points) < 256:
        print(f"Skipping {ply_path}: less than 256 points.")
        continue
    sampled = sample_nearest(points, 256)
    name = os.path.basename(ply_path).replace(".ply", "_0.ply")
    save_ply(os.path.join(output_dir, name), sampled)
    print(f"Saved: {name}")
