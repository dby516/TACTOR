import h5py
import numpy as np

with h5py.File('data/shapenetcorev2_PC_2048/shapenetcorev2_hdf5_2048/test0.h5', 'r') as f:
    print(f.keys())  # see data format
    objects = f['data'][:]  # (2048, 2048, 3)
    labels = f['label'][:]  # (2048, 1)
    n_obj = objects.shape[0]

def save_as_ply(filename, points):
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

def farthest_point_sample(points, k):
    sampled = [np.random.randint(len(points))]
    for _ in range(1, k):
        dist = np.linalg.norm(points - points[sampled[-1]], axis=1)
        min_dist = np.min([np.linalg.norm(points - points[i], axis=1) for i in sampled], axis=0)
        next_idx = np.argmax(min_dist)
        sampled.append(next_idx)
    return points[sampled]

for i in range(8):
    points = objects[i][:]
    label = labels[i]
    print(points.shape, label.shape)
    save_as_ply(f"data/shapenetcorev2_PC_2048/pc_{i}.ply", points)


# FPS 2048 -> 256
# Process all pc_*.ply files
input_dir = "data/shapenetcorev2_PC_2048"
output_dir = "data/shapenetcorev2_PC_2048/pointclouds"
os.makedirs(output_dir, exist_ok=True)

ply_files = sorted(glob(os.path.join(input_dir, "pc_*.ply")))

for ply_path in ply_files:
    points = load_ply(ply_path)
    if len(points) < 256:
        print(f"Skipping {ply_path}: less than 256 points.")
        continue
    sampled = farthest_point_sample(points, 256)
    name = os.path.basename(ply_path).replace(".ply", "_fps.ply")
    save_ply(os.path.join(output_dir, name), sampled)
    print(f"Saved: {name}")