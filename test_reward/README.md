# Reward Function Evaluation for TACTOR

This folder contains handcrafted point cloud examples and evaluation scripts for assessing the effectiveness of the reward function used in the TACTOR policy.

## Purpose

The goal is to verify that the reward function encourages the policy to **actively and evenly explore the object surface**, rather than repeatedly scanning a small region.

## Point Cloud Categories

We construct and evaluate three categories of point cloud distributions per object:

1. **Uniform Coverage (Ideal)**
   - Evenly distributed points across the object's surface.
   - Represents optimal exploratory behavior.

2. **Clustered Region (Redundant)**
   - High-density points in a small localized area.
   - Mimics a policy stuck in a repetitive scan pattern.

3. **Partial Coverage (Incomplete)**
   - Points only on a limited portion (e.g., one hemisphere) of the surface.
   - Simulates limited or biased exploration.

## Evaluation Metrics

Reward signals and feature differences (via PointNet encoder) are compared across the above categories to assess:
- Whether the reward increases with improved surface coverage.
- Whether Chamfer Distance between point clouds over time encourages spatial diversity.
- Whether global feature vectors differ meaningfully between diverse vs redundant scans.

## Notes

All point clouds are either manually sampled or programmatically synthesized from known object meshes. See individual scripts for generation details.

