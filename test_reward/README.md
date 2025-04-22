# Reward Function Evaluation

This module evaluates whether our reward function aligns with the intended exploration behavior: **encouraging comprehensive object surface coverage** while **discouraging redundant or overlapping exploration**.

### Objective

Collect 256 tactile contact points that best represent the overall object surface—**uniformly distributed and maximally informative** within a constrained point budget.

At each timestep, newly acquired contact points are merged with the accumulated point cloud. To maintain a consistent size, we apply fast point sampling (Farthest Point Sampling) to retain the most spatially diverse subset.

### Input

The reward function takes 2 things as its input.

* contact(t): current timestep’s contact readings — pressure vectors shaped `[N, x, y, z, fx, fy, fz]`.
* PC(t): accumulated point cloud from timestep 1 to t.



### Reward Designs

We take 3 designs.

#### 1. PointNet discrepancy

```
reward = α * ||PointNet(PC(t)) - PointNet(PC(t-1))|| + β * len(contact(t))
# Euclidean distance between the 2 features
```

Encourages feature diversity over time using PointNet global embeddings and rewards larger, valid contact sets.

**Pros**:

- Sensitive to global geometry.
- Encourages novelty beyond raw distance.

**Cons**:

- Harder to interpret.
- Computationally heavier than raw metrics.
- Rely on quality of trained PointNetEncoder.

#### 2. Chamfer Distance

```
reward = α * Chamfer(PC(t), PC(t-1)) + β * len(contact(t))
```

Uses geometric dissimilarity as a proxy for exploration gain.

**Pros**:

- Intuitive.
- Easier to compute.

**Cons**:

- May reward noisy outliers.
- Ignores semantic shape changes.

#### 3. Surface Occupancy Gain

```
reward = γ * ΔOccupiedVolume(PC(t), PC(t-1)) + β * len(contact(t))
voxel_index = floor(point / voxel_size) # Map PC(t) to a voxel index
ΔOccupiedVolume = len(occupied_voxels(t)) - len(occupied_voxels(t-1))
```

Tracks increase in occupied volume to directly quantify spatial gain. As we only use 256 points, a low resolution voxel map will be applied.

**Pros:**

* Direct spatial metric rather than proxy distance.
* Easy to interprete

**Cons:**

* Resolution trade-off: too coarse: can't differentiate fine details. too fine: duplicates and noise
* Ignores structure: all newly occupied voxels are treated equally -- can't learn to explore special structures.



### Evaluation Protocol

To assess reward effectiveness, we simulate pairs of point clouds at timestep `t` and `t+1` based on ShapeNetCoreV2 data (sampling code from https://github.com/antao97/PointCloudDatasets).

##### Simulation Setup:

- **PC(t)**: choose 1 random point, add its 255 nearest neighbors → simulates poor exploration.
- **PC(t+1)**: create one of three controlled variants:
  1. **No new contact** — identical to PC(t) → should be punished.
  2. **Redundant contact** — 20 new points close to PC(t) → moderately penalized.
  3. **Exploratory contact** — 20 new points far from PC(t) but close to each other → rewarded.

##### Ideal Behavior:

An optimal policy will:

1. First maximize contact quantity.
2. Then diversify spatial coverage.
3. Eventually saturate the object's surface efficiently.