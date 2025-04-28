import torch


def farthest_point_sampling(points: torch.Tensor, k: int) -> torch.Tensor:
    """
    points: (N, 3) float32 tensor on CUDA
    k: number of points to sample
    returns: (k, 3) tensor of sampled points
    """
    device = points.device
    N = points.shape[0]
    sampled_idx = torch.zeros(k, dtype=torch.long, device=device)
    distances = torch.full((N,), float("inf"), device=device)

    # Initialize with a random point
    sampled_idx[0] = torch.randint(0, N, (1,), device=device)
    farthest = points[sampled_idx[0]].unsqueeze(0)  # [1, 3]

    for i in range(1, k):
        dist = torch.norm(points - farthest, dim=1)  # [N]
        distances = torch.minimum(distances, dist)
        sampled_idx[i] = torch.argmax(distances)
        farthest = points[sampled_idx[i]].unsqueeze(0)

    return points[sampled_idx]  # [k, 3]


# Chamfer Distance
def chamfer_distance(a, b):
    a = a.unsqueeze(1)  # [Na, 1, 3]
    b = b.unsqueeze(0)  # [1, Nb, 3]
    dist = torch.norm(a - b, dim=2)  # [Na, Nb]
    cd = dist.min(dim=1)[0].mean() + dist.min(dim=0)[0].mean()
    return cd

def Reward_ChamferDist(contact_len: int, pc, pc_pre):
    chamfer = chamfer_distance(pc, pc_pre)  # unnormalized
    novelty = chamfer / 0.1
    return novelty


# Voxel Occupancy
def voxelize(points, voxel_size=0.15):
    coords = torch.floor(points / voxel_size).int()
    keys = {tuple(c.tolist()) for c in coords}
    return keys

def Reward_Voxel_Occupancy(contact_len: int, pc, pc_pre, voxel_size=0.05):
    vox_now = voxelize(pc, voxel_size)
    vox_prev = voxelize(pc_pre, voxel_size)
    new_voxels = len(vox_now - vox_prev)

    novelty = new_voxels / 64.0
    return novelty
