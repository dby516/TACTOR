import torch
import torch.nn as nn
from external_tools.point_net.models.pointnet_utils import PointNetEncoder


class PointNetEncoderTrans(nn.Module):
    def __init__(self, output_dim=256):
        super().__init__()
        self.encoder = PointNetEncoder(global_feat=True, feature_transform=False)
        self.projector = nn.Linear(1024, output_dim)

    def forward(self, x):
        feat_1024, _, _ = self.encoder(x)  # x: (B, 3, N)
        feat_256 = self.projector(feat_1024)
        return feat_256

pointnet_model = PointNetEncoderTrans().cuda()
pointnet_model.eval()

def Reward_PointNet(contact, pc, pc_pre):
    device = pc.device
    forces = contact[:, 4:7]
    max_force = torch.norm(forces, dim=1).max()
    force_reward = torch.clamp(max_force, 0.0, 5.0) / 5.0

    pc_feat = pointnet_model(pc.T.unsqueeze(0))         # shape [1, 1024]
    pc_pre_feat = pointnet_model(pc_pre.T.unsqueeze(0)) # shape [1, 1024]
    feat_diff = torch.norm(pc_feat - pc_pre_feat, p=2)

    # Normalize and center: reward ∈ [-0.2, +0.8]
    novelty = feat_diff / 5.0 - 0.2
    total_reward = 0.7 * novelty + 0.3 * (force_reward - 0.2)

    return total_reward

# Chamfer Distance Reward
def chamfer_distance(a, b):
    # a: [Na, 3], b: [Nb, 3]
    a = a.unsqueeze(1)  # [Na, 1, 3]
    b = b.unsqueeze(0)  # [1, Nb, 3]
    dist = torch.norm(a - b, dim=2)  # [Na, Nb]
    cd = dist.min(dim=1)[0].mean() + dist.min(dim=0)[0].mean()
    return cd

def Reward_ChamferDist(contact, pc, pc_pre):
    forces = contact[:, 4:7]
    max_force = torch.norm(forces, dim=1).max()
    force_reward = torch.clamp(max_force, 0.0, 5.0) / 5.0

    chamfer = chamfer_distance(pc, pc_pre)  # unnormalized
    novelty = chamfer / 0.1 - 0.2  # normalized & centered

    return 0.7 * novelty + 0.3 * (force_reward - 0.2)


# Voxel Occupancy Reward
def voxelize(points, voxel_size=0.15):
    coords = torch.floor(points / voxel_size).int()
    keys = {tuple(c.tolist()) for c in coords}
    return keys

def Reward_Voxel_Occupancy(contact, pc, pc_pre, voxel_size=0.15):
    forces = contact[:, 4:7]
    max_force = torch.norm(forces, dim=1).max()
    force_reward = torch.clamp(max_force, 0.0, 5.0) / 5.0

    vox_now = voxelize(pc, voxel_size)
    vox_prev = voxelize(pc_pre, voxel_size)
    new_voxels = len(vox_now - vox_prev)

    novelty = new_voxels / 64.0 - 0.2

    return 0.7 * novelty + 0.3 * (force_reward - 0.2)
