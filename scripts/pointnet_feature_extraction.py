import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset

# PointNet implementation from https://github.com/fxia22/pointnet.pytorch
from pointnet import PointNetEncoder

class PointCloudDataset(Dataset):
    def __init__(self, pointcloud_dir, num_points=1024):
        self.pointcloud_dir = pointcloud_dir
        self.files = [os.path.join(pointcloud_dir, f) for f in os.listdir(pointcloud_dir) if f.endswith('.npy')]
        self.num_points = num_points

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        pc = np.load(self.files[idx])
        if pc.shape[0] > self.num_points:
            indices = np.random.choice(pc.shape[0], self.num_points, replace=False)
        else:
            indices = np.random.choice(pc.shape[0], self.num_points, replace=True)
        pc = pc[indices, :3]  # x,y,z
        return torch.from_numpy(pc).float(), self.files[idx]


def extract_pointnet_features(pointcloud_dir, output_dir, pretrained_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PointCloudDataset(pointcloud_dir)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

    model = PointNetEncoder(global_feat=True, feature_transform=False).to(device)
    model.load_state_dict(torch.load(pretrained_path))
    model.eval()

    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for point_batch, file_names in dataloader:
            point_batch = point_batch.transpose(2, 1).to(device)  # (B, 3, N)
            features, _, _ = model(point_batch)

            for feature_vec, file_path in zip(features, file_names):
                base = os.path.basename(file_path).replace('.npy', '_feat.npy')
                out_path = os.path.join(output_dir, base)
                np.save(out_path, feature_vec.cpu().numpy())
                print(f"Saved feature to {out_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Extract PointNet features from point clouds')
    parser.add_argument('--pc_dir', type=str, required=True, help='Directory with .npy point cloud files')
    parser.add_argument('--out_dir', type=str, required=True, help='Directory to save PointNet features')
    parser.add_argument('--pretrained', type=str, required=True, help='Path to pretrained PointNet weights')
    args = parser.parse_args()

    extract_pointnet_features(args.pc_dir, args.out_dir, args.pretrained)
