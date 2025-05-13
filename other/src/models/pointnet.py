import torch
import torch.nn as nn
import torch.nn.functional as F

class TNet(nn.Module):
    """T-Net for learning input and feature transformations."""
    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.k = k
        
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        
    def forward(self, x):
        batch_size = x.size()[0]
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        
        # Initialize as identity
        iden = torch.eye(self.k).repeat(batch_size, 1, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x.view(-1, self.k, self.k) + iden
        
        return x

class PointNetEncoder(nn.Module):
    """PointNet encoder for feature extraction."""
    def __init__(self, feature_transform=True):
        super(PointNetEncoder, self).__init__()
        self.feature_transform = feature_transform
        
        self.input_transform = TNet(k=3)
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        self.feature_transform = TNet(k=64) if feature_transform else None
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        
    def forward(self, x):
        # x: (B, N, 3)
        batch_size = x.size()[0]
        n_pts = x.size()[1]
        
        # Input transform
        trans = self.input_transform(x.transpose(2, 1))
        x = x.transpose(2, 1)
        x = torch.bmm(x.transpose(2, 1), trans).transpose(2, 1)
        
        # MLP
        x = self.mlp1(x)
        
        # Feature transform
        if self.feature_transform:
            trans_feat = self.feature_transform(x)
            x = torch.bmm(x.transpose(2, 1), trans_feat).transpose(2, 1)
        else:
            trans_feat = None
            
        # MLP
        x = self.mlp2(x)
        
        # Global feature
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        return x, trans, trans_feat

class PointNet(nn.Module):
    """Full PointNet model."""
    def __init__(self, feature_transform=True):
        super(PointNet, self).__init__()
        self.encoder = PointNetEncoder(feature_transform=feature_transform)
        
        # Additional layers for specific tasks can be added here
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        
        self.dropout = nn.Dropout(p=0.3)
        
    def forward(self, x):
        # x: (B, N, 3)
        x, trans, trans_feat = self.encoder(x)
        
        # Additional processing
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x, trans, trans_feat

def feature_transform_regularizer(trans):
    """Compute the regularization loss for feature transform."""
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss 