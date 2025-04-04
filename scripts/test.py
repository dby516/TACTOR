"""
Checkpoint Loader for PointNet++
Author: Benny (adapted by Bingyao)
Date: Apr 2025
"""
from external_tools.point_net.models.pointnet_utils import PointNetEncoder
# import argparse
# import os
# import sys
# import torch
# import importlib

# # Set paths
# ROOT_DIR = '/home/bingyao/tactor/external_tools/point_net'
# sys.path.append(os.path.join(ROOT_DIR, 'models'))


# def parse_args():
#     parser = argparse.ArgumentParser('Load Checkpoint')
#     parser.add_argument('--use_cpu', action='store_true', default=False, help='Use CPU')
#     parser.add_argument('--gpu', type=str, default='0', help='GPU device ID')
#     parser.add_argument('--num_category', type=int, default=40, help='Model class count')
#     parser.add_argument('--log_dir', type=str, required=True, help='Name under log/classification/')
#     parser.add_argument('--use_normals', action='store_true', default=False, help='Use normals')
#     return parser.parse_args()


# def main(args):
#     '''Set device'''
#     if not args.use_cpu:
#         os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

#     '''Set log/checkpoint path'''
#     experiment_dir = os.path.join(ROOT_DIR, 'log/classification', args.log_dir)
#     model_log_path = os.path.join(experiment_dir, 'logs')
#     checkpoint_path = os.path.join(experiment_dir, 'checkpoints', 'best_model.pth')

#     '''Load model'''
#     model_name = os.listdir(model_log_path)[0].split('.')[0]
#     print(f"[INFO] Loading model: {model_name}")
#     model = importlib.import_module(model_name)
#     classifier = model.get_model(args.num_category, normal_channel=args.use_normals)

#     if not args.use_cpu:
#         classifier = classifier.cuda()

#     checkpoint = torch.load(checkpoint_path, map_location='cpu' if args.use_cpu else None)
#     classifier.load_state_dict(checkpoint['model_state_dict'])
#     classifier.eval()

#     print(f"[SUCCESS] Loaded model from {checkpoint_path}")

#     print("model: ", classifier)


# if __name__ == '__main__':
#     args = parse_args()
#     main(args)
