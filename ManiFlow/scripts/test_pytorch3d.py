import numpy as np
import pdb
from PIL import Image, ImageColor
import open3d as o3d
import json
import transforms3d as t3d
import cv2
import torch
import yaml
import trimesh
import math
import os


import pytorch3d.ops as torch3d_ops

def fps(points, num_points=1024, use_cuda=True):
    K = [num_points]
    if use_cuda:
        points = torch.from_numpy(points).cuda()
        sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
        sampled_points = sampled_points.squeeze(0)
        sampled_points = sampled_points.cpu().numpy()
    else:
        points = torch.from_numpy(points)
        sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
        sampled_points = sampled_points.squeeze(0)
        sampled_points = sampled_points.numpy()

    return sampled_points, indices

# Test Pytorch3D FPS in GPU
if __name__ == "__main__":
    import torch
    import numpy as np

    points = np.random.rand(10000, 3).astype(np.float32)

    sampled_points, indices = fps(points, num_points=1024, use_cuda=True)
    print("Sampled Points Shape:", sampled_points.shape)
    print("Indices Shape:", indices.shape)
    print("Sampled Points:", sampled_points)
    print("Indices:", indices)