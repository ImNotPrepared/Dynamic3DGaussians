import torch
import os
import json
import copy
import numpy as np
from PIL import Image
import open3d as o3d
from random import randint
from tqdm import tqdm
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
import matplotlib.pyplot as plt
from gsplat.rendering import rasterization_inria_wrapper
from helpers import setup_camera, l1_loss_v1, l1_loss_v2, weighted_l2_loss_v1, weighted_l2_loss_v2, quat_mult, \
    o3d_knn, params2cpu, save_params, save_params_progressively
from external import calc_ssim, calc_psnr, build_rotation, densify, update_params_and_optimizer
import cv2
import torch.nn.functional as F
from torchmetrics.functional.regression import pearson_corrcoef
import torchvision.transforms as transforms

from ideaII import flow_loss
from sklearn.decomposition import PCA

near, far = 1e-7, 7e0
def vis_feature(features, mask=None):
    # Check the shape of the features tensor: [C, H, W]
    shape = features.shape  # (C, H, W)
    
    # Reshape the features to a 2D matrix: [H*W, C]
    # We want to apply PCA across the channels (C), treating each spatial location (H, W) as a sample
    features = features.reshape(shape[0], -1).T 
    
     # Reshape to [H*W, C], with pixels as samples and channels as features
    if mask:
      mask_flat = mask.flatten()
      masked_features = features[mask_flat == 1]
    # Apply PCA to reduce the number of channels from C to 3 (for RGB)
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(masked_features)  # Result shape will be [H*W, 3]
    
    # Normalize the PCA result to range [0, 255] for visualization as RGB
    pca_features = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())
    pca_features = (pca_features * 255).astype(np.uint8)
    
    # Reshape back to the image format: [H, W, 3]
    pca_features = pca_features.reshape(shape[1], shape[2], 3)
    
    return pca_features

import matplotlib.pyplot as plt
def combine_images(image1, depth1, feat1,  image2, depth2, feat2):
    print(feat1.shape)

  
    # Convert depth maps to 3-channel images
    depth1_3channel = depth1#cv2.cvtColor(depth1, cv2.COLOR_GRAY2RGB)
    depth2_3channel = depth2#cv2.cvtColor(depth2, cv2.COLOR_GRAY2RGB)
    
    # Combine each image and depth side by side
    combined1 = np.hstack((image1, depth1_3channel, feat1))
    combined2 = np.hstack((image2, depth2_3channel, feat2))
    
    # Combine the two combined images in a 2x2 grid
    combined_all = np.vstack((combined1, combined2))
    return combined_all


base_path = '/data3/zihanwa3/Capstone-DSR/Processing/dinov2features/test/'
FFuk_maps = []
def retrieve_feat_map(seq, ts):
  feature_root_path= base_path + seq #undist_cam00_670/000000.npy'
  feature_path = feature_root_path+f'/00{str(ts)}.npy' 
  dinov2_feature = torch.tensor(np.load(feature_path.replace('.jpg', '.npy'))).permute(2, 0, 1)
  gt_feature_map = vis_feature(dinov2_feature)
  print('feature_shape', gt_feature_map.shape)
  return gt_feature_map

base_mask_path = '/data3/zihanwa3/Capstone-DSR/Processing/sam_v2_dyn_mask/'
dyn_masks = []
def retrieve_dyn_mask(seq, ts, resize=None):
    mask_path = base_mask_path + f'{str(seq)}/dyn_mask_{str(ts)}.npz'
    mask = np.load(mask_path)['dyn_mask']
    print('Original mask_shape:', mask.shape)

    # Remove the first dimension (1, h, w) -> (h, w)
    mask = np.squeeze(mask, axis=0)
    
    # Resize the mask if needed
    if resize is not None:
        mask = cv2.resize(mask, resize, interpolation=cv2.INTER_NEAREST)
        print('Resized mask_shape:', mask.shape)

    # Add a new axis to convert (h, w) -> (h, w, 1)
    mask = mask[:, :, np.newaxis]
    print('Converted mask_shape:', mask.shape)

    return mask


ts = 183
FFuk_maps.append(retrieve_feat_map('undist_cam01', ts))
FFuk_maps.append(retrieve_feat_map('undist_cam02', ts))
FFuk_maps.append(retrieve_feat_map('undist_cam03', ts))
FFuk_maps.append(retrieve_feat_map('undist_cam04', ts))

dyn_masks.append(retrieve_dyn_mask(1, ts, resize=(512, 288)))
dyn_masks.append(retrieve_dyn_mask(2, ts, resize=(512, 288)))
dyn_masks.append(retrieve_dyn_mask(3, ts, resize=(512, 288)))
dyn_masks.append(retrieve_dyn_mask(4, ts, resize=(512, 288)))

for feat_map, dyn_mask in zip(FFuk_maps, dyn_masks):
  masked_feat_map = dyn_mask * feat_map
  unmasked_feat_map = feat_map * (1-dyn_mask)
  print(masked_feat_map.shape, unmasked_feat_map.shape)

combined = combine_images(FFuk_maps[0], FFuk_maps[1], FFuk_maps[3], FFuk_maps[2], FFuk_maps[3], FFuk_maps[3])

# Log combined image
wandb.log({
    f"stat_combined_{c-1399}": wandb.Image(combined, caption=f"Rendered image and depth at iteration {i}")
})

