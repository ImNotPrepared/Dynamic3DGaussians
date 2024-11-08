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
import matplotlib.pyplot as plt


base_path = '/data3/zihanwa3/Capstone-DSR/Processing/dinov2features/resized_512_Aligned/'

def retrieve_feat_map(seq, ts):
  feature_root_path= base_path + seq #undist_cam00_670/000000.npy'
  feature_path = feature_root_path+f'/{ts:05d}.npy'
  dinov2_feature = torch.tensor(np.load(feature_path.replace('.jpg', '.npy'))).permute(2, 0, 1)
  gt_feature_map = (dinov2_feature)
  return gt_feature_map

base_mask_path = '/data3/zihanwa3/Capstone-DSR/Processing/sam_v2_dyn_mask/'

def retrieve_dyn_mask(seq, ts, resize=None):
    mask_path = base_mask_path + f'{str(seq)}/dyn_mask_{str(ts)}.npz'
    mask = np.load(mask_path)['dyn_mask']

    # Remove the first dimension (1, h, w) -> (h, w)
    mask = np.squeeze(mask, axis=0)
    
    # Resize the mask if needed
    if resize is not None:
        mask = cv2.resize(mask, resize, interpolation=cv2.INTER_NEAREST)


    # Add a new axis to convert (h, w) -> (h, w, 1)
    mask = mask[:, :, np.newaxis]

    return mask

def vis_feature(features, pca, mask=None):
    # Reshape features
    shape = features.shape  # (C, H, W)
    features_flat = features.reshape(shape[0], -1).T  # Shape: [H*W, C]
    
    if mask is not None:
        mask_flat = mask.flatten().astype(bool)
        features_to_project = features_flat[mask_flat]
    else:
        features_to_project = features_flat

    # Project features onto PCA components
    pca_features = pca.transform(features_to_project)  # Shape: [N, 3]
    
    # Normalize for visualization
    pca_features_norm = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())
    pca_features_norm = (pca_features_norm * 255).astype(np.uint8)
    
    # Reconstruct full image
    full_pca_features = np.zeros((features_flat.shape[0], 3), dtype=np.uint8)
    if mask is not None:
        full_pca_features[mask_flat] = pca_features_norm
    else:
        full_pca_features = pca_features_norm
    
    # Reshape back to image format
    pca_features_image = full_pca_features.reshape(shape[1], shape[2], 3)
    
    return pca_features_image

import os
from tqdm import tqdm
# Step 1: Collect all features across time and cameras


for ts in tqdm(range(123, 423)):
  all_features = []
  all_masks = []
  FFuk_maps = []
  dyn_masks = []
  for cam_id in range(1, 5):
    feat_map = retrieve_feat_map(f'undist_cam0{cam_id}', ts)
    dyn_mask = retrieve_dyn_mask(cam_id, ts, resize=(512, 288))
    # Flatten features and masks
    features_flat = feat_map.reshape(feat_map.shape[0], -1).T  # Shape: [H*W, C]
    mask_flat = dyn_mask.flatten().astype(bool)
    # Optionally apply mask
    masked_features = features_flat[mask_flat]
    all_features.append(masked_features)
    all_masks.append(mask_flat)

  # Concatenate all features
  all_features = np.vstack(all_features)  # Shape: [Total_samples, C]

# Step 2: Apply PCA to the collected dataset
  pca = PCA(n_components=3)
  pca.fit(all_features)

  # Step 3: Visualize features using PCA
  for cam_id in tqdm(range(1, 5), desc=f"Visualizing features for cam {cam_id}"):
      feat_map = retrieve_feat_map(f'undist_cam0{cam_id}', ts)  # Shape: [C, H, W]
      dyn_mask = retrieve_dyn_mask(cam_id, ts, resize=(512, 288))  # Shape: [H, W]

      masked_feat_map = vis_feature(feat_map, pca, mask=dyn_mask)  # Should handle masking internally

      # Convert to image
      masked_image = Image.fromarray(masked_feat_map.astype('uint8'))

      # Save the image
      base_base_path = f'/data3/zihanwa3/Capstone-DSR/Processing/dinov2features/pca_aligned_cam/{cam_id}'
      if not os.path.exists(base_base_path):
          os.makedirs(base_base_path)
      masked_image.save(os.path.join(base_base_path, f'masked_feat_map_{ts}.png'))