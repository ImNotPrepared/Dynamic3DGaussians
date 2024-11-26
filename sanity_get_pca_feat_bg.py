import torch
from PIL import Image
import torchvision.transforms as T
import torchvision
from torchvision.transforms.functional import InterpolationMode, to_pil_image, resize, to_tensor
from sklearn.decomposition import PCA
import numpy as np
import imageio
import math
from itertools import product
from torch.nn import functional as F
import glob
import os
import pickle
import time
import argparse
import pickle

import cv2
near, far = 1e-7, 7e0
import matplotlib.pyplot as plt


base_path = '/data3/zihanwa3/Capstone-DSR/Dynamic3DGaussians/data_ego/cmu_bike/bg_feats/'

def retrieve_feat_map(seq, ts):
  feature_root_path= base_path + seq #undist_cam00_670/000000.npy'
  feature_path = feature_root_path+f'/{ts:05d}.npy'
  dinov2_feature = torch.tensor(np.load(feature_path.replace('.jpg', '.npy'))).permute(2, 0, 1)
  gt_feature_map = (dinov2_feature)
  return gt_feature_map

base_mask_path = '/data3/zihanwa3/Capstone-DSR/Processing_dance/sam_v2_dyn_mask/'


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
    full_pca_features = np.full((features_flat.shape[0], 3), 255, dtype=np.uint8)

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
all_features = []
all_masks = []
FFuk_maps = []
dyn_masks = []
for cam_id in range(1, 5):
  #for ts in tqdm(range(123, 423)):
  for ts in tqdm(range(1477, 1777, 3)):
    feat_map = retrieve_feat_map(f'undist_cam0{cam_id}', ts)
    dyn_mask = retrieve_dyn_mask(cam_id, ts, resize=(512, 288))
    # Flatten features and masks
    features_flat = feat_map.reshape(feat_map.shape[0], -1).T  # Shape: [H*W, C]
    mask_flat = dyn_mask.flatten().astype(bool)
    masked_features = features_flat[mask_flat]
    all_features.append(masked_features)
    all_masks.append(mask_flat)

# Concatenate all features
all_features = np.vstack(all_features)  # Shape: [Total_samples, C]

pca = PCA(n_components=3)
pca.fit(all_features)
# Save the fitted PCA object
with open('/data3/zihanwa3/Capstone-DSR/Processing_dance/dinov2features/fitted_pca_model.pkl', 'wb') as f:
    pickle.dump(pca, f)
print('dumped!!!!')
# Step 3: Visualize features using PCA
#3 for ts in tqdm(range(123, 423), desc=f"Visualizing features for cam {cam_id}"):
for cam_id in range(1, 5):
  for ts in tqdm(range(1477, 1777, 3), desc=f"Visualizing features for cam {cam_id}"):
      feat_map = retrieve_feat_map(f'undist_cam0{cam_id}', ts)  # Shape: [C, H, W]
      dyn_mask = retrieve_dyn_mask(cam_id, ts, resize=(512, 288))  # Shape: [H, W]
      masked_feat_map = vis_feature(feat_map, pca, mask=dyn_mask)  # Should handle masking internally

      # Convert to image
      masked_image = Image.fromarray(masked_feat_map.astype('uint8'))

      # Save the image
      base_base_path = f'/data3/zihanwa3/Capstone-DSR/Processing_dance/dinov2features/cam_temp_white/{cam_id}'
      if not os.path.exists(base_base_path):
          os.makedirs(base_base_path)
      masked_image.save(os.path.join(base_base_path, f'masked_feat_map_{ts}.png'))