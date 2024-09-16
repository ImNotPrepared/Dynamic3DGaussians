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

def retrieve_feat_map(seq, ts):
  feature_root_path= base_path + seq #undist_cam00_670/000000.npy'
  feature_path = feature_root_path+f'/00{str(ts)}.npy' 
  dinov2_feature = torch.tensor(np.load(feature_path.replace('.jpg', '.npy'))).permute(2, 0, 1)
  gt_feature_map = (dinov2_feature)
  print('feature_shape', gt_feature_map.shape)
  return gt_feature_map

base_mask_path = '/data3/zihanwa3/Capstone-DSR/Processing/sam_v2_dyn_mask/'

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

def vis_feature(features, mask=None):
    # Check the shape of the features tensor: [C, H, W]
    shape = features.shape  # (C, H, W)
    
    # Reshape the features to a 2D matrix: [H*W, C]
    features = features.reshape(shape[0], -1).T  # Shape: [H*W, C]
    
    # Initialize masked_features
    masked_features = features
    mask_flat = None

    # Apply mask if provided
    if mask is not None:
        mask_flat = mask.flatten().astype(bool)
        masked_features = features[mask_flat]
    
    # Apply PCA to reduce the number of channels from C to 3 (for RGB)
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(masked_features)  # Shape: [N_masked, 3]
    
    # Normalize the PCA result to range [0, 255] for visualization as RGB
    pca_features_norm = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())
    pca_features_norm = (pca_features_norm * 255).astype(np.uint8)
    
    # Reconstruct the full image
    full_pca_features = np.zeros((features.shape[0], 3), dtype=np.uint8)
    if mask is not None:
        full_pca_features[mask_flat] = pca_features_norm
    else:
        full_pca_features = pca_features_norm
    
    # Reshape back to the image format: [H, W, 3]
    pca_features_image = full_pca_features.reshape(shape[1], shape[2], 3)
    
    return pca_features_image


for ts in [183, 294]:
  FFuk_maps = []
  dyn_masks = []
  FFuk_maps.append(retrieve_feat_map('undist_cam01', ts))
  FFuk_maps.append(retrieve_feat_map('undist_cam02', ts))
  FFuk_maps.append(retrieve_feat_map('undist_cam03', ts))
  FFuk_maps.append(retrieve_feat_map('undist_cam04', ts))

  dyn_masks.append(retrieve_dyn_mask(1, ts, resize=(512, 288)))
  dyn_masks.append(retrieve_dyn_mask(2, ts, resize=(512, 288)))
  dyn_masks.append(retrieve_dyn_mask(3, ts, resize=(512, 288)))
  dyn_masks.append(retrieve_dyn_mask(4, ts, resize=(512, 288)))

  for idx, (feat_map, dyn_mask) in enumerate(zip(FFuk_maps, dyn_masks)):
      # Generate masked and unmasked feature maps
      masked_feat_map = vis_feature(feat_map, mask=dyn_mask)
      unmasked_feat_map = vis_feature(feat_map, mask=np.ones(dyn_mask.shape))

      # Save the masked feature map
      masked_image = Image.fromarray(masked_feat_map)
      masked_image.save(f'./sanity_visuals/masked_feat_map_{ts}_{idx+1}.png')

      # Save the unmasked feature map
      unmasked_image = Image.fromarray(unmasked_feat_map)
      unmasked_image.save(f'./sanity_visuals/unmasked_feat_map_{ts}_{idx+1}.png')

  combined = combine_images(FFuk_maps[0], FFuk_maps[1], FFuk_maps[3], FFuk_maps[2], FFuk_maps[3], FFuk_maps[3])

# Log combined image
wandb.log({
    f"stat_combined_{c-1399}": wandb.Image(combined, caption=f"Rendered image and depth at iteration {i}")
})

