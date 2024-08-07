import torch
import os
import json
import copy
import numpy as np
from PIL import Image
from random import randint
from tqdm import tqdm
import glob
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from helpers import setup_camera, l1_loss_v1, l1_loss_v2, weighted_l2_loss_v1, weighted_l2_loss_v2, quat_mult, \
    o3d_knn, params2cpu, save_params
from external import calc_ssim, calc_psnr, build_rotation, densify, update_params_and_optimizer
import cv2
from torchmetrics.functional.regression import pearson_corrcoef
import torchvision.transforms as transforms

import torch.nn.functional as F
from dyn_utils import *

def get_loss(params, curr_datasss, variables, is_initial_timestep, stat_dataset=None):
    losses = {
        'depth': 0,
        'im': 0,
        'rigid': 0,
        'rot': 0,
        'iso': 0,
        'floor': 0,
        'bg': 0,
        'soft_col_cons': 0
    }
    rendervar = params2rendervar(params)
    rendervar['means2D'].retain_grad()

    for curr_data in curr_datasss:
        im, radius, depth_pred, _ = Renderer(raster_settings=curr_data['cam'])(**rendervar)
        im = im.clip(0, 1)

        H, W = im.shape[1], im.shape[2]
        top_mask = torch.ones((H, W), device=im.device)
        combined_mask = top_mask.type(torch.uint8)

        ground_truth_depth = curr_data['gt_depth']
        depth_pred = F.interpolate(depth_pred.unsqueeze(0), size=(288, 512), mode='bilinear', align_corners=False)
        ground_truth_depth = ground_truth_depth.reshape(-1, 1)
        depth_pred = depth_pred.reshape(-1, 1)

        valid_mask = ground_truth_depth != 0
        depth_pred = depth_pred[valid_mask]
        ground_truth_depth = ground_truth_depth[valid_mask]
        losses['depth'] += (1 - pearson_corrcoef(ground_truth_depth, 1 / (depth_pred + 100)))

        top_mask = top_mask.unsqueeze(0).repeat(3, 1, 1)
        masked_im = im * top_mask
        masked_curr_data_im = curr_data['im'] * top_mask

        losses['im'] += 0.8 * l1_loss_v1(masked_im, masked_curr_data_im) + 0.2 * (1.0 - calc_ssim(masked_im, masked_curr_data_im))

        variables['means2D'] = rendervar['means2D']
        if not is_initial_timestep:
            is_fg = params['label'] == 1
            fg_pts = rendervar['means3D'][is_fg]
            fg_rot = rendervar['rotations'][is_fg]
            rel_rot = quat_mult(fg_rot, variables["prev_inv_rot_fg"])
            rot = build_rotation(rel_rot)
            neighbor_pts = fg_pts[variables["neighbor_indices"]]
            curr_offset = neighbor_pts - fg_pts[:, None]
            curr_offset_in_prev_coord = (rot.transpose(2, 1)[:, None] @ curr_offset[:, :, :, None]).squeeze(-1)

            losses['rigid'] += weighted_l2_loss_v2(curr_offset_in_prev_coord, variables["prev_offset"], variables["neighbor_weight"])
            losses['rot'] += weighted_l2_loss_v2(rel_rot[variables["neighbor_indices"]], rel_rot[:, None], variables["neighbor_weight"])
            curr_offset_mag = torch.sqrt((curr_offset ** 2).sum(-1) + 1e-20)
            losses['iso'] += weighted_l2_loss_v1(curr_offset_mag, variables["neighbor_dist"], variables["neighbor_weight"])
            losses['floor'] += torch.clamp(fg_pts[:, 1], min=0).mean()

            bg_pts = rendervar['means3D'][~is_fg]
            bg_rot = rendervar['rotations'][~is_fg]
            losses['bg'] += l1_loss_v2(bg_pts, variables["init_bg_pts"]) + l1_loss_v2(bg_rot, variables["init_bg_rot"])
            losses['soft_col_cons'] += l1_loss_v2(params['rgb_colors'], variables["prev_col"])

    loss_weights = {'im': 0.1, 'rigid': 0.4, 'rot': 0.4, 'iso': 0.2, 'floor': 0.2, 'bg': 2.0, 'soft_col_cons': 0.01, 'depth': 0.001}
    loss = sum([loss_weights[k] * v for k, v in losses.items()]) / len(curr_datasss)

    seen = radius > 0
    variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
    variables['seen'] = seen

    return loss, variables, losses
  
def toworld_projections(pred_path='/data3/zihanwa3/Capstone-DSR/Appendix/SpaTracker'):
    xyds = get_cat_preds(pred_path=pred_path)
    points=[]
    seq_len = len(xyds[0])
    for i, cam_id in enumerate(['1','2', '4', '4', '4']):
      camera = load_camera(cam_id)
      xyd = xyds[i]

      xyd = xyd.reshape(-1, 3)
      point = camera.unproject_points(xyd, world_coordinates=True)# +torch.tensor([-0.4, 2, -1.1])#/255
      point = point.reshape(seq_len, -1, 3)
      points.append(point)
    points = np.concatenate((points), axis=1)
    return points

def get_cat_preds(path):
    tracks=[]
    for index in [1, 2, 4, 5, 6]:
      file_path = os.path.join(path, f'vis_results_{index}')
      print(path)
      file_pattern = os.path.join(file_path, 'task_dynamic_tossing_spatracker*')
      files = glob.glob(file_pattern)
      files = [file for file in files if not file.endswith('.mp4')]
    
      for file_path in files:
        track=np.load(file_path)
        tracks.append(track)
    return tracks

def train(seq, exp):
    md = json.load(open(f"./data_ego/{seq}/Dy_train_meta.json", 'r'))  # metadata
    num_timesteps = len(md['fn'])
    params, variables, sriud = initialize_params(seq, md, exp, surfix='old_output')
    params, variables =  add_new_gaussians(params, variables, sriud)
    optimizer = initialize_optimizer(params, variables)


    pred_path='/data3/zihanwa3/Capstone-DSR/Appendix/SpaTracker'

    preds=toworld_projections(pred_path)
    ### preds in shape of [timestap, N, (x,y,d)] (37, N, 3)

    assert len(params)-38822 == len(preds[0])
    output_params = []

    initialize_wandb(exp, seq)
    org_params=initialize_params(seq, md, exp, surfix='old_output')
    ### revise
    reversed_range = list(range(110, -1, -3))
    temporal_windows = [reversed_range[i:i+10] for i in range(0, 110, 10)]

    get_batch_window
    for temporal_window in temporal_windows:
        dataset = get_dataset(temporal_window, md, seq, mode='ego_only')
        stat_dataset = None
        todo_dataset = []
        is_initial_timestep = (int(t) == 110)
        if not is_initial_timestep:
            params, variables = initialize_per_timestep(params, variables, optimizer)
        num_iter_per_timestep = int(5.8e3) if is_initial_timestep else int(2.1e3)

        
        curr_data = get_batch_window(todo_dataset, dataset)
        total_loss = 0
        for any_ in window_data:
          params 
          loss, variables, losses = get_loss(params, curr_data, variables, is_initial_timestep, stat_dataset=stat_dataset, org_params=None)
          total_loss += loss

          
        total_loss /= len(window_data)
        total_loss.backward()


        with torch.no_grad():
            #report_progress(params, dataset[0], i, progress_bar)
            report_stat_progress(params, t, i, progress_bar, md)
            #if is_initial_timestep:
            #    params, variables = densify(params, variables, optimizer, i)
            assert ((params['means3D'].shape[0]==0) is False)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
        for key, value in losses.items():
          wandb.log({key: value.item(), "iteration": i})
            
        output_params.append(params2cpu(params, is_initial_timestep))

        if is_initial_timestep:
            variables = initialize_post_first_timestep(params, variables, optimizer)

    save_params(output_params, seq, exp)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run training and visualization for a sequence.")
    parser.add_argument('--exp_name', type=str, required=True, help='Name of the experiment')
    args = parser.parse_args()

    exp_name = args.exp_name
    #for sequence in ["basketball", "boxes", "football", "juggle", "softball", "tennis"]:
    for sequence in ["cmu_bike"]:
        train(sequence, exp_name)
        torch.cuda.empty_cache()
        from visualize import visualize
        visualize(sequence, exp_name)
        from visualize_dyn import visualize
        visualize(sequence, exp_name)
