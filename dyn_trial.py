import torch
import os
import json
import copy
import numpy as np
from PIL import Image
from random import randint
from argparse import ArgumentParser, Namespace
from tqdm import tqdm
from gaussian_renderer import render, network_gui
from helpers import setup_camera, l1_loss_v1, l1_loss_v2, weighted_l2_loss_v1, weighted_l2_loss_v2, quat_mult, \
    o3d_knn, params2rendervar, params2cpu, save_params
from external import calc_ssim, calc_psnr, build_rotation, densify, update_params_and_optimizer
import cv2
from arguments import ModelParams, PipelineParams, OptimizationParams
from torchmetrics.functional.regression import pearson_corrcoef
import torchvision.transforms as transforms
from scene import Scene, GaussianModel
import wandb

def initialize_wandb(exp_name, seq):
    wandb.init(project="Dynamic3D", name=f"{exp_name}_{seq}", config={
        "experiment_name": exp_name,
        "sequence": seq,
    })


def initialize_per_timestep(params, variables, optimizer):
    pts = params['means3D']
    rot = torch.nn.functional.normalize(params['unnorm_rotations'])
    new_pts = pts + (pts - variables["prev_pts"])
    new_rot = torch.nn.functional.normalize(rot + (rot - variables["prev_rot"]))

    is_fg = params['seg_colors'][:, 0] > 0.5
    prev_inv_rot_fg = rot[is_fg]
    prev_inv_rot_fg[:, 1:] = -1 * prev_inv_rot_fg[:, 1:]
    fg_pts = pts[is_fg]
    prev_offset = fg_pts[variables["neighbor_indices"]] - fg_pts[:, None]
    variables['prev_inv_rot_fg'] = prev_inv_rot_fg.detach()
    variables['prev_offset'] = prev_offset.detach()
    variables["prev_pts"] = pts.detach()
    variables["prev_rot"] = rot.detach()
    variables["prev_col"] = params['rgb_colors'].detach()
    new_params = {'means3D': new_pts, 'unnorm_rotations': new_rot}
    params = update_params_and_optimizer(new_params, params, optimizer)

    return params, variables


def initialize_post_first_timestep(gaussians, variables, optimizer, num_knn=20):
    is_fg = torch.ones(len(gaussians.get_xyz))#gaussians.get_seg[:, 0] > 0.5
    init_fg_pts = gaussians.get_xyz[is_fg]
    init_bg_pts = gaussians.get_xyz[~is_fg]

    init_bg_rot = torch.nn.functional.normalize(gaussians.get_rotation[~is_fg])
    neighbor_sq_dist, neighbor_indices = o3d_knn(init_fg_pts.detach().cpu().numpy(), num_knn)
    neighbor_weight = np.exp(-2000 * neighbor_sq_dist)
    neighbor_dist = np.sqrt(neighbor_sq_dist)
    variables["neighbor_indices"] = torch.tensor(neighbor_indices).cuda().long().contiguous()
    variables["neighbor_weight"] = torch.tensor(neighbor_weight).cuda().float().contiguous()
    variables["neighbor_dist"] = torch.tensor(neighbor_dist).cuda().float().contiguous()

    variables["init_bg_pts"] = init_bg_pts.detach()
    variables["init_bg_rot"] = init_bg_rot.detach()
    variables["prev_pts"] = gaussians.get_xyz.detach()
    variables["prev_rot"] = torch.nn.functional.normalize(gaussians.get_rotation).detach()
    params_to_fix = ['logit_opacities', 'log_scales', 'cam_m', 'cam_c']
    for param_group in optimizer.param_groups:
        if param_group["name"] in params_to_fix:
            param_group['lr'] = 0.0
    return variables

def gaussians2params(gaussians):
    params = {
        'means3D': gaussians.get_xyz,
        'rgb_colors': gaussians.get_colors_precomp,
        'unnorm_rotations': gaussians.get_rotation,
        'logit_opacities':  gaussians.get_opacity,
        'log_scales':  gaussians.get_scaling,
    }
    return params

def train(seq, exp, opt=None, pipe=None):
    gaussians = GaussianModel(3)
    scene = Scene(gaussians)
    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] #if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32).cuda()
    initialize_wandb(exp, seq)
    variables=gaussians.variables
    for t in reversed(range(111)):
        t=int(t)
        stat_dataset = None
        viewpoint_stack = None
        is_initial_timestep = (int(t) == 110)
        #if not is_initial_timestep:
        #    params, variables = initialize_per_timestep(params, variables, optimizer)
        num_iter_per_timestep = int(2.8e3) if is_initial_timestep else int(2.1e3)

        for i in tqdm(range(num_iter_per_timestep)):
            losses = {}
            if not viewpoint_stack:
              viewpoint_stack =scene.getTrainCameras().copy()
   
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
            curr_data=(viewpoint_cam)
            viewpoint_cam=viewpoint_cam#['cam']#.copy()

            H, W =viewpoint_cam.image_height ,viewpoint_cam.image_width# 256,  256#im.shape[1], im.shape[2]
            masked_curr_data_im = viewpoint_cam.original_image.cuda() #curr_data['im']# * top_mask
            
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, override_color=True)
            im, viewspace_point_tensor, visibility_filter, radii, depth_pred = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["depth"]
            mask=viewpoint_cam.loss_mask.cuda()
            im=im.cuda()
            masked_im = im * mask


            ground_truth_depth = viewpoint_cam.depth.cuda()
            depth_pred = F.interpolate(depth_pred, size=(144, 256), mode='bilinear', align_corners=False) #*  top_mask
            ground_truth_depth = ground_truth_depth #* top_mask
            depth_pred = depth_pred.squeeze(0)
            depth_pred = depth_pred.reshape(-1, 1)
            ground_truth_depth = ground_truth_depth.reshape(-1, 1)
            depth_pred=depth_pred[depth_pred!=0]
            ground_truth_depth=ground_truth_depth[ground_truth_depth!=0]
            losses['depth'] =   (1 - pearson_corrcoef( ground_truth_depth, 1/(depth_pred+100)))

            #top_mask = top_mask.unsqueeze(0).repeat(3, 1, 1)
            masked_im = im #* top_mask

            masked_curr_data_im = viewpoint_cam.original_image.cuda() #* top_mask

            losses['im'] = 0.8 * l1_loss_v1(masked_im, masked_curr_data_im) + 0.2 * (1.0 - calc_ssim(masked_im, masked_curr_data_im))
            #variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification

            loss_weights = {'im': 0.1, 'rigid': 0.4, 'rot': 0.4, 'iso': 0.2, 'floor': 0.2, 'bg': 2.0, 'soft_col_cons': 0.01, 'depth':0.1}
                            
            loss = sum([loss_weights[k] * v for k, v in losses.items()])
            loss.backward()


            
        progress_bar.close()
        output_params.append(params2cpu(params, is_initial_timestep))
        #print(output_params)
        if is_initial_timestep:
            variables = initialize_post_first_timestep(params, variables, optimizer)

if __name__ == "__main__":
    import argparse
    parser = ArgumentParser(description="Training script parameters")
    #lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument('--exp_name', type=str, required=True, help='Name of the experiment')
    args = parser.parse_args()

    exp_name = args.exp_name
    for sequence in ["cmu_bike"]:
    #(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from)
        train(sequence, exp_name, opt=op.extract(args), pipe=pp)
        torch.cuda.empty_cache()
        from visualize import visualize
        visualize(sequence, exp_name)
        from visualize_dyn import visualize
        visualize(sequence, exp_name)

    print("\nTraining complete.")