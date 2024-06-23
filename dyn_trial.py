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



def get_loss(params, curr_data, variables, is_initial_timestep):
    losses = {}

    feature_map, image, viewspace_point_tensor, visibility_filter, radii, depth_pred = render_pkg["feature_map"], render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["depth"]
        
    curr_id = curr_data['id']
    im=im.clip(0,1)

    H, W =im.shape[1], im.shape[2]
    top_mask = torch.zeros((H, W), device=im.device)
    top_mask[:, :]=1

    if 'antimask' in curr_data.keys():
        antimask=curr_data['antimask'].cuda()
        combined_mask = antimask
        top_mask = combined_mask.type(torch.uint8)
        ground_truth_depth = curr_data['gt_depth']
        depth_pred = depth_pred *  top_mask
        ground_truth_depth = ground_truth_depth * top_mask

        depth_pred = depth_pred.squeeze(0)
        depth_pred = depth_pred.reshape(-1, 1)
        ground_truth_depth = ground_truth_depth.reshape(-1, 1)

        depth_pred=depth_pred[depth_pred!=0]
        ground_truth_depth=ground_truth_depth[ground_truth_depth!=0]
        losses['depth'] =   (1 - pearson_corrcoef( ground_truth_depth, 1/(depth_pred+100)))

    top_mask = top_mask.unsqueeze(0).repeat(3, 1, 1)
    masked_im = im * top_mask

    masked_curr_data_im = curr_data['im'] * top_mask

    losses['im'] = 0.8 * l1_loss_v1(masked_im, masked_curr_data_im) + 0.2 * (1.0 - calc_ssim(masked_im, masked_curr_data_im))


    variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification


    if not is_initial_timestep:
        is_fg = (params['seg_colors'][:, 0] > 0.5).detach()
        #print('perv', rendervar['means3D'].shape)
        fg_pts = rendervar['means3D'][is_fg]
        fg_rot = rendervar['rotations'][is_fg]
        #print('fg_pts', fg_pts.shape)
        rel_rot = quat_mult(fg_rot, variables["prev_inv_rot_fg"])
        rot = build_rotation(rel_rot)
        neighbor_pts = fg_pts[variables["neighbor_indices"]]
        curr_offset = neighbor_pts - fg_pts[:, None]
        curr_offset_in_prev_coord = (rot.transpose(2, 1)[:, None] @ curr_offset[:, :, :, None]).squeeze(-1)


        losses['rigid'] = weighted_l2_loss_v2(curr_offset_in_prev_coord, variables["prev_offset"],
                                              variables["neighbor_weight"])

        losses['rot'] = weighted_l2_loss_v2(rel_rot[variables["neighbor_indices"]], rel_rot[:, None],
                                            variables["neighbor_weight"])

        curr_offset_mag = torch.sqrt((curr_offset ** 2).sum(-1) + 1e-20)
        losses['iso'] = weighted_l2_loss_v1(curr_offset_mag, variables["neighbor_dist"], variables["neighbor_weight"])

        losses['floor'] = torch.clamp(fg_pts[:, 1], min=0).mean()

        bg_pts = rendervar['means3D'][~is_fg]
        bg_rot = rendervar['rotations'][~is_fg]


        losses['bg'] = l1_loss_v2(bg_pts, variables["init_bg_pts"]) + l1_loss_v2(bg_rot, variables["init_bg_rot"])
        losses['soft_col_cons'] = l1_loss_v2(params['rgb_colors'], variables["prev_col"])

    loss_weights = {'im': 0.1, 'rigid': 0.4, 'rot': 0.4, 'iso': 0.2, 'floor': 0.2, 'bg': 2.0, 'soft_col_cons': 0.01, 'depth':0.1}
                    
    loss = sum([loss_weights[k] * v for k, v in losses.items()])
    seen = radius > 0
    variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
    variables['seen'] = seen
    
    return loss, variables


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


def initialize_post_first_timestep(params, variables, optimizer, num_knn=20):
    is_fg = params['seg_colors'][:, 0] > 0.5
    init_fg_pts = params['means3D'][is_fg]
    init_bg_pts = params['means3D'][~is_fg]
    init_bg_rot = torch.nn.functional.normalize(params['unnorm_rotations'][~is_fg])
    neighbor_sq_dist, neighbor_indices = o3d_knn(init_fg_pts.detach().cpu().numpy(), num_knn)
    neighbor_weight = np.exp(-2000 * neighbor_sq_dist)
    neighbor_dist = np.sqrt(neighbor_sq_dist)
    variables["neighbor_indices"] = torch.tensor(neighbor_indices).cuda().long().contiguous()
    variables["neighbor_weight"] = torch.tensor(neighbor_weight).cuda().float().contiguous()
    variables["neighbor_dist"] = torch.tensor(neighbor_dist).cuda().float().contiguous()

    variables["init_bg_pts"] = init_bg_pts.detach()
    variables["init_bg_rot"] = init_bg_rot.detach()
    variables["prev_pts"] = params['means3D'].detach()
    variables["prev_rot"] = torch.nn.functional.normalize(params['unnorm_rotations']).detach()
    params_to_fix = ['logit_opacities', 'log_scales', 'cam_m', 'cam_c']
    for param_group in optimizer.param_groups:
        if param_group["name"] in params_to_fix:
            param_group['lr'] = 0.0
    return variables

import wandb

def initialize_wandb(exp_name, seq):
    wandb.init(project="Dynamic3D", name=f"{exp_name}_{seq}", config={
        "experiment_name": exp_name,
        "sequence": seq,
    })


def gaussians_to_params(gaussians):


     return {
        'means3D': gaussians.get_xyz,
        'rgb_colors': init_pt_cld[:, 3:6], #*255,
        'unnorm_rotations': gaussians.get_rotation, 
        'logit_opacities': gaussians.get_opacity, 
        'log_scales': gaussians.get_scaling, 
    }
def train(seq, exp, opt=None, pipe=None):
    gaussians = GaussianModel(3)
    scene = Scene(gaussians)
    gaussians.training_setup(opt)
    bg_color = [1, 1, 1] #if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    initialize_wandb(exp, seq)


    for t in reversed(range(111)):
        t=int(t)
        stat_dataset = None
        viewpoint_stack = None
        is_initial_timestep = (int(t) == 110)
        if not is_initial_timestep:
            params, variables = initialize_per_timestep(params, variables, optimizer)
        num_iter_per_timestep = int(2.8e3) if is_initial_timestep else int(2.1e3)
        progress_bar = tqdm(range(int(num_iter_per_timestep)), desc=f"timestep {t}")


        for i in range(num_iter_per_timestep):
            if not viewpoint_stack:
              viewpoint_stack = scene.getTrainBatch()
   
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
            curr_data=get_gt_batch(viewpoint_cam)
            print(viewpoint_cam.keys())

            render_pkg = render(viewpoint_cam['cam'], gaussians, pipe, background)


            loss, variables, losses = get_loss(render_pkg, curr_data, variables, is_initial_timestep)

            loss.backward()
            for k, v in variables.items():
              try:
                print(k, v.grad.shape)
              except:
                print('false',  k)
            #variables['means2D'].grad[:38312, :] =  0

            with torch.no_grad():
                report_progress(params, dataset[0], i, progress_bar)
                report_stat_progress(params, t, i, progress_bar, md)
                if is_initial_timestep:
                    params, variables = densify(params, variables, optimizer, i)
                assert ((params['means3D'].shape[0]==0) is False)


                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for key, value in losses.items():
              wandb.log({key: value.item(), "iteration": i})
            
        progress_bar.close()
        output_params.append(params2cpu(params, is_initial_timestep))
        #print(output_params)
        if is_initial_timestep:
            variables = initialize_post_first_timestep(params, variables, optimizer)

    save_params(output_params, seq, exp)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_semantic_feature = self._semantic_feature[selected_pts_mask] 

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_semantic_feature) 



    if iteration < opt.densify_until_iter:
        # Keep track of max radii in image-space for pruning
        gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
        gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

        if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
            size_threshold = 20 if iteration > opt.opacity_reset_interval else None
            gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
        
        if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
            gaussians.reset_opacity()
            








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