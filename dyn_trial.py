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

def train(seq, exp, opt=None, pipe=None):
    gaussians = GaussianModel(3)
    scene = Scene(gaussians)
    gaussians.training_setup(opt)
    bg_color = [1, 1, 1] #if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32).cuda()
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
            losses = {}
            if not viewpoint_stack:
              viewpoint_stack =scene.getTrainCameras().copy()
   
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
            curr_data=(viewpoint_cam)
            viewpoint_cam=viewpoint_cam#['cam']#.copy()

            H, W =viewpoint_cam.image_height ,viewpoint_cam.image_width# 256,  256#im.shape[1], im.shape[2]
            masked_curr_data_im = viewpoint_cam.original_image.cuda() #curr_data['im']# * top_mask
            
            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
            #im, viewspace_point_tensor, visibility_filter, radii, depth_pred = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["depth"]
            mask=viewpoint_cam.loss_mask.cuda()
            torch.cuda.synchronize()
            im=im.cuda()
            masked_im = im * mask


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