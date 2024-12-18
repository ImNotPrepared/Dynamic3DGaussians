import torch
import os
import json
import copy
import numpy as np
import torch.nn.functional as F
from PIL import Image
from random import randint
from tqdm import tqdm
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from helpers import setup_camera, l1_loss_v1, l1_loss_v2, weighted_l2_loss_v1, weighted_l2_loss_v2, quat_mult, \
    o3d_knn, params2cpu, save_params
from external import calc_ssim, calc_psnr, build_rotation, densify, update_params_and_optimizer
import cv2
from torchmetrics.functional.regression import pearson_corrcoef
import torchvision.transforms as transforms

def get_dataset(t, md, seq, mode='stat_only'):
    dataset = []
    t+=183
    if mode=='ego_only':
      for c in range(1,5):
          epsilon=1e-7
          h, w = md['hw'][c]
          k, w2c =  md['k'][t][c], np.linalg.inv(md['w2c'][t][c])
          cam = setup_camera(w, h, k, w2c, near=0.01, far=50)
          fn = md['fn'][t][c] # mask_{fn.split('/')[0]}
          im = np.array(copy.deepcopy(Image.open(f"/ssd0/zihanwa3/data_ego/{seq}/ims/{fn}")))
          im = torch.tensor(im).float().cuda().permute(2, 0, 1) / 255
          im=im.clip(0,1)
          if h == w:
            im = torch.rot90(im, k=1, dims=(1, 2))
          else:
            mask_path=f'/ssd0/zihanwa3/duster_depth/{c}/{t}.npz'
            depth = torch.tensor(np.load(mask_path)['depth']).float().cuda()
            depth=1/(depth+100.)
            dataset.append({'cam': cam, 'im': im, 'id': c, 'gt_depth':depth, 'vis': True}) 

      return dataset

def get_batch(todo_dataset, dataset):
    if not todo_dataset:
        todo_dataset = dataset.copy()
    curr_data = todo_dataset.pop(randint(0, len(todo_dataset) - 1))
    return curr_data

def initialize_params(seq, md, init_pt_path):
    # init_pt_cld_before_dense init_pt_cld
    init_pt_cld = np.load(init_pt_path)["data"]
    #init_pt_cld = np.concatenate((init_pt_cld, init_pt_cld), axis=0)
    print(len(init_pt_cld))
    seg = init_pt_cld[:, 6]
    max_cams = 305
    sq_dist, _ = o3d_knn(init_pt_cld[:, :3], 3)
    mean3_sq_dist = sq_dist.mean(-1).clip(min=0.0000001)

    params = {
        'means3D': init_pt_cld[:, :3],
        'rgb_colors': init_pt_cld[:, 3:6], #*255,
        'seg_colors': np.stack((seg, np.zeros_like(seg), 1 - seg), -1),
        'unnorm_rotations': np.tile([1, 0, 0, 0], (seg.shape[0], 1)),
        'logit_opacities': np.zeros((seg.shape[0], 1)),
        'log_scales': np.tile(np.log(np.sqrt(mean3_sq_dist))[..., None], (1, 3)),
        'cam_m': np.zeros((max_cams, 3)),
        'cam_c': np.zeros((max_cams, 3)),
    }
    print(params['log_scales'].shape)
    params = {k: torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True)) for k, v in
              params.items()}

    params['label']=  ((
        torch.ones(len(params['means3D']), requires_grad=False, device="cuda")
        ))
              
    cam_centers = np.linalg.inv(md['w2c'][0])[:, :3, 3]  # Get scene radius
    scene_radius = 1.1 * np.max(np.linalg.norm(cam_centers - np.mean(cam_centers, 0)[None], axis=-1))
    print('scene_radius', scene_radius)
    variables = {'max_2D_radius': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'scene_radius': scene_radius,
                 'means2D_gradient_accum': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'denom': torch.zeros(params['means3D'].shape[0]).cuda().float()}
    return params, variables

def params2rendervar(params, index=38312):
    ## [org, new_params(person)]
    rendervar = {
        'means3D': params['means3D'],
        'colors_precomp': params['rgb_colors'],
        'rotations': torch.nn.functional.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(params['log_scales']),
        'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device="cuda") + 0,
        'label': params['label']

    }
    return rendervar

def initialize_optimizer(params, variables):
    lrs = {
        'means3D': 0.0000014 * variables['scene_radius'], # 0000014
        'rgb_colors': 0.000028, ###0.0028 will fail
        'unnorm_rotations': 0.00001,
        'seg_colors':0.0,
        'logit_opacities': 0.01,
        'log_scales': 0.005,
        'cam_m': 1e-5,
        'cam_c': 1e-5,
    }
    '''
            'logit_opacities': 0.05,
        'log_scales': 0.001,
    '''
    param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items() if k in lrs.keys()]
    return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)

def get_loss(params, curr_data, variables, is_initial_timestep, stat_dataset=None, org_params=None):
    losses = {}
    rendervar = params2rendervar(params)
    rendervar['means2D'].retain_grad()


    im, radius, depth_pred, _ = Renderer(raster_settings=curr_data['cam'])(**rendervar)
    curr_id = curr_data['id']
    im=im.clip(0,1)

    H, W =im.shape[1], im.shape[2]
    top_mask = torch.zeros((H, W), device=im.device)
    top_mask[:, :]=1


    combined_mask = top_mask
    top_mask = combined_mask.type(torch.uint8)
    
    ground_truth_depth = curr_data['gt_depth']
    #print('depth_loss', depth_pred.shape, ground_truth_depth.shape)
    depth_pred = F.interpolate(depth_pred.unsqueeze(0), size=(288, 512), mode='bilinear', align_corners=False) 
    ground_truth_depth = ground_truth_depth #* top_mask

    depth_pred = depth_pred
    depth_pred = depth_pred.reshape(-1, 1)

    ground_truth_depth = ground_truth_depth.reshape(-1, 1)
    

    depth_pred=depth_pred[ground_truth_depth!=0]
    ground_truth_depth=ground_truth_depth[ground_truth_depth!=0]
    losses['depth'] =   (1 - pearson_corrcoef( ground_truth_depth, 1/(depth_pred+100)))

    #l1_loss_v1(ground_truth_depth, depth_pred)

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

    loss_weights = {'im': 1.0, 'rigid': 0.4, 'rot': 0.4, 'iso': 0.2, 'floor': 0.2, 'bg': 2.0, 'soft_col_cons': 0.01, 'depth':0.0}
                    
    loss = sum([loss_weights[k] * v for k, v in losses.items()])
    seen = radius > 0
    variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
    variables['seen'] = seen
    
    return loss, variables, losses


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


def train(seq, exp):
    md = json.load(open(f"./data_ego/{seq}/Dy_train_meta.json", 'r'))  # metadata

    num_timesteps = len(md['fn'])

    
    init_path='/data3/zihanwa3/Capstone-DSR/Appendix/dust3r/aug_person.npz'


    params, variables = initialize_params(seq, md, init_path)

    optimizer = initialize_optimizer(params, variables)
    output_params = []

    initialize_wandb(exp, seq)


    for t in reversed(range(111)):
        t=int(t)
        dataset = get_dataset(t, md, seq, mode='ego_only')
        stat_dataset = None
        todo_dataset = []
        is_initial_timestep = (int(t) == 110)
        if not is_initial_timestep:
            params, variables = initialize_per_timestep(params, variables, optimizer)
        num_iter_per_timestep = int(2.1e4) if is_initial_timestep else int(2.8e3)
        progress_bar = tqdm(range(int(num_iter_per_timestep)), desc=f"timestep {t}")
        for i in tqdm(range(num_iter_per_timestep), desc=f"timestep {t}"):
            curr_data = get_batch(todo_dataset, dataset)

            loss, variables, losses = get_loss(params, curr_data, variables, is_initial_timestep, stat_dataset=stat_dataset, org_params=None)
            loss.backward()
            with torch.no_grad():
                #report_progress(params, dataset[0], i, progress_bar)
                #report_stat_progress(params, t, i, progress_bar, md)
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
