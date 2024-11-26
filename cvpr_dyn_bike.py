import torch
import os
import json
import copy
from matplotlib import colormaps
import imageio as iio
import numpy as np
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

import torch.nn.functional as F

#### [t, c, ...]\\
def get_dataset(t, md, seq, mode='stat_only'):
    dataset = []
    t+=1543
    if mode=='ego_only':
      for c in range(1,5):
          epsilon=1e-7
          h, w = md['hw'][c]
          k, w2c =  md['k'][t][c], np.linalg.inv(md['w2c'][t][c])
          cam = setup_camera(w, h, k, w2c, near=0.01, far=50)
          fn = md['fn'][t][c] # mask_{fn.split('/')[0]}
          
          fn = fn.split('/')[-1]
          paath = f"/data3/zihanwa3/Capstone-DSR/Processing/undist_data/undist_cam0{c}/{fn}"
          print(paath)

          im = np.array(copy.deepcopy(Image.open(paath)))
          im = torch.tensor(im).float().cuda().permute(2, 0, 1) / 255
          im=im.clip(0,1)
          dataset.append({'cam': cam, 'im': im, 'id': c, 'vis': True}) 

      return dataset
def get_batch(todo_dataset, dataset):
    if not todo_dataset:
        todo_dataset = dataset.copy()
    curr_data = todo_dataset.pop(randint(0, len(todo_dataset) - 1))
    return curr_data



def initialize_params(seq, md):
    dust_pc_path = '/data3/zihanwa3/Capstone-DSR/Appendix/dust3r/duster_depth_clean/290/pc.npz'
    init_pt_cld = np.load(dust_pc_path)["data"]
    seg = init_pt_cld[:, 6]
    max_cams = 50
    sq_dist, _ = o3d_knn(init_pt_cld[:, :3], 3)
    mean3_sq_dist = sq_dist.mean(-1).clip(min=0.0000001)
    params = {
        'means3D': init_pt_cld[:, :3],
        'rgb_colors': init_pt_cld[:, 3:6],
        'seg_colors': np.stack((seg, np.zeros_like(seg), 1 - seg), -1),
        'unnorm_rotations': np.tile([1, 0, 0, 0], (seg.shape[0], 1)),
        'logit_opacities': np.zeros((seg.shape[0], 1)),
        'log_scales': np.tile(np.log(np.sqrt(mean3_sq_dist))[..., None], (1, 3)),
        'cam_m': np.zeros((max_cams, 3)),
        'cam_c': np.zeros((max_cams, 3)),
    }
    params = {k: torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True)) for k, v in
              params.items()}
    params['label']=  ((
        torch.ones(len(params['means3D']), requires_grad=False, device="cuda")
        ))
              
    cam_centers = np.linalg.inv(md['w2c'][0])[:, :3, 3]  # Get scene radius
    scene_radius = 1.1 * np.max(np.linalg.norm(cam_centers - np.mean(cam_centers, 0)[None], axis=-1))
    variables = {'max_2D_radius': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'scene_radius': scene_radius,
                 'means2D_gradient_accum': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'denom': torch.zeros(params['means3D'].shape[0]).cuda().float()}
    return params, variables

def report_stat_progress(params, t, i, progress_bar, md, every_i=2100):
    import matplotlib.pyplot as plt


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

def add_new_gaussians(params, variables, scene_radius):
    '''
    print(k)
    means3D
    rgb_colors
    unnorm_rotations
    logit_opacities
    log_scales
    '''
    path='/data3/zihanwa3/Capstone-DSR/Processing/3D/aug_person.npz'

    path='/data3/zihanwa3/Capstone-DSR/Processing/3D/filtered_person.npz'
    new_pt_cld = np.load(path)["data"]
    print('dyn_len', len(new_pt_cld))
    new_params = initialize_new_params(new_pt_cld)

    
    variables = {'max_2D_radius': torch.zeros(new_params['means3D'].shape[0]).cuda().float(),
                 'scene_radius': scene_radius,
                 'means2D_gradient_accum': torch.zeros(new_params['means3D'].shape[0]).cuda().float(),
                 'denom': torch.zeros(new_params['means3D'].shape[0]).cuda().float()}
    for k, v in new_params.items():
      params_no_grad = params[k]#.requires_grad_(False)  # 使 params[k] 不需要梯度
      if len(params_no_grad.shape) == 3:
        params_no_grad=params_no_grad[0]
      new_params[k] = torch.cat((params_no_grad, v), dim=0)


    for k, v in new_params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            new_params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            new_params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

    #### [org, new_params(person)]
    new_params['label']=  torch.cat((
        torch.zeros(index, requires_grad=False, device="cuda"),
        torch.ones(len(new_params['means3D'])-index, requires_grad=False, device="cuda")
        ))

    variables = {'max_2D_radius': torch.zeros(new_params['means3D'].shape[0]).cuda().float(),
                 'scene_radius': scene_radius,
                 'means2D_gradient_accum': torch.zeros(new_params['means3D'].shape[0]).cuda().float(),
                 'denom': torch.zeros(new_params['means3D'].shape[0]).cuda().float()}
    print('stat_len', len(params['means3D'][0]))
    print('overall_len', len(new_params['means3D']))
    return new_params, variables

def initialize_new_params(new_pt_cld):
    #print(new_pt_cld.shape)
    num_pts = new_pt_cld.shape[0]
    means3D = new_pt_cld[:, :3] # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 4]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
    sq_dist, _ = o3d_knn(new_pt_cld[:, :3], 3)
    mean3_sq_dist = sq_dist.mean(-1).clip(min=0.0000001)
    seg = np.ones((num_pts))
    
    params = {
        'means3D': means3D,
        'rgb_colors': new_pt_cld[:, 3:6],
        'unnorm_rotations': unnorm_rots,
        'seg_colors': np.stack((seg, np.zeros_like(seg), 1 - seg), -1),
        'logit_opacities': logit_opacities,
        'log_scales':  np.tile(np.log(np.sqrt(mean3_sq_dist))[..., None], (1, 3)),
        'label': torch.ones(means3D, requires_grad=False, device="cuda")
    }
    params = {k: torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True)) for k, v in
              params.items()}
    return params

def initialize_optimizer(params, variables):
    lrs = {
        'means3D': 0.00014 * variables['scene_radius'], # 0000014
        'rgb_colors': 0.00028, ###0.0028 will fail
        'unnorm_rotations': 0.00001,
        'seg_colors':0.0,
        'logit_opacities': 0.01,
        'log_scales': 0.005,
        'cam_m': 1e-5,
        'cam_c': 1e-5,
    }
    
    param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items() if k in lrs.keys()]
    return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)
import os.path as osp
def apply_float_colormap(img: torch.Tensor, colormap: str = "turbo") -> torch.Tensor:
    """Convert single channel to a color img.

    Args:
        img (torch.Tensor): (..., 1) float32 single channel image.
        colormap (str): Colormap for img.

    Returns:
        (..., 3) colored img with colors in [0, 1].
    """
    img = torch.nan_to_num(img, 0)
    if colormap == "gray":
        return img.repeat(1, 1, 3)
    img_long = (img * 255).long()
    img_long_min = torch.min(img_long)
    img_long_max = torch.max(img_long)
    assert img_long_min >= 0, f"the min value is {img_long_min}"
    assert img_long_max <= 255, f"the max value is {img_long_max}"
    return torch.tensor(
        colormaps[colormap].colors,  # type: ignore
        device=img.device,
    )[img_long[..., 0]]

def apply_depth_colormap(
    depth,
    acc=None,
    near_plane= None,
    far_plane= None,
):
    """Converts a depth image to color for easier analysis.

    Args:
        depth (torch.Tensor): (..., 1) float32 depth.
        acc (torch.Tensor | None): (..., 1) optional accumulation mask.
        near_plane: Closest depth to consider. If None, use min image value.
        far_plane: Furthest depth to consider. If None, use max image value.

    Returns:
        (..., 3) colored depth image with colors in [0, 1].
    """
    near_plane = near_plane or float(torch.min(depth))
    far_plane = far_plane or float(torch.max(depth))
    depth = (depth - near_plane) / (far_plane - near_plane + 1e-10)
    depth = torch.clip(depth, 0.0, 1.0)
    img = apply_float_colormap(depth, colormap="turbo")
    if acc is not None:
        img = img * acc + (1.0 - acc)
    return img

def save_figs(params, curr_data):
    rendervar = params2rendervar(params)
    rendervar['means2D'].retain_grad()

    for iddx, data in enumerate(curr_data):
      im, radius, depth_pred, _ = Renderer(raster_settings=data['cam'])(**rendervar)
      im=im.clip(0,1)
      depth_min, depth_max = 1e6, 0
      ref_pred_depth = depth_pred.cpu()
      depth_min = min(depth_min, ref_pred_depth.min().item())
      depth_max = max(depth_max, ref_pred_depth.quantile(0.99).item())

      save_dir = 'cvpr_results'
      image_dir = osp.join(save_dir, f"int_images")
      os.makedirs(image_dir, exist_ok=True)

      combined_img = im
      image_path = osp.join(image_dir, f"frame_{iddx:04d}.png")

      print(combined_img.shape, ref_pred_depth.shape)
      iio.imwrite(
          image_path,
          (combined_img.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
      )

      '''ref_pred_depth.permute(1, 2, 0)
      depth_colormap = apply_depth_colormap(
          ref_pred_depth, near_plane=depth_min, far_plane=depth_max
      )
      depth_image_path = osp.join(image_dir, f"depth_{iddx:04d}.png")


      depth_colormap = depth_colormap.detach().cpu().numpy()#.transpose(1, 2, 0)
      print(combined_img.shape, ref_pred_depth.shape)
      iio.imwrite(
          depth_image_path,
          (depth_colormap * 255).astype(np.uint8)
      )'''




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
    

    top_mask = top_mask.unsqueeze(0).repeat(3, 1, 1)
    masked_im = im * top_mask

    masked_curr_data_im = curr_data['im'] * top_mask

    losses['im'] = 0.8 * l1_loss_v1(masked_im, masked_curr_data_im) + 0.2 * (1.0 - calc_ssim(masked_im, masked_curr_data_im))

    variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification
    if not is_initial_timestep:
        is_fg = params['label']==1
        fg_pts = rendervar['means3D'][is_fg]
        fg_rot = rendervar['rotations'][is_fg]
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

    loss_weights = {'im': 0.1, 'rigid': 0.4, 'rot': 0.4, 'iso': 0.2, 'floor': 0.2, 'bg': 2.0, 'soft_col_cons': 0.01,}
                    
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

    is_fg = params['label']==1
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
    is_fg = params['label']==1
    
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
    #if os.path.exists(f"./output/{exp}/{seq}"):
    #    print(f"Experiment '{exp}' for sequence '{seq}' already exists. Exiting.")
    #    return
    md = json.load(open(f"/data3/zihanwa3/Capstone-DSR/Processing_dance/scripts/Dy_train_meta.json", 'r'))  # metadata

    num_timesteps = len(md['fn'])
    params, variables = initialize_params(seq, md)

    optimizer = initialize_optimizer(params, variables)
    output_params = []

    initialize_wandb(exp, seq)

    reversed_range = list(range(0, 5, 3))
    for t in reversed_range:
        t=int(t)
        dataset = get_dataset(t, md, seq, mode='ego_only')
        todo_dataset = []
        is_initial_timestep = (int(t) == 0)
        if not is_initial_timestep:
            params, variables = initialize_per_timestep(params, variables, optimizer)
        num_iter_per_timestep = int(2.3e3) if is_initial_timestep else int(2.1e3)
        progress_bar = tqdm(range(int(num_iter_per_timestep)), desc=f"timestep {t}")
        for i in tqdm(range(num_iter_per_timestep), desc=f"timestep {t}"):
            curr_data = get_batch(todo_dataset, dataset)

            loss, variables, losses = get_loss(params, curr_data, variables, is_initial_timestep, stat_dataset=None, org_params=None)
            loss.backward()
            #variables['means2D'].grad[:38312, :] =  0

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
        save_figs(params, dataset)
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
        from visualize_dyn import visualize
        visualize(sequence, exp_name)