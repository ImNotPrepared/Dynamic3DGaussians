import torch
import os
import json
import copy
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
from dyn_train import MotionBases, feature_bases

near, far = 1e-7, 5e1
#### [t, c, ...]\\

def load_target_tracks(
  self, query_index: int, target_indices: list[int], dim: int = 1
):
  """
  tracks are 2d, occs and uncertainties
  :param dim (int), default 1: dimension to stack the time axis
  return (N, T, 4) if dim=1, (T, N, 4) if dim=0
  """
  q_name = self.frame_names[query_index]
  all_tracks = []
  for ti in target_indices:
      t_name = self.frame_names[ti]
      path = f"{self.tracks_dir}/{q_name}_{t_name}.npy"
      tracks = np.load(path).astype(np.float32)
      all_tracks.append(tracks)
  return torch.from_numpy(np.stack(all_tracks, axis=dim))


def get_dataset(t, md, seq, masks, mode='stat_only'):
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
        depth_path=f'/data3/zihanwa3/Capstone-DSR/Processing/da_v2_disp/{c}/disp_{t}.npz'
        depth = torch.tensor(np.load(depth_path)['depth_map'])
        depth = torch.clamp(depth, min=near, max=far)
        assert depth.shape[1] !=  depth.shape[0]

        mask_path = f'/data3/zihanwa3/Capstone-DSR/Processing/sam_v2_dyn_mask/{c}/dyn_mask_{t}.npz'
        mask = np.load(mask_path)['dyn_mask']


        feature_root_path='/data3/zihanwa3/Capstone-DSR/Processing/dinov2features/resized_512/'
        feature_path = feature_root_path+fn 
        dinov2_feature = torch.tensor(np.load(feature_path.replace('.jpg', '.npy'))).permute(2, 0, 1)


        dataset.append({'cam': cam, 'feature': dinov2_feature, 'im': im, 'id': c-1,  'mask': mask, 'depth': depth, 'vis': True}) 

      print(f'Loaded Dataset of Length {len(dataset)}') 
      return dataset

      
def get_batch(todo_dataset, dataset):
    if not todo_dataset:
        todo_dataset = dataset.copy()
    curr_data = todo_dataset.pop(randint(0, len(todo_dataset) - 1))
    return curr_data


def initialize_params(seq, md, exp):
    # init_pt_cld_before_dense init_pt_cld
    ckpt_path=f'./old_output/no-depth/{seq}/params.npz'
    params = dict(np.load(ckpt_path, allow_pickle=True))
    params = {k: torch.tensor(params[k]).cuda().float().requires_grad_(True) for k in params.keys()}
    cam_centers = np.linalg.inv(md['w2c'][0])[:, :3, 3]  # Get scene radius
    scene_radius = 1.1 * np.max(np.linalg.norm(cam_centers - np.mean(cam_centers, 0)[None], axis=-1))

    variables = {'max_2D_radius': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'scene_radius': scene_radius,
                 'means2D_gradient_accum': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'denom': torch.zeros(params['means3D'].shape[0]).cuda().float()}
    return params, variables, scene_radius


def report_stat_progress(params, t, i, progress_bar, md, every_i=2100):
    import matplotlib.pyplot as plt

    if i % every_i == 0:
        def combine_images(image1, depth1, image2, depth2):
            # Convert depth maps to 3-channel images
            depth1_3channel = cv2.cvtColor(depth1, cv2.COLOR_GRAY2RGB)
            depth2_3channel = cv2.cvtColor(depth2, cv2.COLOR_GRAY2RGB)
            
            # Combine each image and depth side by side
            combined1 = np.hstack((image1, depth1_3channel))
            combined2 = np.hstack((image2, depth2_3channel))
            
            # Combine the two combined images in a 2x2 grid
            combined_all = np.vstack((combined1, combined2))
            return combined_all

        t+=183
        c=0
        h, w = md['hw'][c]
        k, w2c = md['k'][t][c], np.linalg.inv(md['w2c'][t][c])
        cam = setup_camera(w, h, k, w2c, near=0.01, far=50)
        fn = md['fn'][t][c] # mask_{fn.split('/')[0]}
        gt_im = np.array(copy.deepcopy(Image.open(f"/ssd0/zihanwa3/data_ego/cmu_bike/ims/{fn}")))
        gt_im = torch.tensor(gt_im).float().cuda().permute(2, 0, 1) / 255
        gt_im = torch.rot90(gt_im, k=1, dims=(1, 2))
        gt_im = gt_im.permute(1, 2, 0).cpu().numpy() * 255
        gt_im = gt_im.astype(np.uint8)

        gt_im = cv2.resize(gt_im, (256, 256), interpolation=cv2.INTER_LINEAR)
        gt_depth=f'/ssd0/zihanwa3/data_ego/cmu_bike/depth/{int(c)}/depth_{t}.npz'
        gt_depth = torch.tensor(np.load(gt_depth)['depth_map']).float().cuda()  #/ 255
        gt_depth=torch.rot90(gt_depth, k=1, dims=(0, 1))
        
        min_val = torch.min(gt_depth)
        max_val = torch.max(gt_depth)
        gt_depth = (gt_depth - min_val) / (max_val - min_val)
        
        
        # Scale to range [0, 255]
        gt_depth = gt_depth * 255.0
        gt_depth = gt_depth.cpu().numpy() #* 30 #* 255
        ## 

        gt_depth = gt_depth.astype(np.uint8)
        
        gt_depth = cv2.resize(gt_depth, (256, 256), interpolation=cv2.INTER_LINEAR)

        
        im, _, depth, _ = Renderer(raster_settings=cam)(**params2rendervar(params))


                # Process image
        im = torch.rot90(im, k=-1, dims=(1, 2))
        im_wandb = im.permute(1, 2, 0).cpu().numpy() * 255
        im_wandb = im_wandb.astype(np.uint8)
        im_wandb = cv2.resize(im_wandb, (256, 256), interpolation=cv2.INTER_LINEAR)
        
        # Process depth
        depth = torch.rot90(depth, k=-1, dims=(1, 2))
        min_val = torch.min(depth)
        max_val = torch.max(depth)
        depth = (depth - min_val) / (max_val - min_val)


        depth = depth.permute(1, 2, 0).cpu().numpy() * 255
        depth = depth.astype(np.uint8)
        depth = cv2.resize(depth, (256, 256), interpolation=cv2.INTER_LINEAR)
        
        # Combine image and depth
        #print(im_wandb.shape, depth.shape)
        combined = combine_images(gt_im, gt_depth, im_wandb, depth)
        
        # Log combined image
        wandb.log({
            f"ego_{t}": wandb.Image(combined, caption=f"Rendered image and depth at iteration {i}")
        })

        for c in range(1, 5):
            h, w = md['hw'][c]
            k, w2c = md['k'][t][c], np.linalg.inv(md['w2c'][t][c])
            cam = setup_camera(w, h, k, w2c, near=0.01, far=50)
            fn = md['fn'][t][c] # mask_{fn.split('/')[0]}
            
            gt_im = np.array(copy.deepcopy(Image.open(f"/ssd0/zihanwa3/data_ego/cmu_bike/ims/{fn}")))
            gt_im = torch.tensor(gt_im).float().cuda().permute(2, 0, 1) / 255
            gt_im = gt_im.permute(1, 2, 0).cpu().numpy() * 255
            gt_im = gt_im.astype(np.uint8)
            gt_im = cv2.resize(gt_im, (256, 144), interpolation=cv2.INTER_CUBIC)

            gt_depth=f'/ssd0/zihanwa3/data_ego/cmu_bike/depth/{int(c)}/depth_{t}.npz'
            gt_depth = torch.tensor(np.load(gt_depth)['depth_map']).float().cuda()  #/ 255
            gt_depth = gt_depth.cpu().numpy() ##* 30 #* 255

            gt_depth = gt_depth.astype(np.uint8)
            gt_depth = cv2.resize(gt_depth, (256, 144), interpolation=cv2.INTER_CUBIC)

            
            im, _, depth, _ = Renderer(raster_settings=cam)(**params2rendervar(params))
            im=im.clip(0,1)
            im_wandb = im.permute(1, 2, 0).cpu().numpy() * 255
            im_wandb = im_wandb.astype(np.uint8)
            im_wandb = cv2.resize(im_wandb, (256, 144), interpolation=cv2.INTER_CUBIC)
            
            # Process depth
            #depth = torch.rot90(depth, k=-1, dims=(1, 2))
            depth = depth.permute(1, 2, 0).cpu().numpy() * 255
            depth = depth.astype(np.uint8)
            depth = cv2.resize(depth, (256, 144), interpolation=cv2.INTER_CUBIC)
            
            # Combine image and depth
            #print(im_wandb.shape, depth.shape)
            combined = combine_images(gt_im, gt_depth, im_wandb, depth)
         
            # Log combined image
            wandb.log({
                f"stat_combined_{c}_{t}": wandb.Image(combined, caption=f"Rendered image and depth at iteration {i}")
            })

def params2rendervar(params, index=38312):
    ## [org, new_params(person)]
    rendervar = {
        'means3D': params['means3D'],
        'colors_precomp': params['rgb_colors'],
        'rotations': torch.nn.functional.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(params['log_scales']),
        'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device="cuda") + 0,
        'label': params['label'],
        'semantic_feature': params['semantic_feature'],
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
    #path='/data3/zihanwa3/Capstone-DSR/Processing/3D/aug_person.npz'

    path='/data3/zihanwa3/Capstone-DSR/Processing/3D/filtered_person.npz'
    new_pt_cld = np.load(path)["data"]
    print('dyn_len', len(new_pt_cld))
    densified=True
    if densified:
        repeated_pt_cld = []
        for _ in range(7):
            noise = np.random.normal(0, 0.001, new_pt_cld.shape)   
            noisy_pt_cld = new_pt_cld + noise
            repeated_pt_cld.append(noisy_pt_cld)
  
    new_pt_cld = np.vstack(repeated_pt_cld)
    print(new_pt_cld.shape)
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
    index=38312
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
        'semantic_feature': torch.zeros(new_pt_cld.shape[0], 32)
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
        'logit_opacities': 0.002,
        'log_scales': 0.005,
        'semantic_feature':0.003,
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
    losses['im'] = 0
    losses['depth'] = 0 

    im, radius, depth_pred, _ = Renderer(raster_settings=curr_data['cam'])(**rendervar)
    curr_id = curr_data['id']
    im=im.clip(0,1)
    H, W =im.shape[1], im.shape[2]
    top_mask = torch.zeros((H, W), device=im.device)
    top_mask[:, :]=1

    #### MASK
    top_mask = torch.tensor(curr_data['mask'], device=im.device)
    combined_mask = top_mask
    top_mask = combined_mask.type(torch.uint8)
    
    ground_truth_depth = curr_data['depth'].to(im.device)
    depth_mask = top_mask.flatten().bool()
    depth_pred = depth_pred.reshape(-1, 1)[depth_mask]
    depth_pred = torch.clamp(depth_pred, min=near, max=far)
    ground_truth_depth = ground_truth_depth.reshape(-1, 1)[depth_mask]
    depth_pred = 1/depth_pred
    depth_pred = (depth_pred - depth_pred.mean()) / depth_pred.std()
    ground_truth_depth = (ground_truth_depth - ground_truth_depth.mean()) / ground_truth_depth.std()
    #  gt_depth: 1/zoe_depth(metric_depth) -> 1/real_depth; gasussian
    losses['depth'] += (1 - pearson_corrcoef( ground_truth_depth, (depth_pred)))

    #print(top_mask.shape)
    top_mask = top_mask.repeat(3, 1, 1)
    masked_im = im * top_mask

    masked_curr_data_im = curr_data['im'] * top_mask

    losses['im'] = 0.8 * l1_loss_v1(masked_im, masked_curr_data_im) + 0.2 * (1.0 - calc_ssim(masked_im, masked_curr_data_im))

    variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification

    ### (num_points, num_knn)
    if not is_initial_timestep:
        is_fg = params['label']==1
        fg_pts = rendervar['means3D'][is_fg]
        fg_rot = rendervar['rotations'][is_fg]

        rel_rot = quat_mult(fg_rot, variables["prev_inv_rot_fg"])
        rot = build_rotation(rel_rot)
        neighbor_pts = fg_pts[variables["neighbor_indices"]]

        curr_offset = neighbor_pts - fg_pts[:, None]
        # torch.Size([41850, 20, 3]) torch.Size([41850, 3]) torch.Size([41850, 20])   
        #print(neighbor_pts.shape, fg_pts.shape, variables["neighbor_indices"].shape)


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

    loss_weights = {'im': 1.0, 'rigid': 0.4, 'rot': 0.4, 'iso': 0.2, 'floor': 0.2, 'bg': 2.0, 'soft_col_cons': 0.01, 'depth':0.001}
                    
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
    #variables
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
    md = json.load(open(f"./data_ego/{seq}/Dy_train_meta.json", 'r'))  # metadata

    num_timesteps = len(md['fn'])
    params, variables, sriud = initialize_params(seq, md, exp)

    params, variables =  add_new_gaussians(params, variables, sriud)
    optimizer = initialize_optimizer(params, variables)
    output_params = []

    initialize_wandb(exp, seq)
    org_params=initialize_params(seq, md, exp)

    reversed_range = list(range(111, -1, -5))


    means = fg_params['means']
    feats = fg_params['semantic_features']

    coefs, means = feature_bases(means, feats)
    device='cpu'
    num_bases = 49 ### ready_to_tune
    num_frames = len(reversed_range)
    id_rot, rot_dim = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device), 4
    init_rots = id_rot.reshape(1, 1, rot_dim).repeat(num_bases, num_frames, 1)
    init_ts = torch.zeros(num_bases, num_frames, 3, device=device)
    
    bases = MotionBases(init_rots, init_ts) ## [B, F, 3/4]
    transfms = bases.compute_transforms(ts, coefs)
    positions = torch.einsum(
        "pnij,pj->pni",
        transfms,
        F.pad(means, (0, 1), value=1.0),
    )
    ### transfms positions [N, F, 3]

    for t in reversed_range:

        xyz_t =  positions[:, t]
        t=int(t)
        dataset = get_dataset(t, md, seq, masks=None, mode='ego_only')
        stat_dataset = None
        todo_dataset = []
        is_initial_timestep = (int(t) == 111)
        if not is_initial_timestep:
            params, variables = initialize_per_timestep(params, variables, optimizer)


        num_iter_per_timestep = int(4.9e3) if is_initial_timestep else int(2.1e3)
        progress_bar = tqdm(range(int(num_iter_per_timestep)), desc=f"timestep {t}")
        for i in tqdm(range(num_iter_per_timestep), desc=f"timestep {t}"):
            curr_data = get_batch(todo_dataset, dataset)

            loss, variables, losses = get_loss(params, curr_data, variables, is_initial_timestep, stat_dataset=stat_dataset, org_params=None)
            loss.backward()

            with torch.no_grad():
                #report_progress(params, dataset[0], i, progress_bar)
                #report_stat_progress(params, t, i, progress_bar, md)
                #if is_initial_timestep:
                #    params, variables = densify(params, variables, optimizer, i)
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
        #from visualize import visualize
        #visualize(sequence, exp_name)
        from visualize_dyn import visualize
        visualize(sequence, exp_name)
