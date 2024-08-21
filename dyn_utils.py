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

#### [t, c, ...]\\
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

def get_window_dataset(t, md, seq, mode='stat_only'):
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

def get_batch_window(todo_dataset, dataset):
    if not todo_dataset:
        todo_dataset = dataset.copy()
    #curr_data = todo_dataset.pop(randint(0, len(todo_dataset) - 1))
    return todo_dataset #[curr_data] todo_dataset#




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

def params2rendervar(params):
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

def add_new_gaussians(params, scene_radius):
    path='/data3/zihanwa3/Capstone-DSR/Processing/3D/aug_person.npz'
    new_pt_cld = np.load(path)["data"]
    print('dyn_len', len(new_pt_cld))
    new_params = initialize_new_params(new_pt_cld)
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
    num_pts = new_pt_cld.shape[0]
    means3D = new_pt_cld[:, :3] # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 4]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
    sq_dist, _ = o3d_knn(new_pt_cld[:, :3], 3)
    mean3_sq_dist = sq_dist.mean(-1).clip(min=0.0000001)
    seg = new_pt_cld[:, 6]  
    params = {
        'means3D': means3D,
        'rgb_colors': new_pt_cld[:, 3:6],
        'unnorm_rotations': unnorm_rots,
        'seg_colors': np.stack((seg, np.zeros_like(seg), 1 - seg), -1),
        'logit_opacities': logit_opacities,
        'log_scales':  np.tile(np.log(np.sqrt(mean3_sq_dist))[..., None], (1, 3)),
    }
    params = {k: torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True)) for k, v in
              params.items()}
    return params

def initialize_window_params(new_pt_cld, window_length=None):
    num_pts = new_pt_cld.shape[0]
    means3D = new_pt_cld[:, :3] # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 4]
    repeated_rots = np.repeat(unnorm_rots[np.newaxis, :, :], window_length, axis=0)
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
    sq_dist, _ = o3d_knn(new_pt_cld[:, :3], 3)
    mean3_sq_dist = sq_dist.mean(-1).clip(min=0.0000001)
    seg = new_pt_cld[:, 6]  
    params = {
        'means3D': means3D,
        'rgb_colors': new_pt_cld[:, 3:6],
        'unnorm_rotations': unnorm_rots,
        'seg_colors': np.stack((seg, np.zeros_like(seg), 1 - seg), -1),
        'logit_opacities': logit_opacities,
        'log_scales':  np.tile(np.log(np.sqrt(mean3_sq_dist))[..., None], (1, 3)),
    }
    params = {k: torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True)) for k, v in
              params.items()}
    return params


def initialize_optimizer(params, variables):
    lrs = {
        'means3D': 0.0000014 * variables['scene_radius'], # 0000014
        'rgb_colors': 0.000028, ###0.0028 will fail
        'unnorm_rotations': 0.0000001,
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




def initialize_params(seq, md, exp, surfix='output'):
    # init_pt_cld_before_dense init_pt_cld
    ckpt_path=f'./{surfix}/no-depth/{seq}/params.npz'
    params = dict(np.load(ckpt_path, allow_pickle=True))
    params = {k: torch.tensor(params[k]).cuda().float().requires_grad_(True) for k in params.keys()}
    cam_centers = np.linalg.inv(md['w2c'][0])[:, :3, 3]  # Get scene radius
    scene_radius = 1.1 * np.max(np.linalg.norm(cam_centers - np.mean(cam_centers, 0)[None], axis=-1))

    variables = {'max_2D_radius': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'scene_radius': scene_radius,
                 'means2D_gradient_accum': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'denom': torch.zeros(params['means3D'].shape[0]).cuda().float()}
    return params, variables, scene_radius
    
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

def initialize_post_first_timestep_temporal(params, variables, optimizer, num_knn=20):
    is_fg = params['label']==1
    init_fg_pts = params['means3D'][0][is_fg]
    init_bg_pts = params['means3D'][0][~is_fg]
    init_bg_rot = torch.nn.functional.normalize(params['unnorm_rotations'][~is_fg])
    neighbor_sq_dist, neighbor_indices = o3d_knn(init_fg_pts.detach().cpu().numpy(), num_knn)
    neighbor_weight = np.exp(-2000 * neighbor_sq_dist)
    neighbor_dist = np.sqrt(neighbor_sq_dist)
    variables["neighbor_indices"] = torch.tensor(neighbor_indices).cuda().long().contiguous()
    variables["neighbor_weight"] = torch.tensor(neighbor_weight).cuda().float().contiguous()
    variables["neighbor_dist"] = torch.tensor(neighbor_dist).cuda().float().contiguous()

    variables["init_bg_pts"] = init_bg_pts.detach()
    variables["init_bg_rot"] = init_bg_rot.detach()
    variables["prev_pts"] = params['means3D'][0].detach()
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
  
def get_scene_prior(prior_path):
  ### priors in a shape of [N_f , N_g, 3]
  priors = load()
  return priors

def load_camera(cam_id, use_ndc=False, resize=512):
    s_id, e_id = int(cam_id)-1, int(cam_id)
    df = pd.read_csv('/data3/zihanwa3/Capstone-DSR/Dynamic3DGaussians/data_ego/cmu_bike/gopro_calibs.csv')[s_id:e_id]

    intrinsics =np.array(df[['image_width','image_height','intrinsics_0','intrinsics_1','intrinsics_2','intrinsics_3']].values.tolist())
    if resize:
      for item in intrinsics:
        w, h = resize, (2160/3840)*resize
        org_width, org_height, fx, fy, cx, cy = item
        ratio = w/org_width
        assert org_width/w==org_height/h
        fx *= ratio
        fy *= ratio
        cx *= ratio
        cy *= ratio
   
    import torch
    q_values = df[['qw_world_cam', 'qx_world_cam', 'qy_world_cam', 'qz_world_cam']].values[0]#[3,0,1,2]
    t_values = df[['tx_world_cam', 'ty_world_cam', 'tz_world_cam']].values[0]
    q_values_tensor = torch.tensor(q_values, dtype=torch.float64)
    t_values_tensor = torch.tensor(t_values, dtype=torch.float64)
    
    f = fx 
    px, py = cx, cy

    focal_length_ndc = torch.tensor([[f, f]], dtype=torch.float32)
    principal_point_ndc = torch.tensor([[px, py]], dtype=torch.float32)


    from pytorch3d.transforms import quaternion_to_matrix, Translate
    import torch


    R = quaternion_to_matrix(q_values_tensor.unsqueeze(0))  
    T = t_values_tensor.unsqueeze(0)
    print(T)
    camera = PerspectiveCameras(
        R = R,
        T = T,
        focal_length=focal_length_ndc,
        principal_point=principal_point_ndc,
        device=torch.device("cpu") 
    )

    return camera