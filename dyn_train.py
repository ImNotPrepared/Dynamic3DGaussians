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
    o3d_knn, params2rendervar, params2cpu, save_params
from external import calc_ssim, calc_psnr, build_rotation, densify, update_params_and_optimizer
import cv2
from torchmetrics.functional.regression import pearson_corrcoef
import torchvision.transforms as transforms

def report_stat_progress(params, t, i, progress_bar, md, every_i=700):
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

        
        im, _, depth = Renderer(raster_settings=cam)(**params2rendervar(params))


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
            f"ego": wandb.Image(combined, caption=f"Rendered image and depth at iteration {i}")
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

            
            im, _, depth = Renderer(raster_settings=cam)(**params2rendervar(params))
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
                f"stat_combined_{c}": wandb.Image(combined, caption=f"Rendered image and depth at iteration {i}")
            })

#### [t, c, ...]
def get_dataset(t, md, seq, mode='stat_only'):
    dataset = []
    t+=183
    if mode=='ego_only':
      for c in range(5):
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
            mask_path=f'/data3/zihanwa3/Capstone-DSR/Processing/toy_exp/depth/depth_{c}.npz'
            depth = torch.tensor(np.load(mask_path)['depth_map']).float().cuda()
            depth=1/(depth+100.)

          try:
            mask_path=f"/data3/zihanwa3/Capstone-DSR/Processing/toy_exp/mask/cam_{c}.png"
            mask = Image.open(mask_path).convert("L")
            transform = transforms.ToTensor()
            mask_tensor = transform(mask).squeeze(0)
            anti_mask_tensor = mask_tensor > 1e-5
            dataset.append({'cam': cam, 'im': im, 'id': iiiindex, 'antimask': anti_mask_tensor, 'gt_depth':depth, 'vis': True}) 
          except: 
            dataset.append({'cam': cam, 'im': im, 'id': c, 'vis': True})  
      return dataset
def get_batch(todo_dataset, dataset):
    if not todo_dataset:
        todo_dataset = dataset.copy()
    curr_data = todo_dataset.pop(randint(0, len(todo_dataset) - 1))
    return curr_data


def initialize_params(seq, md, exp):
    # init_pt_cld_before_dense init_pt_cld
    ckpt_path=f'./output/+100depth_0.1/{seq}/params.npz'
    params = dict(np.load(ckpt_path, allow_pickle=True))
    params = {k: torch.tensor(params[k]).cuda().float().requires_grad_(True) for k in params.keys()}
    cam_centers = np.linalg.inv(md['w2c'][0])[:, :3, 3]  # Get scene radius
    scene_radius = 1.1 * np.max(np.linalg.norm(cam_centers - np.mean(cam_centers, 0)[None], axis=-1))

    variables = {'max_2D_radius': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'scene_radius': scene_radius,
                 'means2D_gradient_accum': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'denom': torch.zeros(params['means3D'].shape[0]).cuda().float()}
    return params, variables, scene_radius

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
        
      #print(params_no_grad.shape, v.shape)
      new_params[k] = torch.cat((params_no_grad, v), dim=0)
    for k, v in new_params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            new_params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            new_params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))
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
    for k, v in params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

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
    param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items()]
    return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)

def initialize_org_params(seq, md, exp):
    # init_pt_cld_before_dense init_pt_cld
    ckpt_path=f'./output/{exp}/{seq}/params.npz'
    params = dict(np.load(ckpt_path, allow_pickle=True))
    params = {k: torch.tensor(params[k]).cuda().float().requires_grad_(True) for k in params.keys()}
    return params

def get_loss(params, curr_data, variables, is_initial_timestep, stat_dataset=None, org_params=None):
    losses = {}
    rendervar = params2rendervar(params)
    rendervar['means2D'].retain_grad()


    im, radius, depth_pred, = Renderer(raster_settings=curr_data['cam'])(**rendervar)


    def rgb_to_grayscale(tensor):
        # Ensure the input tensor is of shape (N, C, H, W)
        
        # Extract the R, G, B channels
        r, g, b = tensor[0, :, :], tensor[1, :, :], tensor[2, :, :]

        # Use the luminosity method to convert to grayscale
        gray_tensor = 0.2989 * r + 0.5870 * g + 0.1140 * b

        return gray_tensor

    if im.shape[1]==192:
      im = rgb_to_grayscale(im)
      im=im.unsqueeze(0).repeat(3, 1, 1)

    curr_id = curr_data['id']
    #print(params['cam_m'][curr_id], params['cam_c'][curr_id])
    #im = torch.exp(params['cam_m'][curr_id])[:, None, None] * im + params['cam_c'][curr_id][:, None, None]
    im=im.clip(0,1)

    H, W =im.shape[1], im.shape[2]
    top_mask = torch.zeros((H, W), device=im.device)
    top_mask[:, :]=1
    #c_data=curr_data['im']

    if 'antimask' in curr_data.keys():
        antimask=curr_data['antimask'].to(params['cam_c'][curr_id].device)
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

        #print(depth_pred.shape,  top_mask.shape)
        #print(depth_pred.shape, ground_truth_depth.shape)
        #  gt_depth: 1/zoe_depth(metric_depth) -> 1/real_depth; gasussian
        losses['depth'] =   (1 - pearson_corrcoef( ground_truth_depth, 1/(depth_pred+100)))

    #l1_loss_v1(ground_truth_depth, depth_pred)

    top_mask = top_mask.unsqueeze(0).repeat(3, 1, 1)
    masked_im = im * top_mask

    masked_curr_data_im = curr_data['im'] * top_mask

    losses['im'] = 0.8 * l1_loss_v1(masked_im, masked_curr_data_im) + 0.2 * (1.0 - calc_ssim(masked_im, masked_curr_data_im))

    #losses['stat_im'],  =held_stat_loss(stat_dataset, depth_losses)

    variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification

    segrendervar = params2rendervar(params)
    segrendervar['colors_precomp'] = params['seg_colors']
    seg, _, _, = Renderer(raster_settings=curr_data['cam'])(**segrendervar)
    ##losses['seg'] = 0.8 * l1_loss_v1(seg, curr_data['seg']) + 0.2 * (1.0 - calc_ssim(seg, curr_data['seg']))


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

def report_progress(params, data, i, progress_bar, every_i=100):
    if i % every_i == 0:
        im, _, _, = Renderer(raster_settings=data['cam'])(**params2rendervar(params))
        curr_id = data['id']
        im=im.clip(0,1)
        #im = torch.exp(params['cam_m'][curr_id])[:, None, None] * im + params['cam_c'][curr_id][:, None, None]
        psnr = calc_psnr(im, data['im']).mean()
        progress_bar.set_postfix({"train img 0 PSNR": f"{psnr:.{7}f}"})
        progress_bar.update(every_i)

        wandb.log({"PSNR": psnr.item(), "iteration": i})
        
        # Log images to wandb
        im_wandb = im.permute(1, 2, 0).cpu().numpy() * 255
        im_wandb = im_wandb.astype(np.uint8)
        wandb.log({"rendered_image": wandb.Image(im_wandb, caption=f"Rendered image at iteration {i}")})

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

    
    for t in reversed(range(111)):
        t=int(t)
        dataset = get_dataset(t, md, seq, mode='ego_only')
        stat_dataset = None
        todo_dataset = []
        is_initial_timestep = (int(t) == 110)
        if not is_initial_timestep:
            params, variables = initialize_per_timestep(params, variables, optimizer)
        num_iter_per_timestep = int(2.8e3) if is_initial_timestep else int(2.1e3)
        progress_bar = tqdm(range(int(num_iter_per_timestep)), desc=f"timestep {t}")
        for i in range(num_iter_per_timestep):
            curr_data = get_batch(todo_dataset, dataset)

            loss, variables, losses = get_loss(params, curr_data, variables, is_initial_timestep, stat_dataset=stat_dataset, org_params=org_params)
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


if __name__ == "__main__":
    import argparse
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument('--exp_name', type=str, required=True, help='Name of the experiment')
    args = parser.parse_args()

    exp_name = args.exp_name
    #for sequence in ["basketball", "boxes", "football", "juggle", "softball", "tennis"]:
    for sequence in ["cmu_bike"]:
        train(sequence, exp_name, )
        torch.cuda.empty_cache()
        from visualize import visualize
        visualize(sequence, exp_name)
        from visualize_dyn import visualize
        visualize(sequence, exp_name)
