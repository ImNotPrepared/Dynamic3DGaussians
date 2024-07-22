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
    o3d_knn, params2cpu, save_params, save_params_progressively
from external import calc_ssim, calc_psnr, build_rotation, densify, update_params_and_optimizer
import cv2
import torch.nn.functional as F
from torchmetrics.functional.regression import pearson_corrcoef
import torchvision.transforms as transforms
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

def get_dataset(t, md, seq, mode='stat_only'):
    dataset = []
    t += 1111
    def get_jpg_filenames(directory):
        jpg_files = [int(file.split('.')[0]) for file in os.listdir(directory) if file.endswith('.jpg')]
        return jpg_files

    # Specify the directory containing the .jpg files   precise_reduced_im
                # /ssd0/zihanwa3/data_ego/nice100 '/data3/zihanwa3/Capstone-DSR/Appendix/lalalal_new'# /data3/zihanwa3/Capstone-DSR/Appendix/nice10
    dino_mask=False
    if dino_mask:
      directory = '/data3/zihanwa3/Capstone-DSR/Appendix/SR300'
    
    else:
      directory = '/data3/zihanwa3/Capstone-DSR/Appendix/SR_7_pls'
    jpg_filenames = get_jpg_filenames(directory)


    vis_directory = '/data3/zihanwa3/Capstone-DSR/Appendix/SR10'
    vis_filenames = get_jpg_filenames(vis_directory)

    
    if mode=='ego_only':

      t=0

      epsilon=1e-7
      for lis in [jpg_filenames]: # , jpg_filenames_2, jpg_filenames_3
        #print(sorted(lis)[:30])
        for iiiindex, c in sorted(enumerate(lis)):
            h, w = md['hw'][c]
            k, w2c =  md['k'][t][c], np.linalg.inv(md['w2c'][t][c])
            cam = setup_camera(w, h, k, w2c, near=0.01, far=50)
            fn = md['fn'][t][c] # mask_{fn.split('/')[0]}
            if dino_mask:
              mask_path=f"/ssd0/zihanwa3/data_ego/lalalal_newmask/{fn.split('/')[-1]}"
            else:
              mask_path=f"/ssd0/zihanwa3/data_ego/SR_7_mask/{fn.split('/')[-1].replace('.jpg', '.png')}"
            mask = Image.open(mask_path).convert("L")
            transform = transforms.ToTensor()
            mask_tensor = transform(mask).squeeze(0)
            anti_mask_tensor = mask_tensor > 1e-5
            #if c in vis_filenames:
            im = np.array(copy.deepcopy(Image.open(f"/ssd0/zihanwa3/data_ego/{seq}/ims/{fn}")))
            im = torch.tensor(im).float().cuda().permute(2, 0, 1) / 255

            im = torch.rot90(im, k=1, dims=(1, 2))

            '''mask_path=f'/ssd0/zihanwa3/data_ego/cmu_bike/depth/{int(t)}/depth_{c}.npz'
            depth = torch.tensor(np.load(mask_path)['depth_map']).float().cuda()
            depth = torch.rot90(depth, k=1, dims=(0, 1))
            depth = torch.clamp(depth, min=epsilon)
            depth=1/(depth+100.)'''

            anti_mask_tensor=torch.rot90(anti_mask_tensor, k=1, dims=(0, 1))
            #dataset.append({'cam': cam, 'im': im, 'id': iiiindex, 'antimask': anti_mask_tensor, 'gt_depth':depth, 'vis': True})  
            dataset.append({'cam': cam, 'im': im, 'id': iiiindex, 'antimask': anti_mask_tensor, 'vis': True})  

      for c in range(1400, 1404):
          h, w = md['hw'][c]

          k, w2c =  md['k'][t][c], np.linalg.inv(md['w2c'][t][c])
          cam = setup_camera(w, h, k, w2c, near=0.01, far=50)

          fn = md['fn'][t][c]
          im = np.array(copy.deepcopy(Image.open(f"/ssd0/zihanwa3/data_ego/{seq}/ims/{fn}")))
          im = torch.tensor(im).float().cuda().permute(2, 0, 1) / 255
          #print(im.max(),im.min())
          im=im.clip(0,1)
          
          #print(im.shape)
          ############################## First Frame Depth #############################


          '''mask_path=f'/data3/zihanwa3/Capstone-DSR/Appendix/dust3r/patched_stat_imgs/adhoc_depth/{c-1399}.npz'
          depth = torch.tensor(np.load(mask_path)['depth']).float().cuda()
          depth=1/(depth+100)'''
          #np.savez_compressed(, depth_map=new_depth)
          #dataset.append({'cam': cam, 'im': im, 'id': c-1400+100, 'gt_depth':depth, 'vis': True})  
          dataset.append({'cam': cam, 'im': im, 'id': c-1400+100, 'vis': True})  

      return dataset



def get_batch(todo_dataset, dataset):
    if not todo_dataset:
        todo_dataset = dataset.copy()
    curr_data = todo_dataset.pop(randint(0, len(todo_dataset) - 1))
    return [curr_data] #[curr_data] todo_dataset#


    
def initialize_params(seq, md, init_pt_path):
    # init_pt_cld_before_dense init_pt_cld

    size=512
    init_type='fused'
    #init_pt_path=f'/data3/zihanwa3/Capstone-DSR/Appendix/dust3r/duster_{size}_scene.npz'
    #init_pt_path='/data3/zihanwa3/Capstone-DSR/Appendix/dust3r/good_.npz'
    if init_type=='dust':
      init_pt_path='/data3/zihanwa3/Capstone-DSR/Appendix/dust3r/patched_stat_imgs/pc.npz'    
      init_pt_cld = np.load(init_pt_path)["data"]
      #init_pt_cld = np.concatenate((init_pt_cld, init_pt_cld), axis=0)
      print(len(init_pt_cld))
      seg = init_pt_cld[:, 6]
      max_cams = 305

      intrinsics = [
          [1764.094727, 1764.094727, 1920.0, 1080.0],
          [1774.26709, 1774.26709, 1920.0, 1080.0],
          [1764.426025, 1764.426025, 1920.0, 1080.0],
          [1783.065308, 1783.065308, 1920.0, 1080.0]
      ]
      mean3_sq_dist=[]
      #for c in range(1,5):
      #  mask_path=f'/data3/zihanwa3/Capstone-DSR/Processing/filled_complete/{size}/{int(c)-1}.npz'
      for c in range(4):
        mask_path = f'/data3/zihanwa3/Capstone-DSR/Appendix/dust3r/patched_stat_imgs/adhoc_depth/{c+1}.npz'
        depth = torch.tensor(np.load(mask_path)['depth']).float().cuda()
        depth = depth.unsqueeze(0).unsqueeze(0)
        if c<4:
          depth_resized = F.interpolate(depth, size=(int(288*size/512), size), mode='bilinear', align_corners=False)
        else:
          depth_resized = F.interpolate(depth, size=(size, size), mode='bilinear', align_corners=False)
        depth = depth_resized.squeeze(0).squeeze(0)
        print(c)
        scale_gaussian = 4.7*depth / ((intrinsics[c][0] + intrinsics[c][1])/2)
        mean3_sq_dist.append((scale_gaussian**2).flatten().cpu().numpy()) # [H, W] * 4

      mean3_sq_dist=np.concatenate(mean3_sq_dist)
      
    elif init_type=='ego4d':
      init_pt_cld = np.load(f"./data_ego/{seq}/init_correct.npz")["data"]
      #init_pt_cld = np.concatenate((init_pt_cld, init_pt_cld), axis=0)
      print(len(init_pt_cld))
      seg = init_pt_cld[:, 6]
      max_cams = 305
      sq_dist, _ = o3d_knn(init_pt_cld[:, :3], 3)
      mean3_sq_dist = sq_dist.mean(-1).clip(min=0.0000001, max=0.1)
    elif init_type=='fused':
      init_pt_path='/data3/zihanwa3/Capstone-DSR/Appendix/dust3r/patched_stat_imgs/pc.npz'
      org_pt_path=f"./data_ego/{seq}/init_correct.npz"
      org_pt_cld = np.load(org_pt_path)["data"]
      init_pt_cld = np.load(init_pt_path)["data"]
      init_pt_cld = np.concatenate((init_pt_cld, org_pt_cld), axis=0)
      seg = init_pt_cld[:, 6]
      max_cams = 305
      sq_dist, _ = o3d_knn(init_pt_cld[:, :3], 3)
      mean3_sq_dist = sq_dist.mean(-1).clip(min=0.0000001, max=0.01)
    #print(mean3_sq_dist.shape)
    #if gaussian_distribution == "isotropic":
    #  log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1))
    #elif gaussian_distribution == "anisotropic":
    #log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 3))

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

def initialize_optimizer(params, variables):
    lrs = {
        'means3D': 0.000014 *  variables['scene_radius'], # 0000014
        'rgb_colors': 0.00028, ###0.0028 will fail
        'seg_colors': 0.0,
        'unnorm_rotations': 0.001,
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


def get_loss(params, curr_datasss, variables, is_initial_timestep, stat_dataset=None):
    losses = {}
    losses['im'] = 0
    rendervar = params2rendervar(params)
    rendervar['means2D'].retain_grad()

    for curr_data in curr_datasss: 

      im, radius, depth_pred, _ = Renderer(raster_settings=curr_data['cam'])(**rendervar)

      curr_id = curr_data['id']
      #print(params['cam_m'][curr_id], params['cam_c'][curr_id])
      #im = torch.exp(params['cam_m'][curr_id])[:, None, None] * im + params['cam_c'][curr_id][:, None, None]
      im=im.clip(0,1)
      #print(im.max(),im.min())
      #print('-'*100)
      H, W =im.shape[1], im.shape[2]
      top_mask = torch.zeros((H, W), device=im.device)
      top_mask[:, :]=1 
      top_mask = top_mask.unsqueeze(0).repeat(3, 1, 1)
      masked_im = im * top_mask

      masked_curr_data_im = curr_data['im'] * top_mask

      losses['im'] += 0.8 * l1_loss_v1(masked_im, masked_curr_data_im) + 0.2 * (1.0 - calc_ssim(masked_im, masked_curr_data_im))
    losses['im'] /= len(curr_datasss)
    #losses['depth'] /= len(curr_datasss)


    variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification

    loss_weights = {'im': 0.1, 'rigid': 0.0, 'rot': 0.0, 'iso': 0.0, 'floor': 0.0, 'bg': 2.0, 'depth': 0.000,
                    'soft_col_cons': 0.00}
                    
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

def report_stat_progress(params, stat_dataset, i, progress_bar, md, every_i=1400):
    import matplotlib.pyplot as plt
    if i % every_i == 0:
        def get_jpg_filenames(directory):
            jpg_files = [int(file.split('.')[0]) for file in os.listdir(directory) if file.endswith('.jpg')]
            return jpg_files

        # Specify the directory containing the .jpg files   precise_reduced_im
                    # /ssd0/zihanwa3/data_ego/nice100 '/data3/zihanwa3/Capstone-DSR/Appendix/lalalal_new'#
        directory = '/data3/zihanwa3/Capstone-DSR/Appendix/SR10'
        jpg_filenames = get_jpg_filenames(directory)

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

        # Your existing loop
        for item, c in enumerate(jpg_filenames):

            t = 0
            h, w = md['hw'][c]
            k, w2c = md['k'][t][c], np.linalg.inv(md['w2c'][t][c])
            cam = setup_camera(w, h, k, w2c, near=0.01, far=50)
            fn = md['fn'][t][c] # mask_{fn.split('/')[0]}
            gt_im = np.array(copy.deepcopy(Image.open(f"/ssd0/zihanwa3/data_ego/cmu_bike/ims/{fn}")))
            gt_im = torch.tensor(gt_im).float().cuda().permute(2, 0, 1) / 255
            gt_im = torch.rot90(gt_im, k=1, dims=(1, 2))
            mask_path='/data3/zihanwa3/Capstone-DSR/Dynamic3DGaussians/data_ego/masked_cmu_bike/triangular_mask.jpg'
            default_mask= torch.tensor(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE), device=gt_im.device)
            default_mask = default_mask>1e-5

            mask_path=f"/ssd0/zihanwa3/data_ego/lalalal_newmask/{fn.split('/')[-1]}"
            mask = Image.open(mask_path).convert("L")
            transform = transforms.ToTensor()
            mask_tensor = transform(mask).squeeze(0).to(gt_im.device)
            antimask = mask_tensor > 1e-5
            combined_mask = ~torch.logical_or(default_mask, antimask)
            top_mask = combined_mask.type(torch.uint8)
            top_mask=torch.rot90(top_mask, k=1, dims=(0, 1))

            gt_im *= top_mask
            gt_im = torch.rot90(gt_im, k=-1, dims=(1, 2))
            gt_im = gt_im.permute(1, 2, 0).cpu().numpy() * 255
            gt_im = gt_im.astype(np.uint8)

            gt_im = cv2.resize(gt_im, (256, 256), interpolation=cv2.INTER_LINEAR)

            gt_depth=f'/ssd0/zihanwa3/data_ego/cmu_bike/depth/{int(t)}/depth_{c}.npz'
            gt_depth = torch.tensor(np.load(gt_depth)['depth_map']).float().cuda()  #/ 255
            gt_depth=torch.rot90(gt_depth, k=1, dims=(0, 1))
            
            min_val = torch.min(gt_depth)
            max_val = torch.max(gt_depth)
            gt_depth = (gt_depth - min_val) / (max_val - min_val)
            
            
            # Scale to range [0, 255]
            gt_depth = gt_depth * 255.0
            gt_depth *= top_mask
            gt_depth=torch.rot90(gt_depth, k=-1, dims=(0, 1))

            gt_depth = gt_depth.cpu().numpy() #* 30 #* 255
            ## 

            gt_depth = gt_depth.astype(np.uint8)
            
            gt_depth = cv2.resize(gt_depth, (256, 256), interpolation=cv2.INTER_LINEAR)

            
            im, _, depth, _ = Renderer(raster_settings=cam)(**params2rendervar(params))
            
            im=im.clip(0,1)


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
                f"held_out_combined_{item}": wandb.Image(combined, caption=f"Rendered image and depth at iteration {i}")
            })

        for c in range(1400, 1404):
            h, w = md['hw'][c]
            k, w2c = md['k'][t][c], np.linalg.inv(md['w2c'][t][c])
            cam = setup_camera(w, h, k, w2c, near=0.01, far=50)
            fn = md['fn'][t][c] # mask_{fn.split('/')[0]}
            
            gt_im = np.array(copy.deepcopy(Image.open(f"/ssd0/zihanwa3/data_ego/cmu_bike/ims/{fn}")))
            gt_im = torch.tensor(gt_im).float().cuda().permute(2, 0, 1) / 255
            gt_im = gt_im.permute(1, 2, 0).cpu().numpy() * 255
            gt_im = gt_im.astype(np.uint8)
            gt_im = cv2.resize(gt_im, (256, 144), interpolation=cv2.INTER_CUBIC)

            gt_depth=f'/ssd0/zihanwa3/data_ego/cmu_bike/depth/{int(c)-1399}/depth_0.npz'
            gt_depth = torch.tensor(np.load(gt_depth)['depth_map']).float().cuda()  #/ 255
            #min_val = torch.min(gt_depth)
            #max_val = torch.max(gt_depth)
            #gt_depth = (gt_depth - min_val) / (max_val - min_val)

            # Scale to range [0, 255]
            #gt_depth = gt_depth * 255.0

            gt_depth = gt_depth.cpu().numpy() #* 30 #* 255
            ## 

            gt_depth = gt_depth.astype(np.uint8)
            gt_depth = cv2.resize(gt_depth, (256, 144), interpolation=cv2.INTER_CUBIC)

            
            im, _, depth, _ = Renderer(raster_settings=cam)(**params2rendervar(params))
            im=im.clip(0,1)
            
            # Process image
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
                f"stat_combined_{c-1399}": wandb.Image(combined, caption=f"Rendered image and depth at iteration {i}")
            })


def report_progress(params, data, i, progress_bar, every_i=100):
    if i % every_i == 0:
        im, _, _, _ = Renderer(raster_settings=data['cam'])(**params2rendervar(params))
        curr_id = data['id']
        im = torch.exp(params['cam_m'][curr_id])[:, None, None] * im + params['cam_c'][curr_id][:, None, None]
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
    md = json.load(open(f"./data_ego/{seq}/train_meta.json", 'r'))  # metadata
    num_timesteps = len(md['fn'])

    #/data3/zihanwa3/Capstone-DSR/Appendix/dust3r/duster_512_scene.npz

    init_path=False
    params, variables = initialize_params(seq, md, init_path)
    optimizer = initialize_optimizer(params, variables)
    output_params = []
    

    initialize_wandb(exp, seq)

    
    for t in range(7):
        dataset = get_dataset(t, md, seq, mode='ego_only')
        stat_dataset = None
        todo_dataset = []

        is_initial_timestep = (t == 0)
        if not is_initial_timestep:
            params, variables = initialize_per_timestep(params, variables, optimizer)

        num_iter_per_timestep = int(1.7e4) if is_initial_timestep else 2
        progress_bar = tqdm(range(num_iter_per_timestep), desc=f"timestep {t}")
        for i in range(num_iter_per_timestep):
            progressive_params = []
            curr_data = get_batch(todo_dataset, dataset)

            loss, variables, losses = get_loss(params, curr_data, variables, is_initial_timestep, stat_dataset=stat_dataset)
            loss.backward()
            with torch.no_grad():
                report_progress(params, dataset[0], i, progress_bar)
                report_stat_progress(params, curr_data, i, progress_bar,md)
                if is_initial_timestep:
                    params, variables = densify(params, variables, optimizer, i)
                assert ((params['means3D'].shape[0]==0) is False)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for key, value in losses.items():
              wandb.log({key: value.item(), "iteration": i})

            progressive_iter=2e3
            if i % progressive_iter == 0:
              progressive_params.append(params2cpu(copy.deepcopy(params), is_initial_timestep))
              save_params_progressively(progressive_params, seq, exp, i)
            
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