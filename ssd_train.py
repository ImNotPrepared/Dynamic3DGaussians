import torch
import os
import json
import copy
import numpy as np
from PIL import Image
import open3d as o3d
from random import randint
from tqdm import tqdm
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
import matplotlib.pyplot as plt
from gsplat.rendering import rasterization_inria_wrapper
from helpers import setup_camera, l1_loss_v1, l1_loss_v2, weighted_l2_loss_v1, weighted_l2_loss_v2, quat_mult, \
    o3d_knn, params2cpu, save_params, save_params_progressively
from external import calc_ssim, calc_psnr, build_rotation, densify, update_params_and_optimizer
import cv2
import torch.nn.functional as F
from torchmetrics.functional.regression import pearson_corrcoef
import torchvision.transforms as transforms

from ideaII import flow_loss

near, far = 1e-7, 7e0

def nearest_densify(org_pt_cld, init_pt_cld):
  ### Org, Densified
  org_o3d_pcd = o3d.geometry.PointCloud()
  org_o3d_pcd.points = o3d.utility.Vector3dVector(org_pt_cld[:, :3])  # Only use the first 3 dims for KDTree

  init_o3d_pcd = o3d.geometry.PointCloud()
  init_o3d_pcd.points = o3d.utility.Vector3dVector(init_pt_cld[:, :3]) 

  # Build the KDTree for the initial point cloud using Open3D
  kdtree = o3d.geometry.KDTreeFlann(init_o3d_pcd)
  densified_points=[]
  # Find the 5 closest points in init_pt_cld for each point in org_pt_cld
  for i, org_point in enumerate(org_o3d_pcd.points):
      [_, idx, _] = kdtree.search_knn_vector_3d(org_point, 14)
      closest_points = init_pt_cld[idx]
      densified_points.append(closest_points)


  return np.concatenate(densified_points)



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

def get_dataset(t, md, seq, mode='stat_only', clean_img=True, depth_loss=False, debug_mode='no'):
    dataset = []
    json_dataset = []
    t += 1111
    def get_jpg_filenames(directory):
        jpg_files = [int(file.split('.')[0]) for file in os.listdir(directory) if file.endswith('.jpg')]
        return jpg_files

    # Specify the directory containing the .jpg files   precise_reduced_im
                # /ssd0/zihanwa3/data_ego/nice100 '/data3/zihanwa3/Capstone-DSR/Appendix/lalalal_new'# /data3/zihanwa3/Capstone-DSR/Appendix/nice10
    dino_mask=True
    if dino_mask:
      directory = '/data3/zihanwa3/Capstone-DSR/Appendix/SR_49' #SR_49-bikefree
      #directory = '/data3/zihanwa3/Capstone-DSR/Appendix/SR_7_noMask'
    
    else:
      directory = '/data3/zihanwa3/Capstone-DSR/Appendix/SR_7_pls'

    #directory='/data3/zihanwa3/Capstone-DSR/Appendix/SR_7_noMask'
    jpg_filenames = get_jpg_filenames(directory)

    scales_shifts = [(0.0031744423, 0.15567338), (0.0025279315, 0.106763005), (0.0048902677, 0.16312718), (0.0037271702, 0.10191789), (0.002512292, 0.114545256), (0.0029944833, 0.10527076), (0.003602787, 0.14336547), (0.003638356, 0.1080856), (0.0054704025, 0.057398915), (0.0022690576, 0.117439255), (0.002312136, 0.077383846), (0.0054023797, 0.089054525), (0.0050647566, 0.101514965), (0.0036501177, 0.13153434), (0.0008889911, 0.44202688), (0.0025493288, 0.109814465), (0.0024664444, 0.112163335), (-0.00016438629, 0.40732577), (0.0032442464, 0.19807495), (0.0048282435, 0.09168023), (0.002856112, 0.15053965), (0.0020215507, 0.107855394), (0.0030028797, 0.14278293), (0.0024490638, 0.13038686), (0.0024990174, 0.12481204), (0.0057816333, 0.077005506), (0.0019591942, 0.10089706), (0.0013262086, 0.42674613), (0.004126527, 0.13687198), (0.0022844346, 0.097172886), (0.0062575513, 0.12489089), (-0.00014962265, 0.38713253), (0.00086679566, 0.25387546), (0.0021814466, 0.10047534), (0.002019625, 0.10706337), (0.0037505955, 0.13279462), (0.0035237654, 0.12734117), (0.0019494797, 0.14369084), (0.00056177535, 0.28072894), (0.0018662697, 0.10288732), (0.00591453, 0.053784877), (0.002294414, 0.23004633), (0.0014106235, 0.14460064), (0.0013034015, 0.24912238), (0.0015928176, 0.17974892)]
    
    if mode=='ego_only':

      t=0

      epsilon=1e-7
      if debug_mode == 'no' or debug_mode == 'ego':
        for lis in [jpg_filenames]: # , jpg_filenames_2, jpg_filenames_3
          for iiiindex, c in sorted(enumerate(lis)):
              h, w = md['hw'][c]
              k, w2c =  md['k'][t][c], np.linalg.inv(md['w2c'][t][c])
              cam = setup_camera(w, h, k, w2c, near=near, far=far)
              fn = md['fn'][t][c] # mask_{fn.split('/')[0]}
              if clean_img:
                if dino_mask:
                  mask_path=f"/ssd0/zihanwa3/data_ego/lalalal_newmask/{fn.split('/')[-1]}"
                  mask = Image.open(mask_path).convert("L")
                else:
                  mask_path=f"/ssd0/zihanwa3/data_ego/SR_7_mask/{fn.split('/')[-1].replace('.jpg', '.png')}"
                  mask = Image.open(mask_path).convert("L")

                mask_path=f"/data3/zihanwa3/Capstone-DSR/Appendix/SR_49_clean_mask/{fn.split('/')[-1].replace('.jpg', '.npz')}"
                mask = np.load(mask_path)['mask'][0]

              transform = transforms.ToTensor()
              mask_tensor = transform(mask).squeeze(0)
              anti_mask_tensor = mask_tensor > 1e-5

              if clean_img:
                im = np.array(copy.deepcopy(Image.open(f"/ssd0/zihanwa3/data_ego/{seq}/ims/undist_data/{fn}")))
                im = torch.tensor(im).float().cuda().permute(2, 0, 1) / 255

              if depth_loss:
                depth_path=f'/data3/zihanwa3/Capstone-DSR/Processing/da_v2_disp/0/disp_{c}.npz'
                depth = torch.tensor(np.load(depth_path)['depth_map'])
                #depth = torch.clamp(depth, min=near, max=far)


                scale, shift = scales_shifts[iiiindex]
                disp_map = depth
                nonzero_mask = disp_map != 0
                disp_map[nonzero_mask] = disp_map[nonzero_mask] * scale + shift
                valid_depth_mask = (disp_map > 0) & (disp_map <= far)
                disp_map[~valid_depth_mask] = 0
                depth_map = np.full(disp_map.shape, np.inf)
                depth_map[disp_map != 0] = 1 / disp_map[disp_map != 0]
                depth_map[depth_map == np.inf] = 0
                depth_map = depth_map.astype(np.float32)
                depth = torch.tensor(depth_map)
                
                #print(depth.shape)
                assert depth.shape[1] ==  depth.shape[0]
                dataset.append({'cam': cam, 'im': im, 'id': 'ssss', 'antimask': anti_mask_tensor, 'depth': depth, 'vis': True})  
                
              else:
                dataset.append({'cam': cam, 'im': im, 'id': iiiindex, 'antimask': anti_mask_tensor, 'vis': True}) 

      if debug_mode == 'no' or debug_mode == 'stat':
        for c in range(1400, 1404):
            h, w = md['hw'][c]
          

            k, w2c =  md['k'][t][c], np.linalg.inv(md['w2c'][t][c])
            cam = setup_camera(w, h, k, w2c, near=near, far=far)

            fn = md['fn'][t][c]
            im = np.array(copy.deepcopy(Image.open(f"/ssd0/zihanwa3/data_ego/{seq}/ims/{fn}")))
            im = torch.tensor(im).float().cuda().permute(2, 0, 1) / 255
            #print(im.max(),im.min())
            im=im.clip(0,1)
            
            ############################## First Frame Depth #############################
            if depth_loss:
              depth_path=f'/data3/zihanwa3/Capstone-DSR/Processing/da_v2_disp/0/disp_{c}.npz'
              depth = torch.tensor(np.load(depth_path)['depth_map'])
              #depth = torch.clamp(depth, min=near, max=far)
              assert depth.shape[1] !=  depth.shape[0]

              ################DOING REGULAR VIS #################
              scale, shift = scales_shifts[c-1404]
              disp_map = depth
              nonzero_mask = disp_map != 0
              disp_map[nonzero_mask] = disp_map[nonzero_mask] * scale + shift
              valid_depth_mask = (disp_map > 0) & (disp_map <= far)
              disp_map[~valid_depth_mask] = 0
              depth_map = np.full(disp_map.shape, np.inf)
              depth_map[disp_map != 0] = 1 / disp_map[disp_map != 0]
              depth_map[depth_map == np.inf] = 0
              depth_map = depth_map.astype(np.float32)
              depth = torch.tensor(depth_map)

              dataset.append({'cam': cam, 'im': im, 'id': len(jpg_filenames)-1400+c, 'depth': depth, 'vis': True}) 
            else:
              dataset.append({'cam': cam, 'im': im, 'id': c-1400+100, 'vis': True})  


              
      print(f'Loaded Dataset of Length {len(dataset)}')
      return dataset



def get_batch(todo_dataset, dataset):
    if not todo_dataset:
        todo_dataset = dataset.copy()
    curr_data = todo_dataset.pop(randint(0, len(todo_dataset) - 1))
    return [curr_data] #[curr_data] todo_dataset#


    
def initialize_params(seq, md, init_type):
    # init_pt_cld_before_dense init_pt_cld

    size=512
    #init_type='dv2'
    #init_pt_path=f'/data3/zihanwa3/Capstone-DSR/Appendix/dust3r/duster_{size}_scene.npz'
    #init_pt_path='/data3/zihanwa3/Capstone-DSR/Appendix/dust3r/good_.npz'
    if init_type=='dust':
      init_pt_path='/data3/zihanwa3/Capstone-DSR/Appendix/dust3r/patched_stat_imgs/pc.npz'    
      init_pt_cld = np.load(init_pt_path)["data"]
      #init_pt_cld = np.concatenate((init_pt_cld, init_pt_cld), axis=0)
      seg = init_pt_cld[:, 6]
      max_cams = 305

      intrinsics = [
          [1764.09472656, 1764.09472656, 1919.5, 1079.5],
          [1774.26708984, 1774.26708984, 1919.5, 1079.5],
          [1764.42602539, 1764.42602539, 1919.5, 1079.5],
          [1783.06530762, 1783.06530762, 1919.5, 1079.5]
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
        scale_gaussian = 4.7*depth / ((intrinsics[c][0] + intrinsics[c][1])/2)
        mean3_sq_dist.append((scale_gaussian**2).flatten().cpu().numpy()) # [H, W] * 4

      mean3_sq_dist=np.concatenate(mean3_sq_dist)
      
    elif init_type=='ego4d':
      init_pt_cld = np.load(f"./data_ego/{seq}/init_correct.npz")["data"]
      #init_pt_cld = np.concatenate((init_pt_cld, init_pt_cld), axis=0)
      seg = init_pt_cld[:, 6]
      max_cams = 305
      sq_dist, _ = o3d_knn(init_pt_cld[:, :3], 3)
      mean3_sq_dist = sq_dist.mean(-1).clip(min=0.0000001, max=0.1)


    elif init_type=='works':
      init_pt_cld = np.load('/data3/zihanwa3/Capstone-DSR/Appendix/dust3r/works_4.npz')["data"]
      #init_pt_cld = np.concatenate((init_pt_cld, init_pt_cld), axis=0)
      seg = init_pt_cld[:, 6]
      max_cams = 305
      sq_dist, _ = o3d_knn(init_pt_cld[:, :3], 3)
      mean3_sq_dist = sq_dist.mean(-1).clip(min=0.0000001, max=0.1)

    elif init_type=='instat':
      init_pt_path='/data3/zihanwa3/Capstone-DSR/Appendix/dust3r/SR_7_noMask.npz'
      org_pt_path=f"/data3/zihanwa3/Capstone-DSR/Appendix/dust3r/test_filled_img.npz"
      org_pt_cld = np.load(org_pt_path)["data"]
      init_pt_cld = np.load(init_pt_path)["data"]
      init_pt_cld = np.concatenate((init_pt_cld, org_pt_cld), axis=0)
      
      seg = init_pt_cld[:, 6]
      max_cams = 305
      sq_dist, _ = o3d_knn(init_pt_cld[:, :3], 3)
      mean3_sq_dist = sq_dist.mean(-1).clip(min=0.0000001, max=1.0)

    elif init_type=='dv2':
      #init_pt_path='/data3/zihanwa3/Capstone-DSR/Appendix/dust3r/patched_stat_imgs/pc.npz'

      init_pt_path='/data3/zihanwa3/Capstone-DSR/Appendix/Depth-Anything-V2/da_pt_cld.npz'
      org_pt_path=f"./data_ego/{seq}/init_correct.npz"
      org_pt_cld = np.load(org_pt_path)["data"]
      init_pt_cld = np.load(init_pt_path)["data"]
      init_pt_cld = np.concatenate((init_pt_cld, org_pt_cld), axis=0)

      densified=True
      if densified:
          repeated_pt_cld = []
          for _ in range(7):
              noise = np.random.normal(0, 0.01, init_pt_cld.shape)  # Small Gaussian noise
              noisy_pt_cld = init_pt_cld + noise
              repeated_pt_cld.append(noisy_pt_cld)
    
      init_pt_cld = np.vstack(repeated_pt_cld)

      seg = init_pt_cld[:, 6]
      max_cams = 305
      sq_dist, _ = o3d_knn(init_pt_cld[:, :3], 3)
      mean3_sq_dist = sq_dist.mean(-1).clip(min=0.0000001)#.clip(0.1)


    elif init_type=='org_densifed':
      init_pt_path='/data3/zihanwa3/Capstone-DSR/Appendix/dust3r/works_4.npz'
      org_pt_path=f"./data_ego/{seq}/init_correct.npz"
      org_pt_cld = np.load(org_pt_path)["data"]
      init_pt_cld = np.load(init_pt_path)["data"]
      init_pt_cld = nearest_densify(org_pt_cld, init_pt_cld)

      init_pt_cld = np.concatenate((init_pt_cld, org_pt_cld), axis=0)


      densified=True
      if densified:
          repeated_pt_cld = []
          for _ in range(4):
              noise = np.random.normal(0, 0.001, init_pt_cld.shape)  # Small Gaussian noise
              noisy_pt_cld = init_pt_cld + noise
              repeated_pt_cld.append(noisy_pt_cld)
      init_pt_cld = np.vstack(repeated_pt_cld)


      seg = init_pt_cld[:, 6]
      max_cams = 305
      sq_dist, _ = o3d_knn(init_pt_cld[:, :3], 3)
      mean3_sq_dist = sq_dist.mean(-1).clip(min=0.0000001, max=1.0)

    elif init_type=='fused':
      #init_pt_path='/data3/zihanwa3/Capstone-DSR/Appendix/dust3r/patched_stat_imgs/pc.npz'

      init_pt_path='/data3/zihanwa3/Capstone-DSR/Appendix/dust3r/works_4.npz'
      org_pt_path=f"./data_ego/{seq}/init_correct.npz"
      org_pt_cld = np.load(org_pt_path)["data"]
      init_pt_cld = np.load(init_pt_path)["data"]
      init_pt_cld = np.concatenate((init_pt_cld, org_pt_cld), axis=0)

      densified=True
      if densified:
          repeated_pt_cld = []
          for _ in range(7):
              noise = np.random.normal(0, 0.001, init_pt_cld.shape)  # Small Gaussian noise
              noisy_pt_cld = init_pt_cld + noise
              repeated_pt_cld.append(noisy_pt_cld)
    
      init_pt_cld = np.vstack(repeated_pt_cld)

      seg = init_pt_cld[:, 6]
      max_cams = 305
      sq_dist, _ = o3d_knn(init_pt_cld[:, :3], 3)
      mean3_sq_dist = sq_dist.mean(-1).clip(min=0.0000001)#.clip(0.1)


    elif init_type=='duplicated':
      org_pt_path=f"./data_ego/{seq}/init_correct.npz"
      init_pt_cld = np.load(org_pt_path)["data"]
      densified=True
      if densified:
          repeated_pt_cld = []
          for _ in range(100):
              noise = np.random.normal(0, 0.01, init_pt_cld.shape)  # Small Gaussian noise
              noisy_pt_cld = init_pt_cld + noise
              repeated_pt_cld.append(noisy_pt_cld)
    
      init_pt_cld = np.vstack(repeated_pt_cld)

      seg = init_pt_cld[:, 6]
      max_cams = 305
      sq_dist, _ = o3d_knn(init_pt_cld[:, :3], 3)
      mean3_sq_dist = sq_dist.mean(-1).clip(min=0.0000001)#.clip(0.1)

    print(f'NUM of {len(seg)/1e6}M Gaussians are used for initialization')

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

def initialize_optimizer(params, variables):
    lrs = {
        'means3D': 0.0000014 *  variables['scene_radius'], # 0000014
        'rgb_colors': 0.00028, ###0.0028 will fail 00028
        'seg_colors': 0.0,
        'unnorm_rotations': 0.000001, #0.00001(no), 000001
        'logit_opacities': 0.05,
        'log_scales': 0.002,
        'cam_m': 1e-5,
        'cam_c': 1e-5,
    }
    '''
            'logit_opacities': 0.05,
        'log_scales': 0.001,
    '''
    param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items() if k in lrs.keys()]
    return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)


def get_loss(params, curr_datasss, variables, is_initial_timestep, stat_dataset=None, item=None, loss_type='l1', depth_loss=False):
    losses = {}
    losses['im'] = 0
    losses['depth'] = 0 
    losses['flow'] = 0 
    rendervar = params2rendervar(params)
    rendervar['means2D'].retain_grad()




    for curr_data in curr_datasss: 

      im, radius, depth_pred, _ = Renderer(raster_settings=curr_data['cam'])(**rendervar)
      curr_id = curr_data['id']
      im=im.clip(0,1)
      H, W =im.shape[1], im.shape[2]
      top_mask = torch.zeros((H, W), device=im.device)
      top_mask[:, :]=1


      import torchvision.transforms.functional as F
      if 'antimask' in curr_data.keys():
        antimask=curr_data['antimask'].to(im.device)
        combined_mask = ~torch.logical_or(antimask, antimask)# ~torch.logical_or(default_mask, antimask)
        top_mask = combined_mask.type(torch.uint8)


      if depth_loss:
        #if H==W:
        if depth_loss=='l1':
          rl_depth = curr_data['depth'].cuda() * top_mask
          rd_depth = depth_pred * top_mask
          losses['depth'] += l1_loss_v1(rl_depth, rd_depth)

        elif depth_loss=='pearson':
          ground_truth_depth = curr_data['depth'].to(im.device)
          depth_mask = top_mask.flatten().bool()
          depth_pred = depth_pred.reshape(-1, 1)[depth_mask]
          depth_pred = torch.clamp(depth_pred, min=near, max=far)
          ground_truth_depth = ground_truth_depth.reshape(-1, 1)[depth_mask]
          depth_pred = 1/depth_pred
          #depth_pred = (depth_pred - depth_pred.mean()) / depth_pred.std()
          #ground_truth_depth = (ground_truth_depth - ground_truth_depth.mean()) / ground_truth_depth.std()

          #  gt_depth: 1/zoe_depth(metric_depth) -> 1/real_depth; gasussian
          losses['depth'] += (1 - pearson_corrcoef( ground_truth_depth, (depth_pred)))
      

      if item > 4.9e3:
        losses['flow'] += flow_loss(rendervar)


      top_mask = top_mask.unsqueeze(0).repeat(3, 1, 1)


      masked_im = im * top_mask
      masked_curr_data_im = curr_data['im'] * top_mask
      masked_im_np = masked_im.detach().cpu().numpy()
      masked_curr_data_im_np = masked_curr_data_im.detach().cpu().numpy()


      if H == W:
        losses['im'] += l1_loss_v1(masked_im, masked_curr_data_im)               
      else: 
        #loss_type='pearson' # 'l1 pearson
        if loss_type=='l1':
          losses['im'] += 0.8 * l1_loss_v1(masked_im, masked_curr_data_im) + 0.2 * (1.0 - calc_ssim(masked_im, masked_curr_data_im))
        elif loss_type=='pearson':
          zeros_mask = masked_curr_data_im < -100000
          masked_im_flatten = masked_im[~zeros_mask].flatten()
          masked_curr_data_im_flatten = masked_curr_data_im[~zeros_mask].flatten()
          pearson_corr = pearson_corrcoef(masked_im_flatten, masked_curr_data_im_flatten)
          pearson_loss = 1 - pearson_corr
          losses['im'] += 0.8 * pearson_loss + 0.2 * (1.0 - calc_ssim(masked_im, masked_curr_data_im))

    losses['im'] /= len(curr_datasss)
    #losses['depth'] /= len(curr_datasss)


    variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification

    loss_weights = {'im': 0.1, 'rigid': 0.0, 'rot': 0.0, 'iso': 0.0, 'floor': 0.0, 'bg': 2.0, 'depth': 2e-3, 'flow':2e-8,
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
def combine_images(image1, depth1, image2, depth2):
    # Convert depth maps to 3-channel images

    # Combine each image and depth side by side
    combined1 = np.hstack((image1, depth1))
    combined2 = np.hstack((image2, depth2))
    
    # Combine the two combined images in a 2x2 grid
    combined_all = np.vstack((combined1, combined2))
    return combined_all

def vis_depth(depth):
  depth = np.array(depth)
  depth_normalized = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
  
  depth_normalized = depth_normalized.astype(np.uint8)
  colored_depth = (plt.cm.plasma(depth_normalized / 255.0)[:, :, :3] * 255).astype(np.uint8)
  return colored_depth
  
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
            depth1_3channel = depth1#cv2.cvtColor(depth1, cv2.COLOR_GRAY2RGB)
            depth2_3channel = depth2#cv2.cvtColor(depth2, cv2.COLOR_GRAY2RGB)
            
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
            cam = setup_camera(w, h, k, w2c, near=near, far=far)
            fn = md['fn'][t][c] # mask_{fn.split('/')[0]}
            gt_im = np.array(copy.deepcopy(Image.open(f"/ssd0/zihanwa3/data_ego/cmu_bike/ims/undist_data/{fn}")))
            #print(fn)
            gt_im = torch.tensor(gt_im).float().cuda().permute(2, 0, 1) / 255

            mask_path='/data3/zihanwa3/Capstone-DSR/Dynamic3DGaussians/data_ego/masked_cmu_bike/triangular_mask.jpg'
            default_mask= torch.tensor(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE), device=gt_im.device)
            default_mask = default_mask>1e-5
            try:
              mask_path=f"/data3/zihanwa3/Capstone-DSR/Appendix/SR_49_clean_mask/{fn.split('/')[-1].replace('.jpg', '.npz')}"
              mask = np.load(mask_path)['mask'][0]
            except:
              continue

            transform = transforms.ToTensor()
            mask_tensor = transform(mask).squeeze(0).to(gt_im.device)
            antimask = mask_tensor > 1e-5
            combined_mask = ~torch.logical_or(antimask, antimask)# ~torch.logical_or(default_mask, antimask)
            top_mask = combined_mask.type(torch.uint8)


            gt_im *= top_mask
            gt_im = torch.rot90(gt_im, k=-1, dims=(1, 2))
            gt_im = gt_im.permute(1, 2, 0).cpu().numpy() * 255
            gt_im = gt_im.astype(np.uint8)

            gt_im = cv2.resize(gt_im, (256, 256), interpolation=cv2.INTER_LINEAR)

            gt_depth=f'/data3/zihanwa3/Capstone-DSR/Processing/da_v2_disp/0/disp_{c}.npz'
            gt_depth = torch.tensor(np.load(gt_depth)['depth_map']).float().cuda()  #/ 255
            gt_depth=torch.rot90(gt_depth, k=-1, dims=(0, 1))
            gt_depth=torch.clamp(gt_depth, min=near, max=far)

            gt_depth = gt_depth.cpu().numpy() #* 30 #* 255
            gt_depth=vis_depth(gt_depth)
            ## 

            #gt_depth = gt_depth.astype(np.uint8)
            
            gt_depth = cv2.resize(gt_depth, (256, 256), interpolation=cv2.INTER_LINEAR)

            
            im, _, depth, _ = Renderer(raster_settings=cam)(**params2rendervar(params))
            
            im=im.clip(0,1)


                    # Process image
            im = torch.rot90(im, k=-1, dims=(1, 2))

            im_wandb = im.permute(1, 2, 0).cpu().numpy() * 255
            im_wandb = im_wandb.astype(np.uint8)
            im_wandb = cv2.resize(im_wandb, (256, 256), interpolation=cv2.INTER_LINEAR)
            
            # Process depth
            depth = torch.rot90(depth, k=-1, dims=(1, 2))[0]
            #print('real', depth.shape)
            depth = vis_depth(1/np.clip(depth.detach().cpu().numpy(), 0.001, 5))


            #depth = depth.permute(1, 2, 0).cpu().numpy() * 255
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
            cam = setup_camera(w, h, k, w2c, near=near, far=far)
            fn = md['fn'][t][c] # mask_{fn.split('/')[0]}
            
            gt_im = np.array(copy.deepcopy(Image.open(f"/ssd0/zihanwa3/data_ego/cmu_bike/ims/{fn}")))
            gt_im = torch.tensor(gt_im).float().cuda().permute(2, 0, 1) / 255
            gt_im = gt_im.permute(1, 2, 0).cpu().numpy() * 255
            gt_im = gt_im.astype(np.uint8)
            gt_im = cv2.resize(gt_im, (256, 144), interpolation=cv2.INTER_CUBIC)

            #gt_depth=f'/ssd0/zihanwa3/data_ego/cmu_bike/depth/{int(c)-1399}/depth_0.npz'
            #gt_depth = torch.tensor(np.load(gt_depth)['depth_map']).float().cuda()  #/ 255


            gt_depth=f'/data3/zihanwa3/Capstone-DSR/Processing/da_v2_disp/0/disp_{c}.npz'
            gt_depth = torch.tensor(np.load(gt_depth)['depth_map']).float().cuda()  #/ 255

            gt_depth = gt_depth.cpu().numpy() #* 30 #* 255
            gt_depth=vis_depth(gt_depth)

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
            # Process depth
            depth = depth[0]
            #print('real', depth.shape)
            depth = vis_depth(1/np.clip(depth.detach().cpu().numpy(), 0.1, 5))

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


def train(seq, exp, clean_img, init_type, num_timesteps, num_iter_initial, num_iter_subsequent, loss_type, depth_loss, debug_mode):
    if clean_img:
        md = json.load(open(f"./data_ego/{seq}/train_meta_f670.json", 'r'))
    else:
        md = json.load(open(f"./data_ego/{seq}/train_meta.json", 'r'))

    params, variables = initialize_params(seq, md, init_type)
    optimizer = initialize_optimizer(params, variables)
    output_params = []

    initialize_wandb(exp, seq)

    for t in range(num_timesteps):
        dataset = get_dataset(t, md, seq, mode='ego_only', clean_img=clean_img, depth_loss=depth_loss, debug_mode=debug_mode)
        stat_dataset = None
        todo_dataset = []

        is_initial_timestep = (t == 0)
        if not is_initial_timestep:
            params, variables = initialize_per_timestep(params, variables, optimizer)

        num_iter_per_timestep = num_iter_initial if is_initial_timestep else num_iter_subsequent
        progress_bar = tqdm(range(num_iter_per_timestep), desc=f"timestep {t}")
        for i in range(num_iter_per_timestep):
            progressive_params = []
            curr_data = get_batch(todo_dataset, dataset)

            loss, variables, losses = get_loss(params, curr_data, variables, is_initial_timestep, \
            stat_dataset=stat_dataset, item=i, loss_type=loss_type, depth_loss=depth_loss)
            loss.backward()
            with torch.no_grad():
                report_progress(params, dataset[0], i, progress_bar)
                report_stat_progress(params, curr_data, i, progress_bar, md)
                if is_initial_timestep:
                    params, variables = densify(params, variables, optimizer, i)
                assert ((params['means3D'].shape[0] == 0) is False)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            for key, value in losses.items():
                wandb.log({key: value, "iteration": i})

            wandb.log({'num_of_gaussians': len(params['means3D']), "iteration": i})
            progressive_iter = 2e3
            if i % progressive_iter == 0:
                progressive_params.append(params2cpu(copy.deepcopy(params), is_initial_timestep))
                save_params_progressively(progressive_params, seq, exp, i)

        progress_bar.close()
        output_params.append(params2cpu(params, is_initial_timestep))
        if is_initial_timestep:
            variables = initialize_post_first_timestep(params, variables, optimizer)

    save_params(output_params, seq, exp)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run training and visualization for a sequence.")
    parser.add_argument('--exp_name', type=str, required=True, help='Name of the experiment')
    parser.add_argument('--init_type', type=str, default='org_densifed', 
    choices=['dust', 'ego4d', 'works', 'instat', 'dv2', 'org_densifed', 'fused', 'duplicated'], help='Initialization type')
    parser.add_argument('--loss_type', type=str, default='l1')
    parser.add_argument('--depth_loss', type=str, default='l1', 
    choices=['l1', 'pearson'], help='Initialization type')
    parser.add_argument('--debug_mode', type=str, default='no', 
    choices=['stat', 'ego', 'no'], help='Initialization type')

    parser.add_argument('--num_timesteps', type=int, default=7, help='Number of timesteps')
    parser.add_argument('--clean_img', type=bool, default=True, help='Whether to use clean images')
    parser.add_argument('--num_iter_initial', type=int, default=int(2.1e4), help='Number of iterations for the initial timestep')
    parser.add_argument('--num_iter_subsequent', type=int, default=2, help='Number of iterations for subsequent timesteps')
    parser.add_argument('--sequence', type=str, default='cmu_bike', help='Sequence name')


    args = parser.parse_args() 
    exp_name = args.exp_name
    #for sequence in ["basketball", "boxes", "football", "juggle", "softball", "tennis"]:
    for sequence in ["cmu_bike"]:
        #train(args.sequence, args.exp_name, args.clean_img, args.init_type, \
        #args.num_timesteps, args.num_iter_initial, args.num_iter_subsequent, \
        #args.loss_type, args.depth_loss)

        train(
            seq=args.sequence, 
            exp=args.exp_name,  # from args
            clean_img=args.clean_img,   # explicit argument
            init_type=args.init_type,  # explicit argument
            num_timesteps=args.num_timesteps,  # from args
            num_iter_initial=args.num_iter_initial,  # explicit argument
            num_iter_subsequent=args.num_iter_subsequent,  # from args
            loss_type=args.loss_type,  
            depth_loss=args.depth_loss, 
            debug_mode=args.debug_mode,
        )

        torch.cuda.empty_cache()
        from visualize import visualize
        visualize(sequence, exp_name)