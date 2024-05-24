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

def get_dataset(t, md, seq, mode='stat_only'):
    dataset = []
    t += 1111
    def get_jpg_filenames(directory):
        jpg_files = [int(file.split('.')[0]) for file in os.listdir(directory) if file.endswith('.jpg')]
        return jpg_files

    # Specify the directory containing the .jpg files
    directory = '/data3/zihanwa3/Capstone-DSR/Dynamic3DGaussians/depth_im'
    jpg_filenames = get_jpg_filenames(directory)
    print(jpg_filenames)

    if mode=='stat_only':
      for c in range(1, len(md['fn'][t])):
          h, w = md['hw'][c]
          k, w2c =  md['k'][t][c], np.linalg.inv(md['w2c'][t][c])
          cam = setup_camera(w, h, k, w2c, near=0.01, far=50)

          fn = md['fn'][t][c]

          im = np.array(copy.deepcopy(Image.open(f"/scratch/zihanwa3/data3/zihanwa3/Capstone-DSR/Dynamic3DGaussians/data_ego/{seq}/ims/{fn}")))
          im = torch.tensor(im).float().cuda().permute(2, 0, 1) / 255
          seg = np.array(copy.deepcopy(cv2.imread(f"/scratch/zihanwa3/data3/zihanwa3/Capstone-DSR/Dynamic3DGaussians/data_ego/{seq}/seg/{fn.replace('.jpg', '.png')}", cv2.IMREAD_GRAYSCALE))).astype(np.float32)
          seg = cv2.resize(seg, (im.shape[2], im.shape[1]))
          #print(seg.shape)
          ############################## First Frame Depth ##############################
          mask_path=f'/scratch/zihanwa3/data3/zihanwa3/Capstone-DSR/Dynamic3DGaussians/data_ego/cmu_bike/depth/{int(c)}/depth_{t}.npz'
          depth = torch.tensor(np.load(mask_path)['depth_map']).float().cuda()
          #np.savez_compressed(, depth_map=new_depth)

          seg = torch.tensor(seg).float().cuda()
          seg_col = torch.stack((seg, torch.zeros_like(seg), 1 - seg))
          dataset.append({'cam': cam, 'im': im, 'seg': seg_col, 'id': c, 'gt_depth':depth, 'mask':seg})
      
    elif mode=='ego_only':
      t=0
      #for c in range(4204):
      #  interval=27
      #  if c in range(interval) or c in range(1404, 1404+interval) or c in range(2804, 2804+interval):
      for c in jpg_filenames:
        h, w = md['hw'][c]
        k, w2c =  md['k'][t][c], np.linalg.inv(md['w2c'][t][c])
        cam = setup_camera(w, h, k, w2c, near=0.01, far=50)
        fn = md['fn'][t][c]
        if h==192:
          image_path = f"/scratch/zihanwa3/data3/zihanwa3/Capstone-DSR/Dynamic3DGaussians/data_ego/{seq}/ims/{fn}"
          image = Image.open(image_path)
          gray_image = image.convert('L')
          im = np.array(copy.deepcopy(gray_image))
          im = torch.tensor(im).float().cuda() / 255
          im = im.unsqueeze(0).repeat(3, 1, 1)
        else:
          im = np.array(copy.deepcopy(Image.open(f"/scratch/zihanwa3/data3/zihanwa3/Capstone-DSR/Dynamic3DGaussians/data_ego/{seq}/ims/{fn}")))
          im = torch.tensor(im).float().cuda().permute(2, 0, 1) / 255
        ############################## First Frame Depth ##############################
        dataset.append({'cam': cam, 'im': im, 'id': c,})
  
    else:
      t=0
      for c in range(1400):
        h, w = md['hw'][c]
        k, w2c =  md['k'][t][c], (md['w2c'][t][c])
        cam = setup_camera(w, h, k, w2c, near=0.01, far=50)
        fn = md['fn'][t][c]
        im = np.array(copy.deepcopy(Image.open(f"/scratch/zihanwa3/data3/zihanwa3/Capstone-DSR/Dynamic3DGaussians/data_ego/{seq}/ims/{fn}")))
        im = torch.tensor(im).float().cuda().permute(2, 0, 1) / 255
        depth = torch.tensor(np.load(mask_path)['depth_map']).float().cuda()
        dataset.append({'cam': cam, 'im': im, 'id': c, 'gt_depth':depth, 'mask':seg})
      for c in range(1400, 1404):
        h, w = md['hw'][c]
        k, w2c =  md['k'][t][c], (md['w2c'][t][c])
        cam = setup_camera(w, h, k, w2c, near=0.01, far=50)
        fn = md['fn'][t][c]
        im = np.array(copy.deepcopy(Image.open(f"/scratch/zihanwa3/data3/zihanwa3/Capstone-DSR/Dynamic3DGaussians/data_ego/{seq}/ims/{fn}")))
        im = torch.tensor(im).float().cuda().permute(2, 0, 1) / 255
        seg = np.array(copy.deepcopy(cv2.imread(f"/scratch/zihanwa3/data3/zihanwa3/Capstone-DSR/Dynamic3DGaussians/data_ego/{seq}/seg/{fn.replace('.jpg', '.png')}", cv2.IMREAD_GRAYSCALE))).astype(np.float32)
        seg = cv2.resize(seg, (im.shape[2], im.shape[1]))


        #print(seg.shape)
        ############################## First Frame Depth ##############################
        mask_path=f'/scratch/zihanwa3/data3/zihanwa3/Capstone-DSR/Dynamic3DGaussians/data_ego/cmu_bike/depth/{int(c)-1399}/depth_1122.npz'
        depth = torch.tensor(np.load(mask_path)['depth_map']).float().cuda()
        seg = torch.tensor(seg).float().cuda()
        seg_col = torch.stack((seg, torch.zeros_like(seg), 1 - seg))
        dataset.append({'cam': cam, 'im': im, 'id': c, 'gt_depth':depth, 'mask':seg})
    return dataset


def get_stat_dataset(t, md, seq, mode='stat_only'):
    dataset = []
    t=0
    print(
      'wtf'
    )
    if mode=='stat_only':
      for c in range(1400, 1404):
          h, w = md['hw'][c]
          k, w2c =  md['k'][t][c], np.linalg.inv(md['w2c'][t][c])
          cam = setup_camera(w, h, k, w2c, near=0.01, far=50)

          fn = md['fn'][t][c]
          print(fn)
          im = np.array(copy.deepcopy(Image.open(f"/scratch/zihanwa3/data3/zihanwa3/Capstone-DSR/Dynamic3DGaussians/data_ego/{seq}/ims/{fn}")))
          im = torch.tensor(im).float().cuda().permute(2, 0, 1) / 255
          seg = np.array(copy.deepcopy(cv2.imread(f"/scratch/zihanwa3/data3/zihanwa3/Capstone-DSR/Dynamic3DGaussians/data_ego/{seq}/seg/{fn.replace('.jpg', '.png')}", cv2.IMREAD_GRAYSCALE))).astype(np.float32)
          seg = cv2.resize(seg, (im.shape[2], im.shape[1]))
          #print(seg.shape)
        
          ############################## First Frame Depth ##############################
          mask_path=f'/scratch/zihanwa3/data3/zihanwa3/Capstone-DSR/Dynamic3DGaussians/data_ego/cmu_bike/depth/{int(c)-1399}/depth_1122.npz'
          depth = torch.tensor(np.load(mask_path)['depth_map']).float().cuda()
          #np.savez_compressed(, depth_map=new_depth)

          seg = torch.tensor(seg).float().cuda()
          seg_col = torch.stack((seg, torch.zeros_like(seg), 1 - seg))
          dataset.append({'cam': cam, 'im': im, 'seg': seg_col, 'id': c, 'gt_depth':depth, 'mask':seg})
    return dataset

def apply_anti_mask(image_path, mask_path):
    # Load the image and the mask
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    transform = transforms.ToTensor()
    mask_tensor = transform(mask).squeeze(0)
    
    # Create the anti-mask by inverting the mask
    anti_mask_tensor = mask_tensor > 1e-5
    
    # Apply the anti-mask to the image
    image_tensor[:, anti_mask_tensor] = 0

    
    return masked_image


def get_batch(todo_dataset, dataset):
    if not todo_dataset:
        todo_dataset = dataset.copy()
    curr_data = todo_dataset.pop(randint(0, len(todo_dataset) - 1))
    return curr_data


def initialize_params(seq, md):
    init_pt_cld = np.load(f"./data_ego/{seq}/init_pt_cld.npz")["data"]
    #init_pt_cld = np.concatenate((init_pt_cld, init_pt_cld), axis=0)
    print(len(init_pt_cld))
    seg = init_pt_cld[:, 6]
    max_cams = 4204
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
    params = {k: torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True)) for k, v in
              params.items()}
    cam_centers = np.linalg.inv(md['w2c'][0])[:, :3, 3]  # Get scene radius
    scene_radius = 1.1 * np.max(np.linalg.norm(cam_centers - np.mean(cam_centers, 0)[None], axis=-1))
    variables = {'max_2D_radius': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'scene_radius': scene_radius,
                 'means2D_gradient_accum': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'denom': torch.zeros(params['means3D'].shape[0]).cuda().float()}
    return params, variables


def initialize_optimizer(params, variables):
    lrs = {
        'means3D': 0.0000016 * variables['scene_radius'],
        'rgb_colors': 0.00025,
        'seg_colors': 0.0,
        'unnorm_rotations': 0.001,
        'logit_opacities': 0.05,
        'log_scales': 0.001,
        'cam_m': 1e-4,
        'cam_c': 1e-4,
    }
    param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items()]
    return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)


def get_loss(params, curr_data, variables, is_initial_timestep, stat_dataset=None):
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
    im = torch.exp(params['cam_m'][curr_id])[:, None, None] * im + params['cam_c'][curr_id][:, None, None]
    H, W =im.shape[1], im.shape[2]
    top_mask = torch.zeros((H, W), device=im.device)

    top_mask[:, :] = 1
    top_mask = top_mask.unsqueeze(0).repeat(3, 1, 1)

    masked_im = im * top_mask
    masked_curr_data_im = curr_data['im'] * top_mask

    losses['im'] = 0.8 * l1_loss_v1(masked_im, masked_curr_data_im) + 0.2 * (1.0 - calc_ssim(masked_im, masked_curr_data_im))
    def held_stat_loss(stat_dataset):

        losses = 0 
        for i, data in enumerate(stat_dataset):
            im, radius, depth_pred, = Renderer(raster_settings=data['cam'])(**rendervar)
            curr_id = data['id']
            im = torch.exp(params['cam_m'][curr_id])[:, None, None] * im + params['cam_c'][curr_id][:, None, None]
            losses += 0.8 * l1_loss_v1(im, data['im']) + 0.2 * (1.0 - calc_ssim(im, data['im']))
            '''
            ground_truth_depth = data['gt_depth']
            depth_pred = depth_pred.squeeze(0)
            depth_pred = depth_pred.reshape(-1, 1)
            ground_truth_depth = ground_truth_depth.reshape(-1, 1)
            losses+= min(
                            (1 - pearson_corrcoef( - ground_truth_depth, depth_pred)),
                            (1 - pearson_corrcoef(1 / (ground_truth_depth + 200.), depth_pred))
            )
            '''
        return losses/100
    if stat_dataset:
        losses['stat_im']=held_stat_loss(stat_dataset)

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

    loss_weights = {'im': 5.0, 'rigid': 4.0, 'rot': 4.0, 'iso': 2.0, 'floor': 2.0, 'bg': 15.0, 'stat_im':0.005,
                    'soft_col_cons': 0.01}
                    
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

def report_stat_progress(params, stat_dataset, i, progress_bar, every_i=1000):
    if i % every_i == 0:
        for index, data in enumerate(stat_dataset):
            im, _, _, = Renderer(raster_settings=data['cam'])(**params2rendervar(params))
            curr_id = data['id']
            im = torch.exp(params['cam_m'][curr_id])[:, None, None] * im + params['cam_c'][curr_id][:, None, None]
            im_wandb = im.permute(1, 2, 0).cpu().numpy() * 255
            im_wandb = im_wandb.astype(np.uint8)
            wandb.log({f"rendered_image_{index}": wandb.Image(im_wandb, caption=f"Rendered image at iteration {i}")})


def report_progress(params, data, i, progress_bar, every_i=100):
    if i % every_i == 0:
        im, _, _, = Renderer(raster_settings=data['cam'])(**params2rendervar(params))
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
    params, variables = initialize_params(seq, md)
    optimizer = initialize_optimizer(params, variables)
    output_params = []

    initialize_wandb(exp, seq)

    
    for t in range(7):
        dataset = get_dataset(t, md, seq, mode='ego_only')
        stat_dataset = get_stat_dataset(t, md, seq, mode='stat_only')



        stat_dataset=dataset


        todo_dataset = []
        is_initial_timestep = (t == 0)
        if not is_initial_timestep:
            params, variables = initialize_per_timestep(params, variables, optimizer)

        num_iter_per_timestep = int(4.7e3) if is_initial_timestep else 2
        progress_bar = tqdm(range(num_iter_per_timestep), desc=f"timestep {t}")
        for i in range(num_iter_per_timestep):
            curr_data = get_batch(todo_dataset, dataset)

            loss, variables, losses = get_loss(params, curr_data, variables, is_initial_timestep, stat_dataset=stat_dataset)
            loss.backward()
            with torch.no_grad():
                report_progress(params, dataset[0], i, progress_bar)
                report_stat_progress(params, stat_dataset, i, progress_bar)
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