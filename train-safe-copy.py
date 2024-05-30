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

def save_masked_image(image_tensor, mask_tensor, save_path):
    """
    Save the masked image to a file.
    
    :param image_tensor: Tensor of the image (shape: [C, H, W])
    :param mask_tensor: Tensor of the mask (shape: [H, W])
    :param save_path: Path where the image will be saved
    """
    # Convert the image tensor to a PIL Image
    image = image_tensor.cpu().permute(1, 2, 0).numpy() * 255
    image = image.astype(np.uint8)
    image = Image.fromarray(image)
    
    # Convert the mask tensor to a PIL Image
    mask = mask_tensor.cpu().numpy() * 255
    mask = mask.astype(np.uint8)
    mask = Image.fromarray(mask, mode='L')
    
    # Apply the mask to the image
    image.putalpha(mask)
    
    # Save the image
    image.save(save_path)

def get_dataset(t, md, seq, mode='stat_only'):
    dataset = []
    t += 1111
    def get_jpg_filenames(directory):
        jpg_files = [int(file.split('.')[0]) for file in os.listdir(directory) if file.endswith('.jpg')]
        return jpg_files

    # Specify the directory containing the .jpg files   precise_reduced_im
                # /data3/zihanwa3/Capstone-DSR/Dynamic3DGaussians/data_ego/nice100 '/data3/zihanwa3/Capstone-DSR/Appendix/lalalal_new'#
    directory = '/data3/zihanwa3/Capstone-DSR/Dynamic3DGaussians/data_ego/nice100'
    jpg_filenames = get_jpg_filenames(directory)
    if mode=='ego_only':
      t=0
      jpg_filenames_2 = np.array(jpg_filenames)+1404
      jpg_filenames_3 = np.array(jpg_filenames_2)+1400
      for lis in [jpg_filenames]: # , jpg_filenames_2, jpg_filenames_3
        #print(sorted(lis)[:30])
        for c in sorted(lis)[:30]:
            h, w = md['hw'][c]
            k, w2c =  md['k'][t][c], (md['w2c'][t][c])
            cam = setup_camera(w, h, k, w2c, near=0.01, far=50)
            fn = md['fn'][t][c] # mask_{fn.split('/')[0]}
            mask_path=f"/data3/zihanwa3/Capstone-DSR/Appendix/lalalal_mask/{fn.split('/')[-1]}"
            mask = Image.open(mask_path).convert("L")
            transform = transforms.ToTensor()
            mask_tensor = transform(mask).squeeze(0)
            anti_mask_tensor = mask_tensor < 1e-5
            if h==192:
                image_path = f"/scratch/zihanwa3/data_ego/{seq}/ims/{fn}"
                image = Image.open(image_path)
                gray_image = image.convert('L')
                im = np.array(copy.deepcopy(gray_image))
                im = torch.tensor(im).float().cuda() / 255
                im = im.unsqueeze(0).repeat(3, 1, 1)
                dataset.append({'cam': cam, 'im': im, 'id': c, 'antimask': anti_mask_tensor})
            else:
                im = np.array(copy.deepcopy(Image.open(f"/scratch/zihanwa3/data_ego/{seq}/ims/{fn}")))
                im = torch.tensor(im).float().cuda().permute(2, 0, 1) / 255
                mask_path=f'/scratch/zihanwa3/data_ego/cmu_bike/depth/{int(t)}/depth_{c}.npz'
                depth = torch.tensor(np.load(mask_path)['depth_map']).float().cuda()
                dataset.append({'cam': cam, 'im': im, 'id': c, 'antimask': anti_mask_tensor})

            # Save the masked image
            save_path = f"./masked_images/{c}_masked.png"
            save_masked_image(im, anti_mask_tensor, save_path)
            print(f"Saved masked image to {save_path}")

      ## load stat
      for c in range(1400, 1404):
          h, w = md['hw'][c]
          k, w2c =  md['k'][t][c], np.linalg.inv(md['w2c'][t][c])
          cam = setup_camera(w, h, k, w2c, near=0.01, far=50)

          fn = md['fn'][t][c]
          print(fn)
          im = np.array(copy.deepcopy(Image.open(f"/scratch/zihanwa3/data_ego/{seq}/ims/{fn}")))
          im = torch.tensor(im).float().cuda().permute(2, 0, 1) / 255
          seg = np.array(copy.deepcopy(cv2.imread(f"/scratch/zihanwa3/data_ego/{seq}/seg/{fn.replace('.jpg', '.png')}", cv2.IMREAD_GRAYSCALE))).astype(np.float32)
          seg = cv2.resize(seg, (im.shape[2], im.shape[1]))
          mask_path=f'/data3/zihanwa3/Capstone-DSR/Dynamic3DGaussians/data_ego/masked_cmu_bike/{c-1399}/mask.png'
          mask = Image.open(mask_path).convert("L")
          transform = transforms.ToTensor()
          mask_tensor = transform(mask).squeeze(0)
          anti_mask_tensor = mask_tensor < 1e-5

          #print(seg.shape)
        
          ############################## First Frame Depth ##############################
          mask_path=f'/scratch/zihanwa3/data_ego/cmu_bike/depth/{int(c)-1399}/depth_1122.npz'
          depth = torch.tensor(np.load(mask_path)['depth_map']).float().cuda()
          #np.savez_compressed(, depth_map=new_depth)

          seg = torch.tensor(seg).float().cuda()
          seg_col = torch.stack((seg, torch.zeros_like(seg), 1 - seg))
          dataset.append({'cam': cam, 'im': im, 'id': c, 'antimask': anti_mask_tensor, 'gt_depth':depth})

          # Save the masked image
          save_path = f"./masked_images/{c}_masked.png"
          save_masked_image(im, anti_mask_tensor, save_path)
          print(f"Saved masked image to {save_path}")

    return dataset

def train(seq, exp):
    #if os.path.exists(f"./output/{exp}/{seq}"):
    #    print(f"Experiment '{exp}' for sequence '{seq}' already exists. Exiting.")
    #    return
    md = json.load(open(f"./data_ego/{seq}/train_meta.json", 'r'))  # metadata
    num_timesteps = len(md['fn'])
    t=0
    get_dataset(t, md, seq, mode='ego_only')

if __name__ == "__main__":
    import argparse

    exp_name = 'test'
    #for sequence in ["basketball", "boxes", "football", "juggle", "softball", "tennis"]:
    for sequence in ["cmu_bike"]:
        train(sequence, exp_name)
