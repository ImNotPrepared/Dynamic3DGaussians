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

def get_dataset(t, md, seq, mode='stat_only'):
    dataset = []
    t += 1111
    def get_jpg_filenames(directory):
        jpg_files = [int(file.split('.')[0]) for file in os.listdir(directory) if file.endswith('.jpg')]
        return jpg_files
    directory = '/data3/zihanwa3/Capstone-DSR/Dynamic3DGaussians/data_ego/nice100'
    jpg_filenames = get_jpg_filenames(directory)
    if mode=='ego_only':
      t=0
      jpg_filenames_2 = np.array(jpg_filenames)+1404
      jpg_filenames_3 = np.array(jpg_filenames_2)+1400
      for lis in [jpg_filenames]: # , jpg_filenames_2, jpg_filenames_3
        for c in lis:
            h, w = md['hw'][c]
            k, w2c =  md['k'][t][c], np.linalg.inv(md['w2c'][t][c])
            cam = setup_camera(w, h, k, w2c, near=0.01, far=50)
            fn = md['fn'][t][c]
            mask_path=f"/data3/zihanwa3/Capstone-DSR/Appendix/mask_{fn.split('/')[0]}/{fn.split('/')[-1]}"
            mask = Image.open(mask_path).convert("L")
            transform = transforms.ToTensor()
            mask_tensor = transform(mask).squeeze(0)
            anti_mask_tensor = mask_tensor < 1e-5

            im = np.array(copy.deepcopy(Image.open(f"/scratch/zihanwa3/data_ego/{seq}/ims/{fn}")))
            im = torch.tensor(im).float().cuda().permute(2, 0, 1) / 255
            dataset.append({'cam': cam, 'im': im, 'id': c, 'antimask': anti_mask_tensor})



def train(seq, exp):
    #if os.path.exists(f"./output/{exp}/{seq}"):
    #    print(f"Experiment '{exp}' for sequence '{seq}' already exists. Exiting.")
    #    return
    md = json.load(open(f"./data_ego/{seq}/train_meta.json", 'r'))  # metadata
    num_timesteps = len(md['fn'])
    params, variables = initialize_params(seq, md)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run training and visualization for a sequence.")
    parser.add_argument('--exp_name', type=str, required=True, help='Name of the experiment')
    args = parser.parse_args()

    exp_name = args.exp_name
    #for sequence in ["basketball", "boxes", "football", "juggle", "softball", "tennis"]:
    for sequence in ["cmu_bike"]:
        train(sequence, exp_name)