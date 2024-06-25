#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from PIL import Image
import copy
import torch
from torch import nn
import numpy as np
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
import sys
import os
import torch
import os
import open3d as o3d
import numpy as np



def setup_camera(w, h, k, w2c, near=0.01, far=100, im=None, depth=None, loss_mask=None):
    import math
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    w2c = torch.tensor(w2c)#.cuda().float()
    cam_center = torch.inverse(w2c)[:3, 3]
    w2c = np.array(w2c)#.unsqueeze(0).transpose(1, 2)
    R = w2c[:3, :3]  # Rotation matrix
    T = w2c[:3, 3]  # Translation vector
    #print(w2c.shape, R.shape, T.shape)
    FoVx = 2 * math.atan(w / (2 * fx)) * 180 / math.pi  # Field of view in x direction (degrees)
    FoVy = 2 * math.atan(h / (2 * fy)) * 180 / math.pi

    cam=Camera(R, T, FoVx, FoVy, image=im,depth=depth, loss_mask=loss_mask)
    return cam


class Camera(nn.Module):
    def __init__(self, R, T, FoVx, FoVy, image, depth=None, loss_mask=None, gt_alpha_mask=None,
                 trans=np.array([0.0, 0.0, 0.0]),
                 scale=1.0, data_device = "cuda", depth_image = None, mask = None, bounds=None):
        super(Camera, self).__init__()
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.depth_image = depth_image
        self.mask = mask
        self.bounds = bounds

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.depth = depth.to(self.data_device)
        self.loss_mask =loss_mask.to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]



def get_dataset(t, md, seq, mode='stat_only'):
    dataset = []
    t+=183
    if mode=='ego_only':
      for c in range(1,5):
          epsilon=1e-7
          h, w = md['hw'][c]
          k, w2c =  md['k'][t][c], np.linalg.inv(md['w2c'][t][c])
          
          fn = md['fn'][t][c] # mask_{fn.split('/')[0]}
          im = np.array(copy.deepcopy(Image.open(f"/ssd0/zihanwa3/data_ego/{seq}/ims/{fn}")))
         
          im = torch.tensor(im).float().permute(2, 0, 1) / 255
          loss_mask = torch.zeros(h, w)#.cuda()
          loss_mask[:, :]=1
        
          if h == w:
            im = torch.rot90(im, k=1, dims=(1, 2))
          else:
            mask_path=f'/data3/zihanwa3/Capstone-DSR/Processing/toy_exp/depth/depth_{c}.npz'
            depth = torch.tensor(np.load(mask_path)['depth_map']).float()#.cuda()
            depth=1/(depth+100.)#.cuda()

          cam = setup_camera(w, h, k, w2c, near=0.01, far=50, im=im, depth=depth, loss_mask=loss_mask)
          dataset.append(cam)
          '''try:
            mask_path=f"/data3/zihanwa3/Capstone-DSR/Processing/toy_exp/mask/cam_{c}.png"
            mask = Image.open(mask_path).convert("L")
            transform = transforms.ToTensor()
            mask_tensor = transform(mask).squeeze(0)
            anti_mask_tensor = mask_tensor > 1e-5
            dataset.append({'cam': cam, 'im': im, 'id': iiiindex, 'antimask': anti_mask_tensor, 'gt_depth':depth, 'vis': True}) 
          except: 
            dataset.append({'cam': cam, 'im': im, 'id': c, 'vis': True})  '''
      return dataset


class Scene:

    gaussians : GaussianModel

    def __init__(self, gaussians : GaussianModel, seq='cmu_bike', shuffle=True): 
        """b
        :param path: Path to colmap scene main folder.
        """
        self.gaussians = gaussians

        '''        if load_iteration:
                    if load_iteration == -1:
                        self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
                    else:
                        self.loaded_iter = load_iteration
                    print("Loading trained model at iteration {}".format(self.loaded_iter))'''

        #self.train_cameras = {}
        #self.test_cameras = {}
        md = json.load(open(f"./data_ego/cmu_bike/Dy_train_meta.json", 'r'))
        t=294
        self.dataset = get_dataset(t, md, seq, mode='ego_only')
        self.gaussians.init_params(path=f"./data_ego/{seq}/init_correct.npz")

        self.gaussians.init_variables()
        self.train_cameras = []
        self.train_cameras= get_dataset(t, md, seq, mode='ego_only')

        '''if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]'''


        #self.gaussians.load_ply(f"./data_ego/{seq}/init_correct.npz")

        '''if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, scene_info.semantic_feature_dim, args.speedup) 
        '''
    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainBatch(self):
        return self.dataset

    def getTrainCameras(self):
        return self.train_cameras