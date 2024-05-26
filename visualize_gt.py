
import torch
import numpy as np
import pandas as pd
import open3d as o3d
import time
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from colormap import colormap
import matplotlib.pyplot as plt
from pytorch3d.renderer import (

    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    PerspectiveCameras
)
import os
import imageio
import torch
import pytorch3d
from pytorch3d.renderer.cameras import FoVPerspectiveCameras
import cv2

from PIL import Image
from vis_utils import *

near, far = 0.01, 50.0
view_scale = 1

fps = 10
def render_wander_path(c2w):
  """Rendering circular path."""
  hwf = c2w[:, 4:5]
  num_frames = 50
  max_disp = 48.0
import numpy as np

import numpy as np

import numpy as np

def render_wander_path(extrinsics, intrinsics):
    """
    Render a circular path given extrinsic and intrinsic matrices.

    Parameters:
    extrinsics (np.ndarray): 4x4 camera-to-world extrinsic matrix.
    intrinsics (np.ndarray): 3x3 intrinsic matrix.

    Returns:
    list: List of 4x4 extrinsic matrices along the circular path.
    """
    num_frames = 50
    max_disp = 48.0

    # Extract focal length from intrinsics
    focal_length = intrinsics[0, 0]
    max_trans = max_disp / focal_length
    output_poses = []

    for i in range(num_frames):
        x_trans = max_trans * np.sin(2.0 * np.pi * float(i) / float(num_frames))
        y_trans = 0.0
        z_trans = max_trans * np.cos(2.0 * np.pi * float(i) / float(num_frames)) / 2.0

        # Create the transformation matrix for this frame
        i_pose = np.eye(4)
        i_pose[:3, 3] = [x_trans, y_trans, z_trans]

        # Invert the transformation matrix
        i_pose_inv = np.linalg.inv(i_pose)

        # Calculate the new extrinsic matrix
        new_extrinsics = np.dot(extrinsics, i_pose_inv)

        output_poses.append(new_extrinsics)

    return output_poses



def load_data(cam_id, use_ndc=False, resize=True):
    s_id, e_id = int(cam_id)-1, int(cam_id)
    df = pd.read_csv('/data3/zihanwa3/Capstone-DSR/Dynamic3DGaussians/data_ego/cmu_bike/gopro_calibs.csv')[s_id:e_id]

    intrinsics =np.array(df[['image_width','image_height','intrinsics_0','intrinsics_1','intrinsics_2','intrinsics_3']].values.tolist())
    if resize:
      for item in intrinsics:
        w, h = 256, 144
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
    k = 0
    l = min(w, h)  # Ensure 'w' (width) and 'h' (height) are defined appropriately

    focal_length_ndc = torch.tensor([[f, f]], dtype=torch.float32)
    principal_point_ndc = torch.tensor([[px, py]], dtype=torch.float32)


    from pytorch3d.transforms import quaternion_to_matrix, Translate
    import torch

    R = quaternion_to_matrix(q_values_tensor.unsqueeze(0))  
    T = t_values_tensor.unsqueeze(0)
    camera = PerspectiveCameras(
        R = R,
        T = T,
        focal_length=focal_length_ndc,
        principal_point=principal_point_ndc,
        device=torch.device("cpu") 
    )
    return camera, camera, camera, R, T, fx, cx, cy

def get_points_renderer(image_size=512, radius=0.01, background_color=(1, 1, 1)):
    raster_settings = PointsRasterizationSettings(image_size=image_size, radius=radius,)
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color),
    )
    return renderer
def render_360_pc(point_cloud, image_size=256, output_path='./point_cloud.gif', device=None):
    renderer = get_points_renderer(image_size=image_size)
    num_views = 36
    angles = np.linspace(-180, 180, num_views, endpoint=False)
    images = []
    for i in range(num_views):
        R, T = pytorch3d.renderer.look_at_view_transform(
        dist=3,
        elev=0,
        azim=angles[i],
    )
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R,
        T=T,
        device=device
    )
        
        rend = renderer(point_cloud, cameras=cameras)
        rend = rend[0, ..., :3].cpu().numpy()

        image = Image.fromarray((rend * 255).astype(np.uint8))

        images.append(np.array(image))
    imageio.mimsave(output_path, images, fps=5)

def visualize(seq, exp):
    import os
    import time
    import json
    import numpy as np
    import torch
    import imageio
    from PIL import Image
    from pytorch3d.io import save_ply
    from pytorch3d.renderer import PerspectiveCameras
    start_time = time.time()

    # Define base paths
    base_data_path = './data_ego'
    base_visuals_path = f'./visuals/{exp}'+'/visuals'
    base_output_path = './'

    # Ensure directories exist
    os.makedirs(base_visuals_path, exist_ok=True)
    os.makedirs(os.path.join(base_visuals_path, 'sys'), exist_ok=True)
    os.makedirs(os.path.join(base_visuals_path, 'rot'), exist_ok=True)
    os.makedirs(base_output_path, exist_ok=True)

    scene_data, is_fg = load_scene_data(seq, exp)
    file_path = os.path.join(base_data_path, seq, 'train_meta.json') 

    with open(file_path, 'r') as file:
        json_file = json.load(file)

    points_list = []
    rbgs_list = []
    frame_index, cam_index = 0, 0 
    tto = []

    for cam_index in range(1400, 1404):
        h, w = json_file['hw'][cam_index]
        def_pix = torch.tensor(
            np.stack(np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5, 1), -1).reshape(-1, 3)).cuda().float()
        pix_ones = torch.ones(h * w, 1).cuda().float()
        image_size, radius = (h, w), 0.01
        RENDER_MODE = 'color'
        w2c, k = (np.array((json_file['w2c'])[0][cam_index]), np.array(json_file['k'][0][cam_index]))
        w2c = np.linalg.inv(w2c)
        camera = PerspectiveCameras(device="cuda", R=w2c[None, ...], K=k[None, ...])

        im, depth = render(w2c, k, scene_data[0], w, h, near, far)
          
        first_ = np.array(im.detach().cpu().permute(1, 2, 0).numpy()[:, :, ::-1]) * 255
        cv2.imwrite(os.path.join(base_visuals_path, 'sys', f'cam_{cam_index}.png'), first_)
        first_ = np.array(depth.detach().cpu().permute(1, 2, 0).numpy()[:, :, ::-1]) * 255
        cv2.imwrite(os.path.join(base_visuals_path, 'sys', f'depth_{cam_index}.png'), first_)

        pointclouds, pts, cols = rgbd2pcd(im, depth, w2c, k, def_pix, pix_ones, show_depth=(RENDER_MODE == 'depth'))
        point_cloud = Pointclouds(points=[pts], features=[cols]).to('cuda')
        
        points_list.append(pts)
        rbgs_list.append(cols)
        poses = render_wander_path(w2c, k)

        tto = []
        for i, pose in enumerate(poses):
            camera = PerspectiveCameras(device="cuda", R=pose[None, ...], K=k[None, ...])
            im, _ = render(pose, k, scene_data[0], w, h, near, far)
            im = np.array(im.detach().cpu().permute(1, 2, 0).numpy()) * 255
            image = Image.fromarray((im).astype(np.uint8))
            tto.append(image)
        imageio.mimsave(os.path.join(base_visuals_path, 'sys', f'{cam_index-1399}_aaa.gif'), tto, fps=10)
      
    points = torch.cat(points_list, dim=0)
    rgb = torch.cat(rbgs_list, dim=0)
    rgb = rgb.float()
    render_360_pc(point_cloud, image_size=(144, 256), output_path=os.path.join(base_output_path, 'point_cloud.gif'), device='cuda')
    save_ply(os.path.join(base_output_path, 'final_pt_cld.ply'), points)
    data = np.zeros((len(points), 7))
    data[:, :3], data[:, 3:6] = points, rgb
    data[:, 6] = np.ones((len(points)))
    np.savez(os.path.join(base_output_path, "final_pt_cld.npz"), data=data)
    print(f'Saved {len(data)}!')




if __name__ == "__main__":
    import os
    import sys
    sequence = sys.argv[1]
    exp_name = sys.argv[2]
    
    visualize(sequence, exp_name)
    # visualize_train(sequence, exp_name)

