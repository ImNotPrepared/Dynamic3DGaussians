
import torch
import numpy as np
import pandas as pd
import open3d as o3d
import time
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
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

near, far = 1e-7, 50.0
view_scale = 1





def vis_depth(depth):
  depth = np.array(depth)
  depth_normalized = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
  
  depth_normalized = depth_normalized.astype(np.uint8)
  colored_depth = (plt.cm.plasma(depth_normalized / 255.0)[:, :, :3] * 255).astype(np.uint8)
  return colored_depth

fps = 10
def render_wander_path(c2w):
  """Rendering circular path."""
  hwf = c2w[:, 4:5]
  num_frames = 50
  max_disp = 48.0

import numpy as np
C0 = 0.28209479177387814
def rgb_to_spherical_harmonic(rgb):
    return (rgb-0.5) / C0


def spherical_harmonic_to_rgb(sh):
    return sh*C0 + 0.5

def save_ply_splat(path, means, scales, rotations, rgbs, opacities, normals=None):
    if normals is None:
        normals = np.zeros_like(means)


    colors = rgb_to_spherical_harmonic(rgbs)

    if scales.shape[1] == 1:
        scales = np.tile(scales, (1, 3))

    attrs = ['x', 'y', 'z',
             'nx', 'ny', 'nz',
             'f_dc_0', 'f_dc_1', 'f_dc_2',
             'opacity',
             'scale_0', 'scale_1', 'scale_2',
             'rot_0', 'rot_1', 'rot_2', 'rot_3',]

    dtype_full = [(attribute, 'f4') for attribute in attrs]
    elements = np.empty(means.shape[0], dtype=dtype_full)
    for lis in [means, normals, colors, opacities, scales, rotations]:
      print(lis.shape)
    attributes = np.concatenate((means, normals, colors, opacities, scales, rotations), axis=1)
    elements[:] = list(map(tuple, attributes))
    from plyfile import PlyData, PlyElement
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)

    print(f"Saved PLY format Splat to {path}")



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

    print(q_values_tensor)
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

# Function to convert cartesian coordinates to spherical coordinates (longitude, latitude, radius)
def cartesian_to_spherical(cartesian_coords):
    x, y, z = cartesian_coords
    radius = np.sqrt(x**2 + y**2 + z**2)
    lon = np.arctan2(y, x)
    lat = np.arcsin(z / radius)
    return lon, lat, radius

# Convert spherical coordinates (longitude, latitude, radius) to cartesian coordinates
def spherical_to_cartesian(lon, lat, radius):
    x = radius * np.cos(lat) * np.cos(lon)
    y = radius * np.cos(lat) * np.sin(lon)
    z = radius * np.sin(lat)
    return np.array([x, y, z])

# Function to interpolate between two points on a sphere
def spherical_interpolation(p1, p2, num_points):
    # Convert to spherical coordinates
    lon1, lat1, _ = cartesian_to_spherical(p1)
    lon2, lat2, _ = cartesian_to_spherical(p2)
    
    # Interpolate angles
    lons = np.linspace(lon1, lon2, num_points)
    lats = np.linspace(lat1, lat2, num_points)
    
    # Convert back to cartesian coordinates
    points = np.array([spherical_to_cartesian(lon, lat, 1) for lon, lat in zip(lons, lats)])
    return points

# Interpolate rotations using Slerp
def interpolate_rotations(r1, r2, num_points):
    from scipy.spatial.transform import Rotation as R, Slerp
    slerp = Slerp([0, 1], R.from_quat([r1, r2]))
    fractions = np.linspace(0, 1, num_points)
    interpolated_rotations = slerp(fractions).as_quat()
    return interpolated_rotations

# Function to interpolate poses between multiple points on a sphere
def interpolate_between_four(positions, orientations, num_points):
    interpolated_positions = []
    interpolated_orientations = []

    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            p1 = positions[i]
            p2 = positions[j]
            r1 = orientations[i]
            r2 = orientations[j]

            interpolated_positions.append(spherical_interpolation(p1, p2, num_points))
            interpolated_orientations.append(interpolate_rotations(r1, r2, num_points))

    return np.vstack(interpolated_positions), np.vstack(interpolated_orientations)


def visualize_all(seq, exp):
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

    # Define base path
    params_dir = os.path.join('./output', exp, seq)

    params = {}
    for filename in os.listdir(params_dir):
        if filename.endswith(".npz"):
            params_path = os.path.join(params_dir, filename)
            params = dict(np.load(params_path, allow_pickle=True))
            base_data_path = './data_ego'
            base_visuals_path = f'./visuals/{exp}'+'/visuals'
            base_output_path = './'

            # Ensure directories exist
            os.makedirs(base_visuals_path, exist_ok=True)
            os.makedirs(os.path.join(base_visuals_path, filename, 'ego'), exist_ok=True)
            os.makedirs(os.path.join(base_visuals_path, filename, 'sys'), exist_ok=True)
            os.makedirs(os.path.join(base_visuals_path, filename, 'rot'), exist_ok=True)
            os.makedirs(base_output_path, exist_ok=True)



            print(params_path)
            scene_data, is_fg = load_scene_data_knownpath(seq, exp, params_path)
            scene_data=scene_data[0]
            #print(scene_data.keys()) dict_keys(['means3D', 'colors_precomp', 'rotations', 'opacities', 'scales', 'means2D'])
            path = base_visuals_path+'points.ply'
            file_path = os.path.join(base_data_path, seq, 'train_meta.json') 

            with open(file_path, 'r') as file:
                json_file = json.load(file)

            points_list = []
            rbgs_list = []
            frame_index, cam_index = 0, 0 
            tto = []
            depths=[]

            for cam_index in range(1400):
                h, w = json_file['hw'][cam_index]
                def_pix = torch.tensor(
                    np.stack(np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5, 1), -1).reshape(-1, 3)).cuda().float()
                pix_ones = torch.ones(h * w, 1).cuda().float() 
                image_size, radius = (h, w), 0.01
                RENDER_MODE = 'color'
                w2c, k = (np.array((json_file['w2c'])[frame_index][cam_index]), np.array(json_file['k'][frame_index][cam_index]))
                w2c = np.linalg.inv(w2c)
                camera = PerspectiveCameras(device="cuda", R=w2c[None, ...], K=k[None, ...])
                
                im, depth = render(w2c, k, scene_data, w, h, near, far)
                im=im.clip(0,1)
                im = torch.rot90(im, k=-1, dims=(1, 2))
                first_ = np.array(im.detach().cpu().permute(1, 2, 0).numpy()[:, :, ::-1]) * 255
                #cv2.imwrite(os.path.join(base_visuals_path, 'ego', f'cam_{cam_index}.png'), first_)  

                first_ = np.array(im.detach().cpu().permute(1, 2, 0).numpy()) * 255
                first_ = cv2.resize(first_.astype(np.uint8), (256, 256), interpolation=cv2.INTER_LINEAR)

                image = Image.fromarray((first_).astype(np.uint8))
                tto.append(image)
                depth = torch.rot90(depth, k=-1, dims=(1, 2))
                depths.append(np.transpose(vis_depth(1/np.clip(depth.detach().cpu().numpy(), near, far))[0], (0, 2, 1)))
            imageio.mimsave(os.path.join(base_visuals_path, filename, 'sys', 'ego_depth.gif'), depths, fps=21)
            imageio.mimsave(os.path.join(base_visuals_path, filename, 'sys', 'ego.gif'), tto, fps=21)

            interval = 27
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

                im, depth = render(w2c, k, scene_data, w, h, near, far)
                print(depth.max(),depth.min(),depth.shape)
                np.savez(f'/data3/zihanwa3/Capstone-DSR/Processing/toy_exp/gaussian_depth/cam_{cam_index-1399}', depth=depth.detach().cpu())
                
                
                im=im.clip(0,1)
                new_width, new_height = 256, 144  # desired dimensions
                im=im.detach().cpu().permute(1, 2, 0).numpy()
                im = cv2.resize(im, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

                first_ = np.array(im[:, :, ::-1]) * 255
                cv2.imwrite(os.path.join(base_visuals_path, filename, 'sys', f'cam_{cam_index}.png'), first_)
                
                first_ = np.array(depth.detach().cpu().permute(1, 2, 0).numpy()[:, :, ::-1]) * 255
                cv2.imwrite(os.path.join(base_visuals_path, filename,'sys', f'depth_{cam_index}.png'), first_)


                num_frames = 20
                angles = torch.linspace(0, 2 * np.pi, num_frames)

                images = []
                depths = []
                for i, angle in enumerate(angles):
                    cos_a, sin_a = torch.cos(angle), torch.sin(angle)
                    rotation_matrix = torch.tensor([
                        [cos_a, 0, sin_a, 0],
                        [0, 1, 0, 0],
                        [-sin_a, 0, cos_a, 0],
                        [0, 0, 0, 1]
                    ], device="cuda").unsqueeze(0)



                    camera_rotation = rotation_matrix.cpu() @ w2c[None, ...]
                    im, depth = render(camera_rotation[0], k, scene_data, w, h, near, far)
                    im=im.clip(0,1)
                    first_ = np.array(im.detach().cpu().permute(1, 2, 0).numpy()[:, :, ::-1]) * 255
                    im = np.array(im.detach().cpu().permute(1, 2, 0).numpy()) * 255
                    new_width, new_height = 256, 144  # desired dimensions
                    im = cv2.resize(im, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                    image = Image.fromarray((im).astype(np.uint8))

                    cv2.imwrite(os.path.join(base_visuals_path, filename,'rot', f'cam_{angle}.png'), first_)
                    print('image_shape', np.array(image).shape)
                    print('depth_shape', vis_depth(1/np.clip(depth[0].detach().cpu().numpy(), near, far)).shape)
                    images.append(np.array(image))

                    depths.append(vis_depth(1/np.clip(depth[0].detach().cpu().numpy(), near, far)))

                imageio.mimsave(os.path.join(base_visuals_path, filename,'rot', f'cam_{cam_index}_depth.gif'), depths, fps=5)
                imageio.mimsave(os.path.join(base_visuals_path, filename,'rot', f'cam_{cam_index}.gif'), images, fps=5)
                


if __name__ == "__main__":
    import os
    import sys
    sequence = 'cmu_bike'
    exp_name = sys.argv[1]
    
    visualize_all(sequence, exp_name)
