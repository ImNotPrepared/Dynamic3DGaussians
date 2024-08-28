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
import imageio.v3 as iio
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

import numpy as np
from scipy.spatial.transform import Rotation as R

# Function to perform spherical linear interpolation (slerp)
def slerp(p0, p1, t):
    omega = np.arccos(np.dot(p0 / np.linalg.norm(p0), p1 / np.linalg.norm(p1)))
    so = np.sin(omega)
    return np.sin((1.0 - t) * omega) / so * p0 + np.sin(t * omega) / so * p1

# Function to interpolate between two cameras
def interpolate_between_two_cameras(w2c_start, w2c_end, k_start, k_end, t):
    # Interpolate rotation (convert to quaternions)
    quat_start = R.from_matrix(w2c_start[:3, :3]).as_quat()
    quat_end = R.from_matrix(w2c_end[:3, :3]).as_quat()
    interpolated_quat = slerp(quat_start, quat_end, t)
    interpolated_rot = R.from_quat(interpolated_quat).as_matrix()

    # Interpolate translation
    translation_start = w2c_start[:3, 3]
    translation_end = w2c_end[:3, 3]
    interpolated_translation = translation_start * (1 - t) + translation_end * t

    # Combine rotation and translation
    interpolated_w2c = np.eye(4)
    interpolated_w2c[:3, :3] = interpolated_rot
    interpolated_w2c[:3, 3] = interpolated_translation

    # Interpolate intrinsics
    interpolated_intrinsics = k_start * (1 - t) + k_end * t

    return interpolated_w2c, interpolated_intrinsics

def make_video_divisble(
    video, block_size=16
) :
    H, W = video.shape[1:3]
    H_new = H - H % block_size
    W_new = W - W % block_size
    return video[:, :H_new, :W_new]


# Function to interpolate between a list of cameras
def interpolate_cameras(json_file, cam_indices, frame_index=0):
    all_interpolated_cameras = []

    for i in range(len(cam_indices) - 1):
        start_cam_index = cam_indices[i]
        end_cam_index = cam_indices[i + 1]

        #(np.array((json_file['w2c'])[frame_index][cam_index])

        # Get camera parameters
        print(start_cam_index)
        w2c_start = np.array(json_file['w2c'][frame_index][start_cam_index])
        w2c_end = np.array(json_file['w2c'][frame_index][end_cam_index])
        k_start = np.array(json_file['k'][frame_index][start_cam_index])
        k_end = np.array(json_file['k'][frame_index][end_cam_index])

        # Interpolate between the two cameras with t=0.5 for midpoint
        interpolated_w2c, interpolated_intrinsics = interpolate_between_two_cameras(
            w2c_start, w2c_end, k_start, k_end, 0.5
        )

        all_interpolated_cameras.append((interpolated_w2c, interpolated_intrinsics))

    return all_interpolated_cameras


def render_wander_path(c2w):
  """Rendering circular path."""
  hwf = c2w[:, 4:5]
  num_frames = 50
  max_disp = 48.0
import numpy as np

import numpy as np

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



def render_360_pc(point_cloud, image_size=256, output_path='./point_cloud.gif', device=None):
    # Create a point cloud renderer
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from scipy.spatial.transform import Rotation as R, Slerp

    # Load the CSV files
    file_path_calibs = '/data3/zihanwa3/Capstone-DSR/Dynamic3DGaussians/data_ego/cmu_bike/gopro_calibs.csv'

    calib_data = pd.read_csv(file_path_calibs)

    # Extract relevant columns
    camera_position_data = calib_data[['tx_world_cam', 'ty_world_cam', 'tz_world_cam']]
    camera_orientation_data = calib_data[['qx_world_cam', 'qy_world_cam', 'qz_world_cam', 'qw_world_cam']]


    num_interpolated_points = 20

    # Extract positions and orientations from the dataframe
    positions = camera_position_data.values
    orientations = camera_orientation_data.values
    # Interpolate the poses between all four cameras
    interpolated_positions, interpolated_orientations = interpolate_between_four(positions, orientations, num_interpolated_points)


    raster_settings = PointsRasterizationSettings(image_size=image_size, radius=0.003)
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor()
    )

    num_views = len(interpolated_positions)
    images = []

    for i in range(num_views):
        # Get the position and orientation for the current view
        position = interpolated_positions[i]
        orientation = interpolated_orientations[i]

        # Convert orientation quaternion to rotation matrix
        r = R.from_quat(orientation).as_matrix()

        # Convert position and rotation to tensor
        R_tensor = torch.tensor(r, dtype=torch.float32).unsqueeze(0)
        T_tensor = torch.tensor(position, dtype=torch.float32).unsqueeze(0)

        # Create the camera with the current view's rotation and translation
        cameras = FoVPerspectiveCameras(R=R_tensor, T=T_tensor, device=device)

        # Render the point cloud
        rend = renderer(point_cloud, cameras=cameras)
        rend = rend[0, ..., :3].cpu().numpy()

        image = Image.fromarray((rend * 255).astype(np.uint8))
        images.append(np.array(image))

    # Save the rendered images as a GIF
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
    base_visuals_path = f'./dyn_visuals/{exp}'+'/visuals'
    base_output_path = './'

    # Ensure directories exist
    os.makedirs(base_visuals_path, exist_ok=True)
    os.makedirs(os.path.join(base_visuals_path, 'ego'), exist_ok=True)
    os.makedirs(os.path.join(base_visuals_path, 'sys'), exist_ok=True)
    os.makedirs(os.path.join(base_visuals_path, 'rot'), exist_ok=True)
    os.makedirs(base_output_path, exist_ok=True)

    params_path = os.path.join('./output', exp,seq,  "params.npz")

    params = dict(np.load(params_path, allow_pickle=True))


    scene_data, is_fg = load_scene_data(seq, exp)
    scene_data=scene_data[0]
    
    path = base_visuals_path+'points.ply'
    means = params['means3D']
    scales = params['log_scales']
    rotations = params['unnorm_rotations']
    rgbs = params['rgb_colors']
    opacities = params['logit_opacities']
    save_ply_splat(path, means[0], scales, rotations[0], rgbs[0], opacities)


    file_path = os.path.join(base_data_path, seq, 'Dy_train_meta.json') 
    with open(file_path, 'r') as file:
        json_file = json.load(file)

    points_list = []
    rbgs_list = []
    frame_index, cam_index = 0, 0 
    tto = []

    '''    for cam_index in range(1400):
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

        imageio.mimsave(os.path.join(base_visuals_path, 'sys', 'ego.gif'), tto, fps=21)
    '''
    scene_data, is_fg = load_scene_data(seq, exp)

    md=json_file
    # Example usage
    cam_indices = [1, 2, 3, 4, 1]
    reversed_range = list(range(111, -1, -5))


    #video Tracks: # (B,T,N,2)



    from tqdm import tqdm

    interpolated_cameras = interpolate_cameras(json_file, cam_indices)
      
    for cam_index, (w2c, k) in enumerate(interpolated_cameras):
        h, w = md['hw'][3]
        def_pix = torch.tensor(
            np.stack(np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5, 1), -1).reshape(-1, 3)).cuda().float()
        pix_ones = torch.ones(h * w, 1).cuda().float()
        image_size, radius = (h, w), 0.01
        RENDER_MODE = 'color'
        im, depth = render(w2c, k, scene_data[0], w, h, near, far)
        #np.savez(f'/data3/zihanwa3/Capstone-DSR/Processing/toy_exp/gaussian_depth/cam_{cam_index-1399}', depth=depth.detach().cpu())
        im=im.clip(0,1)
        new_width, new_height = 256, 144  # desired dimensions
        im=im.detach().cpu().permute(1, 2, 0).numpy()
        im = cv2.resize(im, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        images = []
        video=[]
        #294-111=183
        reversed_range = list(range(111, -1, -5))
        w2c = np.linalg.inv(w2c)

        for i, frame_index in tqdm(enumerate(reversed_range)):
            #w2c, k = (np.array((json_file['w2c'])[frame_index][cam_index]), np.array(json_file['k'][frame_index][cam_index]))
            
            im, depth = render(w2c, k, scene_data[i], w, h, near, far)
            im=im.clip(0,1)
            first_ = np.array(im.detach().cpu().permute(1, 2, 0).numpy()[:, :, ::-1]) * 255
            



            '''if i>1:
              tracks_3d = (scene_data[i]['means3D'] - scene_data[i-1]['means3D']).unsqueeze(1).detach().cpu().double()#[:1000]  # (N, B, 3)
              print(f"{tracks_3d.shape=}")
              tracks_2d = torch.einsum(
                  "ij,bjk,nbk->nbi", torch.tensor(k).double(), torch.tensor(w2c[None, ...])[:, :3].double(), F.pad(tracks_3d, (0, 1), value=1.0)
              )
              tracks_2d = tracks_2d[..., :2] / tracks_2d[..., 2:]
              print(f"{tracks_2d.shape=}")
              out_img = draw_tracks_2d(im.permute(1, 2, 0), tracks_2d)
              video.append(out_img)'''
            im = np.array(im.detach().cpu().permute(1, 2, 0).numpy()) * 255
            new_width, new_height = 256, 144  # desired dimensions
            im = cv2.resize(im, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            image = Image.fromarray((im).astype(np.uint8))
            images.append(np.array(image))

        #video = np.stack(video, 0)
        #print(video.shape)
        #iio.imwrite(os.path.join(base_visuals_path, 'ego', f'tracks_{cam_index}.mp4'), make_video_divisble(video), fps=5)
        #imageio.mimsave(os.path.join(base_visuals_path, 'ego', f'tracks_{cam_index}.gif'), video, fps=5)
        imageio.mimsave(os.path.join(base_visuals_path, 'ego', f'inter_{cam_index}.gif'), images, fps=5)

      


    for cam_index in range(1,5):
        h, w = md['hw'][cam_index]
        def_pix = torch.tensor(
            np.stack(np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5, 1), -1).reshape(-1, 3)).cuda().float()
        pix_ones = torch.ones(h * w, 1).cuda().float()
        image_size, radius = (h, w), 0.01
        RENDER_MODE = 'color'
        w2c, k = (np.array((json_file['w2c'])[0][cam_index]), np.array(json_file['k'][0][cam_index]))
        w2c = np.linalg.inv(w2c)

        im, depth = render(w2c, k, scene_data[0], w, h, near, far)
        #np.savez(f'/data3/zihanwa3/Capstone-DSR/Processing/toy_exp/gaussian_depth/cam_{cam_index-1399}', depth=depth.detach().cpu())
        im=im.clip(0,1)
        new_width, new_height = 256, 144  # desired dimensions
        im=im.detach().cpu().permute(1, 2, 0).numpy()
        im = cv2.resize(im, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        first_ = np.array(im[:, :, ::-1]) * 255
        cv2.imwrite(os.path.join(base_visuals_path, 'sys', f'cam_{cam_index}.png'), first_)
        
        first_ = np.array(depth.detach().cpu().permute(1, 2, 0).numpy()[:, :, ::-1]) * 255
        cv2.imwrite(os.path.join(base_visuals_path, 'sys', f'depth_{cam_index}.png'), first_)

        images = []
        #294-111=183
        reversed_range = list(range(111, -1, -5))

        for i, frame_index in enumerate(reversed_range):
            print(frame_index,  cam_index)
            w2c, k = (np.array((json_file['w2c'])[frame_index][cam_index]), np.array(json_file['k'][frame_index][cam_index]))
            w2c = np.linalg.inv(w2c)

            im, depth = render(w2c, k, scene_data[i], w, h, near, far)
            im=im.clip(0,1)
            first_ = np.array(im.detach().cpu().permute(1, 2, 0).numpy()[:, :, ::-1]) * 255
            im = np.array(im.detach().cpu().permute(1, 2, 0).numpy()) * 255
            new_width, new_height = 256, 144  # desired dimensions
            im = cv2.resize(im, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            image = Image.fromarray((im).astype(np.uint8))

            #cv2.imwrite(os.path.join(base_visuals_path, 'rot', f'cam_{angle}.png'), first_)
            images.append(np.array(image))
        imageio.mimsave(os.path.join(base_visuals_path, 'rot', f'cam_{cam_index}.gif'), images, fps=5)



def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    from plyfile import PlyData, PlyElement
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)




if __name__ == "__main__":
    import os
    import sys
    import torch.nn.functional as F
    
    
    
    sequence = 'cmu_bike'
    exp_name = sys.argv[1]
    
    visualize(sequence, exp_name)