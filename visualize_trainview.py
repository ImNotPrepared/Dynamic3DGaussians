
import torch
import numpy as np
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
from pytorch3d.renderer.cameras import FoVPerspectiveCameras
import cv2

from PIL import Image
from vis_utils import *

near, far = 1.0, 100.0
view_scale = 1

fps = 20
traj_frac = 90  # 4% of points
traj_length = 15

def extract_y_angle(w2c_matrix):
    """
    Extracts the Y angle (yaw) from a 4x4 world-to-camera (W2C) matrix.
    
    Args:
    w2c_matrix: A 4x4 numpy array representing the world-to-camera transformation matrix.
    
    Returns:
    y_angle: The yaw angle in radians.
    """
    # Ensure the matrix is a NumPy array
    w2c_matrix = np.array(w2c_matrix)

    # Extract the rotation part of the matrix
    R = w2c_matrix[:3, :3]

    # Calculate the yaw angle, assuming a rotation around the Y-axis
    y_angle = np.arctan2(-R[2, 0], R[0, 0])

    return y_angle

def visualize(seq, exp):
    start_time = time.time()
    import json
    #//cmu_bike   ['w', 'h', 'k', 'w2c', 'fn', 'cam_id']

    file_path = os.path.join('./data_ego', seq, 'train_meta.json')
    with open(file_path, 'r') as file:
        json_file = json.load(file)
    w, h = ((json_file['w']), json_file['h'])
    def_pix = torch.tensor(
        np.stack(np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5, 1), -1).reshape(-1, 3)).cuda().float()
    pix_ones = torch.ones(h * w, 1).cuda().float()
    image_size, radius = (h, w), 0.01
    RENDER_MODE='color'
    ### 150 27 4 40
    cam_id=1
    w2c, k = (np.array((json_file['w2c'])[100][cam_id]), np.array(json_file['k'][100][cam_id]))
    scene_data, is_fg = load_scene_data(seq, exp)
    for i in range(len(np.array((json_file['w2c'])))):
      im, depth = render(w2c, k, scene_data[i], w, h, near, far)
      pointclouds = rgbd2pcd(im, depth, w2c, k, def_pix, pix_ones, show_depth=(RENDER_MODE == 'depth'))
      frame=render_pointcloud_pytorch3d(w, h, w2c,k, image_size, radius, pointclouds, extract_y_angle(w2c))
      array = np.clip(frame, 0, 1)  # Assuming the array values are scaled between 0 and 1
      array = (array * 255).astype(np.uint8)  # Scale to 0-255 and convert to uint8
      frame = Image.fromarray(array)
      frame.save(os.path.join(f'/data3/zihanwa3/Capstone-DSR/Dynamic3DGaussians/vis/tr_view/{sequence}', f'scene_{i}_train.png'))


if __name__ == "__main__":
    exp_name = "debug_3"
    for sequence in ["cmu_bike"]:
        visualize(sequence, exp_name)


'''

    frames = []
    angles = [0.0, 3.3599999999999994, 6.719999999999999, 10.08, 13.439999999999998, 16.8, 20.16, 23.52, 26.879999999999995, 30.24, 33.6, 36.96, 40.32, 43.68, 47.04, 50.39999999999999, 53.75999999999999, 57.12, 60.48, 63.84, 67.2, 70.56, 73.92, 77.28, 80.64, 84.0, 87.36, 90.72, 94.08, 97.43999999999998, 100.79999999999998, 104.15999999999998, 107.51999999999998, 110.88, 114.24, 117.6, 120.96, 124.32, 127.68, 131.04, 134.4, 137.76, 141.12, 144.48, 147.84, 151.2, 154.56, 157.92, 161.28, 164.64, 168.0, 171.36, 174.72, 178.08, 181.44, 184.8, 188.16, 191.51999999999998, 194.87999999999997, 198.23999999999998, 201.59999999999997, 204.95999999999998, 208.31999999999996, 211.67999999999998, 215.03999999999996, 218.39999999999998, 221.76, 225.12, 228.48, 231.84, 235.2, 238.56, 241.92, 245.28, 248.64, 252.0, 255.36, 258.72, 262.08, 265.44, 268.8, 272.16, 275.52, 278.88, 282.24, 285.6, 288.96, 292.32, 299.04, 302.4, 305.76, 309.12, 312.48, 315.84, 319.2, 322.56, 325.92, 329.28, 332.64, 336.0, 339.36, 342.72, 346.08, 349.44, 352.8, 356.16, 359.52, 362.88, 366.24, 369.6, 372.96, 376.32, 379.68, 383.03999999999996, 386.4, 389.75999999999993, 393.11999999999995, 396.47999999999996, 399.84, 403.19999999999993, 406.55999999999995, 413.28, 416.63999999999993, 419.99999999999994, 423.35999999999996, 426.71999999999997, 430.0799999999999, 433.43999999999994, 436.79999999999995, 440.16, 443.52, 446.88, 450.24, 453.6, 456.96, 460.32, 463.68, 467.04, 470.4, 473.76, 477.12, 480.48, 483.84, 487.2, 490.56, 493.92, 497.28, 500.64]
    for t, y_angle in tqdm(enumerate(angles[:10])):
      w2c, k = init_camera(y_angle)
      im, depth = render(w2c, k, scene_data[t])
      pointclouds = rgbd2pcd(im, depth, w2c, k, show_depth=(RENDER_MODE == 'depth'))
      frame = render_pointcloud_pytorch3d(pointclouds, y_angle)
      array = np.clip(frame, 0, 1)  # Assuming the array values are scaled between 0 and 1
      array = (array * 255).astype(np.uint8)  # Scale to 0-255 and convert to uint8

      # Now, convert the processed array to a PIL Image
      frame = Image.fromarray(array)
      frames.append(frame)
    gif_path = os.path.join('./vis', "tennis.gif")
    imageio.mimwrite(gif_path, frames, duration=1000.0*(1/10.0), loop=0)
    ####

    

    num_views = 32
    azims = np.linspace(-180, 180, num_views)

    for i in tqdm(range(num_views), desc="Rendering"):

        dist = 6.0
        R, T = look_at_view_transform(dist = dist, azim=azims[i], elev=30.0, up=((0, -1, 0),))
        camera = PerspectiveCameras(
            focal_length=5.0 * dim/2.0, in_ndc=False,
            principal_point=((dim/2, dim/2),),
            R=R, T=T, image_size=(img_size,),
        ).to(args.device)

        img = img.detach().cpu().numpy()
        depth = depth.detach().cpu().numpy()

        img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)

        depth = depth[:, :, 0].astype(np.float32)  # (H, W)
        coloured_depth = colour_depth_q1_render(depth)  # (H, W, 3)

        concat = np.concatenate([img, coloured_depth, mask], axis = 1)
        resized = Image.fromarray(concat).resize((256*3, 256))
        resized.save(debug_path)

        imgs.append(np.array(resized))
        imageio.mimwrite(gif_path, imgs, duration=1000.0*(1/10.0), loop=0)

    start_time = time.time()
    num_timesteps = len(scene_data)
    passed_time = time.time() - start_time
    passed_frames = passed_time * fps
    if ADDITIONAL_LINES == 'trajectories':
        t = int(passed_frames % (num_timesteps - traj_length)) + traj_length  # Skip t that don't have full traj.
    else:
        t = int(passed_frames % num_timesteps)



'''