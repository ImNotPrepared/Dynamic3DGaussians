
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
    start_time = time.time()
    import json
    scene_data, is_fg = load_scene_data(seq, exp)
    file_path = os.path.join('./data_ego', seq, 'train_meta.json') 
    with open(file_path, 'r') as file:
        json_file = json.load(file)
    

    points_list=[]
    rbgs_list=[]
    #### BEGIN ####
    frame_index, cam_index = 0, 0 
    for frame_index in range(5):
      h, w = json_file['hw'][cam_index]
      def_pix = torch.tensor(
        np.stack(np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5, 1), -1).reshape(-1, 3)).cuda().float()
      pix_ones = torch.ones(h * w, 1).cuda().float()
      image_size, radius = (h, w), 0.01
      RENDER_MODE='color'
      w2c, k = (np.array((json_file['w2c'])[0][cam_index]), np.array(json_file['k'][0][cam_index]))
      w2c=np.linalg.inv(w2c)



    for cam_index in range(5):
      h, w = json_file['hw'][cam_index]
      def_pix = torch.tensor(
        np.stack(np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5, 1), -1).reshape(-1, 3)).cuda().float()
      pix_ones = torch.ones(h * w, 1).cuda().float()
      image_size, radius = (h, w), 0.01
      RENDER_MODE='color'
      w2c, k = (np.array((json_file['w2c'])[0][cam_index]), np.array(json_file['k'][0][cam_index]))
      w2c=np.linalg.inv(w2c)

      camera = PerspectiveCameras(device="cuda", R=w2c[None, ...], K=k[None, ...])

      im, depth = render(w2c, k, scene_data[0], w, h, near, far)
          
      first_=np.array(im.detach().cpu().permute(1, 2, 0).numpy()[:, :, ::-1]) * 255
      
      cv2.imwrite(f'./visuals/trainview/sys/cam_{cam_index}.png', first_)
      pointclouds, pts, cols = rgbd2pcd(im, depth, w2c, k, def_pix, pix_ones, show_depth=(RENDER_MODE == 'depth'), )
      point_cloud = Pointclouds(points=[pts], features=[cols]).to('cuda')
      device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      render_360_pc(point_cloud, image_size=(144,256), output_path='./point_cloud.gif', device='cuda')
      points_list.append(pts)
      rbgs_list.append(cols)
      poses  = render_wander_path(w2c, k)

      tto=[]
      for i, pose in enumerate(poses):
        print(pose.shape)
        camera = PerspectiveCameras(device="cuda", R=pose[None, ...], K=k[None, ...])
        im, _ = render(pose, k, scene_data[0], w, h, near, far)
        im=np.array(im.detach().cpu().permute(1, 2, 0).numpy()) * 255
        image = Image.fromarray((im).astype(np.uint8))
        first_=np.array(im[:, :, ::-1]) 
        #cv2.imwrite(f'./visuals/trainview/sys/{cam_index}/{i}.png', first_)
        tto.append(image)
      imageio.mimsave(f'./visuals/trainview/sys/{cam_index}/aaa.gif', tto, fps=10)
      
    num_frames=20
    angles = torch.linspace(0, 2 * np.pi, num_frames)  # 0 to 360 degrees in radians

    # Record all frames
    images=[]
    for i, angle in enumerate(angles):
        cos_a, sin_a = torch.cos(angle), torch.sin(angle)
        rotation_matrix = torch.tensor([
            [cos_a, 0, sin_a, 0],
            [0, 1, 0, 0],
            [-sin_a, 0, cos_a, 0],
            [0, 0, 0, 1]
        ], device="cuda").unsqueeze(0)
 
        camera_rotation = rotation_matrix.cpu() @ w2c[None, ...]
        im, depth = render(camera_rotation[0], k, scene_data[0], w, h, near, far)
        first_=np.array(im.detach().cpu().permute(1, 2, 0).numpy()[:, :, ::-1]) * 255
        im=np.array(im.detach().cpu().permute(1, 2, 0).numpy()) * 255
        image = Image.fromarray((im).astype(np.uint8))

        #image = Image.fromarray(first_)
        cv2.imwrite(f'./visuals/trainview/rot/cam_{angle}.png', first_)
        images.append(np.array(image))
    imageio.mimsave('./visuals/trainview/rot/cam.gif', images, fps=5)



    from pytorch3d.io import save_ply
    points = torch.cat(points_list, dim=0)
    rgb=torch.cat(rbgs_list, dim=0)
    data=np.zeros((len(points), 7))
    data[:, :3], data[:, 3:6] = points, rgb
    data[:, 6] = np.ones((len(points)))
    np.savez("final_pt_cld.npz", data=data)
    print(f'Saved {len(data)}!')

    points = torch.cat(points_list, dim=0)
    rgb = torch.cat(rbgs_list, dim=0)

    # Convert RGB from 0-255 to 0-1 if necessary
    rgb = rgb.float() / 255.0

    # Save to PLY
    save_ply("final_pt_cld.ply", points)
def visualize_train(seq, exp):
    start_time = time.time()
    import json
    scene_data, is_fg = load_scene_data(seq, exp)
    file_path = os.path.join('./data_ego', seq, 'train_meta.json') 
    with open(file_path, 'r') as file:
        json_file = json.load(file)
    h, w = json_file['hw'][1]
    def_pix = torch.tensor(
        np.stack(np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5, 1), -1).reshape(-1, 3)).cuda().float()
    pix_ones = torch.ones(h * w, 1).cuda().float()
    image_size, radius = (h, w), 0.01
    RENDER_MODE='color'
    points_list=[]
    rbgs_list=[]

    #### BEGIN ####
    def initialize_params_zoe():
        #init_pt_cld = np.load(f"init_pt_cld.npz")["data"]
        init_pt_cld = np.load(f"/data3/zihanwa3/Capstone-DSR/Appendix/ZoeDepth/init_pt_cld.npz")["data"]
        
        return torch.tensor(init_pt_cld[:, :3], dtype=torch.float), torch.tensor(init_pt_cld[:, 3:6],dtype=torch.float)
    pts, cols=initialize_params_zoe()
    for cam_index in range(1, 5):
      print(np.array((json_file['w2c'])))
      w2c, k = (np.array((json_file['w2c'])[0][cam_index])[None,...], np.array(json_file['k'][0][cam_index])[None,...])
      w2c, k = (np.array((json_file['w2c'])[0][cam_index]), np.array(json_file['k'][0][cam_index]))
      print(w2c, k)
      w2c=np.linalg.inv(w2c)
      
      _, _, camera, R, t, f_x, c_x, c_y =load_data(cam_index, resize=True)
      print('shape',camera.transform_points(pts[None,...]).shape)
      screen_points = camera.transform_points(pts[None, ...])[0]

    # 初始化深度图和二进制掩码
      depth_map = torch.full((h, w), float('inf'))
      binary_mask = torch.zeros((h, w))

    # 更新深度图和二进制掩码
      for i, (x, y, z) in enumerate(screen_points):
          x_pixel, y_pixel = int(x), int(y)
          if 0 <= x_pixel < w and 0 <= y_pixel < h:
            current_depth = z
            if current_depth < depth_map[y_pixel, x_pixel]:  # 检查深度，仅更新最近的点
              depth_map[y_pixel, x_pixel] = current_depth

      first_=np.array(depth_map.detach().cpu().numpy()) * 255
      
      cv2.imwrite(f'./visuals/trainview/gt/cam_{cam_index}.png', first_)
      
      point_cloud = Pointclouds(points=[pts], features=[cols]).to('cuda')
      device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      render_360_pc(point_cloud, image_size=(144,256), output_path='./point_cloud.gif', device='cuda')
      for i, pose in enumerate(poses):
        print(pose.shape)


if __name__ == "__main__":
    exp_name = "debug_3"
    for sequence in ["cmu_bike"]:
        visualize(sequence, exp_name)
        #visualize_train(sequence, exp_name)


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