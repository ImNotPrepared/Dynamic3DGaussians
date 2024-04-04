
import torch
import numpy as np
import open3d as o3d
import time
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from helpers import setup_camera, quat_mult
from external import build_rotation
from colormap import colormap
import matplotlib.pyplot as plt
from copy import deepcopy
from pytorch3d.structures import Pointclouds
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras,
    PointLights, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
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
from pytorch3d.renderer import (
    look_at_view_transform,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
)
from PIL import Image
from pytorch3d.transforms import RotateAxisAngle
from tqdm import tqdm


def rgbd2pcd(im, depth, w2c, k, def_pix, pix_ones, show_depth=False, project_to_cam_w_scale=None):
    d_near = 1.5
    d_far = 6
    invk = torch.inverse(torch.tensor(k).cuda().float())
    c2w = torch.inverse(torch.tensor(w2c).cuda().float())
    radial_depth = depth[0].reshape(-1)
    def_rays = (invk @ def_pix.T).T
    def_radial_rays = def_rays / torch.linalg.norm(def_rays, ord=2, dim=-1)[:, None]
    pts_cam = def_radial_rays * radial_depth[:, None]
    z_depth = pts_cam[:, 2]
    if project_to_cam_w_scale is not None:
        pts_cam = project_to_cam_w_scale * pts_cam / z_depth[:, None]
    pts4 = torch.concat((pts_cam, pix_ones), 1)
    pts = (c2w @ pts4.T).T[:, :3]
    if show_depth:
        cols = ((z_depth - d_near) / (d_far - d_near))[:, None].repeat(1, 3)
    else:
        cols = torch.permute(im, (1, 2, 0)).reshape(-1, 3)

    pts = pts.contiguous().float().cpu()
    cols = cols.contiguous().float().cpu()
    pointclouds = Pointclouds(points=[pts], features=[cols])

    return pointclouds
def load_scene_data(seq, exp, seg_as_col=False):
    params = dict(np.load(f"./output/{exp}/{seq}/params.npz"))
    params = {k: torch.tensor(v).cuda().float() for k, v in params.items()}
    is_fg = params['seg_colors'][:, 0] > 0.5
    scene_data = []
    for t in range(len(params['means3D'])):
        rendervar = {
            'means3D': params['means3D'][t],
            'colors_precomp': params['rgb_colors'][t] if not seg_as_col else params['seg_colors'],
            'rotations': torch.nn.functional.normalize(params['unnorm_rotations'][t]),
            'opacities': torch.sigmoid(params['logit_opacities']),
            'scales': torch.exp(params['log_scales']),
            'means2D': torch.zeros_like(params['means3D'][0], device="cuda")
        }

        scene_data.append(rendervar)
    return scene_data, is_fg


def make_lineset(all_pts, cols, num_lines):
    linesets = []
    for pts in all_pts:
        lineset = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts, np.float64))
        lineset.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(cols, np.float64))
        pt_indices = np.arange(len(lineset.points))
        line_indices = np.stack((pt_indices, pt_indices - num_lines), -1)[num_lines:]
        lineset.lines = o3d.utility.Vector2iVector(np.ascontiguousarray(line_indices, np.int32))
        linesets.append(lineset)
    return linesets


def calculate_trajectories(scene_data, traj_frac, traj_length, is_fg):
    in_pts = [data['means3D'][is_fg][::traj_frac].contiguous().float().cpu().numpy() for data in scene_data]
    num_lines = len(in_pts[0])
    cols = np.repeat(colormap[np.arange(len(in_pts[0])) % len(colormap)][None], traj_length, 0).reshape(-1, 3)
    out_pts = []
    for t in range(len(in_pts))[traj_length:]:
        out_pts.append(np.array(in_pts[t - traj_length:t + 1]).reshape(-1, 3))
    return make_lineset(out_pts, cols, num_lines)


def calculate_rot_vec(scene_data, traj_frac, traj_length, is_fg):
    in_pts = [data['means3D'][is_fg][::traj_frac].contiguous().float().cpu().numpy() for data in scene_data]
    in_rotation = [data['rotations'][is_fg][::traj_frac] for data in scene_data]
    num_lines = len(in_pts[0])
    cols = colormap[np.arange(num_lines) % len(colormap)]
    inv_init_q = deepcopy(in_rotation[0])
    inv_init_q[:, 1:] = -1 * inv_init_q[:, 1:]
    inv_init_q = inv_init_q / (inv_init_q ** 2).sum(-1)[:, None]
    init_vec = np.array([-0.1, 0, 0])
    out_pts = []
    for t in range(len(in_pts)):
        cam_rel_qs = quat_mult(in_rotation[t], inv_init_q)
        rot = build_rotation(cam_rel_qs).cpu().numpy()
        vec = (rot @ init_vec[None, :, None]).squeeze()
        out_pts.append(np.concatenate((in_pts[t] + vec, in_pts[t]), 0))
    return make_lineset(out_pts, cols, num_lines)


def render(w2c, k, timestep_data, w, h, near, far):
    with torch.no_grad():
        cam = setup_camera(w, h, k, w2c, near, far)
        im, _, depth, = Renderer(raster_settings=cam)(**timestep_data)
        return im, depth


def get_device():
    """
    Checks if GPU is available and returns device accordingly.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device

def init_camera(w, h, y_angle=0., center_dist=2.4, cam_height=1.3, f_ratio=0.82):
    ry = y_angle * np.pi / 180
    w2c = np.array([[np.cos(ry), 0., -np.sin(ry), 0.],
                    [0.,         1., 0.,          cam_height],
                    [np.sin(ry), 0., np.cos(ry),  center_dist],
                    [0.,         0., 0.,          1.]])
    k = np.array([[f_ratio * w, 0, w / 2], [0, f_ratio * w, h / 2], [0, 0, 1]])
    return w2c, k

def setup_pytorch3d_camera(h, w2c, k):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Extract rotation (R) and translation (T) from the w2c matrix
    R = torch.from_numpy(w2c[:3, :3]).float().to(device)
    T = torch.from_numpy(w2c[:3, 3]).float().unsqueeze(0).to(device)

    # Intrinsic parameters (focal length and principal point)
    fx, fy = k[0, 0], k[1, 1]
    cx, cy = k[0, 2], k[1, 2]

    # Convert to PyTorch3D's FoVPerspectiveCameras
    # Note: PyTorch3D uses a slightly different camera model, so some adjustments might be needed
    cameras = FoVPerspectiveCameras(device=device, R=R[None, :], T=T, fov=2*np.arctan(h/(2*fy))*180/np.pi)

    return cameras


def render_pointcloud_pytorch3d(w, h, image_size, radius, pointclouds, y_angle):
    # Assuming the use of a function `get_device` to determine the rendering device (CPU or GPU)
    device = get_device()
    
    # Move the pointcloud to the selected device
    pointclouds = pointclouds.to(device)
    
    # Create a renderer with specified settings
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(
            raster_settings=PointsRasterizationSettings(
                image_size=image_size,  # Specify your desired image size
                radius=radius,  # Specify the radius for points
            )
        ),
        compositor=AlphaCompositor(background_color=(1, 1, 1))  # Specify your desired background color
    ).to(device)

    w2c, k = init_camera(w, h, y_angle)
    cameras = setup_pytorch3d_camera(h, w2c, k)
    images = renderer(pointclouds, cameras=cameras.to(device))
    
    # Convert the rendered tensor to a NumPy array for visualization (e.g., using matplotlib)
    rendered_image = images[0, ..., :3].detach().cpu().numpy()

    return rendered_image

