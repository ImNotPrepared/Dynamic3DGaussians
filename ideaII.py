import numpy as np
import torch
from helpers import setup_camera
import pytorch3d.transforms as transforms
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
import torch.nn as nn
from torchvision.transforms import functional as F
from io import BytesIO
import matplotlib.pyplot as plt
import wandb
from PIL import Image
# Initialize your camera intrinsics (K) and extrinsics (Ex) for each camera
def load_camera_params():
    # Define K and Ex for pseudo cameras
    K = {
        "0": np.array([[fx0, 0, cx0], [0, fy0, cy0], [0, 0, 1]]),
        "a": np.array([[fxa, 0, cxa], [0, fya, cya], [0, 0, 1]]),
        "b": np.array([[fxb, 0, cxb], [0, fyb, cyb], [0, 0, 1]]),
        "c": np.array([[fxc, 0, cxc], [0, fyc, cyc], [0, 0, 1]]),
        "1": np.array([[fx1, 0, cx1], [0, fy1, cy1], [0, 0, 1]])
    }

    Ex = {
        "0": np.eye(4),  # Identity matrix for origin
        "a": get_extrinsic_matrix_for_a(),  # Assume we have a function to get Ex
        "b": get_extrinsic_matrix_for_b(),
        "c": get_extrinsic_matrix_for_c(),
        "1": get_extrinsic_matrix_for_1(),
    }
    return K, Ex



# Calculate flow using RAFT from image 0 to any other pseudo cam
def calculate_raft_flow(raft_model, img1, img2):
    flow = raft_model.estimate_flow(img1, img2)
    return flow

# Calculate ground-truth flow using depth, camera extrinsics, and intrinsics
def calculate_gt_flow(K_src, K_dst, Ex_src, Ex_dst, depth_src):
    # Apply pinhole camera model to get flow vectors
    # src -> dst using depth and camera matrices
    flow = gt_flow_from_depth_and_cameras(K_src, K_dst, Ex_src, Ex_dst, depth_src)
    return flow

# Loss function comparing RAFT and ground-truth flow
def compute_loss(flow_raft, flow_gt):
    loss = torch.mean((flow_raft - flow_gt) ** 2)  # L2 loss
  
    return loss

# Convert pixel coordinates to 3D points in camera coordinates using depth and 3x3 intrinsics
def pixel_to_camera_coordinates(depth, intrinsics):
    h, w = depth.shape
    i, j = torch.meshgrid(torch.arange(w), torch.arange(h), indexing='xy')
    i, j = i.to(depth.device), j.to(depth.device)

    z = depth
    x = (i - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (j - intrinsics[1, 2]) * z / intrinsics[1, 1]

    # Stack into (N, 3) shape where N is the number of pixels
    points_3d = torch.stack([x, y, z], dim=-1)  # Shape: (h, w, 3)
    return points_3d

# Transform 3D points from the camera frame to world coordinates using 4x4 extrinsics
def camera_to_world_coordinates(points_3d, extrinsics):
    R = extrinsics[:3, :3]  # Rotation matrix (3x3)
    t = extrinsics[:3, 3]  # Translation vector (3,)

    # Reshape points to (N, 3), where N is h*w (number of pixels)
    points_3d_flat = points_3d.view(-1, 3)
    
    # Apply extrinsic transformation: P_world = R * P_cam + t
    points_world = torch.matmul(points_3d_flat, R.T) + t
    return points_world.view(points_3d.shape)

# Project 3D points from world coordinates to 2D pixel coordinates in the second camera
def world_to_camera_2d(points_world, intrinsics, extrinsics):
    R = extrinsics[:3, :3]  # Rotation matrix
    t = extrinsics[:3, 3]  # Translation vector

    # Transform world coordinates to the camera frame of camera 2
    points_camera = torch.matmul(points_world.view(-1, 3), R.T) + t

    # Project 3D points into 2D pixel coordinates using camera intrinsics (3x3)
    points_2d = torch.matmul(points_camera, intrinsics.T)
    points_2d = points_2d[:, :2] / points_2d[:, 2:]  # Normalize by depth (divide by z)
    return points_2d.view(points_world.shape[0], points_world.shape[1], 2)

# Compute optical flow between two images with depth and camera parameters (3x3 intrinsics, 4x4 extrinsics)
def compute_optical_flow(depth1, intrinsics1, extrinsics1, depth2, intrinsics2, extrinsics2):
    # Step 1: Convert depth1 to 3D points in camera 1's frame
    points_3d_cam1 = pixel_to_camera_coordinates(depth1, intrinsics1)
    
    # Step 2: Transform 3D points to world coordinates
    points_world = camera_to_world_coordinates(points_3d_cam1, extrinsics1)
    
    # Step 3: Project the world points into camera 2's image plane
    points_2d_cam2 = world_to_camera_2d(points_world, intrinsics2, extrinsics2)
    
    # Step 4: Compute original pixel coordinates in camera 1's image plane
    h, w = depth1.shape
    i, j = torch.meshgrid(torch.arange(w), torch.arange(h), indexing='xy')
    i, j = i.to(depth1.device), j.to(depth1.device)
    points_2d_cam1 = torch.stack([i, j], dim=-1).view(h, w, 2)

    # Step 5: Calculate the optical flow as displacement between pixel coordinates
    flow = points_2d_cam2 - points_2d_cam1
    
    # Reshape flow from [H, W, 2] to [2, H, W]
    flow = flow.permute(2, 0, 1)
    
    return flow


import torch
import torch.nn.functional as F

# Spherical linear interpolation (Slerp) for rotation matrices
def slerp(R1, R2, t):
    """
    Slerp between two rotation matrices R1 and R2 by factor t.
    t is a value between 0 and 1, where 0 gives R1 and 1 gives R2.
    """
    # Compute the quaternion representation of the rotation matrices
    quat1 = transforms.matrix_to_quaternion(R1)
    quat2 = transforms.matrix_to_quaternion(R2)
    quat_slerp = torch.lerp(quat1, quat2, t)


    # quat_slerp = F.normalize(torch.lerp(quat1, quat2, t), dim=0)  # Normalize along the only dimension

    return transforms.quaternion_to_matrix(quat_slerp)

# Linear interpolation for translation vectors
def lerp(t1, t2, alpha):
    """
    Linearly interpolate between two translation vectors t1 and t2.
    alpha is a value between 0 and 1.
    """
    return (1 - alpha) * t1 + alpha * t2

def interpolate_extrinsics(extrinsics1, extrinsics2, alpha):
    """
    Interpolate between two extrinsics matrices (4x4) by factor alpha.
    Returns the interpolated extrinsics matrix.
    """
    # Extract rotation matrices and translation vectors
    R1, t1 = extrinsics1[:3, :3], extrinsics1[:3, 3]
    R2, t2 = extrinsics2[:3, :3], extrinsics2[:3, 3]

    # Interpolate rotation (using Slerp)
    R_interp = slerp(R1, R2, alpha)

    # Interpolate translation (using linear interpolation)
    t_interp = lerp(t1, t2, alpha)

    # Combine interpolated rotation and translation into a new extrinsics matrix
    extrinsics_interp = torch.eye(4, dtype=extrinsics1.dtype).to(extrinsics1.device)
    extrinsics_interp[:3, :3] = R_interp
    extrinsics_interp[:3, 3] = t_interp

    return extrinsics_interp

import torchvision.transforms.functional as F
import torchvision.transforms as T

def preprocess(batch):
    transforms = T.Compose(
        [
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1]
            T.Resize(size=(288, 512)),
        ]
    )
    batch = transforms(batch)
    return batch

def rending_size(batch):
    transforms = T.Compose(
        [
            T.Resize(size=(288, 512)),
        ]
    )
    batch = transforms(batch)
    return batch



def flow_visualize(predicted_flows, img1_batch, index=None):
    from torchvision.utils import flow_to_image

    flow_imgs = flow_to_image(predicted_flows)

    # The images have been mapped into [-1, 1] but for plotting we want them in [0, 1]
    img1_batch = [(img1 + 1) / 2 for img1 in img1_batch]

    grid = [[img1, flow_img] for (img1, flow_img) in zip(img1_batch, flow_imgs)]
    def plot(imgs, **imshow_kwargs):
        if not isinstance(imgs[0], list):
            # Make a 2d grid even if there's just 1 row
            imgs = [imgs]

        num_rows = len(imgs)
        num_cols = len(imgs[0])
        _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
        for row_idx, row in enumerate(imgs):
            for col_idx, img in enumerate(row):
                ax = axs[row_idx, col_idx]
                img = F.to_pil_image(img.to("cpu"))
                ax.imshow(np.asarray(img), **imshow_kwargs)
                ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

        # Save the figure to a buffer
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)

        logging_img=Image.open(buf)
        
        # Log the image to wandb
        wandb.log({f"image_flow_{index}": wandb.Image(logging_img)})
        
        plt.close()

    plot(grid)
import torch.nn.functional as Fun

def warp_flow(flow, displacement):
    """
    Warp the optical flow using the given displacement.

    Args:
        flow (torch.Tensor): The flow to be warped, of shape (2, H, W).
        displacement (torch.Tensor): The displacement to warp the flow, of shape (2, H, W).

    Returns:
        torch.Tensor: The warped flow, of shape (2, H, W).
    """
    # Ensure flow and displacement are in (2, H, W) format
    assert flow.ndimension() == 3 and flow.shape[0] == 2, "Flow must have shape (2, H, W)"
    assert displacement.ndimension() == 3 and displacement.shape[0] == 2, "Displacement must have shape (2, H, W)"

    _, H, W = flow.shape

    # Create a grid of coordinates in pixel space
    device = flow.device
    dtype = flow.dtype
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing='ij'  # Ensure meshgrid uses matrix indexing
    )
    grid = torch.stack((grid_x, grid_y), 0)  # (2, H, W)

    # Compute the coordinates where to sample from flow
    sample_coords = grid + displacement  # (2, H, W)

    # Normalize coordinates to [-1, 1] for grid_sample
    sample_coords_x = 2 * (sample_coords[0, :, :] / (W - 1)) - 1  # (H, W)
    sample_coords_y = 2 * (sample_coords[1, :, :] / (H - 1)) - 1  # (H, W)
    sample_grid = torch.stack((sample_coords_x, sample_coords_y), dim=2)  # (H, W, 2)

    # Expand dimensions to match grid_sample requirements
    sample_grid = sample_grid.unsqueeze(0)  # (1, H, W, 2)

    # Prepare flow for grid_sample: (N, C, H, W)
    flow = flow.unsqueeze(0)  # (1, 2, H, W)

    # Use grid_sample to warp the flow
    warped_flow = Fun.grid_sample(
        flow,
        sample_grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )

    # Remove batch dimension
    warped_flow = warped_flow[0]  # (2, H, W)
    return warped_flow

def accumulate_flows(flows):
    """
    Accumulate a list of flows into a single flow.

    Args:
        flows (list of torch.Tensor): A list of flow tensors to accumulate.

    Returns:
        torch.Tensor: The accumulated flow.
    """
    accumulated_flow = flows[0].clone()
    for i in range(1, len(flows)):
        warped_flow = warp_flow(flows[i], accumulated_flow)
        accumulated_flow = accumulated_flow + warped_flow
    return accumulated_flow

def flow_loss(rendervar):

    total_loss = 0.0
    from pytorch3d.transforms import quaternion_to_matrix, Translate
    import torch
    import json
    import cv2
    import copy
    from PIL import Image
    # Example usage with torch tensors
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('where stuck 1')
    seq='cmu_bike'
    md = json.load(open(f"/data3/zihanwa3/Capstone-DSR/Dynamic3DGaussians/data_ego/{seq}/train_meta.json", 'r'))
    scales_shifts = [(0.0031744423, 0.15567338), (0.0025279315, 0.106763005), (0.0048902677, 0.16312718), (0.0037271702, 0.10191789), (0.002512292, 0.114545256), (0.0029944833, 0.10527076), (0.003602787, 0.14336547), (0.003638356, 0.1080856), (0.0054704025, 0.057398915), (0.0022690576, 0.117439255), (0.002312136, 0.077383846), (0.0054023797, 0.089054525), (0.0050647566, 0.101514965), (0.0036501177, 0.13153434), (0.0008889911, 0.44202688), (0.0025493288, 0.109814465), (0.0024664444, 0.112163335), (-0.00016438629, 0.40732577), (0.0032442464, 0.19807495), (0.0048282435, 0.09168023), (0.002856112, 0.15053965), (0.0020215507, 0.107855394), (0.0030028797, 0.14278293), (0.0024490638, 0.13038686), (0.0024990174, 0.12481204), (0.0057816333, 0.077005506), (0.0019591942, 0.10089706), (0.0013262086, 0.42674613), (0.004126527, 0.13687198), (0.0022844346, 0.097172886), (0.0062575513, 0.12489089), (-0.00014962265, 0.38713253), (0.00086679566, 0.25387546), (0.0021814466, 0.10047534), (0.002019625, 0.10706337), (0.0037505955, 0.13279462), (0.0035237654, 0.12734117), (0.0019494797, 0.14369084), (0.00056177535, 0.28072894), (0.0018662697, 0.10288732), (0.00591453, 0.053784877), (0.002294414, 0.23004633), (0.0014106235, 0.14460064), (0.0013034015, 0.24912238), (0.0015928176, 0.17974892)]
    near, far = 1e-7, 7e0

    #for lis in [[1400, 1401, 1402, 1403]]: 

    def data_prep(lis):
      for iiiindex, c in sorted(enumerate(lis)):
        t=0
        fn = md['fn'][t][c]
        filename=f"/ssd0/zihanwa3/data_ego/{seq}/ims/{fn}"
        raw_image = cv2.imread(filename)
        h, w = md['hw'][c]
        k, w2c =  torch.tensor(md['k'][t][c]), np.linalg.inv(md['w2c'][t][c])

        #cam = setup_camera(w, h, k, w2c, near=near, far=far)

        fn = md['fn'][t][c]
        im = np.array(copy.deepcopy(Image.open(f"/ssd0/zihanwa3/data_ego/{seq}/ims/{fn}")))
        im = torch.tensor(im).float().cuda().permute(2, 0, 1) / 255
        im=im.clip(0,1)
        
        ############################## First Frame Depth #############################
        depth_path=f'/data3/zihanwa3/Capstone-DSR/Processing/da_v2_disp/0/disp_{c}.npz'
        depth = torch.tensor(np.load(depth_path)['depth_map'])
        assert depth.shape[1] !=  depth.shape[0]

        ################DOING REGULAR VIS #################
        scale, shift = scales_shifts[c-1404]
        disp_map = depth
        nonzero_mask = disp_map != 0
        disp_map[nonzero_mask] = disp_map[nonzero_mask] * scale + shift
        valid_depth_mask = (disp_map > 0) & (disp_map <= far)
        disp_map[~valid_depth_mask] = 0
        depth_map = np.full(disp_map.shape, np.inf)
        depth_map[disp_map != 0] = 1 / disp_map[disp_map != 0]
        depth_map[depth_map == np.inf] = 0
        depth_map = depth_map.astype(np.float32)
        depth = torch.tensor(depth_map)

      return depth, torch.tensor(k).float(), torch.tensor(w2c).float(), h, w

    Exs = [(1400, 1401), (1401, 1402), (1402, 1403), (1403, 1400)]
    losses = 0
    for idxxx1, idxxx2 in Exs[:]:
      print('where stuck 2')
      depth1, intrinsics1, extrinsics1, h, w = data_prep([idxxx1])#torch.rand(480, 640).to(device) 
      depth2, intrinsics2, extrinsics2, h, w = data_prep([idxxx2])#torch.rand(480, 640).to(device) 
      #print(intrinsics1, intrinsics2)

      #torch.tensor([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=torch.float32).to(device)  # Camera 1 intrinsics
      #torch.eye(4, dtype=torch.float32).to(device)  # Identity matrix for camera 1 extrinsics
      extrinsics_interps=[]
      alphas = torch.linspace(0, 1, steps=5)
      ### take 
      for alpha in alphas:
        extrinsics_interps.append(interpolate_extrinsics(extrinsics1, extrinsics2, alpha))  # First interpolated camera


      cams = [setup_camera(w, h, (intrinsics1+intrinsics2)/2, w2c_psd, near=near, far=far) for w2c_psd in extrinsics_interps]

      psd_ims = [Renderer(raster_settings=cam)(**rendervar)[0] for cam in cams]

      psd_ims_1 = preprocess(torch.stack(psd_ims[:-1]))
      psd_ims_2 = preprocess(torch.stack(psd_ims[1:]))
      #### INPUT
      from torchvision.models.optical_flow import raft_large

      #list_of_img_batches=[torch.stack([psd_ims[i], psd_ims[i+1]]) for i in range(len(psd_ims)-1)]
      model = raft_large(pretrained=True, progress=False).to(device)
      model = model.eval()
      estimate_flows = model(psd_ims_1.to(device), psd_ims_2.to(device))[-1]
      ### N * [2, H, W]
      # estimate_flow = (torch.sum(estimate_flows, dim=0)).to(device)

      # Get the total flow from a to e
      estimate_flow = accumulate_flows(estimate_flows)

      flow_visualize(estimate_flows, psd_ims_1, index=(idxxx1-1400))
      #print(estimate_flow.shape)

      # torch.Size([2, 288, 512]) torch.Size([2160, 3840, 2])
      print('where stuck 3')
      gt_flow = rending_size(compute_optical_flow(depth1, intrinsics1, extrinsics1, depth2, intrinsics2, extrinsics2)).to(device)
      mse_loss = nn.MSELoss(reduction='mean')

      #print(estimate_flow.shape, gt_flow.shape)
      losses += mse_loss(estimate_flow , gt_flow)
    return losses




if __name__ == "__main__":
    main()
