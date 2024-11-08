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
import json
import cv2
import copy
import pytorch3d.transforms as transforms
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
        wandb.log({f"image_flow_{index}": wandb.Image(logging_img)})
        plt.close()

    plot(grid)

def slerp(q1, q2, t):
    """
    Spherical linear interpolation between two quaternions q1 and q2.
    """
    cos_half_theta = torch.dot(q1, q2)
    if abs(cos_half_theta.item()) >= 1.0:
        return q1

    sin_half_theta = torch.sqrt(1.0 - cos_half_theta * cos_half_theta)
    if sin_half_theta.item() < 0.001:
        # If the angle is small, use linear interpolation
        return (1.0 - t) * q1 + t * q2

    half_theta = torch.acos(cos_half_theta)
    ratio_a = torch.sin((1.0 - t) * half_theta) / sin_half_theta
    ratio_b = torch.sin(t * half_theta) / sin_half_theta

    return ratio_a * q1 + ratio_b * q2


def interpolate_extrinsics(extrinsics1, extrinsics2, alpha):
    """
    Interpolate between two extrinsics matrices (4x4) by factor alpha.
    Returns the interpolated extrinsics matrix.
    """
    # Extract rotation matrices and translation vectors
    R1, t1 = extrinsics1[:3, :3], extrinsics1[:3, 3]
    R2, t2 = extrinsics2[:3, :3], extrinsics2[:3, 3]

    # Convert rotation matrices to quaternions
    quat1 = transforms.matrix_to_quaternion(R1)
    quat2 = transforms.matrix_to_quaternion(R2)

    # Perform SLERP on quaternions
    quat_interp = slerp(quat1, quat2, alpha)
    quat_interp = quat_interp / quat_interp.norm()  # Normalize the quaternion

    # Convert interpolated quaternion back to rotation matrix
    R_interp = transforms.quaternion_to_matrix(quat_interp)

    # Interpolate translation vectors
    t_interp = (1 - alpha) * t1 + alpha * t2

    # Combine into extrinsics matrix
    extrinsics_interp = torch.eye(4, dtype=extrinsics1.dtype).to(extrinsics1.device)
    extrinsics_interp[:3, :3] = R_interp
    extrinsics_interp[:3, 3] = t_interp

    return extrinsics_interp

# Other utility functions remain the same
def pixel_to_camera_coordinates(depth, intrinsics):
    h, w = depth.shape
    i, j = torch.meshgrid(torch.arange(w), torch.arange(h), indexing='xy')
    i, j = i.to(depth.device), j.to(depth.device)
    z = depth
    x = (i - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (j - intrinsics[1, 2]) * z / intrinsics[1, 1]
    points_3d = torch.stack([x, y, z], dim=-1)
    return points_3d

def camera_to_world_coordinates(points_3d, extrinsics):
    R = extrinsics[:3, :3]  # Rotation matrix (3x3)
    t = extrinsics[:3, 3]  # Translation vector (3,)

    # Reshape points to (N, 3), where N is h*w (number of pixels)
    points_3d_flat = points_3d.view(-1, 3)

    # Apply extrinsic transformation: P_world = R * P_cam + t
    points_world = torch.matmul(points_3d_flat, R.T) + t
    return points_world.view(points_3d.shape)

def world_to_camera_2d(points_world, intrinsics, extrinsics):
    R = extrinsics[:3, :3]  # Rotation matrix
    t = extrinsics[:3, 3]  # Translation vector

    # Transform world coordinates to the camera frame
    points_camera = torch.matmul(points_world.view(-1, 3), R.T) + t

    # Project 3D points into 2D pixel coordinates using camera intrinsics
    points_2d = torch.matmul(points_camera, intrinsics.T)
    points_2d = points_2d[:, :2] / points_2d[:, 2:].clamp(min=1e-7)  # Normalize by depth
    return points_2d.view(points_world.shape[0], points_world.shape[1], 2)

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

def warp_flow(flow, displacement):
    # Warp the optical flow using the given displacement
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
    warped_flow = torch.nn.functional.grid_sample(
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
    # Accumulate a list of flows into a single flow
    accumulated_flow = flows[0].clone()
    for i in range(1, len(flows)):
        warped_flow = warp_flow(flows[i], accumulated_flow)
        accumulated_flow = accumulated_flow + warped_flow
    return accumulated_flow
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

def resize_optical_flow(flow, new_size):
    """
    Resize an optical flow map and scale the flow vectors appropriately.

    Args:
        flow (torch.Tensor): Optical flow map of shape [2, H, W] or [H, W, 2].
        new_size (tuple): New size as (new_height, new_width).

    Returns:
        torch.Tensor: Resized flow map with scaled flow vectors.
    """
    if flow.ndim == 3 and flow.shape[0] == 2:
        # Flow shape is [2, H, W]
        _, H, W = flow.shape
    elif flow.ndim == 3 and flow.shape[2] == 2:
        # Flow shape is [H, W, 2], convert to [2, H, W]
        H, W, _ = flow.shape
        flow = flow.permute(2, 0, 1)
    else:
        raise ValueError("Unsupported flow shape: must be [2, H, W] or [H, W, 2]")

    new_h, new_w = new_size
    scale_h = new_h / H
    scale_w = new_w / W

    # Add batch dimension
    flow = flow.unsqueeze(0)

    # Resize the flow map
    flow_resized = torch.nn.functional.interpolate(flow, size=(new_h, new_w), mode='bilinear', align_corners=False)

    # Scale flow vectors
    flow_resized[:, 0, :, :] *= scale_w  # Horizontal component
    flow_resized[:, 1, :, :] *= scale_h  # Vertical component

    # Remove batch dimension
    flow_resized = flow_resized.squeeze(0)

    return flow_resized


def flow_loss(rendervar, raft_model):

    total_loss = 0.0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seq='cmu_bike'
    md = json.load(open(f"/data3/zihanwa3/Capstone-DSR/Dynamic3DGaussians/data_ego/{seq}/train_meta.json", 'r'))
    scales_shifts = [(0.0031744423, 0.15567338), (0.0025279315, 0.106763005), (0.0048902677, 0.16312718), (0.0037271702, 0.10191789), (0.002512292, 0.114545256), (0.0029944833, 0.10527076), (0.003602787, 0.14336547), (0.003638356, 0.1080856), (0.0054704025, 0.057398915), (0.0022690576, 0.117439255), (0.002312136, 0.077383846), (0.0054023797, 0.089054525), (0.0050647566, 0.101514965), (0.0036501177, 0.13153434), (0.0008889911, 0.44202688), (0.0025493288, 0.109814465), (0.0024664444, 0.112163335), (-0.00016438629, 0.40732577), (0.0032442464, 0.19807495), (0.0048282435, 0.09168023), (0.002856112, 0.15053965), (0.0020215507, 0.107855394), (0.0030028797, 0.14278293), (0.0024490638, 0.13038686), (0.0024990174, 0.12481204), (0.0057816333, 0.077005506), (0.0019591942, 0.10089706), (0.0013262086, 0.42674613), (0.004126527, 0.13687198), (0.0022844346, 0.097172886), (0.0062575513, 0.12489089), (-0.00014962265, 0.38713253), (0.00086679566, 0.25387546), (0.0021814466, 0.10047534), (0.002019625, 0.10706337), (0.0037505955, 0.13279462), (0.0035237654, 0.12734117), (0.0019494797, 0.14369084), (0.00056177535, 0.28072894), (0.0018662697, 0.10288732), (0.00591453, 0.053784877), (0.002294414, 0.23004633), (0.0014106235, 0.14460064), (0.0013034015, 0.24912238), (0.0015928176, 0.17974892)]
    near, far = 1e-7, 7e1
    def data_prep(lis):
      for iiiindex, c in sorted(enumerate(lis)):
        t=0
        fn = md['fn'][t][c]
        filename=f"/ssd0/zihanwa3/data_ego/{seq}/ims/{fn}"
        raw_image = cv2.imread(filename)
        h, w = md['hw'][c]
        k, w2c =  torch.tensor(md['k'][t][c]), np.linalg.inv(md['w2c'][t][c])
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

      return depth, torch.tensor(k).float(), torch.tensor(w2c).float(), h, w, im

    Exs = [(1400, 1401), (1401, 1402), (1402, 1403), (1403, 1400)]
    extrinsics_results = {}
    losses = 0
    for idxxx1, idxxx2 in Exs[:]:
      print('where stuck 2')
      depth1, intrinsics1, extrinsics1, h, w, im1 = data_prep([idxxx1])#torch.rand(480, 640).to(device) 
      depth2, intrinsics2, extrinsics2, h, w, im2 = data_prep([idxxx2])#torch.rand(480, 640).to(device) 
      # Concatenate images along width (dim=2) or height (dim=1)
      '''print(im1.shape)
      concatenated_image = torch.cat((im1, im2), dim=2)  # Concatenate along width
      

      # Convert tensor to NumPy array (assuming images are float and need to be scaled to 0-255)
      concatenated_image_np = concatenated_image.cpu().numpy()  # Move tensor to CPU if needed
      concatenated_image_np = np.transpose(concatenated_image_np, (1, 2, 0))
      concatenated_image_np = (concatenated_image_np * 255).astype(np.uint8)  # Scale to [0, 255]

      # Convert NumPy array to PIL Image
      image_to_save = Image.fromarray(concatenated_image_np)

      # Save the image locally
      image_to_save.save(f"/data3/zihanwa3/Capstone-DSR/Dynamic3DGaussians/sanity_flows/concatenated_image_{idxxx1}-{idxxx2}.png")
      print("Image saved as concatenated_image.png")
      '''


      extrinsics_interps=[]
      alphas = torch.linspace(0, 1, steps=7)
      print(alphas)
      ### take 
      adhoc_int = (intrinsics1+intrinsics2)/2
      for alpha in alphas:
        extrinsics_interps.append(interpolate_extrinsics(extrinsics1, extrinsics2, alpha))  # First interpolated camera
      extrinsics_results[f'{idxxx1}_{idxxx2}'] = [(extrinsic.tolist(), adhoc_int.tolist()) for extrinsic in extrinsics_interps]


      cams = [setup_camera(w, h, (intrinsics1+intrinsics2)/2, w2c_psd, near=near, far=far) for w2c_psd in extrinsics_interps]
      psd_ims = [Renderer(raster_settings=cam)(**rendervar)[0].clip(0,1).detach().cpu() for cam in cams]


      
      psd_ims_a = preprocess(torch.stack([im1]))
      psd_ims_d = preprocess(torch.stack([im2]))
      ### a b c d 
      psd_ims_1 = preprocess(torch.stack(psd_ims[:-1]))
      psd_ims_2 = preprocess(torch.stack(psd_ims[1:]))

      estimate_flows = raft_model(psd_ims_1.to(device), psd_ims_2.to(device))[-1] # ([4, 2, 288, 512])
      estimate_flows_1_2 = accumulate_flows(estimate_flows)
      print(estimate_flows.shape)# 5282497.5000 197547.5938

      estimate_flows = raft_model(psd_ims_a.to(device), psd_ims_d.to(device))[-1] # ([1, 2, 288, 512])
      estimate_flows_a_b = accumulate_flows(estimate_flows)
      # print(estimate_flows.shape) #4729003.5000 92035.3203

      ### N * [2, H, W]
      flow_visualize(estimate_flows_1_2, psd_ims_1, index=(idxxx1-1400))

      gt_flow = resize_optical_flow(compute_optical_flow(depth1, intrinsics1, extrinsics1, depth2, intrinsics2, extrinsics2), new_size=(288, 512)).to(device)
      mse_loss = nn.MSELoss(reduction='mean')
      '''for q in list(range(0, 100, 10)):
        print('gt', q, torch.quantile(gt_flow, q/100))

      for q in list(range(0, 100, 10)):
        print('estimate_flows_1_2',  q, torch.quantile(estimate_flows_1_2, q/100))

      for q in list(range(0, 100, 10)):
        print('estimate_flows_a_b',  q, torch.quantile(estimate_flows_a_b, q/100))'''

 
      losses = trimmed_mse_loss(estimate_flows_a_b, estimate_flows_1_2)
      print(losses)#mse_loss(estimate_flows_1_2 , estimate_flows_a_b)
    with open('extrinsics_interpolated.json', 'w') as json_file:
        json.dump(extrinsics_results, json_file, indent=4)
    return losses


def trimmed_mse_loss(pred, gt, quantile=0.9):
    loss = torch.nn.functional.mse_loss(pred, gt, reduction="none").mean(dim=-1)
    #for q in list(range(0, 100, 10)):
    #  print(q, torch.quantile(loss, q/100))
    loss_at_quantile = torch.quantile(loss, quantile)
    trimmed_loss = loss[loss < loss_at_quantile].mean()
    return trimmed_loss

if __name__ == "__main__":
    import wandb
    wandb.init()
    params_path='/data3/zihanwa3/Capstone-DSR/Dynamic3DGaussians/output/explicit_conflict_l1_stat_back/cmu_bike/params_iter_8000.npz'
    params = dict(np.load(params_path, allow_pickle=True))
    from torchvision.models.optical_flow import raft_large
    device='cuda'

    rendervar = {
        'means3D': torch.tensor(params['means3D'], device=device, dtype=torch.float32),
        'colors_precomp': torch.tensor(params['rgb_colors'], device=device, dtype=torch.float32),
        'rotations': torch.nn.functional.normalize(torch.tensor(params['unnorm_rotations'], device=device, dtype=torch.float32)),
        'semantic_feature': torch.tensor(torch.randn(len(params['means3D']), 32), device=device, dtype=torch.float32),
        'opacities': torch.sigmoid(torch.tensor(params['logit_opacities'], device=device, dtype=torch.float32)),
        'scales': torch.exp(torch.tensor(params['log_scales'], device=device, dtype=torch.float32)),
        'means2D': torch.zeros_like(torch.tensor(params['means3D'], device=device, dtype=torch.float32), requires_grad=True, device=device),
        'label': torch.tensor(params['label'], device=device, dtype=torch.int64)  # Assuming 'label' is an integer type
    }

    raft_model = raft_large(pretrained=True, progress=False).to(device)
    raft_model = raft_model.eval()
    loss = flow_loss(rendervar, raft_model)
