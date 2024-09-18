import torch
import torch.nn.functional as F

# Define flow_a_b
flow_a_b = torch.zeros((3, 4, 2), dtype=torch.float32)
flow_a_b[0, :, :] = torch.tensor([1, 0], dtype=torch.float32)
flow_a_b[1, :, :] = torch.tensor([-1, 0], dtype=torch.float32)
flow_a_b[2, :, :] = torch.tensor([1, 0], dtype=torch.float32)

# Define flow_b_c
flow_b_c = torch.zeros((3, 4, 2), dtype=torch.float32)
flow_b_c[:, 0, :] = torch.tensor([0, 1], dtype=torch.float32)
flow_b_c[:, 2, :] = torch.tensor([0, 1], dtype=torch.float32)
flow_b_c[:, 1, :] = torch.tensor([0, -1], dtype=torch.float32)
flow_b_c[:, 3, :] = torch.tensor([0, -1], dtype=torch.float32)

def warp_flow(flow, displacement):
    """
    Warp the optical flow using the given displacement.

    Args:
        flow (torch.Tensor): The flow to be warped, of shape (H, W, 2).
        displacement (torch.Tensor): The displacement to warp the flow, of shape (H, W, 2).

    Returns:
        torch.Tensor: The warped flow, of shape (H, W, 2).
    """
    H, W, _ = flow.shape

    # Create a grid of coordinates in pixel space
    grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W))
    grid_x = grid_x.float()
    grid_y = grid_y.float()
    grid = torch.stack((grid_x, grid_y), 2)  # (H, W, 2)

    # Compute the coordinates where to sample from flow
    sample_coords = grid + displacement  # (H, W, 2)

    # Normalize coordinates to [-1, 1] for grid_sample
    sample_coords_x = 2 * (sample_coords[:, :, 0] / (W - 1)) - 1
    sample_coords_y = 2 * (sample_coords[:, :, 1] / (H - 1)) - 1
    sample_grid = torch.stack((sample_coords_x, sample_coords_y), 2)  # (H, W, 2)

    # Expand dimensions to match grid_sample requirements
    sample_grid = sample_grid.unsqueeze(0)  # (1, H, W, 2)
    flow = flow.permute(2, 0, 1).unsqueeze(0)  # (1, 2, H, W)

    # Use grid_sample to warp the flow
    warped_flow = F.grid_sample(
        flow,
        sample_grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )

    # Remove batch and channel dimensions
    warped_flow = warped_flow[0].permute(1, 2, 0)  # (H, W, 2)
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

# Test the functions
flows = [flow_a_b, flow_b_c]
accumulated_flow = accumulate_flows(flows)

# Print the accumulated flow for verification
print("Accumulated Flow from a to c:")
print(accumulated_flow)

'''
tensor([[[ 1.,  1.],
         [ 1.,  1.],
         [ 1., -1.],
         [ 1.,  1.]],

        [[-1., -1.],
         [-1.,  1.],
         [-1., -1.],
         [-1., -1.]],

        [[ 1.,  1.],
         [ 1.,  1.],
         [ 1., -1.],
         [ 1.,  1.]]])
'''