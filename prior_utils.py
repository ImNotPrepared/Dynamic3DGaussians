'''
    for ii in range(len(nearest_pose_ids)):
      offset = nearest_pose_ids[ii] - idx
      flow, mask = self.read_optical_flow(
          self.scene_path,
          idx,
          start_frame=0,
          fwd=True if offset > 0 else False,
          interval=np.abs(offset),
      )

      flows.append(flow)
      masks.append(mask)
    flows = resize_array(np.stack(flows),new_size)
    masks = resize_array(np.stack(masks), new_size)
    assert flows.shape[1:3] == img_size
    assert masks.shape[1:3] == img_size

'''

import numpy as np 
import os 
def read_optical_flow(basedir, img_i, start_frame, fwd, interval):
  flow_dir = os.path.join(basedir, 'flow_i%d' % interval)

  if fwd:
    fwd_flow_path = os.path.join(
        flow_dir, '%05d_fwd.npz' % (start_frame + img_i)
    )
    fwd_data = np.load(fwd_flow_path)  # , (w, h))
    print("Variables in the NPZ file:", fwd_data.files)

    # Step 3 & 4: Access arrays and inspect their shapes
    for var in fwd_data.files:
        print(f"Shape of '{var}':", fwd_data[var].shape)
    fwd_flow, fwd_mask = fwd_data['flow'], fwd_data['mask']
    fwd_mask = np.float32(fwd_mask)

    return fwd_flow, fwd_mask
  else:
    bwd_flow_path = os.path.join(
        flow_dir, '%05d_bwd.npz' % (start_frame + img_i)
    )
    print(bwd_flow_path)
    bwd_data = np.load(bwd_flow_path)  # , (w, h))
    bwd_flow, bwd_mask = bwd_data['flow'], bwd_data['mask']
    bwd_mask = np.float32(bwd_mask)

    return bwd_flow, bwd_mask

import os
import numpy as np
import cv2

def read_optical_flow_and_calculate_motion(basedir, img_i, start_frame, fwd, interval):
    flow_dir = os.path.join(basedir, 'flow_i%d' % interval)
    if fwd:
      fwd_flow_path = os.path.join(flow_dir, '%05d_fwd.npz' % (start_frame + img_i))
      fwd_data = np.load(fwd_flow_path)
      fwd_flow = fwd_data['flow']
      fwd_mask = np.float32(fwd_data['mask'])
      #forward_warped = cv2.remap(bwd_flow, fwd_flow[..., 0], fwd_flow[..., 1], interpolation=cv2.INTER_LINEAR)
      forward_discrepancy = np.linalg.norm(fwd_flow, axis=2)
      motion_avg = (np.mean(forward_discrepancy))
      motion_max = (np.max(forward_discrepancy))
    else:
      bwd_flow_path = os.path.join(flow_dir, '%05d_bwd.npz' % (start_frame + img_i))
      bwd_data = np.load(bwd_flow_path)
      bwd_flow = bwd_data['flow']
      bwd_mask = np.float32(bwd_data['mask'])
      #backward_warped = cv2.remap(fwd_flow, bwd_flow[..., 0], bwd_flow[..., 1], interpolation=cv2.INTER_LINEAR)
      backward_discrepancy = np.linalg.norm(bwd_flow, axis=2)
      motion_avg = (np.mean(backward_discrepancy))
      motion_max = (np.max(backward_discrepancy))
    # Calculate the warping of the backward flow by the forward flow and vice versa
    
    

    # Calculate discrepancy between original and warped flows



    # Calculate scalar metrics for overall motion
    #motion_avg = (np.mean(forward_discrepancy) + np.mean(backward_discrepancy)) / 2
    #motion_max = max(np.max(forward_discrepancy), np.max(backward_discrepancy))

    return motion_avg, motion_max


def main(idx):
  nearest_pose_ids = [idx + offset for offset in [1, 2, 3, -1, -2, -3]]
  max_step = 3
  # select a nearby time index for cross time rendering
  anchor_pool = [i for i in range(1, max_step + 1)] + [
      -i for i in range(1, max_step + 1)
  ]
  flows, masks = [], []
  anchor_idx = idx + anchor_pool[np.random.choice(len(anchor_pool))]
  anchor_nearest_pose_ids = []

  for offset in [3, 2, 1, 0, -1, -2, -3]:
    if (
        (anchor_idx + offset) < 0
        or (anchor_idx + offset) >= 1400
        or (anchor_idx + offset) == idx
    ):
      continue
    anchor_nearest_pose_ids.append((anchor_idx + offset))

  # occasionally include render image for anchor time index
  if np.random.choice([0, 1], p=[1.0 - 0.005, 0.005]):
    anchor_nearest_pose_ids.append(idx)
  means=[]
  anchor_nearest_pose_ids = np.sort(anchor_nearest_pose_ids)
  for ii in range(len(nearest_pose_ids)):
    offset = nearest_pose_ids[ii] - idx
    flow, mask = read_optical_flow_and_calculate_motion(
        '/data3/zihanwa3/Capstone-DSR/dynibar/preprocessing/dataset/multicam/cam02',
        idx,
        start_frame=0,
        fwd=True if offset > 0 else False,
        interval=np.abs(offset),
    )
    #print(flow, mask)
    #means.append(flow.mean())
    flows.append(flow.mean())
    masks.append(mask.mean())
  #return (np.array(set(means)))
  #print(np.array(flows).mean())

  return flows, masks
if __name__ == '__main__':
  all=[]
  for idx in range(1,100):
    all.append(main(idx=idx)[0])
  print(all)