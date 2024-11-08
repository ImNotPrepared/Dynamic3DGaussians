import math

import torch
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
import torch
import torch.nn as nn
import torch.nn.functional as F


def cont_6d_to_rmat(cont_6d):
    """
    :param 6d vector (*, 6)
    :returns matrix (*, 3, 3)
    """
    x1 = cont_6d[..., 0:3]
    y1 = cont_6d[..., 3:6]

    x = F.normalize(x1, dim=-1)
    y = F.normalize(y1 - (y1 * x).sum(dim=-1, keepdim=True) * x, dim=-1)
    z = torch.linalg.cross(x, y, dim=-1)

    return torch.stack([x, y, z], dim=-1)

class MotionBases(nn.Module):
    def __init__(self, rots, transls):
        super().__init__()

        ## (B, F, 3/4)
        self.num_frames = rots.shape[1]
        self.num_bases = rots.shape[0]

        self.params = nn.ParameterDict(
            {
                "rots": nn.Parameter(rots.float().contiguous().cuda(), requires_grad=True),
                "transls": nn.Parameter(transls.float().contiguous().cuda(), requires_grad=True),
            }
        )


    def compute_transforms(self, ts: torch.Tensor, coefs: torch.Tensor, inverse=True) -> torch.Tensor:
        """
        :param ts (B)
        :param coefs (G, K)
        returns transforms (G, B, 3, 4)
        """
        #result_dict = {key: (0, len(reversed_range)) for key in reversed_range}
        ts = list(range(9))

        print('MotionShape', self.params["transls"].device, self.params["rots"].shape, coefs.shape, coefs.device)
        transls = self.params["transls"][:, ts]  # (K, B, 3)
        rots = self.params["rots"][:, ts]  # (K, B, 6)
        transls = torch.einsum("pk,kni->pni", coefs, transls)
        rots = torch.einsum("pk,kni->pni", coefs, rots)  # (G, B, 6)
        rotmats = cont_6d_to_rmat(rots)  # (K, B, 3, 3)
        return torch.cat([rotmats, transls[..., None]], dim=-1)

def similarity_mapping(feats, K=10):
    '''
    Args: 
        feats --> [N, E] torch.Tensor
        K     --> Number of clusters
    Returns:
        output --> [N, E, B] torch.Tensor
    '''
    
    '''# [num_bases, num_gaussians]
    if mode == "kmeans":
        model = KMeans(n_clusters=num_bases)
    else:
        model = HDBSCAN(min_cluster_size=20, max_cluster_size=num_tracks // 4)

    ### k means labels and 
    model.fit(vel_dirs)
    labels = model.labels_'''
    import open3d as o3d
    import numpy as np

    feats = feats.clone().detach().cpu()
    feats_normalized = feats / feats.norm(dim=1, keepdim=True) # Normalize the feature vectors
    feats_normalized = torch.ones_like(feats_normalized) - torch.randn(feats_normalized.shape)

    print(feats_normalized.shape)
    from tqdm import tqdm
    batch_size = 40000
    N = feats_normalized.size(0)  # Number of rows in feats_normalized
    cos_sim_matrix = torch.zeros(N, N, device=feats_normalized.device) 
    cos_sim_matrix = torch.mm(feats_normalized, feats_normalized.t()) # Pre-allocate the full similarity matrix on the same device
    #for i in tqdm(range(0, N, batch_size)):
      # Determine the size of the current batch
    #  end_idx = min(i + batch_size, N)  # Ensure we don't go out of bounds
    #  cos_sim_batch = torch.mm(feats_normalized, feats_normalized.t()[:, i:end_idx])
      
      # Copy the result to the appropriate slice of the cos_sim_matrix
    #  cos_sim_matrix[:, i:end_idx] = cos_sim_batch


    K = 49  # Number of clusters you want
    spectral = SpectralClustering(n_clusters=K, affinity='precomputed', random_state=42)
    labels = spectral.fit_predict(cos_sim_matrix)

    '''
    for i in tqdm(range(0, N, batch_size)):
        # Determine the size of the current batch
        end_idx = min(i + batch_size, N)  # Ensure we don't go out of bounds
        cos_sim_batch = torch.mm(feats_normalized, feats_normalized.t()[:, i:end_idx])
        
        # Copy the result to the appropriate slice of the cos_sim_matrix
        cos_sim_matrix[:, i:end_idx] = cos_sim_batch
    
    # Perform spectral clustering
    clustering = SpectralClustering(n_clusters=K, affinity='precomputed', random_state=42)
    labels = clustering.fit_predict(cos_sim_matrix)
    '''
    labels_expanded = torch.tensor(labels)
    
    return labels_expanded

##
# num_frames = tracks_3d.xyz.shape[1]
##

def feature_bases(means, feats, cano_t=49, mode='kmeans'):

    labels = similarity_mapping(feats)

    means_cano = means# means[:, cano_t].clone()  # [num_gaussians, 3]


    num_bases = labels.max().item() + 1

    sampled_centers = torch.stack(
        [
            means_cano[torch.tensor(labels == i)].median(dim=0).values
            for i in range(num_bases)
        ]
    )[None]
    scene_center = means_cano.median(dim=0).values
    dists = torch.norm(means_cano - scene_center, dim=-1)
    dists_th = torch.quantile(dists, 1.0) #######33
    valid_mask = dists < 1e6 ##########
    means_cano = means_cano[valid_mask]
    ### sampled_centers, num_bases, labels = ###
    #### compute cluster weight
    ids, counts = labels.unique(return_counts=True)
    ##################changed freq###################
    ids = ids[counts > 0].long()
    num_bases = len(ids)
    sampled_centers = sampled_centers[:, ids]
    print(means_cano.shape, sampled_centers.shape)
    dists2centers = torch.norm(means_cano[:, None] - sampled_centers, dim=-1) # [N, 10]

    print(dists2centers.shape)
    motion_coefs = 10 * torch.exp(-dists2centers)


    #### definie motion coeff


    #if motion_coefs is not None:
    #  params_dict["motion_coefs"] = nn.Parameter(motion_coefs)

    return nn.Parameter(motion_coefs), means_cano




if __name__ == '__main__':

  import numpy as np
  feature_root_path='/data3/zihanwa3/Capstone-DSR/Processing/dinov2features/resized_512/' #undist_cam00_670/000000.npy'
  fn = 'undist_cam01/00111.npy'
  feature_path = feature_root_path+fn 
  dinov2_feature = torch.tensor(np.load(feature_path.replace('.jpg', '.npy'))).reshape(-1, 32)[:100, ...]
  ### [288, 512, 32]

  feats = dinov2_feature
  tracks_3d = torch.randn(feats.shape[0], 50, 3)
  means = torch.randn(feats.shape[0], 3)
  coefs, means = feature_bases(tracks_3d, means, feats)


  device='cpu'
  num_bases = 10
  num_frames = 50
  id_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
  rot_dim = 4
  init_rots = id_rot.reshape(1, 1, rot_dim).repeat(num_bases, num_frames, 1)
  init_ts = torch.zeros(num_bases, num_frames, 3, device=device)
  bases = MotionBases(init_rots, init_ts)



  ts = ts = torch.arange(0, num_frames, device=device) ## ts [F], coefs [N, B]

  transfms = bases.compute_transforms(ts, coefs)
  print(ts.shape, coefs.shape,transfms.shape)
  ##### transfms transforms (G, B, 3, 4)
  ### G: num, B: motion bases 3: 4: 
  ### rotmats = cont_6d_to_rmat(rots)  # (K, B, 3, 3)
  ### return torch.cat([rotmats, transls[..., None]], dim=-1)
  ###   fg means (G, 3) pad -> (G, 4)
  ##### returned (G, B, 4)
  #means = torch.randn(feats.shape[0], 3)
  ## [N, F, 3]
  positions = torch.einsum(
      "pnij,pj->pni",
      transfms,
      F.pad(means, (0, 1), value=1.0),
  )

  print(positions.shape)



'''
####
### shape-of-motion velocity intialized code 
####
def sample_initial_bases_centers(
    mode: str, cano_t: int, tracks_3d: TrackObservations, num_bases: int
):
    velocities = xyz_interp[:, 1:] - xyz_interp[:, :-1]
    vel_dirs = (
        velocities / (cp.linalg.norm(velocities, axis=-1, keepdims=True) + 1e-5)
    ).reshape((num_tracks, -1))

    # [num_bases, num_gaussians]
    if mode == "kmeans":
        model = KMeans(n_clusters=num_bases)
    else:
        model = HDBSCAN(min_cluster_size=20, max_cluster_size=num_tracks // 4)
    model.fit(vel_dirs)

    labels = model.labels_
    num_bases = labels.max().item() + 1

    sampled_centers = torch.stack(
        [
            means_canonical[torch.tensor(labels == i)].median(dim=0).values
            for i in range(num_bases)
        ]
    )[None]
    print("number of {} clusters: ".format(mode), num_bases)
    return sampled_centers, num_bases, torch.tensor(labels)

  ### used only in compute_transforms
  def compute_means(ts, fg: GaussianParams, bases: MotionBases):
      transfms = bases.compute_transforms(ts, fg.get_coefs())
      means = torch.einsum(
          "pnij,pj->pni",
          transfms,
          F.pad(fg.params["means"], (0, 1), value=1.0),
      )
      return means
'''