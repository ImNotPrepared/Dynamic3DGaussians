import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class MotionBases(nn.Module):
    def __init__(self, rots, transls):
        super().__init__()
        self.num_frames = rots.shape[1]
        self.num_bases = rots.shape[0]

        self.params = nn.ParameterDict(
            {
                "rots": nn.Parameter(rots),
                "transls": nn.Parameter(transls),
            }
        )

    @staticmethod
    def init_from_state_dict(state_dict, prefix="params."):
        param_keys = ["rots", "transls"]
        assert all(f"{prefix}{k}" in state_dict for k in param_keys)
        args = {k: state_dict[f"{prefix}{k}"] for k in param_keys}
        return MotionBases(**args)

    def compute_transforms(self, ts: torch.Tensor, coefs: torch.Tensor) -> torch.Tensor:
        """
        :param ts (B)
        :param coefs (G, K)
        returns transforms (G, B, 3, 4)
        """
        transls = self.params["transls"][:, ts]  # (K, B, 3)
        rots = self.params["rots"][:, ts]  # (K, B, 6)
        transls = torch.einsum("pk,kni->pni", coefs, transls)
        rots = torch.einsum("pk,kni->pni", coefs, rots)  # (G, B, 6)
        rotmats = cont_6d_to_rmat(rots)  # (K, B, 3, 3)
        return torch.cat([rotmats, transls[..., None]], dim=-1)



import torch
from sklearn.cluster import SpectralClustering

def similarity_mapping(feats, K=10):
    '''
    Args: 
        feats --> [N, E] torch.Tensor
        K     --> Number of clusters
    Returns:
        output --> [N, E, K] torch.Tensor
    '''
    
    feats_normalized = feats / feats.norm(dim=1, keepdim=True) # Normalize the feature vectors
    cos_sim_matrix = torch.mm(feats_normalized, feats_normalized.t())
    cos_sim_matrix_np = cos_sim_matrix.cpu().numpy()
    
    # Perform spectral clustering
    clustering = SpectralClustering(n_clusters=K, affinity='precomputed', random_state=42)
    labels = clustering.fit_predict(cos_sim_matrix_np)
    
    # Convert labels to one-hot encoding
    labels_one_hot = torch.nn.functional.one_hot(torch.tensor(labels), num_classes=K)  # Shape [N, K]
    
    # Expand dimensions to enable broadcasting
    feats_expanded = feats.unsqueeze(2)                  # Shape [N, E, 1]
    labels_expanded = labels_one_hot.unsqueeze(1)        # Shape [N, 1, K]
    
    # Multiply features with one-hot labels to get the output
    output = feats_expanded * labels_expanded            # Shape [N, E, K]
    
    return output

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
'''

def loading():
    motion_bases = MotionBases.init_from_state_dict(
        state_dict, prefix=f"{prefix}motion_bases.params."
    )



if __name__ == '__main__':

  if motion_coefs is not None:
    params_dict["motion_coefs"] = nn.Parameter(motion_coefs)

  feats = ...
  read_feature

  clustered_means = similarity_mapping(feats)

  means_cano = tracks_3d.xyz[:, cano_t].clone()  # [num_gaussians, 3]
  # remove outliers
  scene_center = means_cano.median(dim=0).values
  dists = torch.norm(means_cano - scene_center, dim=-1)
  dists_th = torch.quantile(dists, 0.95)
  valid_mask = dists < dists_th
  means_cano = means_cano[valid_mask]
  
  ### 
  sampled_centers, num_bases, labels = ###

  # assign each point to the label to compute the cluster weight
  ids, counts = labels.unique(return_counts=True)
  ids = ids[counts > 100]
  num_bases = len(ids)
  sampled_centers = sampled_centers[:, ids]

  dists2centers = torch.norm(means_cano[:, None] - sampled_centers, dim=-1)
  motion_coefs = 10 * torch.exp(-dists2centers)

  id_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
  rot_dim = 4
  init_rots = id_rot.reshape(1, 1, rot_dim).repeat(num_bases, num_frames, 1)
  init_ts = torch.zeros(num_bases, num_frames, 3, device=device)

  ## init
  bases = MotionBases(init_rots, init_ts)

  ## Inside the LOOP
  coefs = fg.get_coefs()
  transfms = bases.compute_transforms(ts, coefs)



  ##### transfms transforms (G, B, 3, 4)
  ### G: num, B: motion bases 3: 4: 
  ### rotmats = cont_6d_to_rmat(rots)  # (K, B, 3, 3)
  ### return torch.cat([rotmats, transls[..., None]], dim=-1)
  ###   fg means (G, 3) pad -> (G, 4)
  ##### returned (G, B, 4)
  positions = torch.einsum(
      "pnij,pj->pni",
      transfms,
      F.pad(fg.params["means"], (0, 1), value=1.0),
  )



  pred_2d, pred_depth = project_2d_tracks(
      positions.swapaxes(0, 1), Ks, w2cs, return_depth=True
  )
  pred_2d = pred_2d.swapaxes(0, 1)
  pred_depth = pred_depth.swapaxes(0, 1)


  ### used only in compute_transforms
  def compute_means(ts, fg: GaussianParams, bases: MotionBases):
      transfms = bases.compute_transforms(ts, fg.get_coefs())
      means = torch.einsum(
          "pnij,pj->pni",
          transfms,
          F.pad(fg.params["means"], (0, 1), value=1.0),
      )
      return means

  def compute_transforms(
      self, ts: torch.Tensor, inds: torch.Tensor | None = None
  ) -> torch.Tensor:

      self.motion_coef_activation = lambda x: F.softmax(x, dim=-1)
      def get_coefs(self) -> torch.Tensor:
          return self.motion_coef_activation(self.params["motion_coefs"])

      if inds is not None:
          coefs = coefs[inds]
      transfms = motion_bases.compute_transforms(ts, coefs)  # (G, B, 3, 4)
      return transfms

def compute_transforms(self, ts: torch.Tensor, coefs: torch.Tensor) -> torch.Tensor:
    """
    :param ts (B)
    :param coefs (G, K)
    returns transforms (G, B, 3, 4)
    """
    transls = self.params["transls"][:, ts]  # (K, B, 3)
    rots = self.params["rots"][:, ts]  # (K, B, 6)
    transls = torch.einsum("pk,kni->pni", coefs, transls)
    rots = torch.einsum("pk,kni->pni", coefs, rots)  # (G, B, 6)
    rotmats = cont_6d_to_rmat(rots)  # (K, B, 3, 3)
    return torch.cat([rotmats, transls[..., None]], dim=-1)