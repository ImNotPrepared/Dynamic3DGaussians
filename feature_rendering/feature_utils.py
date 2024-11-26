

import torch
import torch.nn.functional as F


## to this end we get (N*N)
tensor_normalized = F.normalize(tensor, p=2, dim=1)
cosine_similarity_matrix = torch.matmul(tensor_normalized, tensor_normalized.T)


k = 5  

topk_values, topk_indices = torch.topk(cosine_similarity_matrix, k=k+1, dim=1)

topk_indices = topk_indices[:, 1:]  # Shape: [N, k]



    def compute_transforms(
        self, ts: torch.Tensor, inds: torch.Tensor | None = None
    ) -> torch.Tensor:
        coefs = self.fg.get_coefs()  # (G, K)
        if inds is not None:
            coefs = coefs[inds]
        transfms = self.motion_bases.compute_transforms(ts, coefs)  # (G, B, 3, 4)
        return transfms