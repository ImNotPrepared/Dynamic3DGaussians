        if motion_coefs is not None:
            params_dict["motion_coefs"] = nn.Parameter(motion_coefs)
        self.params = nn.ParameterDict(params_dict)
        self.quat_activation = lambda x: F.normalize(x, dim=-1, p=2)
        self.color_activation = torch.sigmoid
        self.scale_activation = torch.exp
        self.opacity_activation = torch.sigmoid
        self.motion_coef_activation = lambda x: F.softmax(x, dim=-1)


        def get_coefs(self) -> torch.Tensor:
            assert "motion_coefs" in self.params
            return self.motion_coef_activation(self.params["motion_coefs"])

### During Training:
    motion_bases, motion_coefs, tracks_3d = init_motion_params_with_procrustes(
        tracks_3d, num_motion_bases, rot_type, cano_t, vis=vis, port=port
    )
    motion_bases = motion_bases.to(device)

    fg_params = init_fg_from_tracks_3d(cano_t, tracks_3d, motion_coefs)
    fg_params = fg_params.to(device)


def init_motion_params_with_procrustes(
    tracks_3d: TrackObservations,
    num_bases: int,
    cano_t: int,
    min_mean_weight: float = 0.1,
) -> tuple[MotionBases, torch.Tensor, TrackObservations]:
    device = tracks_3d.xyz.device
    num_frames = tracks_3d.xyz.shape[1]
    # sample centers and get initial se3 motion bases by solving procrustes
    means_cano = tracks_3d.xyz[:, cano_t].clone()  # [num_gaussians, 3]

    # remove outliers
    scene_center = means_cano.median(dim=0).values
    print(f"{scene_center=}")
    dists = torch.norm(means_cano - scene_center, dim=-1)
    dists_th = torch.quantile(dists, 0.95)
    valid_mask = dists < dists_th

    # remove tracks that are not visible in any frame
    valid_mask = valid_mask & tracks_3d.visibles.any(dim=1)
    print(f"{valid_mask.sum()=}")

    tracks_3d = tracks_3d.filter_valid(valid_mask)


    means_cano = means_cano[valid_mask]

    sampled_centers, num_bases, labels = sample_initial_bases_centers(
        cluster_init_method, cano_t, tracks_3d, num_bases
    )

    # assign each point to the label to compute the cluster weight
    ids, counts = labels.unique(return_counts=True)
    ids = ids[counts > 100]
    num_bases = len(ids)
    sampled_centers = sampled_centers[:, ids]
    print(f"{num_bases=} {sampled_centers.shape=}")

    # compute basis weights from the distance to the cluster centers
    dists2centers = torch.norm(means_cano[:, None] - sampled_centers, dim=-1)
    motion_coefs = 10 * torch.exp(-dists2centers)

    init_rots, init_ts = [], []

    id_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    rot_dim = 4
    init_rots = id_rot.reshape(1, 1, rot_dim).repeat(num_bases, num_frames, 1)
    init_ts = torch.zeros(num_bases, num_frames, 3, device=device)
    errs_before = np.full((num_bases, num_frames), -1.0)
    errs_after = np.full((num_bases, num_frames), -1.0)

    tgt_ts = list(range(cano_t - 1, -1, -1)) + list(range(cano_t, num_frames))
    print(f"{tgt_ts=}")
    skipped_ts = {}
    for n, cluster_id in enumerate(ids):
        mask_in_cluster = labels == cluster_id
        cluster = tracks_3d.xyz[mask_in_cluster].transpose(
            0, 1
        )  # [num_frames, n_pts, 3]
        visibilities = tracks_3d.visibles[mask_in_cluster].swapaxes(
            0, 1
        )  # [num_frames, n_pts]
        confidences = tracks_3d.confidences[mask_in_cluster].swapaxes(
            0, 1
        )  # [num_frames, n_pts]
        weights = get_weights_for_procrustes(cluster, visibilities)
        prev_t = cano_t
        cluster_skip_ts = []
        for cur_t in tgt_ts:
            # compute pairwise transform from cano_t
            procrustes_weights = (
                weights[cano_t]
                * weights[cur_t]
                * (confidences[cano_t] + confidences[cur_t])
                / 2
            )
            if procrustes_weights.sum() < min_mean_weight * num_frames:
                init_rots[n, cur_t] = init_rots[n, prev_t]
                init_ts[n, cur_t] = init_ts[n, prev_t]
                cluster_skip_ts.append(cur_t)
            else:
                se3, (err, err_before) = solve_procrustes(
                    cluster[cano_t],
                    cluster[cur_t],
                    weights=procrustes_weights,
                    enforce_se3=True,
                    rot_type=rot_type,
                )
                init_rot, init_t, _ = se3
                assert init_rot.shape[-1] == rot_dim
                # double cover
                if torch.linalg.norm(
                    init_rot - init_rots[n][prev_t]
                ) > torch.linalg.norm(-init_rot - init_rots[n][prev_t]):
                    init_rot = -init_rot
                init_rots[n, cur_t] = init_rot
                init_ts[n, cur_t] = init_t
                if err == np.nan:
                    print(f"{cur_t=} {err=}")
                    print(f"{procrustes_weights.isnan().sum()=}")
                if err_before == np.nan:
                    print(f"{cur_t=} {err_before=}")
                    print(f"{procrustes_weights.isnan().sum()=}")
                errs_after[n, cur_t] = err
                errs_before[n, cur_t] = err_before
            prev_t = cur_t
        skipped_ts[cluster_id.item()] = cluster_skip_ts


    bases = MotionBases(init_rots, init_ts)
    return bases, motion_coefs, tracks_3d

class MotionBases(nn.Module):
    def __init__(self, rots, transls):
        super().__init__()
        self.num_frames = rots.shape[1]
        self.num_bases = rots.shape[0]
        assert check_bases_sizes(rots, transls)
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


def load_target_tracks(
    self, query_index: int, target_indices: list[int], dim: int = 1
):
    """
    tracks are 2d, occs and uncertainties
    :param dim (int), default 1: dimension to stack the time axis
    return (N, T, 4) if dim=1, (T, N, 4) if dim=0
    """
    q_name = self.frame_names[query_index]
    all_tracks = []
    for ti in target_indices:
        t_name = self.frame_names[ti]
        path = f"{self.tracks_dir}/{q_name}_{t_name}.npy"
        tracks = np.load(path).astype(np.float32)
        all_tracks.append(tracks)
    return torch.from_numpy(np.stack(all_tracks, axis=dim))

def get_tracks_3d(
    self, num_samples: int, start: int = 0, end: int = -1, step: int = 1, **kwargs
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_frames = self.num_frames
    if end < 0:
        end = num_frames + 1 + end
    query_idcs = list(range(start, end, step))
    target_idcs = list(range(start, end, step))


    masks = torch.stack([self.get_mask(i) for i in target_idcs], dim=0)
    fg_masks = (masks == 1).float()
    depths = torch.stack([self.get_depth(i) for i in target_idcs], dim=0)
    inv_Ks = torch.linalg.inv(self.Ks[target_idcs])
    c2ws = torch.linalg.inv(self.w2cs[target_idcs])

    num_per_query_frame = int(np.ceil(num_samples / len(query_idcs)))
    cur_num = 0
    tracks_all_queries = []
    for q_idx in query_idcs:

        ### For EveryTimestep
        # (N, T, 4)
        tracks_2d = self.load_target_tracks(q_idx, target_idcs)
        num_sel = int(
            min(num_per_query_frame, num_samples - cur_num, len(tracks_2d))
        )
        if num_sel < len(tracks_2d):
            sel_idcs = np.random.choice(len(tracks_2d), num_sel, replace=False)
            tracks_2d = tracks_2d[sel_idcs]
        cur_num += tracks_2d.shape[0]
        img = self.get_image(q_idx)
        tidx = target_idcs.index(q_idx)

        tracks_tuple = get_tracks_3d_for_query_frame(
            tidx, img, tracks_2d, depths, fg_masks, inv_Ks, c2ws
        )
        tracks_all_queries.append(tracks_tuple)

    tracks_3d, colors, visibles, invisibles, confidences = map(
        partial(torch.cat, dim=0), zip(*tracks_all_queries)
    )
    return tracks_3d, visibles, invisibles, confidences, colors

def get_tracks_3d_for_query_frame(
    query_index: int,
    query_img: torch.Tensor,
    tracks_2d: torch.Tensor,
    depths: torch.Tensor,
    masks: torch.Tensor,
    inv_Ks: torch.Tensor,
    c2ws: torch.Tensor,
):
    """
    :param query_index (int)
    :param query_img [H, W, 3]
    :param tracks_2d [N, T, 4]
    :param depths [T, H, W]
    :param masks [T, H, W]
    :param inv_Ks [T, 3, 3]
    :param c2ws [T, 4, 4]
    returns (
        tracks_3d [N, T, 3]
        track_colors [N, 3]
        visibles [N, T]
        invisibles [N, T]
        confidences [N, T]
    )
    """
    T, H, W = depths.shape
    query_img = query_img[None].permute(0, 3, 1, 2)  # (1, 3, H, W)
    tracks_2d = tracks_2d.swapaxes(0, 1)  # (T, N, 4)
    tracks_2d, occs, dists = (
        tracks_2d[..., :2],
        tracks_2d[..., 2],
        tracks_2d[..., 3],
    )
    # visibles = postprocess_occlusions(occs, dists)
    # (T, N), (T, N), (T, N)
    visibles, invisibles, confidences = parse_tapir_track_info(occs, dists)
    # Unproject 2D tracks to 3D.
    # (T, 1, H, W), (T, 1, N, 2) -> (T, 1, 1, N)
    track_depths = F.grid_sample(
        depths[:, None],
        normalize_coords(tracks_2d[:, None], H, W),
        align_corners=True,
        padding_mode="border",
    )[:, 0, 0]
    tracks_3d = (
        torch.einsum(
            "nij,npj->npi",
            inv_Ks,
            F.pad(tracks_2d, (0, 1), value=1.0),
        )
        * track_depths[..., None]
    )
    tracks_3d = torch.einsum("nij,npj->npi", c2ws, F.pad(tracks_3d, (0, 1), value=1.0))[
        ..., :3
    ]
    # Filter out out-of-mask tracks.
    # (T, 1, H, W), (T, 1, N, 2) -> (T, 1, 1, N)
    is_in_masks = (
        F.grid_sample(
            masks[:, None],
            normalize_coords(tracks_2d[:, None], H, W),
            align_corners=True,
        )[:, 0, 0]
        == 1
    )
    visibles *= is_in_masks
    invisibles *= is_in_masks
    confidences *= is_in_masks.float()

    # valid if in the fg mask at least 40% of the time
    # in_mask_counts = is_in_masks.sum(0)
    # t = 0.25
    # thresh = min(t * T, in_mask_counts.float().quantile(t).item())
    # valid = in_mask_counts > thresh
    valid = is_in_masks[query_index]
    # valid if visible 5% of the time
    visible_counts = visibles.sum(0)
    valid = valid & (
        visible_counts
        >= min(
            int(0.05 * T),
            visible_counts.float().quantile(0.1).item(),
        )
    )

    # Get track's color from the query frame.
    # (1, 3, H, W), (1, 1, N, 2) -> (1, 3, 1, N) -> (N, 3)
    track_colors = F.grid_sample(
        query_img,
        normalize_coords(tracks_2d[query_index : query_index + 1, None], H, W),
        align_corners=True,
        padding_mode="border",
    )[0, :, 0].T
    return (
        tracks_3d[:, valid].swapdims(0, 1),
        track_colors[valid],
        visibles[:, valid].swapdims(0, 1),
        invisibles[:, valid].swapdims(0, 1),
        confidences[:, valid].swapdims(0, 1),
    )