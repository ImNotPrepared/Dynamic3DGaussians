



def interpolate_extrinsics(extrinsics):
    interpolated_extrinsics = []

    for i in range(len(extrinsics) - 1):
        R1 = extrinsics[i][:3, :3]
        R2 = extrinsics[i+1][:3, :3]
        t1 = extrinsics[i][:3, 3]
        t2 = extrinsics[i+1][:3, 3]

        # Convert rotations to quaternions
        q1 = R.from_matrix(R1).as_quat()
        q2 = R.from_matrix(R2).as_quat()

        # Spherical linear interpolation of rotations
        slerp = R.from_quat([q1, q2]).mean()

        # Linear interpolation of translations
        t = (t1 + t2) / 2

        # Combine interpolated rotation and translation into a 4x4 matrix
        interpolated_extrinsic = np.eye(4)
        interpolated_extrinsic[:3, :3] = slerp.as_matrix()
        interpolated_extrinsic[:3, 3] = t

        interpolated_extrinsics.append(interpolated_extrinsic)

    return interpolated_extrinsics

def (org_pose, int_pose):
      def held_stat_loss(stat_dataset):
        losses = 0
        for data in stat_dataset:
          
            im, radius, _, = Renderer(raster_settings=data['cam'])(**rendervar)
            curr_id = data['id']
            cam = setup_camera(w, h, k, w2c, near=0.01, far=50)
            im = torch.exp(params['cam_m'][curr_id])[:, None, None] * im + params['cam_c'][curr_id][:, None, None]
            losses += 0.8 * l1_loss_v1(im, data['im']) + 0.2 * (1.0 - calc_ssim(im, data['im']))
        return losses

    losses['stat_im']=held_stat_loss(stat_dataset)
