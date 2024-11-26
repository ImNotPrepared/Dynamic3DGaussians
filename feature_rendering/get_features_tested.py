import torch
from PIL import Image
import torchvision.transforms as T
import torchvision
from torchvision.transforms.functional import InterpolationMode, to_pil_image, resize, to_tensor
from sklearn.decomposition import PCA
import numpy as np
import imageio
import math
from itertools import product
from torch.nn import functional as F
import glob
import os
import pickle
import time
import argparse

def generate_crop_boxes_quadratic(
    im_size, n_layers: int, overlap_ratio: float, num_crops_l0=2
):
    """
    Generates a list of crop boxes of different sizes. Each layer
    has (2**i)**2 boxes for the ith layer.
    """
    crop_boxes, layer_idxs = [], []
    im_h, im_w = im_size
    short_side = min(im_h, im_w)

    # Original image
    crop_boxes.append([
        int((im_w / 2) - (short_side / 2)),
        int((im_h / 2) - (short_side / 2)),
        int((im_w / 2) + (short_side / 2)),
        int((im_h / 2) + (short_side / 2))])
    layer_idxs.append(0)

    def crop_len(orig_len, n_crops, overlap):
        return int(math.ceil((overlap * (n_crops - 1) + orig_len) / n_crops))

    def reverse_overlap(orig_len, n_crops, crop):
        return int((crop * n_crops - orig_len) / (n_crops - 1))

    for i_layer in range(n_layers):
        n_crops_per_side_w = num_crops_l0 ** (i_layer + 1) + 1 ** (i_layer)
        n_crops_per_side_h = num_crops_l0 ** (i_layer + 1)

        overlap_w = int(overlap_ratio * im_w * (2 / n_crops_per_side_w))
        overlap_h = int(overlap_ratio * im_h * (2 / n_crops_per_side_h))

        crop_w = crop_len(im_w, n_crops_per_side_w, overlap_w)
        crop_h = crop_len(im_h, n_crops_per_side_h, overlap_h)
        crop = max(crop_w, crop_h)

        if im_w > im_h:
            overlap_h = reverse_overlap(im_h, n_crops_per_side_h, crop)
        else:
            overlap_w = reverse_overlap(im_w, n_crops_per_side_w, crop)

        crop_box_x0 = [int((crop - overlap_w) * i) for i in range(n_crops_per_side_w)]
        crop_box_y0 = [int((crop - overlap_h) * i) for i in range(n_crops_per_side_h)]

        # Crops in XYWH format
        for x0, y0 in product(crop_box_x0, crop_box_y0):
            box = [x0, y0, min(x0 + crop, im_w), min(y0 + crop, im_h)]
            crop_boxes.append(box)
            layer_idxs.append(i_layer + 1)

    return crop_boxes, layer_idxs

def generate_im_feats(
    image: np.ndarray,
    model,
    transforms,
    output_size=(180, 320),
    num_crops_l0=4,
    crop_n_layers=2,
    model_input_size=896,
    crop_overlap_ratio=512 / 1500,
    embedding_dim=384,
    device="cuda:0"
):
    orig_size = image.shape[:2]

    crop_boxes, layer_idxs = generate_crop_boxes_quadratic(
        orig_size, crop_n_layers, crop_overlap_ratio, num_crops_l0=num_crops_l0
    )

    if output_size is None:
        output_size = orig_size
        scale_h = 1
        scale_w = 1
    else:
        scale_h = output_size[0] / orig_size[0]
        scale_w = output_size[1] / orig_size[1]

    image_features = torch.zeros(1, embedding_dim, output_size[0], output_size[1]).to(device)
    image_features_sum = torch.zeros(1, 1, output_size[0], output_size[1]).to(device)

    for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
        # get image features
        x0, y0, x1, y1 = crop_box
        cropped_im = image[y0:y1, x0:x1, :]
        cropped_im_size = cropped_im.shape[:2]
        transformed_im = preprocess(cropped_im, model_input_size)
        transformed_im_size = (transformed_im.shape[2], transformed_im.shape[3])


        crop_feat = predict(cropped_im, transforms, model, device)

        if model_input_size == 224:
            crop_feat = crop_feat.reshape(crop_feat.shape[0], 16, 16, crop_feat.shape[2]).permute(0, 3, 1, 2)
        else:
            crop_feat = crop_feat.reshape(crop_feat.shape[0], 64, 64, crop_feat.shape[2]).permute(0, 3, 1, 2)

        scaled_size = (int(cropped_im_size[0] * scale_h), int(cropped_im_size[1] * scale_w))

        crop_feat = postprocess_masks(
            crop_feat,
            transformed_im_size,
            scaled_size,
            model_input_size
        )

        # add features, upscaled embedding and mask data
        y0, x0 = int(scale_h * y0), int(scale_w * x0)
        y1, x1 = y0 + scaled_size[0], x0 + scaled_size[1]

        image_features[:, :, y0:y1, x0:x1] += crop_feat
        image_features_sum[:, :, y0:y1, x0:x1] += 1

    image_features = image_features / image_features_sum
    return image_features.cpu()

def postprocess_masks(feats: torch.Tensor, input_size, original_size, img_size) -> torch.Tensor:
    feats = F.interpolate(feats, img_size, mode="bilinear", align_corners=False)
    feats = feats[:, :, :input_size[0], :input_size[1]]
    feats = F.interpolate(feats, original_size, mode="bilinear", align_corners=False)
    return feats

def preprocess(x, model_input_size) -> torch.Tensor:
    target_size = get_preprocess_shape(x.shape[0], x.shape[1], model_input_size)
    x = np.array(resize(to_pil_image(x), target_size))
    x = torch.as_tensor(x)
    x = x.permute(2, 0, 1).contiguous()[None, :, :, :]
    # Pad
    h, w = x.shape[-2:]
    padh = model_input_size - h
    padw = model_input_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x

def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int):
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    return int(newh + 0.5), int(neww + 0.5)

def predict(img, transforms, model, device):
    img = to_pil_image(img)
    img = transforms(img).unsqueeze(0).to(device)
    print(img.shape)

    with torch.no_grad():
        features = model.forward_features(img)["x_norm_patchtokens"]
    return features

def main(args):
    output_size = (args.output_height, args.output_width)
    seqs = os.listdir(args.input_dir)

    # model and transforms
    # model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(args.device)

    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg').to(args.device)

    transforms = T.Compose([
        T.Resize(args.model_input_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(args.model_input_size),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    for j, seq in enumerate(seqs):
        try:
          int(seq[-1])
          print(f'processing:{seq}')
        except:
          continue
          
        if int(seq[-1]) != int(args.seq):
          continue

        path = f'{args.input_dir}/{seq}/00294.jpg'
        #paths = glob.glob(path)

        initial_scale = torchvision.transforms.Resize(
            output_size, InterpolationMode.BILINEAR)
        pca = None

        for i, p in enumerate(sorted([path])):
            img = to_tensor(Image.open(p))
            img = initial_scale(img).permute(1, 2, 0).numpy()

            features = generate_im_feats(
                img,
                model,
                transforms,
                output_size=output_size,
                model_input_size=args.model_input_size,
                num_crops_l0=args.num_crops_l0,
                crop_n_layers=args.crop_n_layers,
                embedding_dim=args.embedding_dim,
                device=args.device
            )

            features = features.permute(0, 2, 3, 1)
            features = features.cpu().squeeze().numpy()
            if features.shape[-1] != args.num_dims:
                shape = features.shape
                features = features.reshape(-1, shape[2])
                if pca is None:
                    pca = PCA(n_components=args.num_dims)
                    pca.fit(features)
                ###  first frame pca
                features = pca.transform(features)
                features = features.reshape(shape[0], shape[1], args.num_dims)



            path = p.replace(args.input_dir, args.save_dir)
            path = path.replace('.jpg', '')
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.save(path, features.squeeze())
            print(f'saved_to: {path}')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--do_pca", action='store_true', help="If just visualizing pca visualization of features of first image.")
    parser.add_argument("--save_feats", action='store_false', help="If features should be saved.")
    parser.add_argument("--num_crops_l0", default=4, type=int, help="How many crops in layer 0.")
    parser.add_argument("--crop_n_layers", default=1, type=int, help="How many layers.")
    parser.add_argument("--num_dims", default=32, type=int, help="Number of dimensions features should be downscaled to.")
    parser.add_argument("--device", default='cuda:0', type=str, help="Device to use.")
    parser.add_argument("--seq", default=1, type=str, help="Device to use.")
    parser.add_argument("--output_height", default=288, type=int, help="Desired output height of feature map.")
    parser.add_argument("--output_width", default=512, type=int, help="Desired output width of feature map.")
    parser.add_argument("--embedding_dim", default=384, type=int, help="Desired embedding dim for dino extractor.")
    # /ssd0/zihanwa3/data_ego/cmu_bike/ims/undist_data/undist_cam01 /ssd0/zihanwa3/data_ego/cmu_bike/ims/1/00111.jpg
    parser.add_argument("--model_input_size", default=896, type=int, choices=[896, 518, 224], help="Input images size, larger gives right res feature map.")
    parser.add_argument("--input_dir", default='/data3/zihanwa3/Capstone-DSR/Processing/undist_data', type=str, help="Directory with input sequences, ie, data/dataset/<sequences>/*.jpg.")
    parser.add_argument("--save_dir", default='/data3/zihanwa3/Capstone-DSR/Processing/dinov2features/test', type=str, help="Directory to save features, ie, data/dataset_features/<sequences>/*.jpg.")

    args = parser.parse_args()

    main(args)
    print('finished')
