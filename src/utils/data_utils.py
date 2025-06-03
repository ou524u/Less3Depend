import random
import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict as edict
from einops import rearrange
import imageio
from src.utils.camera_utils import pose_encoding_to_c2w, c2w_to_pose_encoding


def create_video_from_frames(frames, output_video_file, framerate=30):
    """
    Creates a video from a sequence of frames.

    Parameters:
        frames (numpy.ndarray): Array of image frames (shape: N x H x W x C).
        output_video_file (str): Path to save the output video file.
        framerate (int, optional): Frames per second for the video. Default is 30.
    """
    frames = np.asarray(frames)

    # Normalize frames if values are in [0,1] range
    if frames.max() <= 1:
        frames = (frames * 255).astype(np.uint8)

    imageio.mimsave(output_video_file, frames, fps=framerate, quality=8)

def rays_from_c2w(c2w, fxfycxcy, h=224, w=224, device="cuda"):
    """
    Args:
        c2w (torch.tensor): [b, v, 4, 4]
        fxfycxcy (torch.tensor): [b, v, 4]
        h (int): height of the image
        w (int): width of the image
    Returns:
        ray_o (torch.tensor): [b, v, 3, h, w]
        ray_d (torch.tensor): [b, v, 3, h, w]
    """

    b, v = c2w.size()[:2]
    c2w = c2w.reshape(b * v, 4, 4)

    fx, fy, cx, cy = fxfycxcy[:,:, 0], fxfycxcy[:,:,  1], fxfycxcy[:,:,  2], fxfycxcy[:,:,  3]

    fxfycxcy = fxfycxcy.reshape(b * v, 4)
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    y, x = y.to(device), x.to(device)
    x = x[None, :, :].expand(b * v, -1, -1).reshape(b * v, -1)
    y = y[None, :, :].expand(b * v, -1, -1).reshape(b * v, -1)
    x = (x + 0.5 - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]
    y = (y + 0.5 - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]
    z = torch.ones_like(x)
    ray_d = torch.stack([x, y, z], dim=2)  # [b*v, h*w, 3]
    ray_d = torch.bmm(ray_d, c2w[:, :3, :3].transpose(1, 2))  # [b*v, h*w, 3]
    # ray_d = ray_d / torch.norm(ray_d, dim=2, keepdim=True).clamp(min=1e-3)  # [b*v, h*w, 3]

    # NOTE: use fp32 when doing normalize!
    training_dtype = ray_d.dtype
    ray_d_fp32 = ray_d.to(torch.float32)
    ray_d_norm = ray_d_fp32.norm(dim=2, keepdim=True).clamp(min=1e-3)
    ray_d = (ray_d_fp32 / ray_d_norm).to(training_dtype)


    ray_o = c2w[:, :3, 3][:, None, :].expand_as(ray_d)  # [b*v, h*w, 3]

    ray_o = rearrange(ray_o, "(b v) (h w) c -> b v c h w", b=b, v=v, h=h, w=w, c=3)
    ray_d = rearrange(ray_d, "(b v) (h w) c -> b v c h w", b=b, v=v, h=h, w=w, c=3)

    return ray_o, ray_d
    


class ProcessData(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    @torch.no_grad()
    def compute_rays(self, c2w, fxfycxcy, h=None, w=None, device="cuda"):
        """
        Args:
            c2w (torch.tensor): [b, v, 4, 4]
            fxfycxcy (torch.tensor): [b, v, 4]
            h (int): height of the image
            w (int): width of the image
        Returns:
            ray_o (torch.tensor): [b, v, 3, h, w]
            ray_d (torch.tensor): [b, v, 3, h, w]
        """

        b, v = c2w.size()[:2]
        c2w = c2w.reshape(b * v, 4, 4)

        fx, fy, cx, cy = fxfycxcy[:,:, 0], fxfycxcy[:,:,  1], fxfycxcy[:,:,  2], fxfycxcy[:,:,  3]
        h_orig = int(2 * cy.max().item())  # Original height (estimated from the intrinsic matrix)
        w_orig = int(2 * cx.max().item())  # Original width (estimated from the intrinsic matrix)
        if h is None or w is None:
            h, w = h_orig, w_orig

        # in case the ray/image map has different resolution than the original image
        if h_orig != h or w_orig != w:
            fx = fx * w / w_orig
            fy = fy * h / h_orig
            cx = cx * w / w_orig
            cy = cy * h / h_orig

        fxfycxcy = fxfycxcy.reshape(b * v, 4)
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        y, x = y.to(device), x.to(device)
        x = x[None, :, :].expand(b * v, -1, -1).reshape(b * v, -1)
        y = y[None, :, :].expand(b * v, -1, -1).reshape(b * v, -1)
        x = (x + 0.5 - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]
        y = (y + 0.5 - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]
        z = torch.ones_like(x)
        ray_d = torch.stack([x, y, z], dim=2)  # [b*v, h*w, 3]
        ray_d = torch.bmm(ray_d, c2w[:, :3, :3].transpose(1, 2))  # [b*v, h*w, 3]
        # ray_d = ray_d / torch.norm(ray_d, dim=2, keepdim=True).clamp(min=1e-3)  # [b*v, h*w, 3]

        # NOTE: use fp32 when doing normalize!
        training_dtype = ray_d.dtype
        ray_d_fp32 = ray_d.to(torch.float32)
        ray_d_norm = ray_d_fp32.norm(dim=2, keepdim=True).clamp(min=1e-3)
        ray_d = (ray_d_fp32 / ray_d_norm).to(training_dtype)

        ray_o = c2w[:, :3, 3][:, None, :].expand_as(ray_d)  # [b*v, h*w, 3]

        ray_o = rearrange(ray_o, "(b v) (h w) c -> b v c h w", b=b, v=v, h=h, w=w, c=3)
        ray_d = rearrange(ray_d, "(b v) (h w) c -> b v c h w", b=b, v=v, h=h, w=w, c=3)

        return ray_o, ray_d
    
    def fetch_views(self, data_batch, has_target_image=False, target_has_input=True):
        """
        Splits the input data batch into input and target sets.
        
        Args:
            data_batch (dict): Contains input tensors with the following keys:
                - 'image' (torch.Tensor): Shape [b, v, c, h, w], optional for some target views
                - 'fxfycxcy' (torch.Tensor): Shape [b, v, 4]
                - 'c2w' (torch.Tensor): Shape [b, v, 4, 4]
            target_has_input (bool): If True, target includes input views.

        Returns:
            tuple: (input_dict, target_dict), both as EasyDict objects.

        """
        # randomize input views if dynamic_input_view_num is True and not in inference mode
        if (self.config.training.get("dynamic_input_view_num", False) 
            and (not self.config.inference.get("if_inference", False))):
            self.config.training.num_input_views = np.random.randint(2, 5)
        

        input_dict, target_dict = {}, {}
        # index = [] save for future use if we want to select specific views

        num_target_views, num_views, bs = self.config.training.num_target_views, data_batch["c2w"].size(1), data_batch["image"].size(0)
        # assert num_target_views < num_views, f"We have {num_views} views, but we want to select {num_target_views} target views. This is more than the total number of views we have."
        
        # Decide the target view indices
        if target_has_input: 
            # Randomly sample target views across all views
            index = torch.tensor([
                random.sample(range(num_views), num_target_views)
                for _ in range(bs)
            ], dtype=torch.long, device=data_batch["image"].device) # [b, num_target_views]

        else:
            num_target_views = 3
            num_views = 5
            assert (
                self.config.training.num_input_views + num_target_views <= num_views
            ), f"We have {num_views} views in total, but we want to select {self.config.training.num_input_views} input views and {num_target_views} target views. This is more than the total number of views we have."
            
            index = torch.tensor([
                [num_views - 1 - j for j in range(num_target_views)]
                for _ in range(bs)
            ], dtype=torch.long, device=data_batch["image"].device)
            index = torch.sort(index, dim=1).values # [b, num_target_views]


        for key, value in data_batch.items():
            if key == "scene_name":
                input_dict[key] = value
                target_dict[key] = value
                continue
            
            input_dict[key] = value[:, :self.config.training.num_input_views, ...]

            to_expand_dim = value.shape[2:] # [b, v, (value dim)] -> [value dim], e.g. [c, h, w] or [4] or [4, 4]
            expanded_index = index.view(index.shape[0], index.shape[1], *(1,) * len(to_expand_dim)).expand(-1, -1, *to_expand_dim)

            # Don't have target image supervision 
            if key == "image" and not has_target_image:                
                continue
            else:
                target_dict[key] = torch.gather(value, dim=1, index=expanded_index)
        
        height, width = data_batch["image"].shape[3], data_batch["image"].shape[4]
        input_dict["image_h_w"] = (height, width)
        target_dict["image_h_w"] = (height, width)

        input_dict, target_dict = edict(input_dict), edict(target_dict)

        return input_dict, target_dict


    
    @torch.no_grad()
    def forward(self, data_batch, has_target_image=True, target_has_input=True, compute_rays=True):
        """
        Preprocesses the input data batch and (optionally) computes ray_o and ray_d.

        Args:
            data_batch (dict): Contains input tensors with the following keys:
                - 'image' (torch.Tensor): Shape [b, v, c, h, w]
                - 'fxfycxcy' (torch.Tensor): Shape [b, v, 4]
                - 'c2w' (torch.Tensor): Shape [b, v, 4, 4]
            has_target_image (bool): If True, target views have image supervision.
            target_has_input (bool): If True, target views can be sampled from input views.
            compute_rays (bool): If True, compute ray_o and ray_d.
                
        Returns:
            Input and Target data_batch (dict): Contains processed tensors with the following keys:
                - 'image' (torch.Tensor): Shape [b, v, c, h, w]
                - 'fxfycxcy' (torch.Tensor): Shape [b, v, 4]
                - 'c2w' (torch.Tensor): Shape [b, v, 4, 4]
                - 'ray_o' (torch.Tensor): Shape [b, v, 3, h, w]
                - 'ray_d' (torch.Tensor): Shape [b, v, 3, h, w]
                - 'image_h_w' (tuple): (height, width)
        """
        # # NOTE: this is to test if our code could handle errors
        # if torch.rand(1).item() < 0.5:
        #     print("Error: test if our code could handle errors")
        #     return None, None

        input_dict, target_dict = self.fetch_views(data_batch, has_target_image=has_target_image, target_has_input=target_has_input)

        noise_scale = self.config.get("unposed", {}).get("noise_scale", -1.0)
        # NOTE: use fp32 when doing noising
        if noise_scale > 0:
            # noise input camera
            training_dtype = input_dict["c2w"].dtype
            input_pose_enc = c2w_to_pose_encoding((input_dict["c2w"]).to(torch.float32))
            noised_pose_enc = input_pose_enc + torch.randn_like(input_pose_enc) * noise_scale
            input_dict["c2w"] = pose_encoding_to_c2w(noised_pose_enc).to(training_dtype)

            # noise target camera
            target_pose_enc = c2w_to_pose_encoding(target_dict["c2w"].to(torch.float32))
            noised_target_pose_enc = target_pose_enc + torch.randn_like(target_pose_enc) * noise_scale
            target_dict["c2w"] = pose_encoding_to_c2w(noised_target_pose_enc).to(training_dtype)

            # print(f"Noised input camera: {input_dict['c2w'][0, 0]}, target camera: {target_dict['c2w'][0, 0]}")


        canonical_camera = self.config.get("unposed", {}).get("canonical_camera", False)
        # for unposedLVSM models only.

        # NOTE: use fp32 when doing canonical transformation
        if canonical_camera:
            training_dtype = input_dict["c2w"].dtype

            input_extrinsics = input_dict["c2w"].to(torch.float32)
            target_extrinsics = target_dict["c2w"].to(torch.float32)

            # for all c2w matrices, set the first camera as the canonical camera. 
            C0 = input_extrinsics[:, :1]

            # need to check first if the first camera is ill-conditioned.
            conds = torch.linalg.cond(C0.squeeze(1))  # [B], each is cond of a [4, 4] matrix
            if not torch.all(conds < 1e4):
                print("Error: the first camera is ill-conditioned")
                return None, None
            
            C0_inv = torch.linalg.inv(C0)
            input_dict["c2w"] = torch.matmul(C0_inv, input_extrinsics).to(training_dtype)
            target_dict["c2w"] = torch.matmul(C0_inv, target_extrinsics).to(training_dtype)
            # print(f"Canonical camera set to the first view. Identity matrix: {input_dict['c2w'][0, 0]}")

        if compute_rays:
            for dict in [input_dict, target_dict]:
                c2w = dict["c2w"]
                fxfycxcy = dict["fxfycxcy"]
                image_height, image_width = dict["image_h_w"]

                ray_o, ray_d = self.compute_rays(c2w, fxfycxcy, image_height, image_width, device=data_batch["image"].device)
                dict["ray_o"], dict["ray_d"] = ray_o, ray_d

        return input_dict, target_dict

      
      