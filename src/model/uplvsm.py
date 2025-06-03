import os
import torch
import torch.nn as nn
from easydict import EasyDict as edict
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
import traceback
from src.utils import camera_utils, data_utils
from .transformer import QK_Norm_TransformerBlock, init_weights
from .loss import LossComputer
from .ptlvsm import dino_tokenizer
from src.utils.camera_utils import pose_encoding_to_c2w, c2w_to_pose_encoding
from src.utils.data_utils import rays_from_c2w
from .estimator import ViTEstimator

class unposedLVSM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.process_data = data_utils.ProcessData(config)

        # Initialize both input tokenizers, and output de-tokenizer
        self._init_tokenizers()
        
        # Initialize transformer blocks
        self._init_transformer()
        
        # Initialize estimator
        self._init_estimator()

        # Initialize loss computer
        self.loss_computer = LossComputer(config)

    def _create_dino_tokenizer(self, dino_qknorm=False):
        """Helper function to create a tokenizer with given config"""
        tokenizer = nn.Sequential(
            Rearrange(
                "b v c h w -> (b v) c h w",
            ),
            dino_tokenizer(dino_qknorm=dino_qknorm),
        )
        return tokenizer

    def _create_tokenizer(self, in_channels, patch_size, d_model):
        """Helper function to create a tokenizer with given config"""
        tokenizer = nn.Sequential(
            Rearrange(
                "b v c (hh ph) (ww pw) -> (b v) (hh ww) (ph pw c)",
                ph=patch_size,
                pw=patch_size,
            ),
            nn.Linear(
                in_channels * (patch_size**2),
                d_model,
                bias=False,
            ),
        )
        tokenizer.apply(init_weights)

        return tokenizer

    def _init_tokenizers(self):
        """Initialize the image and target pose tokenizers, and image token decoder"""
        # Image tokenizer
        dino_qknorm = self.config.get("unposed", {}).get("dino_qknorm", True)
        self.image_tokenizer = self._create_dino_tokenizer(dino_qknorm)
        
        # Target pose tokenizer
        self.target_pose_tokenizer = self._create_tokenizer(
            in_channels = 6,
            patch_size = 14,
            d_model = self.config.model.transformer.d
        )
        
        decoder_as_canonical = self.config.get("unposed", {}).get("decoder_as_canonical", False)
        if decoder_as_canonical:
            self.canonical_pose_tokenizer = self.target_pose_tokenizer
        else:
            self.canonical_pose_tokenizer = self._create_tokenizer(
                in_channels = 6,
                patch_size = 14,
                d_model = self.config.model.transformer.d
            )
        
        # Image token decoder (decode image tokens into pixels)
        self.image_token_decoder = nn.Sequential(
            nn.LayerNorm(self.config.model.transformer.d, bias=False),
            nn.Linear(
                self.config.model.transformer.d,
                (self.config.model.target_pose_tokenizer.patch_size**2) * 3,
                bias=False,
            ),
            nn.Sigmoid()
        )
        self.image_token_decoder.apply(init_weights)


    def _init_transformer(self):
        """Initialize transformer blocks"""
        config = self.config.model.transformer
        use_qk_norm = config.get("use_qk_norm", False)

        # latent vectors for LVSM encoder-decoder
        self.n_light_field_latent = nn.Parameter(
            torch.randn(
                config.n_latent_vectors,
                config.d,
            )
        )
        nn.init.trunc_normal_(self.n_light_field_latent, std=0.02)

        # Create transformer blocks
        self.transformer_encoder = [
            QK_Norm_TransformerBlock(
                config.d, config.d_head, use_qk_norm=use_qk_norm
            ) for _ in range(config.encoder_n_layer)
        ]

        self.transformer_decoder = [
            QK_Norm_TransformerBlock(
                config.d, config.d_head, use_qk_norm=use_qk_norm
            ) for _ in range(config.decoder_n_layer)
        ]
        
        # Apply special initialization if configured
        if config.get("special_init", False):
            # Encoder
            for idx, block in enumerate(self.transformer_encoder):
                if config.get("depth_init", False):
                    weight_init_std = 0.02 / (2 * (idx + 1)) ** 0.5
                else:
                    weight_init_std = 0.02 / (2 * config.encoder_n_layer) ** 0.5
                block.apply(lambda module: init_weights(module, weight_init_std))

            # Decoder
            for idx, block in enumerate(self.transformer_decoder):
                if config.get("depth_init", False):
                    weight_init_std = 0.02 / (2 * (idx + 1)) ** 0.5
            else:
                weight_init_std = 0.02 / (2 * config.decoder_n_layer) ** 0.5
            block.apply(lambda module: init_weights(module, weight_init_std))  
        else:
            # Encoder
            for block in self.transformer_encoder:
                block.apply(init_weights)

            # Decoder
            for block in self.transformer_decoder:
                block.apply(init_weights)

                
        self.transformer_encoder = nn.ModuleList(self.transformer_encoder)
        self.transformer_decoder = nn.ModuleList(self.transformer_decoder)
        self.transformer_input_layernorm_decoder = nn.LayerNorm(config.d, bias=False)

    def _init_estimator(self):
        self.estimator = ViTEstimator(self.config)
        self.estimator.apply(init_weights)


    def train(self, mode=True):
        """Override the train method to keep the loss computer in eval mode"""
        super().train(mode)
        self.loss_computer.eval()


    
    def pass_layers(self, transformer_blocks, input_tokens, gradient_checkpoint=False, checkpoint_every=1):
        """
        Helper function to pass input tokens through all transformer blocks with optional gradient checkpointing.
        
        Args:
            input_tokens: Tensor of shape [batch_size, num_views * num_patches, hidden_dim]
                The input tokens to process through the transformer blocks.
            gradient_checkpoint: bool, default False
                Whether to use gradient checkpointing to save memory during training.
            checkpoint_every: int, default 1 
                Number of transformer layers to group together for gradient checkpointing.
                Only used when gradient_checkpoint=True.
                
        Returns:
            Tensor of shape [batch_size, num_views * num_patches, hidden_dim]
                The processed tokens after passing through all transformer blocks.
        """
        num_layers = len(transformer_blocks)
        
        if not gradient_checkpoint:
            # Standard forward pass through all layers
            for layer in transformer_blocks:
                input_tokens = layer(input_tokens)
            return input_tokens
            
        # Gradient checkpointing enabled - process layers in groups
        def _process_layer_group(tokens, start_idx, end_idx):
            """Helper to process a group of consecutive layers."""
            for idx in range(start_idx, end_idx):
                tokens = transformer_blocks[idx](tokens)
            return tokens
            
        # Process layer groups with gradient checkpointing
        for start_idx in range(0, num_layers, checkpoint_every):
            end_idx = min(start_idx + checkpoint_every, num_layers)
            input_tokens = torch.utils.checkpoint.checkpoint(
                _process_layer_group,
                input_tokens,
                start_idx,
                end_idx,
                use_reentrant=False
            )
            
        return input_tokens
            

    def get_posed_input(self, images=None, ray_o=None, ray_d=None, method="default_plucker"):
        '''
        Args:
            images: [b, v, c, h, w]
            ray_o: [b, v, 3, h, w]
            ray_d: [b, v, 3, h, w]
            method: Method for creating pose conditioning
        Returns:
            posed_images: [b, v, c+6, h, w] or [b, v, 6, h, w] if images is None
        '''

        if method == "custom_plucker":
            o_dot_d = torch.sum(-ray_o * ray_d, dim=2, keepdim=True)
            nearest_pts = ray_o + o_dot_d * ray_d
            pose_cond = torch.cat([ray_d, nearest_pts], dim=2)
            
        elif method == "aug_plucker":
            o_dot_d = torch.sum(-ray_o * ray_d, dim=2, keepdim=True)
            nearest_pts = ray_o + o_dot_d * ray_d
            o_cross_d = torch.cross(ray_o, ray_d, dim=2)
            pose_cond = torch.cat([o_cross_d, ray_d, nearest_pts], dim=2)
            
        else:  # default_plucker
            o_cross_d = torch.cross(ray_o, ray_d, dim=2)
            pose_cond = torch.cat([o_cross_d, ray_d], dim=2)

        if images is None:
            return pose_cond
        else:
            return torch.cat([images * 2.0 - 1.0, pose_cond], dim=2)

    def get_pose_from_enc(self, pose_enc, pose_enc_type="absT_quaR", known_fxfycxcy=None, image_size=224):
        # pose_enc: [b, v, 7] or [b, v, 8]
        # pose_enc_type: "absT_quaR" or "absT_quaR_FoV"
        if pose_enc_type == "absT_quaR":
            if known_fxfycxcy is None:
                raise ValueError("known_fxfycxcy is required for absT_quaR")
            estimated_c2w = pose_encoding_to_c2w(pose_enc)
            estimated_ray_o, estimated_ray_d = rays_from_c2w(estimated_c2w, known_fxfycxcy, h=image_size, w=image_size)
        
        elif pose_enc_type == "absT_quaR_FoV":
            raise NotImplementedError
        else:
            raise ValueError(f"Unknown pose encoding type: {pose_enc_type}")
        
        return self.get_posed_input(ray_o=estimated_ray_o, ray_d=estimated_ray_d)

    def get_pose_from_c2w(self, c2w, known_fxfycxcy=None, image_size=224):
        if known_fxfycxcy is None:
            raise ValueError("known_fxfycxcy is required for get_pose_from_c2w")
        estimated_ray_o, estimated_ray_d = rays_from_c2w(c2w, known_fxfycxcy, h=image_size, w=image_size)
        return self.get_posed_input(ray_o=estimated_ray_o, ray_d=estimated_ray_d)

    
    def forward(self, data_batch, has_target_image=True, target_has_input=True):

        input, target = self.process_data(data_batch, has_target_image=has_target_image, target_has_input = target_has_input, compute_rays=True)
        if input is None:
            return None

        checkpoint_every = self.config.training.grad_checkpoint_every
        n_latent_vectors = self.config.model.transformer.n_latent_vectors
        
        # Process input images
        unposed_input_images = input.image
        b, v_input, c, h, w = unposed_input_images.size()
        input_img_tokens = self.image_tokenizer(unposed_input_images)  # [b*v, n_patches, d]

        _, n_patches, d = input_img_tokens.size()  # [b*v, n_patches, d]
        input_img_tokens = input_img_tokens.reshape(b, v_input * n_patches, d)  # [b, v*n_patches, d]
                
        # canonical_pose_cond = self.get_posed_input(ray_o=input.ray_o, ray_d=input.ray_d) # 
        # canonical_pose_tokens = self.canonical_pose_tokenizer(canonical_pose_cond) # [b*v, n_patches, d]
        # canonical_pose_tokens = canonical_pose_tokens.reshape(b, v_input * n_patches, d) # [b, v*n_patches, d]
        identity_c2w = torch.eye(4, device=input.image.device).repeat(b, v_input, 1, 1)
        canonical_pose_cond = self.get_pose_from_c2w(identity_c2w, known_fxfycxcy=input.fxfycxcy, image_size=h)
        canonical_pose_tokens = self.canonical_pose_tokenizer(canonical_pose_cond) # [b*v, n_patches, d]
        canonical_pose_tokens = canonical_pose_tokens.reshape(b, v_input * n_patches, d) # [b, v*n_patches, d]

        # add canonical marker
        input_img_tokens[:, :n_patches] = input_img_tokens[:, :n_patches] + canonical_pose_tokens[:, :n_patches]

        # get encoder input
        latent_vector_tokens = self.n_light_field_latent.expand(b, -1, -1) # [b, n_latent_vectors, d]
        encoder_input_tokens = torch.cat((latent_vector_tokens, input_img_tokens), dim=1) # [b, n_latent_vectors + v*n_patches, d]

        # encoder forward pass
        intermediate_tokens = self.pass_layers(self.transformer_encoder, encoder_input_tokens, gradient_checkpoint=True, checkpoint_every=checkpoint_every)

        latent_tokens, input_img_tokens = intermediate_tokens.split(
            [self.config.model.transformer.n_latent_vectors, v_input * n_patches], dim=1
        ) # [b, n_latent_vectors, d], [b, v*n_patches, d]
            
        # after passing through encoder, we get latent scene.
        b, v_target, c, h, w = target.image.size()
        repeated_latent_tokens = repeat(
                                latent_tokens,
                                'b nl d -> (b v_target) nl d', 
                                v_target=v_target) 
        # fix the estimator thing
        pose_enc_type = self.config.get("unposed", {}).get("pose_enc_type", "absT_quaR")

        # estimate pose encodings for target views
        target_img_tokens = self.image_tokenizer(target.image) # [b*v_target, n_patches, d]
        estimated_pose_enc = self.estimator(target_img_tokens, repeated_latent_tokens) # [b*v, 7] or [b*v, 8]
        estimated_pose_enc = rearrange(estimated_pose_enc, "(b v) d -> b v d", v=v_target) # [b, v, 7] or [b, v, 8]
        estimated_pose_cond = self.get_pose_from_enc(estimated_pose_enc, pose_enc_type=pose_enc_type, known_fxfycxcy=target.fxfycxcy, image_size=h)
        target_pose_tokens = self.target_pose_tokenizer(estimated_pose_cond) # [b*v_target, n_patches, d]

        
        decoder_input_tokens = torch.cat((target_pose_tokens, repeated_latent_tokens), dim=1) # [b*v_target, n_latent_vectors + n_patches, d]
        decoder_input_tokens = self.transformer_input_layernorm_decoder(decoder_input_tokens)

        transformer_output_tokens = self.pass_layers(self.transformer_decoder, decoder_input_tokens, gradient_checkpoint=True, checkpoint_every=checkpoint_every)

        # Discard the latent tokens
        target_image_tokens, _ = transformer_output_tokens.split(
            [n_patches, n_latent_vectors], dim=1
        ) # [b*v_target, n_patches, d], [b*v_target, n_latent_vectors, d]

        # [b*v_target, n_patches, p*p*3]
        rendered_images = self.image_token_decoder(target_image_tokens)
        
        height, width = target.image_h_w

        patch_size = self.config.model.target_pose_tokenizer.patch_size
        rendered_images = rearrange(
            rendered_images, "(b v) (h w) (p1 p2 c) -> b v c (h p1) (w p2)",
            v=v_target,
            h=height // patch_size, 
            w=width // patch_size, 
            p1=patch_size, 
            p2=patch_size, 
            c=3
        )
        if has_target_image:
            loss_metrics = self.loss_computer(
                rendered_images,
                target.image
            )
        else:
            loss_metrics = None

        result = edict(
            input=input,
            target=target,
            loss_metrics=loss_metrics,
            render=rendered_images        
            )
        
        return result


    @torch.no_grad()
    def render_video(self, data_batch, traj_type="interpolate", num_frames=60, loop_video=False, order_poses=False):
        """
        Render a video from the model.
        
        Args:
            result: Edict from forward pass or just data
            traj_type: Type of trajectory
            num_frames: Number of frames to render
            loop_video: Whether to loop the video
            order_poses: Whether to order poses
            
        Returns:
            result: Updated with video rendering
        """
    
        if data_batch.input is None:
            input, target = self.process_data(data_batch, has_target_image=False, target_has_input=self.config.training.target_has_input, compute_rays=True)
            data_batch = edict(input=input, target=target)
        else:
            input, target = data_batch.input, data_batch.target
        
        # Prepare unposed input images
        unposed_input_images = input.image
        bs, v_input, c, h, w = unposed_input_images.size()
        input_img_tokens = self.image_tokenizer(unposed_input_images)  # [b*v_input, n_patches, d]
        
        
        _, n_patches, d = input_img_tokens.size()  # [b*v, n_patches, d]
        input_img_tokens = input_img_tokens.reshape(bs, v_input * n_patches, d)  # [b, v*n_patches, d]
                
        identity_c2w = torch.eye(4, device=input.image.device).repeat(bs, v_input, 1, 1)
        canonical_pose_cond = self.get_pose_from_c2w(identity_c2w, known_fxfycxcy=input.fxfycxcy, image_size=h)
        canonical_pose_tokens = self.canonical_pose_tokenizer(canonical_pose_cond) # [b*v, n_patches, d]
        canonical_pose_tokens = canonical_pose_tokens.reshape(bs, v_input * n_patches, d) # [b, v*n_patches, d]

        # add canonical marker
        input_img_tokens[:, :n_patches] = input_img_tokens[:, :n_patches] + canonical_pose_tokens[:, :n_patches]

        # get encoder input
        latent_vector_tokens = self.n_light_field_latent.expand(bs, -1, -1) # [b, n_latent_vectors, d]
        encoder_input_tokens = torch.cat((latent_vector_tokens, input_img_tokens), dim=1) # [b, n_latent_vectors + v*n_patches, d]

        # encoder forward pass
        intermediate_tokens = self.pass_layers(self.transformer_encoder, encoder_input_tokens, gradient_checkpoint=True)

        latent_tokens, input_img_tokens = intermediate_tokens.split(
            [self.config.model.transformer.n_latent_vectors, v_input * n_patches], dim=1
        ) # [b, n_latent_vectors, d], [b, v*n_patches, d]
         
        
        if traj_type == "interpolate":
            # interpolate across views
            total_images = target.image
            _, v_total, _, _, _ = total_images.size()
            fxfycxcy_total = target.fxfycxcy


            # do the estimator thing
            repeated_latent_tokens = repeat(latent_tokens, 'b nl d -> (b v) nl d', v=v_total)
            input_img_tokens = self.image_tokenizer(total_images) # [b*v_input, n_patches, d]
            estimated_pose_enc = self.estimator(input_img_tokens, repeated_latent_tokens) # [b*v, 7] or [b*v, 8]
            estimated_pose_enc = rearrange(estimated_pose_enc, "(b v) d -> b v d", v=v_total) # [b, v, 7] or [b, v, 8]
            c2ws = pose_encoding_to_c2w(estimated_pose_enc) # [b, v, 4, 4]
            fxfycxcy = fxfycxcy_total #  [b, v, 4]
            device = target.c2w.device

            # Create intrinsics from fxfycxcy
            intrinsics = torch.zeros((c2ws.shape[0], c2ws.shape[1], 3, 3), device=device) # [b, v, 3, 3]
            intrinsics[:, :,  0, 0] = fxfycxcy[:, :, 0]
            intrinsics[:, :,  1, 1] = fxfycxcy[:, :, 1]
            intrinsics[:, :,  0, 2] = fxfycxcy[:, :, 2]
            intrinsics[:, :,  1, 2] = fxfycxcy[:, :, 3]

            # Loop video if requested
            if loop_video:
                c2ws = torch.cat([c2ws, c2ws[:, [0], :]], dim=1)
                intrinsics = torch.cat([intrinsics, intrinsics[:, [0], :]], dim=1)

            c2ws = c2ws.to(torch.float32)
            intrinsics = intrinsics.to(torch.float32)

            # Interpolate camera poses
            all_c2ws, all_intrinsics = [], []
            for b in range(input.image.size(0)):
                cur_c2ws, cur_intrinsics = camera_utils.get_interpolated_poses_many(
                    c2ws[b, :, :3, :4], intrinsics[b], num_frames, order_poses=order_poses
                )
                all_c2ws.append(cur_c2ws.to(device))
                all_intrinsics.append(cur_intrinsics.to(device))

            all_c2ws = torch.stack(all_c2ws, dim=0) # [b, num_frames, 3, 4]
            all_intrinsics = torch.stack(all_intrinsics, dim=0) # [b, num_frames, 3, 3]

            # Add homogeneous row to c2ws
            homogeneous_row = torch.tensor([[[0, 0, 0, 1]]], device=device).expand(all_c2ws.shape[0], all_c2ws.shape[1], -1, -1)
            all_c2ws = torch.cat([all_c2ws, homogeneous_row], dim=2)

            # Convert intrinsics to fxfycxcy format
            all_fxfycxcy = torch.zeros((all_intrinsics.shape[0], all_intrinsics.shape[1], 4), device=device)
            all_fxfycxcy[:, :, 0] = all_intrinsics[:, :, 0, 0]  # fx
            all_fxfycxcy[:, :, 1] = all_intrinsics[:, :, 1, 1]  # fy
            all_fxfycxcy[:, :, 2] = all_intrinsics[:, :, 0, 2]  # cx
            all_fxfycxcy[:, :, 3] = all_intrinsics[:, :, 1, 2]  # cy

        _, num_views, _, _ = all_c2ws.size() # [b, v, 4, 4]

        # Compute rays for rendering
        rendering_ray_o, rendering_ray_d = self.process_data.compute_rays(
            fxfycxcy=all_fxfycxcy, c2w=all_c2ws, h=h, w=w, device=device
        )
        # Get pose conditioning for target views
        target_pose_cond = self.get_posed_input(
            ray_o=rendering_ray_o.to(input.image.device), 
            ray_d=rendering_ray_d.to(input.image.device)
        )
        _, num_views, c, h, w = target_pose_cond.size()
        target_pose_tokens = self.target_pose_tokenizer(target_pose_cond) # [b*v_target, n_patches, d]

            
        _, n_patches, d = target_pose_tokens.size()  # [b*v_target, n_patches, d]
        target_pose_tokens = target_pose_tokens.reshape(bs, num_views * n_patches, d)  # [b, v_target*n_patches, d]

        view_chunk_size = 4
        video_rendering_list = []
        
        for cur_chunk in range(0, num_views, view_chunk_size):
            cur_view_chunk_size = min(view_chunk_size, num_views - cur_chunk)
            
            # Get current chunk of target pose tokens
            start_idx, end_idx = cur_chunk * n_patches, (cur_chunk + cur_view_chunk_size) * n_patches
            cur_target_pose_tokens = rearrange(target_pose_tokens[:, start_idx:end_idx,: ], 
                                               "b (v_chunk p) d -> (b v_chunk) p d", 
                                               v_chunk=cur_view_chunk_size, p=n_patches)

            cur_repeated_latent_tokens = repeat(
                latent_tokens,
                'b nl d -> (b v_chunk) nl d', 
                v_chunk=cur_view_chunk_size
                )

            decoder_input_tokens = torch.cat((cur_target_pose_tokens, cur_repeated_latent_tokens), dim=1)
            decoder_input_tokens = self.transformer_input_layernorm_decoder(decoder_input_tokens)

            transformer_output_tokens = self.pass_layers(
                self.transformer_decoder, 
                decoder_input_tokens, 
                gradient_checkpoint=False
            )

            target_image_tokens, _ = transformer_output_tokens.split(
                [n_patches, self.config.model.transformer.n_latent_vectors], dim=1
            )

            # Decode to images
            height, width = target.image_h_w
            patch_size = self.config.model.target_pose_tokenizer.patch_size
            
            video_rendering = self.image_token_decoder(target_image_tokens)
            video_rendering = rearrange(
                video_rendering, "(b v) (h w) (p1 p2 c) -> b v c (h p1) (w p2)",
                v=cur_view_chunk_size,
                h=height // patch_size, 
                w=width // patch_size, 
                p1=patch_size, 
                p2=patch_size, 
                c=3
            ).cpu()

            video_rendering_list.append(video_rendering)

        # Combine all chunks
        video_rendering = torch.cat(video_rendering_list, dim=1)
        data_batch.video_rendering = video_rendering

        return data_batch

    @torch.no_grad()
    def load_ckpt(self, load_path):
        if os.path.isdir(load_path):
            ckpt_names = [file_name for file_name in os.listdir(load_path) if file_name.endswith(".pt")]
            ckpt_names = sorted(ckpt_names, key=lambda x: x)
            ckpt_paths = [os.path.join(load_path, ckpt_name) for ckpt_name in ckpt_names]
        else:
            ckpt_paths = [load_path]
        try:
            if torch.distributed.get_rank() == 0:
                print(f"Checkpoint loaded from {ckpt_paths[-1]}")
            checkpoint = torch.load(ckpt_paths[-1], map_location="cpu", weights_only=True)
        except:
            traceback.print_exc()
            print(f"Failed to load {ckpt_paths[-1]}")
            return None
        
        self.load_state_dict(checkpoint["model"], strict=False)
        return 0


    @torch.no_grad()
    def visualize_attention_maps(self, data_batch, selected_patch_idx=None):
        """Visualize attention maps from the last encoder layer.
        
        Args:
            data_batch: Input data batch containing context images
            
        Returns:
            dict containing:
                - context_image_0: Original context image 0 with highlighted patch
                - context_image_1: Original context image 1
                - attention_map: Attention heatmap for the selected patch
        """
        input, target = self.process_data(data_batch, has_target_image=False, target_has_input=True, compute_rays=True)
        if input is None:
            return None

        # Process input images
        unposed_input_images = input.image  # [b, v_input, c, h, w]
        b, v_input, c, h, w = unposed_input_images.size()
        input_img_tokens = self.image_tokenizer(unposed_input_images)  # [b*v, n_patches, d]

        _, n_patches, d = input_img_tokens.size()  # [b*v, n_patches, d]
        input_img_tokens = input_img_tokens.reshape(b, v_input * n_patches, d)  # [b, v*n_patches, d]

        # canonical_pose_cond = self.get_posed_input(ray_o=input.ray_o, ray_d=input.ray_d) # 
        # canonical_pose_tokens = self.canonical_pose_tokenizer(canonical_pose_cond) # [b*v, n_patches, d]
        # canonical_pose_tokens = canonical_pose_tokens.reshape(b, v_input * n_patches, d) # [b, v*n_patches, d]
        identity_c2w = torch.eye(4, device=input.image.device).repeat(b, v_input, 1, 1)
        canonical_pose_cond = self.get_pose_from_c2w(identity_c2w, known_fxfycxcy=input.fxfycxcy, image_size=h)
        canonical_pose_tokens = self.canonical_pose_tokenizer(canonical_pose_cond) # [b*v, n_patches, d]
        canonical_pose_tokens = canonical_pose_tokens.reshape(b, v_input * n_patches, d) # [b, v*n_patches, d]

        # add canonical marker
        input_img_tokens[:, :n_patches] = input_img_tokens[:, :n_patches] + canonical_pose_tokens[:, :n_patches]

        # get encoder input
        latent_vector_tokens = self.n_light_field_latent.expand(b, -1, -1) # [b, n_latent_vectors, d]
        encoder_input_tokens = torch.cat((latent_vector_tokens, input_img_tokens), dim=1) # [b, n_latent_vectors + v*n_patches, d]

        # Process through all encoder layers except the last one
        for layer in self.transformer_encoder[:-1]:
            encoder_input_tokens = layer(encoder_input_tokens)

        # Get attention maps from the last layer
        last_layer = self.transformer_encoder[-1]
        encoder_input_tokens, attention_maps = last_layer.attn(
            last_layer.norm1(encoder_input_tokens),
            return_attention=True
        ) 
        # print(f"attention_maps.shape: {attention_maps.shape}") # torch.Size([4, 12, 1536, 1536]) # [b, n_heads, n_patches, n_patches]
        
        # Select a random patch from context image 0
        n_patches_per_image = n_patches

        if selected_patch_idx is None:
            random_patch_idx = self.config.model.transformer.n_latent_vectors + torch.randint(0, n_patches_per_image, (1,)).item()
        else:
            random_patch_idx = self.config.model.transformer.n_latent_vectors + selected_patch_idx

        image_patches_start = self.config.model.transformer.n_latent_vectors + n_patches_per_image
        image_patches_end = image_patches_start + n_patches_per_image*2
        
        # Get attention weights for the selected patch
        # attention_maps shape: [b, num_heads, n_total_patches, n_total_patches]
        # We want to get attention from the selected patch to all patches in image 1
        complete_attention_maps = attention_maps[:, :, self.config.model.transformer.n_latent_vectors:, self.config.model.transformer.n_latent_vectors:]
        complete_attention_maps = complete_attention_maps.mean(dim=1)
        selected_patch_attention = attention_maps[:, :, random_patch_idx, image_patches_start:image_patches_end]  # [b, num_heads, n_patches_per_image]
        
        # Average attention across heads
        attention_heatmap = selected_patch_attention.mean(dim=1)  # [b, n_patches_per_image]
        
        # Reshape attention heatmap to image size
        patch_size = 14  # as specified in the requirements
        h_patches = h // patch_size
        w_patches = w // patch_size
        attention_heatmap = attention_heatmap.reshape(b, h_patches, w_patches)
        
        # Create visualization
        context_image_0 = unposed_input_images[:, 0].permute(0, 2, 3, 1)  # [b, h, w, c]
        context_image_1 = unposed_input_images[:, 1].permute(0, 2, 3, 1)  # [b, h, w, c]
        
        # Create highlighted patch in context_image_0
        highlighted_image_0 = context_image_0.clone()
        # Calculate patch coordinates in the first image
        patch_idx_in_first_image = random_patch_idx - self.config.model.transformer.n_latent_vectors
        patch_h = (patch_idx_in_first_image // w_patches) * patch_size
        patch_w = (patch_idx_in_first_image % w_patches) * patch_size
        
        # Create a semi-transparent red overlay for the selected patch in image 0
        alpha = 0.5  # transparency factor
        for b_idx in range(b):
            # Blend the original image with red overlay using alpha blending
            patch = highlighted_image_0[b_idx, patch_h:patch_h+patch_size, patch_w:patch_w+patch_size]
            red_overlay = torch.tensor([1.0, 0.0, 0.0], device=patch.device)
            highlighted_image_0[b_idx, patch_h:patch_h+patch_size, patch_w:patch_w+patch_size] = \
                (1 - alpha) * patch + alpha * red_overlay

        # Apply attention-based color overlay to image 1
        highlighted_image_1 = context_image_1.clone()
        
        # Convert attention weights to colors using viridis colormap
        # First normalize attention weights to [0, 1]
        attention_weights = attention_heatmap.reshape(b, h_patches, w_patches)
        attention_min = attention_weights.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
        attention_max = attention_weights.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
        normalized_attention = (attention_weights - attention_min) / (attention_max - attention_min + 1e-8)
        
        # Convert to viridis colormap (approximated)
        # viridis colormap goes from dark blue to yellow
        viridis_colors = torch.tensor([
            [0.267004, 0.004874, 0.329415],  # dark blue
            [0.127568, 0.566949, 0.550556],  # cyan
            [0.369214, 0.788888, 0.382914],  # green
            [0.993248, 0.906157, 0.143936]   # yellow
        ], device=normalized_attention.device)
        
        # Interpolate colors based on attention weights
        color_indices = normalized_attention * (len(viridis_colors) - 1)
        lower_indices = torch.floor(color_indices).long()
        upper_indices = torch.ceil(color_indices).long()
        weights = color_indices - lower_indices.float()
        
        # Expand dimensions for broadcasting
        weights = weights.unsqueeze(-1)
        lower_colors = viridis_colors[lower_indices]
        upper_colors = viridis_colors[upper_indices]
        
        # Interpolate colors
        attention_colors = (1 - weights) * lower_colors + weights * upper_colors
        
        # Apply colors to patches
        alpha = 0.5  # transparency factor
        for b_idx in range(b):
            for h_idx in range(h_patches):
                for w_idx in range(w_patches):
                    patch_h = h_idx * patch_size
                    patch_w = w_idx * patch_size
                    patch = highlighted_image_1[b_idx, patch_h:patch_h+patch_size, patch_w:patch_w+patch_size]
                    color = attention_colors[b_idx, h_idx, w_idx]
                    highlighted_image_1[b_idx, patch_h:patch_h+patch_size, patch_w:patch_w+patch_size] = \
                        (1 - alpha) * patch + alpha * color
        
        return {
            'context_image_0': context_image_0,  # [b, h, w, c]
            'context_image_1': context_image_1,  # [b, h, w, c]
            'highlighted_image_0': highlighted_image_0,  # [b, h, w, c]
            'highlighted_image_1': highlighted_image_1,  # [b, h, w, c]
            'attention_map': complete_attention_maps.to(torch.float32),      # [b, h_patches, w_patches]
            'selected_patch_idx': random_patch_idx
        }
