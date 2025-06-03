# ViT estimator

import torch
import torch.nn as nn
from .transformer import QK_Norm_TransformerBlock, init_weights
from .dinov2 import Mlp, Mlp2Layer
import torch.nn.functional as F

class ViTEstimator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.pose_enc_type = self.config.get("unposed", {}).get("pose_enc_type", "absT_quaR")
        self.pose_enc_dim = 7 if self.pose_enc_type == "absT_quaR" else 8
        

        self.transformer_estimator = nn.ModuleList([
            QK_Norm_TransformerBlock(
                config.model.transformer.d, config.model.transformer.d_head, use_qk_norm=config.model.transformer.use_qk_norm
            ) for _ in range(config.model.transformer.estimator_n_layer)
        ])

        self.absT_head = Mlp(config.model.transformer.d, config.model.transformer.d // 2, 3)
        self.quaR_head = Mlp(config.model.transformer.d, config.model.transformer.d // 2, 4)   

        if self.pose_enc_type == "absT_quaR_FoV":
            self.FoV_head = Mlp(config.model.transformer.d, config.model.transformer.d // 2, 1)


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
            
    def forward(self, target_img_tokens, repeated_latent_tokens, target_pose_enc=None, detach_estimator=True):
        # all [b*v, xx] tensors

        b_v_target, n_patches, d = target_img_tokens.size()

        estimator_input_tokens = torch.cat((target_img_tokens, repeated_latent_tokens), dim=1)

        # custom gradient checkpointing
        estimator_output_tokens = self.pass_layers(self.transformer_estimator, estimator_input_tokens, gradient_checkpoint=True, checkpoint_every=1)

        estimator_latent, _ = estimator_output_tokens.split(
            [n_patches, self.config.model.transformer.n_latent_vectors], dim=1
        )
        estimator_latent = torch.mean(estimator_latent, dim=1, keepdim=False)

        pred_absT = self.absT_head(estimator_latent)
        pred_quaR = self.quaR_head(estimator_latent)

        # # NOTE: quaterion need to be normalized! AND we use fp32 here.
        # pred_quaR = pred_quaR / torch.norm(pred_quaR, dim=-1, keepdim=True).clamp(min=1e-3)
        training_dtype = pred_quaR.dtype
        pred_quaR_fp32 = pred_quaR.to(torch.float32)
        quaR_norm = pred_quaR_fp32.norm(dim=-1, keepdim=True).clamp(min=1e-3)
        pred_quaR = (pred_quaR_fp32 / quaR_norm).to(training_dtype)

        if self.pose_enc_type == "absT_quaR_FoV":
            pred_FoV = self.FoV_head(estimator_latent)
            
        
        if self.pose_enc_type == "absT_quaR":
            pred_pose_enc = torch.cat((pred_absT, pred_quaR), dim=1)
        elif self.pose_enc_type == "absT_quaR_FoV":
            pred_pose_enc = torch.cat((pred_absT, pred_quaR, pred_FoV), dim=1)
        
        return pred_pose_enc # [b*v, 7] or [b*v, 8]
