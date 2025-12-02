"""
SwinUNETR variant with SwiGLU in MLP blocks and ReLU² in convolutional layers.

This module provides SwinUNETR_SwiGLU_ReLU2, which is equivalent to SwinUNETR
but uses:
- SwiGLU activation in MLP blocks (instead of standard MLP with GELU)
- ReLU² activation in convolutional encoder/decoder blocks (instead of ReLU)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from collections.abc import Sequence

from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.networks.layers import DropPath, trunc_normal_
from monai.utils import ensure_tuple_rep, look_up_option, optional_import

# Import necessary components from the base swin_unetr module
from .swin_unetr import SwinTransformer, MERGING_MODE
import numpy as np
from ..layers.layers import SwiGLU
from ..layers.relu2 import ReLU2

rearrange, _ = optional_import("einops", name="rearrange")


def _replace_relu_with_relu2(module):
    """
    Recursively replace all ReLU activations with ReLU² in a module.
    
    This function traverses the module tree and replaces any nn.ReLU
    instances with ReLU2.
    
    Args:
        module: The module to process
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.ReLU):
            # Replace ReLU with ReLU²
            setattr(module, name, ReLU2(inplace=child.inplace))
        else:
            # Recursively process child modules
            _replace_relu_with_relu2(child)


class SwinUNETR_SwiGLU_ReLU2(nn.Module):
    """
    Swin UNETR variant with SwiGLU in MLP blocks and ReLU² in convolutional layers.
    
    This is equivalent to SwinUNETR but uses:
    - SwiGLU activation in transformer MLP blocks
    - ReLU² activation in convolutional encoder/decoder blocks
    
    Based on: "Hatamizadeh et al.,
    Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images
    <https://arxiv.org/abs/2201.01266>"
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int = 2,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        window_size: Sequence[int] | int = 7,
        qkv_bias: bool = True,
        mlp_ratio: float = 4.0,
        feature_size: int = 24,
        norm_name: tuple | str = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        norm_layer: type[nn.LayerNorm] = nn.LayerNorm,
        patch_norm: bool = False,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        downsample: str | nn.Module = "merging",
        use_v2: bool = False,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            patch_size: size of the patch token.
            feature_size: dimension of network feature size.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            window_size: local window size.
            qkv_bias: add a learnable bias to query, key, value.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            norm_name: feature normalization type and arguments.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            dropout_path_rate: drop path rate.
            normalize: normalize output intermediate features in each stage.
            norm_layer: normalization layer.
            patch_norm: whether to apply normalization to the patch embedding. Default is False.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims: number of spatial dims.
            downsample: module used for downsampling.
            use_v2: using swinunetr_v2, which adds a residual convolution block at the beginning of each swin stage.
        """
        super().__init__()

        if spatial_dims not in (2, 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        self.patch_size = patch_size

        patch_sizes = ensure_tuple_rep(self.patch_size, spatial_dims)
        window_size = ensure_tuple_rep(window_size, spatial_dims)

        if not (0 <= drop_rate <= 1):
            raise ValueError("dropout rate should be between 0 and 1.")

        if not (0 <= attn_drop_rate <= 1):
            raise ValueError("attention dropout rate should be between 0 and 1.")

        if not (0 <= dropout_path_rate <= 1):
            raise ValueError("drop path rate should be between 0 and 1.")

        if feature_size % 12 != 0:
            raise ValueError("feature_size should be divisible by 12.")

        self.normalize = normalize

        # Create SwinTransformer with SwiGLU in MLP blocks
        self.swinViT = SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_sizes,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            mlp_type="SwiGLU",  # Use SwiGLU instead of standard MLP
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=norm_layer,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            downsample=look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample,
            use_v2=use_v2,
        )

        # Create encoder blocks (will be patched to use ReLU²)
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder10 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=16 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        # Create decoder blocks (will be patched to use ReLU²)
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)

        # Replace ReLU with ReLU² in all encoder and decoder blocks
        for encoder in [self.encoder1, self.encoder2, self.encoder3, self.encoder4, self.encoder10]:
            _replace_relu_with_relu2(encoder)
        
        for decoder in [self.decoder5, self.decoder4, self.decoder3, self.decoder2, self.decoder1]:
            _replace_relu_with_relu2(decoder)
        
        # Also replace in v2 residual blocks if used
        if use_v2:
            for layer_list in [self.swinViT.layers1c, self.swinViT.layers2c, 
                              self.swinViT.layers3c, self.swinViT.layers4c]:
                for block in layer_list:
                    _replace_relu_with_relu2(block)

    @torch.jit.unused
    def _check_input_size(self, spatial_shape):
        img_size = np.array(spatial_shape)
        remainder = (img_size % np.power(self.patch_size, 5)) > 0
        if remainder.any():
            wrong_dims = (np.where(remainder)[0] + 2).tolist()
            raise ValueError(
                f"spatial dimensions {wrong_dims} of input image (spatial shape: {spatial_shape})"
                f" must be divisible by {self.patch_size}**5."
            )

    def forward(self, x_in):
        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            self._check_input_size(x_in.shape[2:])
        hidden_states_out = self.swinViT(x_in, self.normalize)
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])
        dec3 = self.decoder5(dec4, hidden_states_out[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)
        logits = self.out(out)
        return logits

