# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

################################################################################
# This model was imported directly from the MONAI library at this
#
# Source:
# https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/unetr.py
#
# It was adapted with a wrapper to allow us to import and use it in this repository.
################################################################################

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Optional

import torch
import torch.nn as nn
from cell_observatory_finetune.models.layers.utils import pack_time, unpack_time
from cell_observatory_finetune.training.losses import get_loss_fn
from cell_observatory_platform.models.patch_embeddings import calc_num_patches
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import (
    UnetrBasicBlock,
    UnetrPrUpBlock,
    UnetrUpBlock,
)
from monai.networks.nets.vit import ViT
from monai.utils import ensure_tuple_rep


class UNETR(nn.Module):
    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Sequence[int] | int,
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        proj_type: str = "conv",
        norm_name: tuple | str = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        qkv_bias: bool = False,
        save_attn: bool = False,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size. Defaults to 16.
            hidden_size: dimension of hidden layer. Defaults to 768.
            mlp_dim: dimension of feedforward layer. Defaults to 3072.
            num_heads: number of attention heads. Defaults to 12.
            proj_type: patch embedding layer type. Defaults to "conv".
            norm_name: feature normalization type and arguments. Defaults to "instance".
            conv_block: if convolutional block is used. Defaults to True.
            res_block: if residual block is used. Defaults to True.
            dropout_rate: fraction of the input units to drop. Defaults to 0.0.
            spatial_dims: number of spatial dims. Defaults to 3.
            qkv_bias: apply the bias term for the qkv linear layer in self attention block. Defaults to False.
            save_attn: to make accessible the attention in self attention block. Defaults to False.

        Examples::

            # for single channel input 4-channel output with image size of (96,96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=(96,96,96), feature_size=32, norm_name='batch')

             # for single channel input 4-channel output with image size of (96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=96, feature_size=32, norm_name='batch', spatial_dims=2)

            # for 4-channel input 3-channel output with image size of (128,128,128), conv position embedding and instance norm
            >>> net = UNETR(in_channels=4, out_channels=3, img_size=(128,128,128), proj_type='conv', norm_name='instance')

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.num_layers = 12
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.patch_size = ensure_tuple_rep(16, spatial_dims)
        self.feat_size = tuple(
            img_d // p_d for img_d, p_d in zip(img_size, self.patch_size)
        )
        self.hidden_size = hidden_size
        self.classification = False
        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            proj_type=proj_type,
            classification=self.classification,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
            qkv_bias=qkv_bias,
            save_attn=save_attn,
        )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=out_channels,
        )
        self.proj_axes = (0, spatial_dims + 1) + tuple(
            d + 1 for d in range(spatial_dims)
        )
        self.proj_view_shape = list(self.feat_size) + [self.hidden_size]

    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x

    def forward(self, x_in):
        x, hidden_states_out = self.vit(x_in)
        enc1 = self.encoder1(x_in)
        x2 = hidden_states_out[3]
        enc2 = self.encoder2(self.proj_feat(x2))
        x3 = hidden_states_out[6]
        enc3 = self.encoder3(self.proj_feat(x3))
        x4 = hidden_states_out[9]
        enc4 = self.encoder4(self.proj_feat(x4))
        dec4 = self.proj_feat(x)
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)
        return self.out(out)


##############################################################
# Cell Observatory Fine-tuning Framework Integration
# This wrapper adapts MONAI's UNETR (designed for 3D medical imaging)
# to work with Cell Observatory's microscopy data pipeline.
#
# KEY TRANSLATION JOBS:
# 1. Data Format: Framework uses TZYXC (channel-last) -> Model needs BCZYX (channel-first)
# 2. Time Dimension: Framework has data [B,T,Z,Y,X,C] -> Model expects [B,C,Z,Y,X]
#                    Solution: Merge batch+time into one dimension [B*T,C,Z,Y,X]
# 3. Input Structure: Framework passes data_sample dict -> Model expects raw tensor
# 4. Output: Model returns predictions ->  Framework expects (loss_dict, predictions)
##############################################################


# Model size configurations
# Select via model_template parameter in config YAML
CONFIGS = {
    "unetr-tiny": {
        "feature_size": 12,
        "hidden_size": 384,
        "mlp_dim": 1536,
        "num_heads": 6,
    },
    "unetr-small": {
        "feature_size": 14,
        "hidden_size": 512,
        "mlp_dim": 2048,
        "num_heads": 8,
    },
    "unetr-base": {
        "feature_size": 16,
        "hidden_size": 768,
        "mlp_dim": 3072,
        "num_heads": 12,
    },
    "unetr-large": {
        "feature_size": 32,
        "hidden_size": 1024,
        "mlp_dim": 4096,
        "num_heads": 16,
    },
}


class FinetuneUNETR(nn.Module):
    """
    Wrapper for UNETR model to integrate with the Cell Observatory Fine-Tuning repository.

    This class handles:
    - Dimension translation between repository format (TZYXC/ZYXC) and UNETR format (BCDHW/BCHW)
    - Task-specific preprocessing and loss computation
    - Integration with repository's data pipeline

    Note: decoder and decoder_args parameters are kept for API compatibility with other
    models (MAE/JEPA) but are unused since UNETR has an integrated decoder architecture.
    """

    def __init__(
        self,
        decoder_args: dict,
        decoder: Literal["vit", "linear", "dense_predictor"],
        task: Literal[
            "channel_split", "upsample_space", "upsample_spacetime", "upsample_time"
        ],
        output_channels: Optional[int],
        model_template: Literal[
            "unetr",  # custom: use feature_size, hidden_size, mlp_dim, num_heads to config model
            "unetr-tiny",
            "unetr-small",
            "unetr-base",
            "unetr-large",
        ] = "unetr",
        input_fmt: str = "ZYXC",
        input_shape: tuple = (128, 128, 128, 2),
        patch_shape: tuple = (4, 16, 16, 16),
        in_channels: int = 1,
        out_channels: int = 2,
        img_size: Sequence[int] | int = (128, 128, 128),
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        proj_type: str = "conv",
        norm_name: str = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        qkv_bias: bool = False,
        save_attn: bool = False,
        loss_fn: str = "l2_masked",
    ):
        """
        Args:
            decoder_args: Dictionary of decoder arguments (kept for framework compatibility)
            decoder: Decoder type (kept for framework compatibility, UNETR has built-in decoder)
            task: Fine-tuning task type
            output_channels: Number of output channels for the task
            model_template: Pre-configured model size
            input_fmt: Input format string (e.g., 'ZYXC', 'TZYXC')
            input_shape: Shape of input data
            patch_shape: Shape of patches for patch embedding
            in_channels: Number of input channels for UNETR
            out_channels: Number of output channels for UNETR
            img_size: Spatial dimensions of input image for UNETR
            feature_size: Feature size for UNETR
            hidden_size: Hidden dimension size for transformer
            mlp_dim: MLP dimension in transformer
            num_heads: Number of attention heads
            proj_type: Projection type for patch embedding
            norm_name: Normalization layer type
            conv_block: Whether to use convolutional blocks
            res_block: Whether to use residual blocks
            dropout_rate: Dropout rate
            spatial_dims: Number of spatial dimensions (2 or 3)
            qkv_bias: Whether to use bias in QKV projection
            save_attn: Whether to save attention weights
            loss_fn: Loss function name
        """
        super().__init__()

        # Apply model template config if specified
        if model_template in CONFIGS.keys():
            config = CONFIGS[model_template]
            self.feature_size = config["feature_size"]
            self.hidden_size = config["hidden_size"]
            self.mlp_dim = config["mlp_dim"]
            self.num_heads = config["num_heads"]
        else:
            self.feature_size = feature_size
            self.hidden_size = hidden_size
            self.mlp_dim = mlp_dim
            self.num_heads = num_heads

        # ========== Data Format Configuration ==========
        self.task = task
        self.input_fmt = input_fmt
        self.input_shape = input_shape
        self.patch_shape = patch_shape
        self.output_channels = output_channels
        self.spatial_dims = spatial_dims

        # Parse input format to extract dimensions
        axis_to_value = dict(zip(input_fmt, input_shape))
        self.in_chans = axis_to_value.get("C", 1)
        self.num_frames = axis_to_value.get("T", None)

        # Initialize MONAI UNETR
        self.unetr = UNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            feature_size=self.feature_size,
            hidden_size=self.hidden_size,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            proj_type=proj_type,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
            qkv_bias=qkv_bias,
            save_attn=save_attn,
        )

        # Initialize loss function
        self.loss_fn = get_loss_fn(loss_fn)

    def _convert_tensor_format(self, x: torch.Tensor):
        """
        Convert tensor from framework format to model format using existing utilities.

        The model expects channels in position 1, but framework has them last.
        For 4D data, we also merge batch and time into a single dimension since
        UNETR doesn't know about time - it treats each timepoint as a separate sample.

        Conversions:
        - 3D: [B, Z, Y, X, C] -> [B, C, Z, Y, X]
        - 4D: [B, T, Z, Y, X, C] -> [B*T, C, Z, Y, X]

        Args:
            x: Input tensor in framework format

        Returns:
            tuple: (converted_tensor, batch_size, time_steps)
                - converted_tensor: Model-ready tensor with channels in position 1
                - batch_size: Original batch size (needed to split B*T later)
                - time_steps: Original time steps, or None for 3D data

        Example:
            Input:  [2, 16, 128, 128, 128, 2] (2 samples, 16 timepoints, 2 channels)
            Output: [32, 2, 128, 128, 128]    (32 "samples", 2 channels)
                    B=2, T=16 (stored for later reconstruction)
        """
        B = x.shape[0]

        if "T" in self.input_fmt:
            # 4D data with time: [B, T, Z, Y, X, C] -> [B*T, C, Z, Y, X]
            # Use existing pack_time utility
            x, B, T = pack_time(x, input_format=self.input_fmt, output_format="TCZYX")
            return x, B, T
        else:
            # 3D data without time: [B, Z, Y, X, C] -> [B, C, Z, Y, X]
            if self.input_fmt == "ZYXC":
                # Move channel from last to second position
                x = x.permute(0, 4, 1, 2, 3).contiguous()
            elif self.input_fmt == "YXC":
                # 2D case: [B, Y, X, C] -> [B, C, Y, X]
                x = x.permute(0, 3, 1, 2).contiguous()
            else:
                raise ValueError(
                    f"Unsupported input format for 3D data: {self.input_fmt}"
                )
            return x, B, None

    def _convert_tensor_back(self, x: torch.Tensor, B: int, T: int):
        """
        Convert tensor from model format back to framework format using existing utilities.

        Handles both 3D and 4D data:
        - 3D: [B, C, Z, Y, X] -> [B, Z, Y, X, C]
        - 4D: [B*T, C, Z, Y, X] -> [B, T, Z, Y, X, C]

        Args:
            x: Tensor in model format
            B: Original batch size
            T: Original time steps (None if no time dimension)

        Returns:
            Tensor in framework format (ready for loss computation)

        Example:
            Input:  [32, 4, 128, 128, 128]    (32 "samples", 4 output channels)
                    B=2, T=16
            Output: [2, 16, 128, 128, 128, 4] (2 samples, 16 timepoints, 4 channels)
        """
        if T is not None:
            # 4D data: [B*T, C, Z, Y, X] -> [B, T, Z, Y, X, C]
            # Use existing unpack_time utility
            x = unpack_time(x, B, T, input_format="TCZYX", output_format=self.input_fmt)
            return x
        else:
            # 3D data: [B, C, Z, Y, X] -> [B, Z, Y, X, C]
            if self.input_fmt == "ZYXC":
                # Move channel from second to last position
                x = x.permute(0, 2, 3, 4, 1).contiguous()
            elif self.input_fmt == "YXC":
                # 2D case: [B, C, Y, X] -> [B, Y, X, C]
                x = x.permute(0, 2, 3, 1).contiguous()
            else:
                raise ValueError(
                    f"Unsupported input format for 3D data: {self.input_fmt}"
                )
            return x

    @torch.jit.ignore
    def get_num_patches(self):
        """Calculate the number of patches for loss computation."""
        num_patches, _ = calc_num_patches(
            input_fmt=self.input_fmt,
            input_shape=self.input_shape,
            patch_shape=(16, 16, 16) if self.spatial_dims == 3 else (16, 16),
        )
        return num_patches

    def forward(self, data_sample: dict):
        """
        Forward pass for training with loss computation.

        Flow:
        1. Extract data from framework dict
        2. Convert TZYXC -> BCZYX (and merge B*T for 4D)
        3. Run through UNETR
        4. Convert BCZYX -> TZYXC (and split B*T for 4D)
        5. Compute loss in framework format
        6. Return (loss_dict, predictions) as framework expects

        Args:
            data_sample: Framework data dict with keys:
                - 'data_tensor': Input in TZYXC or ZYXC format
                - 'metainfo': Dict with 'targets' and 'masks'

        Returns:
            tuple: (loss_dict, predictions)
                - loss_dict: {'step_loss': scalar_tensor}
                - predictions: Tensor in framework format (TZYXC/ZYXC)
        """
        inputs = data_sample["data_tensor"]
        meta = data_sample["metainfo"]
        targets = meta.get("targets", [None])[0]

        # Handle channel split task: average channels for input
        if self.task == "channel_split":
            # Find channel dimension (accounting for batch)
            channel_dim = None
            for i, ax in enumerate(self.input_fmt):
                if ax == "C":
                    channel_dim = i + 1  # +1 for batch dimension
                    break
            if channel_dim is None:
                raise ValueError("Channel axis 'C' not present in input_format")
            # Average across channels for input
            model_input = inputs.mean(dim=channel_dim, keepdim=True)
        else:
            model_input = inputs

        # Convert input tensor format and track original dimensions
        inputs_converted, B, T = self._convert_tensor_format(model_input)

        # Forward through UNETR
        predictions = self.unetr(inputs_converted)

        # Convert predictions back to framework format
        predictions = self._convert_tensor_back(predictions, B, T)

        # Compute task-specific loss
        if self.task == "channel_split":
            loss = self.loss_fn(
                predictions, targets, num_patches=self.get_num_patches()
            )
        elif self.task == "upsample_space":
            loss = self.loss_fn(
                predictions, targets, num_patches=self.get_num_patches()
            )
        elif self.task == "upsample_time":
            # For time upsampling, only supervise masked timepoints
            target_masks = meta.get("target_masks", [None])[0]
            if target_masks is not None:
                loss = self.loss_fn(
                    predictions, targets, num_patches=target_masks.sum()
                )
            else:
                loss = self.loss_fn(
                    predictions, targets, num_patches=self.get_num_patches()
                )
        elif self.task == "upsample_spacetime":
            # For spacetime upsampling
            loss = self.loss_fn(
                predictions, targets, num_patches=self.get_num_patches()
            )
        else:
            raise ValueError(f"Unknown task: {self.task}")

        loss_dict = {"step_loss": loss}
        return loss_dict, predictions

    def predict(self, data_sample: dict):
        """
        Inference-only forward pass (no loss computation).

        Same conversion process as forward(), but skips loss calculation.
        Used during evaluation or when generating predictions.

        Args:
            data_sample: Dictionary with 'data_tensor' key

        Returns:
            Predictions tensor in framework format (TZYXC/ZYXC)
        """
        inputs = data_sample["data_tensor"]

        # Handle channel split task
        if self.task == "channel_split":
            # Find channel dimension (accounting for batch)
            channel_dim = None
            for i, ax in enumerate(self.input_fmt):
                if ax == "C":
                    channel_dim = i + 1  # +1 for batch dimension
                    break
            if channel_dim is None:
                raise ValueError("Channel axis 'C' not present in input_format")
            model_input = inputs.mean(dim=channel_dim, keepdim=True)
        else:
            model_input = inputs

        # Convert: TZYXC -> BCZYX (merge B*T if needed)
        inputs_converted, B, T = self._convert_tensor_format(model_input)

        # Model forward
        predictions = self.unetr(inputs_converted)

        # Convert: BCZYX -> TZYXC (split B*T if needed)
        predictions = self._convert_tensor_back(predictions, B, T)

        return predictions
