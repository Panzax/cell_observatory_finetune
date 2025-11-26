"""
Adapted from:
https://github.com/IDEA-Research/MaskDINO/maskdino/utils/utils.py#L36
https://github.com/facebookresearch/detectron2/detectron2/layers/wrappers.py#L103
"""


import torch
from torch import nn
import torch.nn.functional as F

from cell_observatory_platform.models.patch_embeddings import calc_num_patches


class LayerNorm3D(nn.Module):
    def __init__(self, normalized_shape, norm_layer=nn.LayerNorm):
        super().__init__()
        self.ln = norm_layer(normalized_shape) if norm_layer is not None else nn.Identity()

    def forward(self, x):
        """
        x: N C D H W
        """
        x = x.permute(0, 2, 3, 4, 1)
        x = self.ln(x)
        x = x.permute(0, 4, 1, 2, 3)
        return x


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class SwiGLU(nn.Module):
    # Adapted from: https://github.com/FadiZidiDz/LoLA-SpecViT-Model/blob/main/improved_GCPE.py
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(input_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # SwiGLU is
        # Linear3(Swish(Linear1(x)) * Linear2(X))
        # Where Swish(x) = x * sigmoid(x)
        return self.fc3(F.silu(self.fc1(x)) * self.fc2(x))

def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)


class Conv3d(torch.nn.Conv3d):
    def __init__(self, *args, **kwargs):
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x: torch.Tensor):
        x = F.conv3d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


def compute_num_pixels_per_patch(channels, temporal_patch_size, axial_patch_size, lateral_patch_size, input_fmt):
    pixels_per_patch = channels
    pixels_per_patch *= temporal_patch_size if temporal_patch_size is not None else 1
    pixels_per_patch *= axial_patch_size if axial_patch_size is not None else 1
    pixels_per_patch *= lateral_patch_size ** 2 if input_fmt is not "XC" else lateral_patch_size
    return pixels_per_patch


def patchify(inputs, 
             input_fmt, 
             temporal_patch_size, 
             axial_patch_size, 
             lateral_patch_size,
             channels,
             reshape=True
):

    num_patches, token_shape = calc_num_patches(
        input_fmt=input_fmt,
        input_shape=inputs.shape[1:],
        lateral_patch_size=lateral_patch_size,
        axial_patch_size=axial_patch_size,
        temporal_patch_size=temporal_patch_size,
    )
    pixels_per_patch = compute_num_pixels_per_patch(channels, 
                                                    temporal_patch_size, 
                                                    axial_patch_size, 
                                                    lateral_patch_size, 
                                                    input_fmt)
    
    b = inputs.shape[0]
    t, z, y, x, c = token_shape

    if input_fmt == "TZYXC":
        if reshape:
            patches = inputs.reshape(shape=(
                b,
                t, temporal_patch_size,
                z, axial_patch_size,
                y, lateral_patch_size,
                x, lateral_patch_size,
                channels,
            ))
            patches = torch.einsum("btizjykxvc->btzyxijkvc", patches)
        else:
            patches = inputs.unfold(1, temporal_patch_size, temporal_patch_size) \
                .unfold(2, axial_patch_size, axial_patch_size) \
                .unfold(3, lateral_patch_size, lateral_patch_size) \
                .unfold(4, lateral_patch_size, lateral_patch_size) \

    elif input_fmt == "ZYXC":
        if reshape:
            patches = inputs.reshape(shape=(
                b,
                z, axial_patch_size,
                y, lateral_patch_size,
                x, lateral_patch_size,
                channels,
            ))
            patches = torch.einsum("bzjykxvc->bzyxjkvc", patches)
        else:
            patches = inputs.unfold(1, axial_patch_size, axial_patch_size) \
                .unfold(2, lateral_patch_size, lateral_patch_size) \
                .unfold(3, lateral_patch_size, lateral_patch_size)

    elif input_fmt == "TYXC":
        if reshape:
            patches = inputs.reshape(shape=(
                b,
                t, temporal_patch_size,
                y, lateral_patch_size,
                x, lateral_patch_size,
                channels,
            ))
            patches = torch.einsum("btiykxvc->btyxikvc", patches)
        else:
            patches = inputs.unfold(1, temporal_patch_size, temporal_patch_size) \
                .unfold(2, lateral_patch_size, lateral_patch_size) \
                .unfold(3, lateral_patch_size, lateral_patch_size)

    elif input_fmt == "YXC":
        if reshape:
            patches = inputs.reshape(shape=(
                b,
                y, lateral_patch_size,
                x, lateral_patch_size,
                channels,
            ))
            patches = torch.einsum("bykxvc->byxkvc", patches)
        else:
            patches = inputs.unfold(1, lateral_patch_size, lateral_patch_size) \
                .unfold(2, lateral_patch_size, lateral_patch_size)

    elif input_fmt == "XC":
        if reshape:
            patches = inputs.reshape(shape=(
                b,
                x, lateral_patch_size,
                channels,
            ))
        else:
            patches = inputs.unfold(1, lateral_patch_size, lateral_patch_size)
    else:
        raise NotImplementedError

    # NOTE: if tensor is already in the specified memory format, 
    #       contiguous returns the tensor
    patches = patches.contiguous().view(b, num_patches, pixels_per_patch)
    return patches


class ReLUSquared(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.relu(x) ** 2


class MLPReLUSquaredBlock(nn.Module):
    """
    A multi-layer perceptron block with ReLU^2 activation, matching MONAI's MLPBlock structure.
    
    This class is designed to be a drop-in replacement for MONAI's MLPBlock when using
    ReLUSquared activation, maintaining the same interface and dropout behavior.
    
    Based on MONAI's MLPBlock structure:
    - linear1 -> activation -> drop1 -> linear2 -> drop2
    - For "swin" mode, drop1 and drop2 share the same instance
    """
    
    def __init__(
        self, 
        hidden_size: int, 
        mlp_dim: int, 
        dropout_rate: float = 0.0, 
        dropout_mode: str = "swin"
    ) -> None:
        """
        Args:
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer. If 0, `hidden_size` will be used.
            dropout_rate: fraction of the input units to drop.
            dropout_mode: dropout mode, can be "vit", "swin", or "vista3d".
                "vit" mode uses two dropout instances.
                "swin" corresponds to one shared instance (default).
                "vista3d" mode does not use dropout.
        """
        super().__init__()
        
        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")
        
        mlp_dim = mlp_dim or hidden_size
        
        self.linear1 = nn.Linear(hidden_size, mlp_dim)
        self.linear2 = nn.Linear(mlp_dim, hidden_size)
        self.fn = ReLUSquared()
        
        # Set up dropout based on mode (matching MONAI's MLPBlock)
        if dropout_mode == "vit":
            self.drop1 = nn.Dropout(dropout_rate)
            self.drop2 = nn.Dropout(dropout_rate)
        elif dropout_mode == "swin":
            self.drop1 = nn.Dropout(dropout_rate) if dropout_rate > 0.0 else nn.Identity()
            self.drop2 = self.drop1  # Same instance for "swin" mode
        elif dropout_mode == "vista3d":
            self.drop1 = nn.Identity()
            self.drop2 = nn.Identity()
        else:
            raise ValueError(f"dropout_mode should be one of {{'vit', 'swin', 'vista3d'}}, got {dropout_mode}")
    
    def forward(self, x):
        """
        Forward pass matching MONAI's MLPBlock structure:
        x -> linear1 -> activation -> drop1 -> linear2 -> drop2
        """
        x = self.linear1(x)
        x = self.fn(x)
        x = self.drop1(x)
        x = self.linear2(x)
        x = self.drop2(x)
        return x
