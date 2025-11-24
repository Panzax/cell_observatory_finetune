from pathlib import Path

import pytest

import torch

from cell_observatory_finetune.models.meta_arch.swin_unetr import FinetuneSwinUNETR
from cell_observatory_finetune.data.utils import instance_map_to_boundary

CUDA_AVAILABLE = torch.cuda.is_available()


def _get_model_3d(
    B=1,
    Z=128,
    Y=128,
    X=128,
    C=2,
    patch_size=2,
    model_template="swin-unetr-small",
    device="cpu",
):
    """Helper to create a 3D SwinUNETR model (ZYXC format)."""
    model = FinetuneSwinUNETR(
        decoder_args={},
        decoder="vit",
        task="boundary_segmentation",
        output_channels=None,
        model_template=model_template,
        input_fmt="ZYXC",
        input_shape=(Z, Y, X, C),
        patch_shape=(patch_size, patch_size, patch_size),
        feature_size=24,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        spatial_dims=3,
    ).to(device)
    return model


def _get_model_4d(
    B=1,
    T=4,
    Z=128,
    Y=128,
    X=128,
    C=2,
    patch_size=2,
    model_template="swin-unetr-small",
    device="cpu",
):
    """Helper to create a 4D SwinUNETR model (TZYXC format)."""
    model = FinetuneSwinUNETR(
        decoder_args={},
        decoder="vit",
        task="boundary_segmentation",
        output_channels=None,
        model_template=model_template,
        input_fmt="TZYXC",
        input_shape=(T, Z, Y, X, C),
        patch_shape=(1, patch_size, patch_size, patch_size),
        feature_size=24,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        spatial_dims=3,
    ).to(device)
    return model


def _create_instance_map(B, *spatial_dims, device="cpu"):
    """Create a synthetic instance map for testing."""
    instance_map = torch.zeros((B, *spatial_dims), dtype=torch.long, device=device)
    # Create a few instances
    if len(spatial_dims) == 3:
        Z, Y, X = spatial_dims
        # Instance 1: small cube
        instance_map[:, Z//4:Z//4+Z//8, Y//4:Y//4+Y//8, X//4:X//4+X//8] = 1
        # Instance 2: another cube
        instance_map[:, 3*Z//4:3*Z//4+Z//8, 3*Y//4:3*Y//4+Y//8, 3*X//4:3*X//4+X//8] = 2
    elif len(spatial_dims) == 4:
        T, Z, Y, X = spatial_dims
        # Instance 1: small cube
        instance_map[:, T//2, Z//4:Z//4+Z//8, Y//4:Y//4+Y//8, X//4:X//4+X//8] = 1
        # Instance 2: another cube
        instance_map[:, T//2, 3*Z//4:3*Z//4+Z//8, 3*Y//4:3*Y//4+Y//8, 3*X//4:3*X//4+X//8] = 2
    return instance_map


# ==================== Boundary Segmentation Task Tests ====================

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for this test")
def test_boundary_segmentation_loss_computation():
    """Test that boundary_segmentation task correctly computes GeneralizedDiceLoss."""
    device = "cuda"
    torch.manual_seed(0)
    
    B, Z, Y, X, C = 2, 128, 128, 128, 2
    model = _get_model_3d(B=B, Z=Z, Y=Y, X=X, C=C, device=device)
    model.train()
    
    # Create input: [B, Z, Y, X, C]
    inputs = torch.randn(B, Z, Y, X, C, dtype=torch.float32, device=device)
    
    # Create instance map and convert to boundary mask (simulating preprocessor)
    instance_map = _create_instance_map(B, Z, Y, X, device=device)
    boundary_masks = instance_map_to_boundary(instance_map.float(), boundary_width=2)
    
    # Simulate the exact data structure from PretrainDatasourceRay + FinetunePreprocessor
    data_sample = {
        "data_tensor": inputs,
        "metainfo": {
            "targets": [boundary_masks],  # List with one element, shape [B, Z, Y, X]
            "masks": [None],
        },
    }
    
    loss_dict, predictions = model.forward(data_sample)
    
    # Verify loss is computed
    assert "step_loss" in loss_dict
    assert torch.is_tensor(loss_dict["step_loss"])
    assert loss_dict["step_loss"].ndim == 0
    assert torch.isfinite(loss_dict["step_loss"]), "Loss is NaN/Inf"
    assert loss_dict["step_loss"].item() > 0, "Loss should be positive"
    
    # Verify predictions shape
    assert predictions.shape == (B, Z, Y, X, 1)
    assert torch.isfinite(predictions).all(), "Predictions contain NaN/Inf"


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for this test")
def test_boundary_segmentation_4d_loss_computation():
    """Test boundary_segmentation with 4D data (TZYXC format)."""
    device = "cuda"
    torch.manual_seed(0)
    
    B, T, Z, Y, X, C = 2, 4, 128, 128, 128, 2
    model = _get_model_4d(B=B, T=T, Z=Z, Y=Y, X=X, C=C, device=device)
    model.train()
    
    # Create input: [B, T, Z, Y, X, C]
    inputs = torch.randn(B, T, Z, Y, X, C, dtype=torch.float32, device=device)
    
    # Create instance map and convert to boundary mask
    # For 4D data, process each timepoint separately since instance_map_to_boundary
    # only supports 2D/3D spatial inputs
    instance_map = _create_instance_map(B, T, Z, Y, X, device=device)
    # Reshape to [B*T, Z, Y, X] for processing, then reshape back
    instance_map_reshaped = instance_map.view(B * T, Z, Y, X)
    boundary_masks_reshaped = instance_map_to_boundary(instance_map_reshaped.float(), boundary_width=2)
    boundary_masks = boundary_masks_reshaped.view(B, T, Z, Y, X)
    
    # Simulate data structure from preprocessor
    data_sample = {
        "data_tensor": inputs,
        "metainfo": {
            "targets": [boundary_masks],  # [B, T, Z, Y, X]
            "masks": [None],
        },
    }
    
    loss_dict, predictions = model.forward(data_sample)
    
    # Verify loss
    assert "step_loss" in loss_dict
    assert torch.isfinite(loss_dict["step_loss"])
    assert loss_dict["step_loss"].item() > 0
    
    # Verify predictions shape: [B, T, Z, Y, X, 1]
    assert predictions.shape == (B, T, Z, Y, X, 1)
    assert torch.isfinite(predictions).all()


def test_input_format_validation():
    """Test that invalid input formats raise appropriate errors."""
    device = "cpu"
    
    # Valid formats should work
    model_zyxc = _get_model_3d(device=device)
    assert model_zyxc.input_fmt == "ZYXC"
    
    model_tzyxc = _get_model_4d(device=device)
    assert model_tzyxc.input_fmt == "TZYXC"
    
    # Invalid format should raise ValueError
    with pytest.raises(ValueError, match="Unsupported input format"):
        FinetuneSwinUNETR(
            decoder_args={},
            decoder="vit",
            task="boundary_segmentation",
            output_channels=None,
            model_template="swin-unetr-small",
            input_fmt="INVALID",
            input_shape=(128, 128, 128, 2),
            patch_shape=(2, 2, 2),
        )


def test_task_validation():
    """Test that invalid tasks raise appropriate errors."""
    device = "cpu"
    
    # Valid task should work
    model = _get_model_3d(device=device)
    assert model.task == "boundary_segmentation"
    
    # Invalid task should raise ValueError
    with pytest.raises(ValueError, match="Unknown task"):
        FinetuneSwinUNETR(
            decoder_args={},
            decoder="vit",
            task="invalid_task",
            output_channels=None,
            model_template="swin-unetr-small",
            input_fmt="ZYXC",
            input_shape=(128, 128, 128, 2),
            patch_shape=(2, 2, 2),
        )


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for this test")
def test_data_sample_structure_compatibility():
    """Test that the model correctly handles the exact data structure from PretrainDatasourceRay."""
    device = "cuda"
    torch.manual_seed(0)
    
    B, Z, Y, X, C = 2, 128, 128, 128, 2
    model = _get_model_3d(B=B, Z=Z, Y=Y, X=X, C=C, device=device)
    model.eval()
    
    # Create input matching what PretrainDatasourceRay provides
    inputs = torch.randn(B, Z, Y, X, C, dtype=torch.float32, device=device)
    
    # Create boundary masks (simulating what FinetunePreprocessor creates)
    instance_map = _create_instance_map(B, Z, Y, X, device=device)
    boundary_masks = instance_map_to_boundary(instance_map.float(), boundary_width=2)
    
    # Exact structure from PretrainDatasourceRay + FinetunePreprocessor
    # Note: targets is a list with one element, masks is also a list
    data_sample = {
        "data_tensor": inputs,
        "metainfo": {
            "targets": [boundary_masks],  # List with shape [B, Z, Y, X]
            "masks": [None],  # List (even if None)
            # Additional metadata that might be present
            "image_sizes": [(Z, Y, X)] * B,
            "orig_image_sizes": [(Z, Y, X)] * B,
        },
    }
    
    with torch.no_grad():
        loss_dict, predictions = model.forward(data_sample)
    
    # Verify the model handles this structure correctly
    assert "step_loss" in loss_dict
    assert predictions.shape == (B, Z, Y, X, 1)
    assert torch.isfinite(predictions).all()
    assert torch.isfinite(loss_dict["step_loss"])


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for this test")
def test_loss_differentiability():
    """Test that loss is differentiable for gradient computation."""
    device = "cuda"
    torch.manual_seed(0)
    
    B, Z, Y, X, C = 2, 64, 64, 64, 2
    model = _get_model_3d(B=B, Z=Z, Y=Y, X=X, C=C, device=device)
    model.train()
    
    inputs = torch.randn(B, Z, Y, X, C, dtype=torch.float32, device=device, requires_grad=True)
    instance_map = _create_instance_map(B, Z, Y, X, device=device)
    boundary_masks = instance_map_to_boundary(instance_map.float(), boundary_width=2)
    
    data_sample = {
        "data_tensor": inputs,
        "metainfo": {
            "targets": [boundary_masks],
            "masks": [None],
        },
    }
    
    loss_dict, predictions = model.forward(data_sample)
    loss = loss_dict["step_loss"]
    
    # Verify loss is differentiable
    loss.backward()
    
    # Check that gradients exist for model parameters
    has_grad = False
    for param in model.parameters():
        if param.grad is not None:
            has_grad = True
            assert torch.isfinite(param.grad).all(), "Gradients contain NaN/Inf"
            break
    
    assert has_grad, "No gradients computed"


def test_output_channels_validation_boundary_segmentation():
    """Test that output_channels validation works correctly for boundary_segmentation."""
    device = "cpu"
    
    # For boundary_segmentation, output_channels must be None (will be set to 1 automatically)
    model = FinetuneSwinUNETR(
        decoder_args={},
        decoder="vit",
        task="boundary_segmentation",
        output_channels=None,  # Must be None
        model_template="swin-unetr-small",
        input_fmt="ZYXC",
        input_shape=(128, 128, 128, 2),
        patch_shape=(2, 2, 2),
    ).to(device)
    
    assert model.output_channels == 1
    
    # If explicitly set to 1, should raise ValueError (must be None)
    with pytest.raises(ValueError, match="For semantic segmentation, output_channels must be 1 but got 1"):
        FinetuneSwinUNETR(
            decoder_args={},
            decoder="vit",
            task="boundary_segmentation",
            output_channels=1,  # Should raise error - must be None
            model_template="swin-unetr-small",
            input_fmt="ZYXC",
            input_shape=(128, 128, 128, 2),
            patch_shape=(2, 2, 2),
        )
    
    # If set to something other than 1, should raise ValueError
    with pytest.raises(ValueError, match="For semantic segmentation, output_channels must be 1 but got 2"):
        FinetuneSwinUNETR(
            decoder_args={},
            decoder="vit",
            task="boundary_segmentation",
            output_channels=2,  # Should raise error
            model_template="swin-unetr-small",
            input_fmt="ZYXC",
            input_shape=(128, 128, 128, 2),
            patch_shape=(2, 2, 2),
        )

