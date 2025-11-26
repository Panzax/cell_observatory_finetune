from pathlib import Path

import pytest

import torch

from cell_observatory_finetune.models.meta_arch.swin_unetr import FinetuneSwinUNETR

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
        decoder_args={},  # Unused but required for API compatibility
        decoder="vit",  # Unused but required for API compatibility
        task="boundary_segmentation",
        output_channels=None,  # Will be set to 1 for boundary_segmentation
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
        decoder_args={},  # Unused but required for API compatibility
        decoder="vit",  # Unused but required for API compatibility
        task="boundary_segmentation",
        output_channels=None,  # Will be set to 1 for boundary_segmentation
        model_template=model_template,
        input_fmt="TZYXC",
        input_shape=(T, Z, Y, X, C),
        patch_shape=(1, patch_size, patch_size, patch_size),  # T_patch=1, spatial patches
        feature_size=24,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        spatial_dims=3,
    ).to(device)
    return model


# ==================== 3D Data Format Tests (ZYXC) ====================

@pytest.mark.parametrize("B,Z,Y,X,C", [
    (1, 128, 128, 128, 2),
    (2, 128, 128, 128, 2),
    (1, 64, 64, 64, 2),
])
def test_swin_unetr_3d_forward_shapes(B, Z, Y, X, C):
    """Test forward pass with 3D data (ZYXC format) - verify output shapes."""
    device = "cuda" if CUDA_AVAILABLE else "cpu"
    torch.manual_seed(0)
    
    model = _get_model_3d(B=B, Z=Z, Y=Y, X=X, C=C, device=device)
    model.eval()
    
    # Input: [B, Z, Y, X, C]
    inputs = torch.randn(B, Z, Y, X, C, dtype=torch.float32, device=device)
    
    # Target: [B, Z, Y, X] (no channel dimension for binary segmentation)
    targets = torch.randint(0, 2, (B, Z, Y, X), dtype=torch.float32, device=device)
    
    data_sample = {
        "data_tensor": inputs,
        "metainfo": {
            "targets": [targets],
            "masks": [None],
        },
    }
    
    with torch.no_grad():
        loss_dict, predictions = model.forward(data_sample)
    
    # Verify loss
    assert "step_loss" in loss_dict
    assert torch.is_tensor(loss_dict["step_loss"])
    assert loss_dict["step_loss"].ndim == 0
    assert torch.isfinite(loss_dict["step_loss"]), "Loss is NaN/Inf"
    
    # Verify predictions shape: [B, Z, Y, X, 1] (output_channels=1 for boundary_segmentation)
    assert predictions.shape == (B, Z, Y, X, 1)
    assert torch.isfinite(predictions).all(), "Predictions contain NaN/Inf"


@pytest.mark.parametrize("B,Z,Y,X,C", [
    (1, 128, 128, 128, 2),
    (2, 128, 128, 128, 2),
])
def test_swin_unetr_3d_predict_shapes(B, Z, Y, X, C):
    """Test predict method with 3D data (ZYXC format) - inference only."""
    device = "cuda" if CUDA_AVAILABLE else "cpu"
    torch.manual_seed(0)
    
    model = _get_model_3d(B=B, Z=Z, Y=Y, X=X, C=C, device=device)
    model.eval()
    
    # Input: [B, Z, Y, X, C]
    inputs = torch.randn(B, Z, Y, X, C, dtype=torch.float32, device=device)
    
    data_sample = {
        "data_tensor": inputs,
    }
    
    with torch.no_grad():
        predictions = model.predict(data_sample)
    
    # Verify predictions shape: [B, Z, Y, X, 1]
    assert predictions.shape == (B, Z, Y, X, 1)
    assert torch.isfinite(predictions).all(), "Predictions contain NaN/Inf"


# ==================== 4D Data Format Tests (TZYXC) ====================

@pytest.mark.parametrize("B,T,Z,Y,X,C", [
    (1, 4, 128, 128, 128, 2),
    (2, 4, 128, 128, 128, 2),
    (1, 8, 128, 128, 128, 2),
])
def test_swin_unetr_4d_forward_shapes(B, T, Z, Y, X, C):
    """Test forward pass with 4D data (TZYXC format) - verify output shapes."""
    device = "cuda" if CUDA_AVAILABLE else "cpu"
    torch.manual_seed(0)
    
    model = _get_model_4d(B=B, T=T, Z=Z, Y=Y, X=X, C=C, device=device)
    model.eval()
    
    # Input: [B, T, Z, Y, X, C]
    inputs = torch.randn(B, T, Z, Y, X, C, dtype=torch.float32, device=device)
    
    # Target: [B, T, Z, Y, X] (no channel dimension for binary segmentation)
    targets = torch.randint(0, 2, (B, T, Z, Y, X), dtype=torch.float32, device=device)
    
    data_sample = {
        "data_tensor": inputs,
        "metainfo": {
            "targets": [targets],
            "masks": [None],
        },
    }
    
    with torch.no_grad():
        loss_dict, predictions = model.forward(data_sample)
    
    # Verify loss
    assert "step_loss" in loss_dict
    assert torch.is_tensor(loss_dict["step_loss"])
    assert loss_dict["step_loss"].ndim == 0
    assert torch.isfinite(loss_dict["step_loss"]), "Loss is NaN/Inf"
    
    # Verify predictions shape: [B, T, Z, Y, X, 1]
    assert predictions.shape == (B, T, Z, Y, X, 1)
    assert torch.isfinite(predictions).all(), "Predictions contain NaN/Inf"


@pytest.mark.parametrize("B,T,Z,Y,X,C", [
    (1, 4, 128, 128, 128, 2),
    (2, 4, 128, 128, 128, 2),
])
def test_swin_unetr_4d_predict_shapes(B, T, Z, Y, X, C):
    """Test predict method with 4D data (TZYXC format) - inference only."""
    device = "cuda" if CUDA_AVAILABLE else "cpu"
    torch.manual_seed(0)
    
    model = _get_model_4d(B=B, T=T, Z=Z, Y=Y, X=X, C=C, device=device)
    model.eval()
    
    # Input: [B, T, Z, Y, X, C]
    inputs = torch.randn(B, T, Z, Y, X, C, dtype=torch.float32, device=device)
    
    data_sample = {
        "data_tensor": inputs,
    }
    
    with torch.no_grad():
        predictions = model.predict(data_sample)
    
    # Verify predictions shape: [B, T, Z, Y, X, 1]
    assert predictions.shape == (B, T, Z, Y, X, 1)
    assert torch.isfinite(predictions).all(), "Predictions contain NaN/Inf"


# ==================== Data Flow Tests ====================

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for this test")
def test_swin_unetr_3d_data_flow():
    """Test complete data flow for 3D data: format conversion -> model -> format conversion."""
    device = "cuda"
    torch.manual_seed(0)
    
    B, Z, Y, X, C = 2, 128, 128, 128, 2
    model = _get_model_3d(B=B, Z=Z, Y=Y, X=X, C=C, device=device)
    model.eval()
    
    # Create input in framework format: [B, Z, Y, X, C]
    inputs = torch.randn(B, Z, Y, X, C, dtype=torch.float32, device=device)
    targets = torch.randint(0, 2, (B, Z, Y, X), dtype=torch.float32, device=device)
    
    data_sample = {
        "data_tensor": inputs,
        "metainfo": {
            "targets": [targets],
            "masks": [None],
        },
    }
    
    # Test internal conversion methods
    inputs_converted, B_orig, T_orig = model._convert_tensor_format(inputs)
    
    # Verify conversion: [B, Z, Y, X, C] -> [B, C, Z, Y, X]
    assert inputs_converted.shape == (B, C, Z, Y, X)
    assert B_orig == B
    assert T_orig is None
    
    # Test reverse conversion
    predictions_model_format = torch.randn(B, 1, Z, Y, X, dtype=torch.float32, device=device)
    predictions_framework_format = model._convert_tensor_back(predictions_model_format, B_orig, T_orig)
    
    # Verify reverse conversion: [B, C, Z, Y, X] -> [B, Z, Y, X, C]
    assert predictions_framework_format.shape == (B, Z, Y, X, 1)
    
    # Test full forward pass
    with torch.no_grad():
        loss_dict, predictions = model.forward(data_sample)
    
    assert predictions.shape == (B, Z, Y, X, 1)
    assert torch.isfinite(predictions).all()
    assert torch.isfinite(loss_dict["step_loss"])


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for this test")
def test_swin_unetr_4d_data_flow():
    """Test complete data flow for 4D data: format conversion -> model -> format conversion."""
    device = "cuda"
    torch.manual_seed(0)
    
    B, T, Z, Y, X, C = 2, 4, 128, 128, 128, 2
    model = _get_model_4d(B=B, T=T, Z=Z, Y=Y, X=X, C=C, device=device)
    model.eval()
    
    # Create input in framework format: [B, T, Z, Y, X, C]
    inputs = torch.randn(B, T, Z, Y, X, C, dtype=torch.float32, device=device)
    targets = torch.randint(0, 2, (B, T, Z, Y, X), dtype=torch.float32, device=device)
    
    data_sample = {
        "data_tensor": inputs,
        "metainfo": {
            "targets": [targets],
            "masks": [None],
        },
    }
    
    # Test internal conversion methods
    inputs_converted, B_orig, T_orig = model._convert_tensor_format(inputs)
    
    # Verify conversion: [B, T, Z, Y, X, C] -> [B*T, C, Z, Y, X]
    assert inputs_converted.shape == (B * T, C, Z, Y, X)
    assert B_orig == B
    assert T_orig == T
    
    # Test reverse conversion
    predictions_model_format = torch.randn(B * T, 1, Z, Y, X, dtype=torch.float32, device=device)
    predictions_framework_format = model._convert_tensor_back(predictions_model_format, B_orig, T_orig)
    
    # Verify reverse conversion: [B*T, C, Z, Y, X] -> [B, T, Z, Y, X, C]
    assert predictions_framework_format.shape == (B, T, Z, Y, X, 1)
    
    # Test full forward pass
    with torch.no_grad():
        loss_dict, predictions = model.forward(data_sample)
    
    assert predictions.shape == (B, T, Z, Y, X, 1)
    assert torch.isfinite(predictions).all()
    assert torch.isfinite(loss_dict["step_loss"])


# ==================== Model Template Tests ====================

@pytest.mark.parametrize("model_template", [
    "swin-unetr-small",
    "swin-unetr-base",
    "swin-unetr-large",
])
def test_swin_unetr_model_templates(model_template):
    """Test different model size templates."""
    device = "cuda" if CUDA_AVAILABLE else "cpu"
    torch.manual_seed(0)
    
    B, Z, Y, X, C = 1, 128, 128, 128, 2
    model = _get_model_3d(
        B=B, Z=Z, Y=Y, X=X, C=C,
        model_template=model_template,
        device=device
    )
    model.eval()
    
    inputs = torch.randn(B, Z, Y, X, C, dtype=torch.float32, device=device)
    targets = torch.randint(0, 2, (B, Z, Y, X), dtype=torch.float32, device=device)
    
    data_sample = {
        "data_tensor": inputs,
        "metainfo": {
            "targets": [targets],
            "masks": [None],
        },
    }
    
    with torch.no_grad():
        loss_dict, predictions = model.forward(data_sample)
    
    assert predictions.shape == (B, Z, Y, X, 1)
    assert torch.isfinite(predictions).all()
    assert torch.isfinite(loss_dict["step_loss"])


# ==================== Loss Computation Tests ====================

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for this test")
def test_swin_unetr_loss_computation():
    """Test that loss is computed correctly and is differentiable."""
    device = "cuda"
    torch.manual_seed(0)
    
    B, Z, Y, X, C = 2, 128, 128, 128, 2
    model = _get_model_3d(B=B, Z=Z, Y=Y, X=X, C=C, device=device)
    model.train()  # Set to training mode for gradient computation
    
    inputs = torch.randn(B, Z, Y, X, C, dtype=torch.float32, device=device, requires_grad=True)
    targets = torch.randint(0, 2, (B, Z, Y, X), dtype=torch.float32, device=device)
    
    data_sample = {
        "data_tensor": inputs,
        "metainfo": {
            "targets": [targets],
            "masks": [None],
        },
    }
    
    loss_dict, predictions = model.forward(data_sample)
    loss = loss_dict["step_loss"]
    
    # Verify loss is a scalar tensor
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    
    # Verify loss is differentiable
    loss.backward()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()


# ==================== Edge Cases ====================

def test_swin_unetr_get_num_patches():
    """Test get_num_patches method."""
    device = "cpu"
    
    # Test 3D
    model_3d = _get_model_3d(Z=128, Y=128, X=128, C=2, patch_size=2, device=device)
    num_patches_3d = model_3d.get_num_patches()
    assert isinstance(num_patches_3d, int)
    assert num_patches_3d > 0
    
    # Test 4D
    model_4d = _get_model_4d(T=4, Z=128, Y=128, X=128, C=2, patch_size=2, device=device)
    num_patches_4d = model_4d.get_num_patches()
    assert isinstance(num_patches_4d, int)
    assert num_patches_4d > 0


def test_swin_unetr_output_channels_validation():
    """Test that output_channels validation works correctly."""
    device = "cpu"
    
    # For boundary_segmentation, output_channels must be None (will be set to 1 automatically)
    model = FinetuneSwinUNETR(
        decoder_args={},
        decoder="vit",
        task="boundary_segmentation",
        output_channels=None,  # Must be None, will be set to 1
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

