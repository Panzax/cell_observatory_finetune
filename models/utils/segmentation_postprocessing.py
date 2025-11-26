"""
Post-processing utilities for converting boundary predictions to instance segmentation.
Uses GPU-accelerated Voronoi-Otsu labeling from pyclesperanto.

Based on: https://github.com/clEsperanto/pyclesperanto
Reference: Voronoi-Otsu labeling guide from clEsperanto documentation
"""

import numpy as np
import torch

try:
    import pyclesperanto as cle
    CLESPERANTO_AVAILABLE = True
    # Initialize GPU on import
    print(f"clEsperanto using: {cle.select_device()}")
except ImportError:
    CLESPERANTO_AVAILABLE = False
    print("Warning: pyclesperanto not installed. Install with: pip install pyclesperanto")


def make_isotropic_if_needed(image, voxel_size=(1.0, 1.0, 1.0)):
    """
    Make image isotropic if voxel sizes differ significantly.
    
    clEsperanto recommendation: make 3D data isotropic before Voronoi-Otsu labeling.
    
    Args:
        image: 3D array [Z, Y, X]
        voxel_size: Tuple of (z_size, y_size, x_size) in physical units
        
    Returns:
        Tuple of (isotropic_image, scale_factors_used)
    """
    z_size, y_size, x_size = voxel_size
    
    # Check if already approximately isotropic
    min_size = min(voxel_size)
    ratios = [s / min_size for s in voxel_size]
    
    # If all ratios are close to 1.0, no need to rescale
    if all(0.8 < r < 1.2 for r in ratios):
        return image, (1.0, 1.0, 1.0)
    
    # Calculate target isotropic shape
    target_voxel_size = min_size
    new_z = int(image.shape[0] * z_size / target_voxel_size)
    new_y = int(image.shape[1] * y_size / target_voxel_size)
    new_x = int(image.shape[2] * x_size / target_voxel_size)
    
    # Rescale using clEsperanto
    image_gpu = cle.push(image)
    isotropic_gpu = cle.scale(image_gpu, factor_x=new_x/image.shape[2], 
                               factor_y=new_y/image.shape[1], 
                               factor_z=new_z/image.shape[0],
                               auto_size=True)
    isotropic = cle.pull(isotropic_gpu)
    
    scale_factors = (new_z/image.shape[0], new_y/image.shape[1], new_x/image.shape[2])
    return isotropic, scale_factors


def voronoi_otsu_segmentation_3d(
    boundary_predictions,
    spot_sigma=5.0,
    outline_sigma=1.0,
    make_isotropic=True,
    voxel_size=(1.0, 1.0, 1.0),
    use_gpu=True
):
    """
    Apply Voronoi-Otsu segmentation to 3D boundary predictions.
    
    This converts boundary predictions (or intensity images) into instance segmentation 
    labels where each cell interior gets a unique integer ID.
    
    Algorithm (from clEsperanto):
    1. Gaussian blur with spot_sigma to find cell centers
    2. Detect local maxima as seeds
    3. Gaussian blur with outline_sigma for boundary refinement
    4. Otsu threshold to create binary mask
    5. Filter seeds to those within the mask
    6. Voronoi diagram within the masked region
    
    Args:
        boundary_predictions: Array-like [Z, Y, X] or [Z, Y, X, C]
                             Can be:
                             - Intensity image (bright spots = cell centers)
                             - Boundary predictions (will be inverted internally)
        spot_sigma: Gaussian blur sigma for spot detection (cell centers)
                   Typical: 2-10 depending on cell size in pixels
                   Higher = detect larger, more separated cells
        outline_sigma: Gaussian blur sigma for outline detection (boundaries)
                      Typical: 0-2
                      Higher = smoother boundaries, more tolerance for noise
        make_isotropic: Whether to rescale to isotropic voxels first (recommended for 3D)
        voxel_size: Tuple of (z_size, y_size, x_size) in same units
                   Only used if make_isotropic=True
        use_gpu: Whether to use GPU acceleration (if available)
        
    Returns:
        labels: np.ndarray [Z, Y, X] with integer labels for each cell (0=background)
                If isotropic rescaling was applied, this is in original coordinates
    """
    if not CLESPERANTO_AVAILABLE:
        raise ImportError(
            "pyclesperanto not installed. Run: pip install pyclesperanto"
        )
    
    # Convert to numpy if needed
    if isinstance(boundary_predictions, torch.Tensor):
        boundary_predictions = boundary_predictions.cpu().numpy()
    
    # Handle channel dimension - take first channel if multi-channel
    if boundary_predictions.ndim == 4:
        boundary_predictions = boundary_predictions[..., 0]
    
    # Ensure float32
    image = boundary_predictions.astype(np.float32)
    original_shape = image.shape
    
    # Make isotropic if requested (recommended for 3D)
    if make_isotropic and image.ndim == 3:
        image, scale_factors = make_isotropic_if_needed(image, voxel_size)
        if not all(s == 1.0 for s in scale_factors):
            print(f"Made isotropic: {original_shape} -> {image.shape}, scales: {scale_factors}")
    
    # Push to GPU
    img_gpu = cle.push(image)
    
    # Apply Voronoi-Otsu labeling
    # This is the single-function approach from clEsperanto
    labels_gpu = cle.voronoi_otsu_labeling(
        img_gpu,
        spot_sigma=spot_sigma,
        outline_sigma=outline_sigma
    )
    
    # Pull back to CPU
    labels = cle.pull(labels_gpu)
    
    # Scale back to original resolution if we made it isotropic
    if make_isotropic and image.ndim == 3 and labels.shape != original_shape:
        labels_gpu = cle.push(labels)
        labels_original_gpu = cle.scale(
            labels_gpu,
            factor_x=original_shape[2]/labels.shape[2],
            factor_y=original_shape[1]/labels.shape[1],
            factor_z=original_shape[0]/labels.shape[0],
            auto_size=True,
            interpolate=False  # Use nearest neighbor for labels
        )
        labels = cle.pull(labels_original_gpu)
    
    return labels.astype(np.int32)


def postprocess_batch(
    predictions,
    spot_sigma=5.0,
    outline_sigma=1.0,
    make_isotropic=True,
    voxel_size=(1.0, 1.0, 1.0),
    use_gpu=True
):
    """
    Process a batch of predictions through Voronoi-Otsu segmentation.
    
    Handles both 3D and 4D (time-series) data automatically.
    
    Args:
        predictions: Tensor or array in framework format
                    - 3D: [B, Z, Y, X, C]
                    - 4D: [B, T, Z, Y, X, C]
        spot_sigma: Sigma for spot detection (2-10 typical)
        outline_sigma: Sigma for outline detection (0-2 typical)
        make_isotropic: Make data isotropic before segmentation (recommended)
        voxel_size: Physical voxel size (z, y, x) for isotropic rescaling
        use_gpu: Use GPU acceleration
        
    Returns:
        labels: np.ndarray with same shape as predictions (minus channel dim)
                - 3D: [B, Z, Y, X]
                - 4D: [B, T, Z, Y, X]
                Each cell has unique integer ID (0=background)
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    
    batch_results = []
    
    # Handle batch dimension
    for i in range(predictions.shape[0]):
        # Check if we have time dimension
        if predictions.ndim == 6:  # [B, T, Z, Y, X, C]
            time_results = []
            for t in range(predictions.shape[1]):
                frame = predictions[i, t]  # [Z, Y, X, C]
                labels = voronoi_otsu_segmentation_3d(
                    frame,
                    spot_sigma=spot_sigma,
                    outline_sigma=outline_sigma,
                    make_isotropic=make_isotropic,
                    voxel_size=voxel_size,
                    use_gpu=use_gpu
                )
                time_results.append(labels)
            batch_results.append(np.stack(time_results))
            
        elif predictions.ndim == 5:  # [B, Z, Y, X, C]
            frame = predictions[i]  # [Z, Y, X, C]
            labels = voronoi_otsu_segmentation_3d(
                frame,
                spot_sigma=spot_sigma,
                outline_sigma=outline_sigma,
                make_isotropic=make_isotropic,
                voxel_size=voxel_size,
                use_gpu=use_gpu
            )
            batch_results.append(labels)
        else:
            raise ValueError(f"Unexpected prediction shape: {predictions.shape}")
    
    return np.stack(batch_results)


def postprocess_single_frame(
    prediction,
    spot_sigma=5.0,
    outline_sigma=1.0,
    make_isotropic=True,
    voxel_size=(1.0, 1.0, 1.0),
    use_gpu=True
):
    """
    Process a single frame (convenience function for testing/debugging).
    
    Args:
        prediction: Single 3D volume [Z, Y, X] or [Z, Y, X, C]
        spot_sigma: Sigma for spot detection
        outline_sigma: Sigma for outline detection
        make_isotropic: Make data isotropic before segmentation
        voxel_size: Physical voxel size (z, y, x)
        use_gpu: Use GPU acceleration
        
    Returns:
        labels: [Z, Y, X] with integer labels for each cell
    """
    return voronoi_otsu_segmentation_3d(
        prediction,
        spot_sigma=spot_sigma,
        outline_sigma=outline_sigma,
        make_isotropic=make_isotropic,
        voxel_size=voxel_size,
        use_gpu=use_gpu
    )

