"""
ReLU² activation function.

ReLU² is defined as relu(x)^2, which is a smooth, non-negative activation
that has been shown to work well in certain architectures.
"""

import torch
import torch.nn as nn


class ReLU2(nn.Module):
    """
    ReLU² activation: relu(x)^2
    
    This activation function applies ReLU and then squares the result.
    It produces smooth, non-negative outputs and has been used in recent
    vision transformer architectures.
    """
    
    def __init__(self, inplace: bool = False):
        """
        Args:
            inplace: If True, performs the operation in-place. Default: False
        """
        super().__init__()
        self.inplace = inplace
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply ReLU² activation: relu(x)^2
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with ReLU² applied
        """
        return torch.relu(x).pow(2) if not self.inplace else torch.relu_(x).pow_(2)
    
    def extra_repr(self) -> str:
        return f"inplace={self.inplace}"

