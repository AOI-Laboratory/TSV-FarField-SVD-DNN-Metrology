"""
BFP raw 3-wavelength model with log1p (BFP-raw-w3).

Architecture:
    ResNet18 backbone (skip_first_pool=False) → shared Linear projection
    → AdaptiveAvgPool1d → regression head

Input: (batch, 3, 4000, 3000) — 3 selected wavelength slices from 30 total.
Output: (batch, 6) — [retch, Rtop, Rbot, n, height, b]

Wavelength selection: indices [0, 15, 29] from 30 wavelengths (first, mid, last).
Applies np.log1p() to input images before tensor conversion.

Key difference from Model_BFP_raw3_1:
    - Uses shared Linear projection (not per-position MLPs)
    - No progressive AvgPool + 1×1 Conv stages
    - Applies log1p transformation to input

Paper: "SVD of Far-Field EM Data for Compressing and Accelerating DNNs
       in TSV Geometric Parameter Prediction" (IEEE TIM, MEAS_D_2613346)
"""
import math
import torch
from torch import nn
from ResNet import FeatureNet18


class DUVModel(nn.Module):
    """BFP raw 3-wavelength model with log1p input transformation.

    Processes 3 full-resolution BFP images (4000×3000) through a shared
    ResNet18 backbone, then uses a shared linear projection + global
    average pooling + regression head for OCD prediction.
    """

    def __init__(self, input_shape=(3, 4000, 3000), out_dim=6):
        """
        Args:
            input_shape: (channels, height, width) of input images.
            out_dim: Number of OCD parameters to predict (6).
        """
        super(DUVModel, self).__init__()
        self.resnet = FeatureNet18(skip_first_pool=False)

        # Infer channel count with a small dummy input (memory-efficient)
        dummy = torch.zeros(1, input_shape[0], 256, 256)
        with torch.no_grad():
            out = self.resnet(dummy)
        _, c, _, _ = out.shape

        # Compute actual spatial dimensions after backbone (scale factor = 32)
        scale_factor = 32
        h = input_shape[1] // scale_factor
        w = input_shape[2] // scale_factor

        self.enc_ocd = Encoder_OCD(inplanes=h * w, in_channels=c, out_dim=out_dim)

    def forward(self, x):
        """Forward pass.

        Args:
            x: (batch, 3, 4000, 3000) — 3 wavelength slices (log1p applied).

        Returns:
            ocd: (batch, 6) — predicted OCD parameters.
        """
        x = self.resnet(x)
        x = x.view(x.shape[0], x.shape[1], -1)
        ocd = self.enc_ocd(x)
        return ocd


class Encoder_OCD(nn.Module):
    """OCD regression head with shared linear projection.

    Uses a single shared Linear layer to project channel features,
    followed by global average pooling and a 2-layer regression head.
    Simpler than the per-position MLP variant (Model_BFP_raw3_1).
    """

    def __init__(self, inplanes, in_channels, hidden_dim=256, out_dim=6, activation=nn.ReLU()):
        """
        Args:
            inplanes: Number of spatial positions (H' * W').
            in_channels: Backbone output channels C.
            hidden_dim: Hidden dimension for projection and regression.
            out_dim: Number of OCD parameters to predict.
        """
        super(Encoder_OCD, self).__init__()
        self.inplanes = inplanes
        self.in_channels = in_channels

        # Shared linear projection: C → hidden_dim
        self.feature_linear = nn.Linear(in_channels, hidden_dim)

        # Global average pooling over spatial positions
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            activation,
            nn.Linear(hidden_dim // 2, out_dim)
        )

    def forward(self, x):
        """
        Args:
            x: (batch, channels, inplanes) — backbone features.

        Returns:
            (batch, out_dim) — predicted OCD parameters.
        """
        x = x.permute(0, 2, 1)           # (batch, inplanes, channels)
        x = self.feature_linear(x)       # (batch, inplanes, hidden_dim)

        # Global average pooling
        x = x.permute(0, 2, 1)           # (batch, hidden_dim, inplanes)
        x = self.pool(x).squeeze(-1)     # (batch, hidden_dim)

        # Regression head
        out = self.regressor(x)
        return out