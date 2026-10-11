"""
BFP SVD-compressed model (30 wavelengths, p=1) — BFP-SVD-w30-p1.

Architecture:
    1×1 Conv (1→3 channels) → ResNet18 backbone (skip_first_pool=True)
    → shared Linear projection → LayerNorm → AdaptiveAvgPool1d → regression head

Input: (batch, 1, 60, 4000) = [pad(Us); Vt]
    - Us: (30, 3000) → padded to (30, 4000)
    - Vt: (30, 4000)
Output: (batch, 6) — [retch, Rtop, Rbot, n, height, b]

SVD rank: p=1 (top-1 singular vectors).
All 30 wavelengths used (no subsampling).

Paper: "SVD of Far-Field EM Data for Compressing and Accelerating DNNs
       in TSV Geometric Parameter Prediction" (IEEE TIM, MEAS_D_2613346)
"""
import math
import torch
from torch import nn
from ResNet import FeatureNet18


class DUVModel(nn.Module):
    """BFP SVD-compressed model (30 wavelengths, p=1).

    Uses a 1×1 convolution to expand the single-channel SVD input to 3
    channels for the ResNet18 backbone. The backbone uses skip_first_pool=True
    to preserve spatial resolution. A shared linear projection + global
    average pooling + regression head produces the OCD prediction.
    """

    def __init__(self, n_lambda=30, width=4000, out_dim=6, hidden_dim=256):
        """
        Args:
            n_lambda: Number of wavelengths (one side). Input height = n_lambda * 2.
            width: Spatial width of BFP images (4000).
            out_dim: Number of OCD parameters to predict (6).
            hidden_dim: Hidden dimension for the regression head.
        """
        super(DUVModel, self).__init__()
        self.n_lambda = n_lambda
        self.width = width
        self.out_dim = out_dim

        # Expand single-channel input to 3 channels for ResNet
        self.up_channel = nn.Conv2d(1, 3, kernel_size=1, stride=1)

        # ResNet18 backbone (skip first max-pool to preserve resolution)
        self.resnet = FeatureNet18(skip_first_pool=True)

        # Infer backbone output shape with a dummy forward pass
        dummy = torch.zeros(1, 1, n_lambda * 2, width)
        with torch.no_grad():
            dummy = self.up_channel(dummy)
            out = self.resnet(dummy)
        _, c, h, w = out.shape

        # OCD regression head
        self.enc_ocd = Encoder_OCD(inplanes=h * w, in_channels=c,
                                   hidden_dim=hidden_dim, out_dim=out_dim)

    def forward(self, x):
        """Forward pass.

        Args:
            x: (batch, 1, n_lambda*2, width) — SVD features [pad(Us); Vt].

        Returns:
            ocd: (batch, 6) — predicted OCD parameters.
        """
        x = self.up_channel(x)
        x = self.resnet(x)
        x = x.view(x.shape[0], x.shape[1], -1)
        ocd = self.enc_ocd(x)
        return ocd


class Encoder_OCD(nn.Module):
    """OCD regression head with shared linear projection.

    Uses a single shared Linear layer (instead of per-position MLPs) to
    project channel features, followed by LayerNorm, global average pooling,
    and a 2-layer regression head.
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

        # Normalization and regularization
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(p=0.1)

        # Global average pooling over spatial positions
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            activation,
            nn.Dropout(0.1),
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
        x = self.norm(x)
        x = torch.relu(x)
        x = self.dropout(x)

        # Global average pooling
        x = x.permute(0, 2, 1)           # (batch, hidden_dim, inplanes)
        x = self.pool(x).squeeze(-1)     # (batch, hidden_dim)

        # Regression head
        out = self.regressor(x)
        return out