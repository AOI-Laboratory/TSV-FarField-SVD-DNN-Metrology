"""
Polar (θ,φ) SVD-compressed far-field diffraction model (89 wavelengths, rank p=1).

Input: concatenated [pad(Us); Vt] → (batch, 1, 178, 361)
  - Us: top-1 left singular vectors, shape (89, 181), zero-padded to (89, 361)
  - Vt: top-1 right singular vectors, shape (89, 361)
  - Concatenated along height: (178, 361) → (1, 178, 361)

Architecture:
  Conv2d(1→3, k=1) → FeatureNet18 (skip_first_pool) → Encoder_OCD

This is the primary model used in the paper.

Paper: "SVD of Far-Field EM Data for Compressing and Accelerating DNNs
       in TSV Geometric Parameter Prediction" (IEEE TIM, MEAS_D_2613346)
"""
import math

import torch
from torch import nn
from ResNet import FeatureNet18


class DUVModel(nn.Module):
    """ResNet-18 based predictor for TSV OCD parameters from SVD features.

    Args:
        n_lambda: Number of wavelengths (default 89 for this variant).
        width: Angular width in phi dimension (default 361).
        out_dim: Number of OCD parameters to predict (default 6).
        hidden_dim: Hidden layer width in Encoder_OCD (default 256).
    """

    def __init__(self, n_lambda=89, width=361, out_dim=6, hidden_dim=256):
        super(DUVModel, self).__init__()
        self.n_lambda = n_lambda
        self.width = width
        self.out_dim = out_dim

        # Project single-channel SVD input to 3 channels for ResNet backbone
        self.up_channel = nn.Conv2d(1, 3, kernel_size=1, stride=1)

        # Feature extraction backbone (ResNet-18 without first max-pool)
        self.resnet = FeatureNet18(skip_first_pool=True)

        # Infer backbone output dimensions via dummy forward pass
        dummy = torch.zeros(1, 1, n_lambda * 2, width)
        with torch.no_grad():
            dummy = self.up_channel(dummy)          # (1, 3, H, W)
            out = self.resnet(dummy)                # (1, C, H', W')
        _, c, h, w = out.shape

        # Regression head: (batch, C, H'*W') → (batch, out_dim)
        self.enc_ocd = Encoder_OCD(inplanes=h * w, in_channels=c,
                                   hidden_dim=hidden_dim, out_dim=out_dim)

    def forward(self, x):
        """Forward pass.

        Args:
            x: (batch, 1, n_lambda*2, width) — concatenated [pad(Us); Vt]

        Returns:
            ocd: (batch, out_dim) — predicted OCD parameters (normalized)
        """
        x = self.up_channel(x)          # (batch, 3, H, W)
        x = self.resnet(x)              # (batch, C, H', W')
        x = x.view(x.shape[0], x.shape[1], -1)  # (batch, C, H'*W')
        ocd = self.enc_ocd(x)           # (batch, out_dim)
        return ocd


class Encoder_OCD(nn.Module):
    """Regression head that maps backbone features to OCD parameters.

    Applies a shared linear projection to each spatial position, followed by
    global average pooling and a 2-layer MLP regression head.

    Args:
        inplanes: Number of spatial positions (H' * W').
        in_channels: Number of input channels from backbone.
        hidden_dim: Width of the shared projection layer.
        out_dim: Number of output OCD parameters.
    """

    def __init__(self, inplanes, in_channels, hidden_dim=256, out_dim=6,
                 activation=nn.ReLU()):
        super(Encoder_OCD, self).__init__()
        self.inplanes = inplanes
        self.in_channels = in_channels

        # Shared linear projection: C → hidden_dim (applied per spatial position)
        self.feature_linear = nn.Linear(in_channels, hidden_dim)

        # Normalization and regularization
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(p=0.1)

        # Global average pooling over spatial positions
        self.pool = nn.AdaptiveAvgPool1d(1)

        # 2-layer regression head
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            activation,
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, out_dim)
        )

    def forward(self, x):
        """
        Args:
            x: (batch, C, inplanes) — flattened backbone features

        Returns:
            out: (batch, out_dim) — predicted OCD parameters
        """
        # Project each spatial position independently
        x = x.permute(0, 2, 1)               # (batch, inplanes, C)
        x = self.feature_linear(x)           # (batch, inplanes, hidden_dim)
        x = self.norm(x)
        x = torch.relu(x)
        x = self.dropout(x)

        # Aggregate spatial positions via global average pooling
        x = x.permute(0, 2, 1)               # (batch, hidden_dim, inplanes)
        x = self.pool(x).squeeze(-1)         # (batch, hidden_dim)

        # Regression head
        out = self.regressor(x)              # (batch, out_dim)
        return out