"""
Polar (θ,φ) raw far-field diffraction model (89 wavelengths, baseline).

Input: (batch, 89, 181, 361) — 89 selected wavelength slices of norm_E2_angle
  (every 9th wavelength from the full 801)

Architecture:
  FeatureNet18 (skip_first_pool, init_channels=89) → Encoder_OCD

This is the primary raw baseline used in the paper for comparison
against the SVD-compressed models.

Paper: "SVD of Far-Field EM Data for Compressing and Accelerating DNNs
       in TSV Geometric Parameter Prediction" (IEEE TIM, MEAS_D_2613346)
"""
import math

import torch
from torch import nn
from ResNet import FeatureNet18


class DUVModel(nn.Module):
    """ResNet-18 based predictor for TSV OCD parameters from raw far-field data.

    Args:
        input_shape: (n_lambda, theta, phi) — default (89, 181, 361).
        out_dim: Number of OCD parameters to predict (default 6).
    """

    def __init__(self, input_shape=(89, 181, 361), out_dim=6):
        super(DUVModel, self).__init__()

        # Feature extraction backbone with 89 input channels
        self.resnet = FeatureNet18(skip_first_pool=True, init_channels=input_shape[0])

        # Infer backbone output dimensions via dummy forward pass
        dummy = torch.zeros(1, *input_shape)
        with torch.no_grad():
            out = self.resnet(dummy)                # (1, C, H', W')
        _, c, h, w = out.shape

        # Regression head: (batch, C, H'*W') → (batch, out_dim)
        self.enc_ocd = Encoder_OCD(inplanes=h * w, in_channels=c, out_dim=out_dim)

    def forward(self, x):
        """Forward pass.

        Args:
            x: (batch, 89, 181, 361) — raw far-field intensity slices

        Returns:
            ocd: (batch, out_dim) — predicted OCD parameters (normalized)
        """
        x = self.resnet(x)                          # (batch, C, H', W')
        x = x.view(x.shape[0], x.shape[1], -1)      # (batch, C, H'*W')
        ocd = self.enc_ocd(x)                       # (batch, out_dim)
        return ocd


class Encoder_OCD(nn.Module):
    """Regression head that maps backbone features to OCD parameters.

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

        # Global average pooling over spatial positions
        self.pool = nn.AdaptiveAvgPool1d(1)

        # 2-layer regression head
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            activation,
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

        # Aggregate spatial positions via global average pooling
        x = x.permute(0, 2, 1)               # (batch, hidden_dim, inplanes)
        x = self.pool(x).squeeze(-1)         # (batch, hidden_dim)

        # Regression head
        out = self.regressor(x)              # (batch, out_dim)
        return out