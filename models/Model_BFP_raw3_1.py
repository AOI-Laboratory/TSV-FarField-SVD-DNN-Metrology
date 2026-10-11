"""
BFP raw 3-wavelength model (BFP-SVD-w3-p1).

Architecture:
    ResNet18 backbone (no skip_first_pool) → 3× AvgPool2d(3, stride=2)
    → 3× Conv2d(512, 512, 1×1) → Encoder_OCD (per-position MLP + regression head)

Input: (batch, 3, 4000, 3000) — 3 selected wavelength slices from 30 total.
Output: (batch, 6) — [retch, Rtop, Rbot, n, height, b]

Wavelength selection: indices [0, 14, 29] from 30 wavelengths (first, mid, last).
No log1p transformation applied.

Paper: "SVD of Far-Field EM Data for Compressing and Accelerating DNNs
       in TSV Geometric Parameter Prediction" (IEEE TIM, MEAS_D_2613346)
"""
import torch
from torch import nn
from ResNet import FeatureNet18


class DUVModel(nn.Module):
    """BFP raw 3-wavelength model with ResNet18 backbone.

    Processes 3 full-resolution BFP images (4000×3000) through a shared
    ResNet18 backbone, followed by progressive pooling and 1×1 convolutions
    to reduce spatial dimensions, then a per-position MLP encoder for
    OCD parameter regression.
    """

    def __init__(self, input_shape=(3, 4000, 3000), out_dim=6):
        """
        Args:
            input_shape: (channels, height, width) of input images.
            out_dim: Number of OCD parameters to predict (6).
        """
        super(DUVModel, self).__init__()

        # ResNet18 backbone (standard, with first max-pool)
        self.resnet = FeatureNet18()

        # Progressive spatial reduction: 3 rounds of AvgPool + 1×1 Conv
        self.pool_1 = nn.AvgPool2d(3, stride=2, padding=0)
        self.conv2d_1 = nn.Conv2d(512, 512, kernel_size=1, stride=1)
        self.pool_2 = nn.AvgPool2d(3, stride=2, padding=0)
        self.conv2d_2 = nn.Conv2d(512, 512, kernel_size=1, stride=1)
        self.pool_3 = nn.AvgPool2d(3, stride=2, padding=0)
        self.conv2d_3 = nn.Conv2d(512, 512, kernel_size=1, stride=1)

        # OCD regression head: input spatial size = 14×10 after 3 pools
        self.enc_ocd = Encoder_OCD(inplanes=14 * 10)

    def forward(self, x):
        """Forward pass.

        Args:
            x: (batch, 3, 4000, 3000) — 3 wavelength slices.

        Returns:
            ocd: (batch, 6) — predicted OCD parameters.
        """
        x = self.resnet(x)
        x = self.pool_1(x)
        x = self.conv2d_1(x)
        x = self.pool_2(x)
        x = self.conv2d_2(x)
        x = self.pool_3(x)
        x = self.conv2d_3(x)
        x = torch.reshape(x, shape=(x.shape[0], x.shape[1], -1))
        ocd = self.enc_ocd(x)
        return ocd


class Encoder_Feature_Linear(nn.Module):
    """Per-position MLP: projects 512-dim features to 8-dim embedding."""

    def __init__(self, in_size, out_size, activation=nn.ReLU()):
        super(Encoder_Feature_Linear, self).__init__()
        self.linear1 = nn.Linear(in_size, out_size * 4, bias=True)
        self.linear2 = nn.Linear(out_size * 4, out_size, bias=True)
        self.activation = activation

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        return torch.unsqueeze(x, dim=1)


class Encoder_OCD(nn.Module):
    """OCD regression head with per-position MLPs.

    Each spatial position has its own 2-layer MLP (512→32→8),
    followed by a shared regression head (112→512→256→6).
    """

    def __init__(self, inplanes=14 * 10, activation=nn.ReLU()):
        super(Encoder_OCD, self).__init__()
        out_size = 8
        dims = [out_size * inplanes, 512, 256, 6]

        self.inplanes = inplanes
        # Per-position MLPs (one per spatial position)
        self.feature_linears = nn.ModuleList(
            [Encoder_Feature_Linear(in_size=512, out_size=out_size) for _ in range(inplanes)]
        )
        self.flatten = nn.Flatten()

        # Shared regression head
        self.ocd_linear1 = nn.Linear(dims[0], dims[1], bias=True)
        self.ocd_linear2 = nn.Linear(dims[1], dims[2], bias=True)
        self.ocd_linear3 = nn.Linear(dims[2], dims[3], bias=False)

        self.activation = activation
        self.relu6 = nn.ReLU6()

    def forward(self, x):
        """
        Args:
            x: (batch, 512, inplanes) — features from backbone.

        Returns:
            (batch, 6) — predicted OCD parameters.
        """
        features = list()
        for i in range(self.inplanes):
            features.append(self.feature_linears[i](x[:, :, i]))

        x = torch.cat(features, dim=1)
        x = self.flatten(x)

        x = self.ocd_linear1(x)
        x = self.activation(x)
        x = self.ocd_linear2(x)
        x = self.activation(x)
        x = self.ocd_linear3(x)
        x = self.relu6(x)
        return x