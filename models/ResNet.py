"""
ResNet-18 feature backbone for far-field diffraction models.

Provides FeatureNet18, a modified ResNet-18 that:
  - Skips the first max-pooling layer (skip_first_pool=True) to preserve
    spatial resolution for the 181×361 angular grid.
  - Accepts configurable input channels (init_channels) for raw multi-channel
    far-field inputs.

Used by all model variants in this repository:
  - SVD models: Conv2d(1→3, k=1) → FeatureNet18(skip_first_pool=True)
  - Raw models: FeatureNet18(skip_first_pool=True, init_channels=N)

Paper: "SVD of Far-Field EM Data for Compressing and Accelerating DNNs
       in TSV Geometric Parameter Prediction" (IEEE TIM, MEAS_D_2613346)
"""
import torch.nn as nn


class basic_block(nn.Module):
    """Standard ResNet basic block (two 3×3 convolutions with residual connection)."""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride, downsample):
        super(basic_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class FeatureNet18(nn.Module):
    """ResNet-18 backbone adapted for far-field diffraction feature extraction.

    Args:
        layers: Number of basic blocks in each stage (default (2,2,2,2)).
        net_block: Block type (default basic_block).
        skip_first_pool: If True, skip the initial max-pooling to preserve
            spatial resolution (recommended for 181×361 angular grids).
        init_channels: Number of input channels (3 for SVD models after
            1×1 conv up-projection; N for raw models with N wavelengths).

    Output:
        Feature map of shape (batch, 512, H', W') where H' and W' depend
        on input size and pooling configuration.
    """

    def __init__(self, layers=(2, 2, 2, 2), net_block=basic_block, skip_first_pool=False, init_channels=3):
        super(FeatureNet18, self).__init__()
        self.skip_first_pool = skip_first_pool
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels=init_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.net_block_layer(net_block, 64, layers[0])
        self.layer2 = self.net_block_layer(net_block, 128, layers[1], stride=2)
        self.layer3 = self.net_block_layer(net_block, 256, layers[2], stride=2)
        self.layer4 = self.net_block_layer(net_block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def net_block_layer(self, net_block, out_channels, num_blocks, stride=1):
        """Build a sequential stage of basic blocks with optional downsample."""
        downsample = None

        if stride != 1 or self.in_channels != out_channels * net_block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * net_block.expansion, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(out_channels * net_block.expansion))

        layers = []
        layers.append(net_block(self.in_channels, out_channels, stride, downsample))
        if net_block.expansion != 1:
            self.in_channels = out_channels * net_block.expansion

        else:
            self.in_channels = out_channels

        for i in range(1, num_blocks):
            layers.append(net_block(self.in_channels, out_channels, 1, None))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.skip_first_pool:
            x = self.maxpooling(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x