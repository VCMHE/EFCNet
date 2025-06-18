import torch
import torch.nn as nn

class SmoothEdgeExtra(nn.Module):
    def __init__(self, in_channels=8, reduction=8):
        super(SmoothEdgeExtra, self).__init__()

        self.up = nn.Sequential(
            nn.Conv2d(1, in_channels, kernel_size=3, padding=1, bias=False),  # 3x3
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        out_channels = in_channels // reduction
        #Reduce the channel dimension, usually by a factor of 8 or down to 2
        self.conv_efg = nn.Sequential(
            nn.Conv2d(in_channels, 2, kernel_size=1, bias=False),  # 1x1
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),

            nn.Conv2d(2, 2, kernel_size=3, padding=1, bias=False),  # 3x3
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),

            nn.Conv2d(2, in_channels, kernel_size=1, bias=False),  # 1x1
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),  # 1x1
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.conv_1X1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),  # 1x1
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.conv_revert = nn.Sequential(
            nn.Conv2d(out_channels*2, in_channels, kernel_size=1, bias=False),  # 1x1
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        self.down = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=3, padding=1, bias=False),  # 3x3
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.up(x)
        x1 = x
        x = self.conv_efg(x)
        x1 = self.conv_1X1(x1)
        x = torch.cat((x, x1), dim=1)
        x = self.conv_revert(x)
        x = self.down(x)
        return x
