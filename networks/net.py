import torch
from torch import nn
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from networks.Fusion import Fusion

class Conv1(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv1, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0, stride=1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, input):
        out = self.conv(input)
        out = self.bn(out)
        out = self.relu(out)
        return out


class Conv3_3(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv3_3, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, stride=1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, input):
        out = self.conv(input)
        out = self.bn(out)
        out = self.relu(out)
        return out


class Conv5_5(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv5_5, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=5, padding=2, stride=1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, input):
        out = self.conv(input)
        out = self.bn(out)
        out = self.relu(out)
        return out


class Conv7_7(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv7_7, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=7, padding=3, stride=1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, input):
        out = self.conv(input)
        out = self.bn(out)
        out = self.relu(out)
        return out


class Convlutioanl_out1(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Convlutioanl_out1, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, stride=1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, input):
        out = self.conv(input)
        out = self.bn(out)
        out = self.relu(out)
        return out


class Convlutioanl_out2(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Convlutioanl_out2, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        out = self.conv(input)
        out = self.sigmoid(out)
        return out


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class MODEL(nn.Module):
    def __init__(self, in_channel=1, out_channel_16=16, out_channel_32=32, out_channel_64=64, out_channel_128=128,
                 out_channel_256=256, out_channel_512=512, out_channel_336=336, output_channel_64=64, output_channel_16=16,
                 output_channel_1=1):
        super(MODEL, self).__init__()
        self.conv3_3 = Conv3_3(in_channel, out_channel_16)
        self.conv5_5 = Conv5_5(out_channel_16, out_channel_64)
        self.conv7_7 = Conv7_7(out_channel_64, out_channel_256)

        self.ca1 = CoordAtt(inp=out_channel_16, oup=out_channel_16)
        self.ca2 = CoordAtt(inp=out_channel_64, oup=out_channel_64)
        self.ca3 = CoordAtt(inp=out_channel_256, oup=out_channel_256)

        self.conv1 = Conv1(out_channel_32, out_channel_16)
        self.conv2 = Conv1(out_channel_128, out_channel_64)
        self.conv3 = Conv1(out_channel_512, out_channel_256)
        self.conv4 = Conv1(out_channel_336, output_channel_64)

        self.SU1 = Fusion(in_dim=out_channel_16, out_dim=out_channel_16)
        self.SU2 = Fusion(in_dim=out_channel_64, out_dim=out_channel_64)
        self.SU3 = Fusion(in_dim=out_channel_256, out_dim=out_channel_256)
        self.SU4 = Fusion(in_dim=output_channel_64, out_dim=output_channel_64)

        self.convolutional_out1 = Convlutioanl_out1(output_channel_64, output_channel_16)
        self.convolutional_out2 = Convlutioanl_out2(output_channel_16, output_channel_1)

    def forward(self, spect, mri):
        input_I1 = self.conv3_3(spect)
        input_IC1 = self.ca1(input_I1)
        input_I2 = self.conv5_5(input_I1)
        input_IC2 = self.ca2(input_I2)
        input_I3 = self.conv7_7(input_I2)
        input_IC3 = self.ca3(input_I3)

        input_V1 = self.conv3_3(mri)
        input_VC1 = self.ca1(input_V1)
        input_V2 = self.conv5_5(input_V1)
        input_VC2 = self.ca2(input_V2)
        input_V3 = self.conv7_7(input_V2)
        input_VC3 = self.ca3(input_V3)

        input_F1 = torch.cat((input_IC1, input_VC1), dim=1)
        input_F1 = self.conv1(input_F1)
        input_F1 = self.SU1(input_F1)

        input_F2 = torch.cat((input_IC2, input_VC2), dim=1)
        input_F2 = self.conv2(input_F2)
        input_F2 = self.SU2(input_F2)

        input_F3 = torch.cat((input_IC3, input_VC3), dim=1)
        input_F3 = self.conv3(input_F3)
        input_F3 = self.SU3(input_F3)

        input_F4 = torch.cat((input_F1, input_F2, input_F3), dim=1)
        input_F4 = self.conv4(input_F4)
        input_F4 = self.SU4(input_F4)

        out = self.convolutional_out1(input_F4)
        out = self.convolutional_out2(out)

        return out, input_V1, input_V2, input_V3
