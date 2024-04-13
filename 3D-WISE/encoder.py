# -*- coding:utf-8 -*-
# -----------------------------------------
#   Filename: encoder.py
#   Author  : Qing Wu
#   Email   : wuqing@shanghaitech.edu.cn
#   Date    : 2021/9/20
# -----------------------------------------
import torch.nn as nn
import torch


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc = nn.Sequential(nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class cbam(nn.Module):
    def __init__(self,planes):
        super(cbam, self).__init__()
        self.ca=ChannelAttention(planes)
        self.sa=SpatialAttention()
    def forward(self,x):
        x=self.ca(x)*x
        x=self.sa(x)*x
        return x


# -------------------------------
# RDN encoder network
# <Zhang, Yulun, et al. "Residual dense network for image super-resolution.">
# Here code is modified from: https://github.com/yjn870/RDN-pytorch/blob/master/models.py
# -------------------------------
class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=3 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)


class RDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(RDB, self).__init__()
        self.layers = nn.Sequential(
            *[DenseLayer(in_channels + growth_rate * i, growth_rate) for i in range(num_layers)])
        # local feature fusion
        self.lff = nn.Conv3d(in_channels + growth_rate * num_layers, growth_rate, kernel_size=1)
        self.cbam=cbam(64)
    def forward(self, x):
        x_1=self.layers(x)
        x_2=self.cbam(x_1)
        return x + self.lff(x_2)  # local residual learning


class RDN(nn.Module):
    def __init__(self, feature_dim=32, num_features=16, growth_rate=16, num_blocks=8, num_layers=3):
        super(RDN, self).__init__()
        self.G0 = num_features
        self.G = growth_rate
        self.D = num_blocks
        self.C = num_layers
        # shallow feature extraction
        self.sfe1 = nn.Conv3d(1, num_features, kernel_size=3, padding=3 // 2)
        self.sfe2 = nn.Conv3d(num_features, num_features, kernel_size=3, padding=3 // 2)
        # residual dense blocks
        self.rdbs = nn.ModuleList([RDB(self.G0, self.G, self.C)])
        for _ in range(self.D - 1):
            self.rdbs.append(RDB(self.G, self.G, self.C))
        # global feature fusion
        self.gff = nn.Sequential(
            nn.Conv3d(self.G * self.D, self.G0, kernel_size=1),
            nn.Conv3d(self.G0, self.G0, kernel_size=3, padding=3 // 2)
        )
        self.output = nn.Conv3d(self.G0, feature_dim, kernel_size=3, padding=3 // 2)
        self.cbam=cbam(128)
    def forward(self, x):
        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)
        x = sfe2
        local_features = []
        for i in range(self.D):
            x = self.rdbs[i](x)
            local_features.append(x)

        x=torch.cat(local_features, 1)
        x=self.cbam(x)
        x = self.gff(x) + sfe1  # global residual learning
        x = self.output(x)
        return x


# -------------------------------
# ResCNN encoder network
# <Du, Jinglong, et al. "Super-resolution reconstruction of single
# anisotropic 3D MR images using residual convolutional neural network.">
# -------------------------------
class ResCNN(nn.Module):
    def __init__(self, feature_dim=32):
        super(ResCNN, self).__init__()
        self.conv_start = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True)
        )
        self.block1 = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True)
        )
        self.block3 = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True)
        )
        self.conv_end = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=feature_dim, kernel_size=3, padding=3 // 2),
        )

    def forward(self, x):
        in_block1 = self.conv_start(x)
        out_block1 = self.block1(in_block1)
        in_block2 = out_block1 + in_block1
        out_block2 = self.block2(in_block2)
        in_block3 = out_block2 + in_block2
        out_block3 = self.block3(in_block3)
        res_img = self.conv_end(out_block3 + in_block3)
        return x + res_img


# -------------------------------
# SRResNet
# <Ledig, Christian, et al. "Photo-realistic single image super-resolution
# using a generative adversarial network.">
# -------------------------------
def conv(ni, nf, kernel_size=3, actn=False):
    layers = [nn.Conv3d(ni, nf, kernel_size, padding=kernel_size // 2)]
    if actn: layers.append(nn.ReLU(True))
    return nn.Sequential(*layers)


class ResSequential(nn.Module):
    def __init__(self, layers, res_scale=1.0):
        super().__init__()
        self.res_scale = res_scale
        self.m = nn.Sequential(*layers)

    def forward(self, x): return x + self.m(x) * self.res_scale


def res_block(nf):
    return ResSequential(
        [conv(nf, nf, actn=True), conv(nf, nf)],
        1.0)  # this is best one


class SRResnet(nn.Module):
    def __init__(self, nf=16, feature_dim=32):
        super().__init__()
        features = [conv(1, nf)]
        for i in range(18): features.append(res_block(nf))
        features += [conv(nf, nf),
                     conv(nf, feature_dim)]
        self.features = nn.Sequential(*features)

    def forward(self, x):
        return self.features(x)
