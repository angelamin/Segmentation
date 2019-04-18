from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision
import numpy as np
# from torchvision import models
import torch.nn.init as init

import sys

def conv3x3(in_, out):
    return nn.Conv2d(in_, out, kernel_size=3, padding=1, bias=True,)

def conv3x3hole(in_, out, hole):
    return nn.Conv2d(in_, out, kernel_size=3, padding=hole, bias=True, dilation=hole)

class ConvRelu(nn.Module):
    def __init__(self, in_, out, BatchNorm = nn.BatchNorm2d):
        super(ConvRelu, self).__init__()
        self.conv = conv3x3(in_, out)
        self.bn = BatchNorm(out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class ConvRelu2(nn.Module):
    def __init__(self, in_, out, BatchNorm=nn.BatchNorm2d):
        super(ConvRelu2, self).__init__()
        self.conv1 = conv3x3(in_, out)
        self.bn1 = BatchNorm(out)
        self.conv2 = conv3x3(out, out)
        self.bn2 = BatchNorm(out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        return x

class ConvRelu4(nn.Module):
    def __init__(self, in_, out, BatchNorm=nn.BatchNorm2d):
        super(ConvRelu4, self).__init__()
        self.conv1 = conv3x3(in_, out)
        self.bn1 = BatchNorm(out)
        self.conv2 = conv3x3(out, out)
        self.bn2 = BatchNorm(out)
        self.conv3 = conv3x3(out, out)
        self.bn3 = BatchNorm(out)
        self.conv4 = conv3x3(out, out)
        self.bn4 = BatchNorm(out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.activation(x)
        return x

class ConvHole(nn.Module):
    def __init__(self, in_, out, hole, BatchNorm = nn.BatchNorm2d):
        super(ConvHole, self).__init__()
        self.conv1 = conv3x3hole(in_, out, hole)
        self.bn1 = BatchNorm(out)
        self.conv2 = conv3x3hole(out, out, hole)
        self.bn2 = BatchNorm(out)
        self.conv3 = conv3x3hole(out, out, hole)
        self.bn3 = BatchNorm(out)
        self.conv4 = conv3x3hole(out, out, hole)
        self.bn4 = BatchNorm(out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.activation(x)
        return x

class DecoderBlockV2(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True, BatchNorm=nn.BatchNorm2d):
        super(DecoderBlockV2, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            """
                Paramaters for Deconvolution were chosen to avoid artifacts, following
                link https://distill.pub/2016/deconv-checkerboard/
            """
            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels, BatchNorm),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
                ConvRelu(middle_channels, out_channels, BatchNorm),
            )
        else:
            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels, BatchNorm),
                ConvRelu(middle_channels, out_channels, BatchNorm),
                nn.Upsample(scale_factor=2, mode='bilinear'),
            )

    def forward(self, x):
        return self.block(x)

class UNet19(nn.Module):
    def __init__(self, num_classes=21, num_filters=32, pretrained=True, is_deconv=False, syncbn=False, group_size=1, group=None, sync_stats=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network used
            True - encoder pre-trained with VGG16
        :is_deconv:
            False: bilinear interpolation is used in decoder
            True: deconvolution is used in decoder
        """
        self.init_cell = []
        super(UNet19, self).__init__()

        from torch.nn import BatchNorm2d as BatchNorm
        models.BatchNorm = BatchNorm

        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2, 2)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(
            ConvRelu2(3, 64, BatchNorm),
        )
        self.conv2 = nn.Sequential(
            self.pool,
            ConvRelu2(64, 128, BatchNorm)
        )
        self.conv3 = nn.Sequential(
            self.pool,
            ConvRelu4(128, 256, BatchNorm)
        )
        self.conv4 = nn.Sequential(
            self.pool,
            ConvRelu4(256, 512, BatchNorm)
        )
        self.conv5 = nn.Sequential(
            self.pool,
            ConvHole(512, 512, 2, BatchNorm)
        )

        self.dec5 = DecoderBlockV2(512, 1024, 512, is_deconv=False, BatchNorm=BatchNorm)
        self.dec4 = DecoderBlockV2(1024, 512, 512, is_deconv=False, BatchNorm=BatchNorm)
        self.dec3 = DecoderBlockV2(1024, 512, 256, is_deconv=False, BatchNorm=BatchNorm)
        self.dec2 = DecoderBlockV2(512, 256, 128, is_deconv=False, BatchNorm=BatchNorm)
        self.dec1 = DecoderBlockV2(256, 128, 64, is_deconv=False, BatchNorm=BatchNorm)
        self.dec0 = ConvRelu(128, 64, BatchNorm)
        self.dec = ConvRelu(64, 64, BatchNorm)

        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

        self.init_cell.append(self.dec5)
        self.init_cell.append(self.dec4)
        self.init_cell.append(self.dec3)
        self.init_cell.append(self.dec2)
        self.init_cell.append(self.dec1)
        self.init_cell.append(self.dec0)
        self.init_cell.append(self.dec)
        self.init_cell.append(self.final)

        for list in self.init_cell:
            for m in list.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                    init.xavier_normal(m.weight.data)
                    if m.bias is not None:
                        init.constant(m.bias.data, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, SynchronizedBatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        center = self.dec5(self.pool(conv5)) #512 1024 512

        dec5 = self.dec4(torch.cat([center, conv5], 1)) #1024 512 512

        dec4 = self.dec3(torch.cat([dec5, conv4], 1))  #1024 512 256
        dec3 = self.dec2(torch.cat([dec4, conv3], 1))  #512 256 128
        dec2 = self.dec1(torch.cat([dec3, conv2], 1))  #256 128 64
        dec1 = self.dec0(torch.cat([dec2, conv1], 1))   #128 64 64

        dec0 = self.dec(dec1)

        x_out = F.log_softmax(self.final(dec0), dim=1)
        return x_out
