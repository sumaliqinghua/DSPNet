# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------
'''
tongdaozhuyili
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
# import os
import pprint
import os
import logging

import torch
import torch.nn as nn
# 【c】怎么接上的，怎么换其他
BN_MOMENTUM = 0.1  # bn层的不是adam那个
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False
    )


class Depthwise(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False):
        super(Depthwise, self).__init__()
        self.conv_group = nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size, stride=stride,
                                    padding=padding, groups=in_planes, bias=bias)
        self.conv_1x1 = nn.Conv2d(
            in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv_group(x)
        x = self.conv_1x1(x)
        return x


class DwTrans(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=0, bias=False):
        super(DwTrans, self).__init__()
        self.in_channels = in_channels
        self.conv_group = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=in_channels,
            bias=False
        )
        self.conv_1x1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        # print(self.in_channels)
        x = self.conv_group(x)
        x = self.conv_1x1(x)
        return x

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fcs = nn.Sequential(nn.Linear(channel, int(channel/reduction)),
                                 nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                 nn.Linear(int(channel/reduction), channel),
                                 nn.Sigmoid())

    def forward(self, x):
        bahs, chs, _, _ = x.size()
        # Returns a new tensor with the same data as the self tensor but of a different size.
        y = self.avg_pool(x).view(bahs, chs)
        y = self.fcs(y).view(bahs, chs, 1, 1)
        return torch.mul(x, y)

class SSEBlock(nn.Module):
    def __init__(self, inchannel):
        super(SSEBlock, self).__init__()

        self.spatial_se = nn.Sequential(nn.Conv2d(inchannel, 1, kernel_size=1,
                                                stride=1, padding=0, bias=False),
                                        nn.Sigmoid())

    def forward(self, x):
        # Returns a new tensor with the same data as the self tensor but of a different size.
        spa_se = self.spatial_se(x)#权重
        # spa_se = torch.mul(x, spa_se)#乘上权重后的结果；各自独立的话就要这一句
        return spa_se


class BasicBlock(nn.Module):
    expansion = 1  # 【c】哪儿用了

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride  # 【c】哪儿用了

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


class Bottleneck(nn.Module):
    expansion = 4  # 为啥是4，为啥要写出来;不是2？

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)  # 【see】
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class MultiResNet(nn.Module):

    def __init__(self, block, layers, cfg, **kwargs):  # **kwargs代表什么含义
        self.inplanes = 64
        extra = cfg.MODEL.EXTRA
        self.deconv_with_bias = extra.DECONV_WITH_BIAS  # false
        self.method = extra.LOSS_TYPE
        super(MultiResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)  # 【see】
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1)
        # 【】这个换成conv
        self.layer1 = self._make_layer(block, 64, layers[0])  # 3
        # 从2开始每个layer的第一个bottlen分辨率下降2
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # 4
        self.deconv_layers2 = self._make_deconv_layer(
            extra.STAGE2.NUM_DECONV_LAYERS,  # 3
            extra.STAGE2.NUM_DECONV_FILTERS,  # 256 256 256
            extra.STAGE2.NUM_DECONV_KERNELS,  # 4 4 4
        )
        # self.change_channel_2 =nn.Conv2d(
        #     in_channels=512,  # 【】把deconv改为resnet对应的通道数
        #     out_channels=256,
        #     kernel_size=1,  # 1
        #     stride=1,
        #     padding=0
        # )
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # 6
        self.deconv_layers3 = self._make_deconv_layer(
            extra.STAGE3.NUM_DECONV_LAYERS,  # 3
            extra.STAGE3.NUM_DECONV_FILTERS,  # 256 256 256
            extra.STAGE3.NUM_DECONV_KERNELS,  # 4 4 4
        )
        # self.change_channel_3 =nn.Conv2d(#参数量太大了：只有1M
        #     in_channels=1024,  # 【】把deconv改为resnet对应的通道数
        #     out_channels=256,
        #     kernel_size=1,  # 1
        #     stride=1,
        #     padding=0
        # )
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # 3
        self.deconv_layers4 = self._make_deconv_layer(
            extra.STAGE4.NUM_DECONV_LAYERS,  # 3
            extra.STAGE4.NUM_DECONV_FILTERS,  # 256 256 256
            extra.STAGE4.NUM_DECONV_KERNELS,  # 4 4 4
        )
        # self.sse_attention = SSEBlock(256)
        # used for deconv layers

        self.final_layer = nn.Conv2d(
            in_channels=256,  # 【】把deconv改为resnet对应的通道数
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,  # 1
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0  # 0
        )
        self.channel_att2 = SEBlock(16,1)
        self.channel_att3 = SEBlock(16,1)
        self.channel_att4 = SEBlock(16,1)



    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )  # 保持分辨率 通道一致

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
            # print(i,self.inplanes)

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1  # 【】这啥
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        in_channels = self.inplanes
        for i in range(num_layers):  # 只3个？
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)  # 返回4 1 0

            planes = num_filters[i]  # 256 256 256
            layers.append(
                nn.Sequential(
                DwTrans(
                    in_channels,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias),
            nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            # 新增后面的卷积
            # nn.Conv2d(256,256,3,1,1),
            # nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(256,256,3,1,1),
            # nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
            # nn.ReLU(inplace=True),
            ))
            in_channels = planes

        return nn.ModuleList(layers)  # 【see】改

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # print(self.inplanes)
        x1 = self.layer1(x)
        # print(x.shape)
        # print(self.inplanes)
        x2 = self.layer2(x1)
        # print(x.shape)
        # print(self.inplanes)
        x3 = self.layer3(x2)
        # print(x.shape)
        x4 = self.layer4(x3)  # 前4层就是一般的resnet结构？：是
        # print(x.shape,'over')

        # x4 = self.layer4(x3)#这句话就有13M参数量
        xd2 = self.deconv_layers2[0](x2)  # 跳接+x1

        # x2 = self.change_channel_2(x2)
        xd3_first = self.deconv_layers3[0](x3)
        # print(len(self.deconv_layers3),xd3_first.shape,xd2.shape)
        xd3 = self.deconv_layers3[1](xd3_first)

        # x3 =self.change_channel_3(x3)#deconv3用了之后再改通道
        xd4 = self.deconv_layers4[0](x4)
        xd4 = self.deconv_layers4[1](xd4)  # x2还是xd3_first
        xd4 = self.deconv_layers4[2](xd4)#还是xd3??

        #深指导浅 #【l】
        # xd4 = torch.mul(xd4,self.sse_attention(xd4))
        # xd3 = torch.mul(xd3,self.sse_attention(xd3))
        # xd2 = torch.mul(xd2,self.sse_attention(xd2))#顶部
        # #钱指导深 #【c】
        # xd4 = torch.mul(xd4,self.sse_attention(xd2))
        # xd3 = torch.mul(xd3,self.sse_attention(xd2))
        # xd2 = torch.mul(xd2,self.sse_attention(xd2))#顶部

        if self.method == 'zhongji_loss':
            out = []
            out.append(xd2)
            out.append(xd3)
            out.append(xd4)
            for i in range(len(out)):
                out[i] = self.final_layer(out[i])
            out[0] = self.channel_att2(out[0])
            out[1] = self.channel_att3(out[1])
            out[2] = self.channel_att4(out[2])
            return out
        elif self.method == 'sum_loss':
            x = xd4+xd3+xd2
            x = self.final_layer(x)  # 1x1
            return x

    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            logger.info('=> init deconv2 weights from normal distribution')
            for name, m in self.deconv_layers2.named_modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.normal_(m.weight, std=0.001)
                    logger.info(
                        '=> init {}.weight as normal(0, 0.001)'.format(name))
                if isinstance(m, nn.ConvTranspose2d):
                    logger.info(
                        '=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    logger.info('=> init {}.weight as 1'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

            logger.info('=> init deconv3 weights from normal distribution')
            for name, m in self.deconv_layers3.named_modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.normal_(m.weight, std=0.001)
                if isinstance(m, nn.ConvTranspose2d):
                    logger.info(
                        '=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    logger.info('=> init {}.weight as 1'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

            logger.info('=> init deconv4 weights from normal distribution')
            for name, m in self.deconv_layers4.named_modules():  # 【】有没有简便一点的对几个层都处理的
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.normal_(m.weight, std=0.001)
                if isinstance(m, nn.ConvTranspose2d):
                    logger.info(
                        '=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    logger.info('=> init {}.weight as 1'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

            logger.info('=> init final conv weights from normal distribution')
            for m in self.final_layer.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    logger.info(
                        '=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)

            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(
                pretrained))  # 初始化的是后半段，预训练的是resnet
            self.load_state_dict(pretrained_state_dict, strict=False)
        else:
            logger.info('=> init weights from normal distribution')
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.normal_(m.weight, std=0.001)
                    # nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        # 干嘛不直接共用上部分：上半部分只初始化了后半部分
                        nn.init.constant_(m.bias, 0)


resnet_spec = {
    18: (BasicBlock, [2, 2, 2, 2]),  # 18是键名
    34: (BasicBlock, [3, 4, 6, 3]),
    50: (Bottleneck, [3, 4, 6, 3]),  # 【】和34的差别在？Bottleneck？
    101: (Bottleneck, [3, 4, 23, 3]),
    152: (Bottleneck, [3, 8, 36, 3])
}

# 测试


def get_pose_net(cfg, is_train, **kwargs):
    num_layers = cfg.MODEL.EXTRA.NUM_LAYERS

    # 怎么对应上50的 Bottleneck是哪儿：下面一句
    block_class, layers = resnet_spec[num_layers]

    model = MultiResNet(block_class, layers, cfg, **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)  # 类的函数要调用了才执行

    return model
