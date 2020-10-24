'''
通道注意力
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
from torch import nn
from torch.nn import functional as F

from .utils import (
    relu_fn,
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv2d,
    get_model_params,
    efficientnet_params,
    load_pretrained_weights,
)

import argparse
# import os
import pprint
import os
import logging

import torch.nn as nn
# 【c】怎么接上的，怎么换其他
BN_MOMENTUM = 0.1  # bn层的不是adam那个
logger = logging.getLogger(__name__)

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

class DwTrans(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,stride=2,padding=1,output_padding=0,bias=False):
        super(DwTrans,self).__init__()
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
        self.conv_1x1 = nn.Conv2d(in_channels,out_channels,kernel_size=1,bias=False)

    def forward(self,x):
        x = self.conv_group(x)
        x = self.conv_1x1(x)
        return x


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block

    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above

    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # skip connection and drop connect

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Expansion phase
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Output phase
        final_oup = self._block_args.output_filters
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = relu_fn(self._bn0(self._expand_conv(inputs)))
        x = relu_fn(self._bn1(self._depthwise_conv(x)))

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(relu_fn(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x


class EfficientNet(nn.Module):
    """
    An EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods

    Args:
        blocks_args (list): A list of BlockArgs to construct blocks
        global_params (namedtuple): A set of GlobalParams shared between blocks

    Example:
        model = EfficientNet.from_pretrained('efficientnet-b0')

    """

    def __init__(self, cfg,blocks_args=None, global_params=None,**kwargs):#不会被前面两个参数干扰了吧
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args
        #【see】
        # self.inplanes = 320#deconv前的通道数
        extra = cfg.MODEL.EXTRA
        self.deconv_with_bias = extra.DECONV_WITH_BIAS  # false
        self.relu = nn.ReLU(inplace=True)
        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(32, self._global_params)  # number of output channels
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self._global_params))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params))


        #【see】
        self.method = extra.LOSS_TYPE
        self.outplane_idx = extra.OUTPLANE_IDX
        # print(self.outplane_idx)
        self.deconv_layers2 = self._make_deconv_layer(
            extra.STAGE2.NUM_DECONV_LAYERS,  # 3
            extra.STAGE2.NUM_DECONV_FILTERS,  # 256 256 256
            extra.STAGE2.NUM_DECONV_KERNELS,  # 4 4 4
            extra.DECONV_INCHANNELS[0]
        )
        self.deconv_layers3 = self._make_deconv_layer(
            extra.STAGE3.NUM_DECONV_LAYERS,  # 3
            extra.STAGE3.NUM_DECONV_FILTERS,  # 256 256 256
            extra.STAGE3.NUM_DECONV_KERNELS,  # 4 4 4
            extra.DECONV_INCHANNELS[1]
        )
        # self. change_channel_2 = self._change_channel(extra.DECONV_INCHANNELS[0],256)
        # self. change_channel_3 = self._change_channel(extra.DECONV_INCHANNELS[1],256)
        self.deconv_layers4 = self._make_deconv_layer(
            extra.STAGE4.NUM_DECONV_LAYERS,  # 3
            extra.STAGE4.NUM_DECONV_FILTERS,  # 256 256 256
            extra.STAGE4.NUM_DECONV_KERNELS,  # 4 4 4
            extra.DECONV_INCHANNELS[2]
        )
        self.final_layer = nn.Conv2d(
            in_channels=extra.STAGE4.NUM_DECONV_FILTERS[-1],  # 256
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,  # 1
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0  # 0
        )
        # self.channel_att = SEBlock(cfg.MODEL.NUM_JOINTS,16)
        self.channel_att2 = SEBlock(16,1)
        self.channel_att3 = SEBlock(16,1)
        self.channel_att4 = SEBlock(16,1)

    def extract_features(self, inputs):
        """ Returns output of the final convolution layer """

        # Stem
        x = relu_fn(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            # if idx == 3:#右4
            #     x1 = x
            #【see】
            # print(x.shape)
            if idx == self.outplane_idx[0]:#3 分辨率32
                x2 = x
            if idx == self.outplane_idx[1]:#5 16
                x3 = x
            if idx == self.outplane_idx[2]:#15 8
                x4 = x
            #看看idx多长，是所有层数:15对
        # Head
        # x = relu_fn(self._bn1(self._conv_head(x)))【】这个要不要
        # print('idx is {}'.format(idx))
        return x2,x3,x4
        # Head
        # x = relu_fn(self._bn1(self._conv_head(x)))
    def _change_channel(self,inplane,outplane):
        change_channel = nn.Sequential(
        nn.Conv2d(
            in_channels=inplane,  # 【】把deconv改为resnet对应的通道数
            out_channels=outplane,
            kernel_size=1,  # 1
            stride=1,
            padding=0
        ),
        nn.BatchNorm2d(outplane,momentum=BN_MOMENTUM)
        )
        return change_channel


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

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels,in_channel):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):  # 只3个？
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)  # 返回4 1 0

            planes = num_filters[i]  # 256 256 256
            layers.append(
                nn.Sequential(
                DwTrans(
                    in_channels=in_channel,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias),
            nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)))
            in_channel = planes

        # return nn.Sequential(*layers)
        return nn.ModuleList(layers)#【see】改为列表

    def forward(self, inputs):
        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """

        # Convolution layers
        x2,x3,x4 = self.extract_features(inputs)
        # x2_ = self.change_channel_2(x2)
        # x3_ =self.change_channel_3(x3)#左
        xd2 = self.deconv_layers2[0](x2)  #【】+xd4吗还是xd3y;用xd3_first吗不用的话就x2
        xd3 = self.deconv_layers3[0](x3)#这儿是x3不是x2；加x2吗？
        xd3 = self.deconv_layers3[1](xd3)#【y】+xd4吗
        xd4 = self.deconv_layers4[0](x4)#
        xd4 = self.deconv_layers4[1](xd4)  #因为d4刚经过了relu值会比较小
        xd4 = self.deconv_layers4[2](xd4)#【】加x1吗

        if self.method == 'zhongji_loss':
            out = []
            out.append(xd2)
            # out.append(xd3)
            # out.append(xd4)
            for i in range(len(out)):
                # print(out[i].shape)
                out[i] = self.final_layer(out[i])
                # out[i] = self.channel_att(out[i])#要不要加？通道的不，空间的要
            out[0] = self.channel_att2(out[0])
            # out[1] = self.channel_att3(out[1])
            # out[2] = self.channel_att4(out[2])
            return out
        elif self.method == 'sum_loss':
            x = (x2+x3+x4)/3
            x = self.final_layer(x)  # 1x1
            return x

        # # Pooling and final linear layer
        # x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        # if self._dropout:
        #     x = F.dropout(x, p=self._dropout, training=self.training)
        # x = self._fc(x)
        # return x

    def init_weights(self, deconvpretrained=True):
        if deconvpretrained:#是否要对后面的deconv进行与训练
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


        else:
            raise NotImplementedError

    @classmethod
    def from_name(cls, model_name,cfg, override_params=None):#【】cls是啥？
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        #【see】在这儿写传入的deconv的参数
        return EfficientNet(cfg,blocks_args, global_params)

    @classmethod
    def from_pretrained(cls, model_name,cfg, num_classes=100):
        model = EfficientNet.from_name(model_name,cfg, override_params={'num_classes': num_classes})
        load_pretrained_weights(model, model_name, load_fc=(num_classes == 1000))
        return model

    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name, also_need_pretrained_weights=False):
        """ Validates model name. None that pretrained weights are only available for
        the first four models (efficientnet-b{i} for i in 0,1,2,3) at the moment. """
        num_models = 4 if also_need_pretrained_weights else 8
        valid_models = ['efficientnet_b'+str(i) for i in range(num_models)]
        if model_name.replace('-','_') not in valid_models:
            raise ValueError('model_name should be one of: ' + ', '.join(valid_models))
