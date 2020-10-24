# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn

BN_MOMENTUM = 0.1  # 【】bn层都用这个：init的才用，干嘛的
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1  # 【c】这啥

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)  # 【c】看看每个stage的输入输出
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)  # 通道不变
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride  # 【c】这还有啥用

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual  # 分辨率通道数均一样
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4  # 【】通道扩展倍数；为啥是4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=1, bias=False)  # 【l】什么时候不用bias
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
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

        out += residual  # 【】通道数也要一致吗
        out = self.relu(out)

        return out


class IncepHRModule(nn.Module):

    def __init__(self, num_blocks, block, cur_stage, original_channel, multi_scale_output=True):
        super().__init__()
        # stage == branch num_blocks[前后blocks] original_channel n
        self.multi_scale_output = multi_scale_output  # 是否要多级融合
        self.block = block  # 【】block怎么传的
        self.cur_stage = cur_stage  # 当前
        self.num_blocks = num_blocks
        self.cur_channel = 2**(cur_stage-1) * \
            original_channel  # stage2 2n 3 4n 4 8n
        self.middle_channels = [self.cur_channel//2]
        self.relu = nn.ReLU(inplace=True)
        for i in range(self.cur_stage-1):
            self.middle_channels.append(self.cur_channel)  # 尾部最后的
        self.out_channels = [i*2 for i in self.middle_channels]
        self.out_channels.append(2*self.cur_channel)
        # 前段
        self.stem_layers_first = self._make_stem(
            self.block, self.num_blocks[0])
        self.expand_layers_middle = self._expand_layers(
            self.cur_channel, self.cur_stage, Is_middle=True)
        self.fuse_layer_middle = self._make_fuse_layers(self.middle_channels)
        # 后半段
        self.stem_layers_second = self._make_stem(
            self.block, self.num_blocks[-1])
        self.expand_layers_tail = self._expand_layers(
            self.cur_channel, self.cur_stage, Is_middle=False)
        self.fuse_layer_tail = self._make_fuse_layers(self.out_channels,self.multi_scale_output)
        self.change_channel_middle = self._change_channel_layers(
            self.middle_channels, 2)
        self.change_channel_tail = self._change_channel_layers(
            self.out_channels, 1)
        # self.change_channel_stem = nn.Sequential(nn.Conv2d(
        #     4*self.cur_channel, cur_channel,  # 【】4n时都改为 4*cur
        #     1, 1, 0, bias=False),
        #     nn.BatchNorm2d(cur_channel),
        #     nn.ReLU(inplace=True)
        # )

    def forward(self, x):  # x是个列表
        x_head = x  # head 混后
        # print(self.cur_stage,len(x))
        # for i in range(len(x)):
        #     print(x[i].shape)
        x = self.stem_layers_first(x[-1])
        x_expand_middle = []
        for i in range(self.cur_stage):  # 将末节点生成中间分叉
            if i == self.cur_stage-1:
                y = x
            else:
                # print(self.cur_stage,i)
                # print(self.expand_layers_middle[i](x).shape,x_head[i].shape)
                # 这个expand is_middle=true【】 中间层加head吗
                y = self.relu(self.expand_layers_middle[i](x))+x_head[i]
            x_expand_middle.append(y)#【】xhead已经relu过了这样好吗

        x_middle_fuse = []
        for i in range(len(self.fuse_layer_middle)):
            y = x_expand_middle[0] if i == 0 else self.fuse_layer_middle[i][0](
                x_expand_middle[0])  # 由前面j第0层汇聚到 第0层的
            for j in range(1, self.cur_stage):
                if i == j:
                    y = y + x_expand_middle[j]  # 是同一层就直接+其他各层要进行相应的变换（上/下采样）
                else:  # 【see】i代表汇聚到第i层，j代表参与汇聚的各分支
                    y = y + self.fuse_layer_middle[i][j](x_expand_middle[j])
            x_middle_fuse.append(self.relu(y))  # 得到融合了的各分支的结果
        x = self.stem_layers_second(x_middle_fuse[-1])

        x_expand_tail = []
        for i in range(self.cur_stage+1):  # 有一个下采样层
            if i == self.cur_stage-1:
                y = x
            else:
                y = self.expand_layers_tail[i](x)  # 这个expand is_middle=false
            x_expand_tail.append(y)  # 末层分叉 先不relu【】cat前relu还是后还是都

        x_tail = []  # 【】会不会耗显存？不造
        for i in range(len(x_head)):  # 将首节点cat给末尾
            x = torch.cat((x_expand_tail[i], x_head[i]), 1)
            x_tail.append(self.relu(x))  # 【】cat后relu 不加bn?
        x_tail.append(self.relu(x_expand_tail[-1]))  # 尾部多一节点

        for i in range(len(x_tail)):  # 1x1
            # fuse前要relu 1x1里没有relu
            x_tail[i] = self.relu(self.change_channel_tail[i](
                x_tail[i]))  # 【】要不要1x1还是单独bn relu 马上fuse 这句还有没有必要？

        x_tail_fuse = []
        for i in range(len(self.fuse_layer_tail)):
            y = x_tail[0] if i == 0 else self.fuse_layer_tail[i][0](
                x_tail[0])  # 汇聚到第i层的第0个分支
            for j in range(1, self.cur_stage+1):
                if i == j:
                    y = y + x_tail[j]  # 是同一层就直接+其他各层要进行相应的变换（上/下采样）
                else:  # i前j后
                    y = y + self.fuse_layer_tail[i][j](x_tail[j])
            # 得到融合了的各分支的结果 保留relu 在1x1那里去掉relu
            x_tail_fuse.append(self.relu(y))

        for i in range(len(x_tail_fuse)):  # 1x1
            # fuse后relu了 1x1里没有relu
            x_tail_fuse[i] = self.change_channel_tail[i](
                x_tail_fuse[i])  # 要和middle加所以1x1去掉relu

        x_head = []  # 节省显存
        len_middel_sum_tail = len(x_middle_fuse) if self.multi_scale_output==True else 1#最后一个stg只有一个分支
        for i in range(len_middel_sum_tail):
            x = self.change_channel_middle[i](
                x_middle_fuse[i])+x_tail_fuse[i]
            x_head.append(self.relu(x))
        # 两个for可以合并 没合并时上一步是1x1所以也要relu
        if self.multi_scale_output==True:#如果是最后一个stg的就不用下支路了
            x_head.append(self.relu(x_tail_fuse[-1]))

        return x_head  # 其实返回的是尾部

    def _make_fuse_layers(self, in_channels,multi_scale_output=True):
        num_branches = len(in_channels)  # 【see】
        if num_branches == 1:
            return None  # 单分支不用融合
        num_inchannels = in_channels
        fuse_layers = []
        for i in range(num_branches if multi_scale_output==True else 1):
            # 如果不多级输出的话只要第一层（结尾那儿）;不断遍历i层
            fuse_layer = []  # 【】再过一下这个层？
            for j in range(num_branches):  # 不断遍历其余各层融合到i层里
                if j > i:  # i下面的层：转换i通道（1x1），分辨率（upsample）为j的
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_inchannels[j],
                                num_inchannels[i],
                                1, 1, 0, bias=False  # 核大小，步长，pad；
                            ),
                            nn.BatchNorm2d(num_inchannels[i]),  # 【c】这个没用moment
                            # 【see】可以换模式不，双线性插值
                            nn.Upsample(scale_factor=2**(j-i), mode='nearest')
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)  # 自己就不加了，在forward里加的
                else:  # i上面的层
                    conv3x3s = []
                    for k in range(i-j):  # 第j层通过多个conv来降采样到i层分辨率其中最后一个conv输出通道与i层一致（前面的与j一致）
                        if k == i - j - 1:  # 最后一组
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,  # 输出通道为第i层的
                                        3, 2, 1, bias=False  # 核大小，步长，pad；
                                    ),
                                    # 【c】这个没用moment
                                    nn.BatchNorm2d(num_outchannels_conv3x3)
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,  # 输入输出通道相等
                                        3, 2, 1, bias=False
                                    ),
                                    # 【c】这个没用moment
                                    nn.BatchNorm2d(num_outchannels_conv3x3),
                                    nn.ReLU(inplace=True)
                                )
                            )
                    # 第一个else结束第一个for结束
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def _change_channel_layers(self, in_channels, scale_factor):  # 中间层通道x2
        out_list = []
        for i in range(len(in_channels)):
            in_channel = in_channels[i]
            out_channel = scale_factor*in_channel
            out_list.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channel,
                        out_channel,
                        1, 1, 0, bias=False  # 核大小，步长，pad；
                    ),
                    nn.BatchNorm2d(out_channel),  # 【c】这个没用moment
                    # nn.ReLU(True)  #  中部和尾部加 所以不要relu
                ))
        return nn.ModuleList(out_list)

    def _expand_layers(self, cur_channel, cur_stage, Is_middle=False):
        # 分支数 固定通道n数 当前stage数
        # stem传来的单个节点 通道数n
        expand_layers = []
        # 上采样
        for j in range(cur_stage-1):  # stage2 1层上
            up_channel = cur_channel//2 if j == 0 else cur_channel  # 顶层通道是当前的一半
            expand_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        cur_channel,
                        up_channel,
                        1, 1, 0, bias=False  # 核大小，步长，pad；
                    ),
                    nn.BatchNorm2d(up_channel),  # 【c】这个没用moment
                    # 【see】可以换模式不，双线性插值
                    nn.Upsample(scale_factor=2 **
                                (cur_stage-1-j), mode='nearest')
                )
            )
        # 当前层 1x1?还是传前面的
        expand_layers.append(None)  # 当为空时传前面的x
        if(Is_middle == False):
            # 下采样
            down_channel = cur_channel*2
            expand_layers.append(nn.Sequential(
                nn.Conv2d(
                    cur_channel,
                    down_channel,  # 输入输出通道相等
                    3, 2, 1, bias=False
                ),
                # 【c】这个没用moment
                nn.BatchNorm2d(down_channel)
                # nn.ReLU(True) #【】加relu吗,M:不加在forward里统一加的
            ))
        return nn.ModuleList(expand_layers)

    def _make_stem(self, block, num_block, stride=1):  # 【see】要用两次前部和后部
        bottlten_input = self.cur_channel
        bottlen_output = bottlten_input
        downsample = None
        if stride != 1 or \
           bottlten_input != bottlen_output * block.expansion:  # 【】如果分叉那儿用n的话;也就是说步长为1时且前后通道数一致时不降采样
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.bottlten_input,
                    bottlen_output * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),  # 【see】使步长通道数与主路一致
                nn.BatchNorm2d(
                    bottlen_output * block.expansion,
                    momentum=BN_MOMENTUM
                ),
            )

        layers = []  # 接block
        layers.append(
            block(
                bottlten_input,  # 【】输入全是这个？那这个也是1x1：对
                bottlen_output,  # 【】不是有个啥膨胀4倍?
                stride,
                downsample,
            )
        )
        for i in range(1, num_block):  # 该分支后面的block
            layers.append(
                block(
                    bottlen_output,  # 【】输入全是这个？那这个也是1x1：对
                    bottlen_output
                )  # 【c】basicblock
            )
        return nn.Sequential(*layers)

blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}  # 调用类


class IncepHRNet(nn.Module):

    def __init__(self, cfg, **kwargs):
        # self.original_channel = cfg.MODEL
        extra = cfg.MODEL.EXTRA  # 【c】干嘛的
        super(IncepHRNet, self).__init__()
        self.inplanes = 64
        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 4)  # 居然只用了一次

        self.stage2_cfg = cfg['MODEL']['EXTRA']['STAGE2']
        self.original_channel =self.stage2_cfg['ORIGINAL_CHANNEL']
        self.multi_scale_output =self.stage2_cfg['MULTI_SCALE_OUTPUT']
        self.block =blocks_dict[self.stage2_cfg['BLOCK']]
        self.stage1_make_first = self.stage1_make(self.original_channel)

        self.num_blocks =self.stage2_cfg['NUM_BLOCKS']
        self.cur_stage =self.stage2_cfg['CUR_STAGE']
        self.stage2 = IncepHRModule(self.num_blocks,self.block,self.cur_stage,self.original_channel,self.multi_scale_output)

        self.stage3_cfg = cfg['MODEL']['EXTRA']['STAGE3']
        self.num_blocks =self.stage3_cfg['NUM_BLOCKS']
        self.cur_stage =self.stage3_cfg['CUR_STAGE']
        self.stage3 = IncepHRModule(self.num_blocks,self.block,self.cur_stage,self.original_channel,self.multi_scale_output)

        self.stage4_cfg = cfg['MODEL']['EXTRA']['STAGE4']
        self.num_blocks =self.stage4_cfg['NUM_BLOCKS']
        self.cur_stage =self.stage4_cfg['CUR_STAGE']
        self.stage4 = IncepHRModule(self.num_blocks,self.block,self.cur_stage,self.original_channel,False)

        self.final_layer = nn.Conv2d(
            in_channels=8*self.original_channel,  #【】
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,  # 1
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )

        self.pretrained_layers = cfg['MODEL']['EXTRA']['PRETRAINED_LAYERS']

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)
        x_list = []
        for i in range(len(self.stage1_make_first)):
            x_list.append(self.stage1_make_first[i](x))

        x_list = self.stage2(x_list)
        x_list = self.stage3(x_list)
        x_list = self.stage4(x_list)

        x = self.final_layer(x_list[0])  # 1x1卷积变16通道

        return x

    def _make_layer(self, block, planes, blocks, stride=1):  # 【】和make one branch 有点像啊64,4
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion,
                               momentum=BN_MOMENTUM),  # 有mom
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            # 【c】self.inplanes=planes？
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def stage1_make(self,original_channel):
        stage1_list = []
        stage1_list.append(nn.Sequential(nn.Conv2d(
            256, original_channel,  # 上面为n
            1, 1, 0, bias=False),
            nn.BatchNorm2d(original_channel),
            nn.ReLU(inplace=True)
        ))
        stage1_list.append(nn.Sequential(nn.Conv2d(
            256, original_channel*2,  # 【】4n时都改为 4*cur
            3, 2, 1, bias=False),
            nn.BatchNorm2d(original_channel*2),
            nn.ReLU(inplace=True)))
        return nn.ModuleList(stage1_list)

    def init_weights(self, pretrained=''):  # 【c】怎么用的
            # 【see】把运行信息打印在log里
        logger.info('=> init weights from normal distribution')
        for m in self.modules():  # modules是哪个参数
            if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:  # 【c】为啥是in
                        nn.init.constant_(m.bias, 0)  # 有bias才设为0
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)  # 【】不能整个网络一起初始化？

        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))

            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers \
                or self.pretrained_layers[0] is '*':  # 'conv1''layer1'一类；*代表全部层都要复制。人为定义的
                    need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict,  # 继承自module所以有这个函数
                                strict=False)  # 把参数给有对应结构的网络
        elif pretrained:  # 【see】文件不存在，但pretrained字符串不为空时
            logger.error('=> please download pre-trained models first!')
            raise ValueError('{} is not exist!'.format(pretrained))


def get_pose_net(cfg, is_train, **kwargs):  # 【c】哪儿用了这个函数？
    model = IncepHRNet(cfg, **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)

    return model
