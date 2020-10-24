"""
The implementation of GHM-C and GHM-R losses.
Details can be found in the paper `Gradient Harmonized Single-stage Detector`:
https://arxiv.org/abs/1811.05181
Copyright (c) 2018 Multimedia Laboratory, CUHK.
Licensed under the MIT License (see LICENSE for details)
Written by Buyu Li
"""

import torch
import torch.nn.functional as F
import torch.nn as nn

import os
debug_file = './tmp/debug'#怎么写到config里



class GHMC_Loss(nn.Module):
    def __init__(self, bins=10, momentum=0):
        self.bins = bins  # 份数
        self.momentum = momentum
        self.edges = [float(x) / bins for x in range(bins+1)]  # 0.1 0.2...1
        self.edges[-1] += 1e-6  # 1.000001
        if momentum > 0:  # 【】=0呢？
            self.acc_sum = [0.0 for _ in range(bins)]  # 初始化，存啥的

    def forward(self, input, target, mask):
        """ Args:
        input [batch_num, class_num]:
            The direct prediction of classification fc layer.
        target [batch_num, class_num]:
            Binary target (0 or 1) for each sample each class. The value is -1
            when the sample is ignored.
        """
        edges = self.edges
        mmt = self.momentum
        weights = torch.zeros_like(input)

        # gradient length
        g = torch.abs(input.sigmoid().detach() - target)

        valid = mask > 0
        tot = max(valid.float().sum().item(), 1.0)
        n = 0  # n valid bins
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i+1]) & valid
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] \
                        + (1 - mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
                n += 1
        if n > 0:
            weights = weights / n

        loss = F.binary_cross_entropy_with_logits(
            input, target, weights, reduction='sum') / tot
        return loss

# class GHMC_Loss(nn.Module):
#     def __init__(self, bins=10, momentum=0):
#         self.bins = bins  # 份数
#         self.momentum = momentum
#         self.edges = [float(x) / bins for x in range(bins+1)]  # 0.1 0.2...1
#         self.edges[-1] += 1e+3 #【】怎么划分梯度呢 统计一下
#         if momentum > 0:  # 【】=0呢？
#             self.acc_sum = [0.0 for _ in range(bins)]  #存带动量的β

#     def calc(self, input, target, mask):
#         """ Args:
#         input [batch_num, class_num]:
#             The direct prediction of classification fc layer.
#         target [batch_num, class_num]:
#             Binary target (0 or 1) for each sample each class. The value is -1
#             when the sample is ignored.
#         """
#         edges = self.edges
#         mmt = self.momentum
#         weights = torch.zeros_like(input)

#         # gradient length
#         g = torch.abs(input.sigmoid().detach() - target)#应该是总loss

#         # valid = mask > 0
#         tot = input.size(0)
#         n = 0  # n valid bins
#         for i in range(self.bins):
#             inds = (g >= edges[i]) & (g < edges[i+1])
#             num_in_bin = inds.sum().item()#落在这一梯度范围的所有样本个数
#             if num_in_bin > 0:
#                 if mmt > 0:
#                     self.acc_sum[i] = mmt * self.acc_sum[i] \
#                         + (1 - mmt) * num_in_bin
#                     weights[inds] = tot / self.acc_sum[i]
#                 else:
#                     weights[inds] = tot / num_in_bin
#                 n += 1
#         if n > 0:
#             weights = weights / n

#         loss = F.binary_cross_entropy_with_logits(
#             input, target, weights, reduction='sum') / tot
#         return loss

# class GHMR_Loss(nn.Module):
#     def __init__(self, mu=0.02, bins=10, momentum=0):
#         super().__init__()
#         self.mu = mu
#         self.bins = bins
#         self.edges = [float(x) / bins for x in range(bins+1)]
#         self.edges[-1] = 1e3  # 怎么不是+=了？:是让所有点都落在这个范围内
#         self.momentum = momentum
#         if momentum > 0:
#             self.acc_sum = [0.0 for _ in range(bins)]

    # def calc(self, input, target):
    #     """ Args:
    #     input [batch_num, 4 (* class_num)]:
    #         The prediction of box regression layer. Channel number can be 4 or
    #         (4 * class_num) depending on whether it is class-agnostic.
    #     target [batch_num, 4 (* class_num)]:#batch,通道x？
    #         The target regression values with the same size of input.
    #     """
    #     mu = self.mu
    #     edges = self.edges
    #     mmt = self.momentum
    #     # if os.path.exists(debug_file):
    #     #     import ipdb
    #     #     ipdb.set_trace()
    #     # ASL1 loss
    #     diff = input - target
    #     loss = torch.sqrt(diff * diff + mu * mu) - mu

    #     # gradient length
    #     g = torch.abs(diff / torch.sqrt(mu * mu + diff * diff)).detach()  # 一个batch的所有梯度
    #     weights = torch.zeros_like(g)  # β

    #     # valid = mask > 0  # 【】mask怎么来
    #     # tot = max(mask.float().sum().item(), 1.0)  # total总样本数，至少要大于1
    #     tot = input.size(0)*input.size(1)
    #     # tot = 1.0 #【see】
    #     n = 0  # n: valid bins#总有效样本数
    #     for i in range(self.bins):
    #         inds = (g >= edges[i]) & (g < edges[i+1])  #此处梯度是否属于该bin
    #         num_in_bin = inds.sum().item()#落在第i个bin里的样本数
    #         if num_in_bin > 0:
    #             n += 1
    #             if mmt > 0:#带动量的话就指数平均
    #                 self.acc_sum[i] = mmt * self.acc_sum[i] \
    #                     + (1 - mmt) * num_in_bin
    #                 weights[inds] = tot / self.acc_sum[i]#更合理的β [10]
    #             else:
    #                 weights[inds] = tot / num_in_bin #【see】梯度在这一bin的才赋值，也就是该样本的β
    #     if n > 0:
    #         weights /= n

    #     loss = loss * weights #【】不该元素乘积吗
    #     loss = loss.sum() / tot
    #     return loss
class GHMR_Loss(nn.Module):
    def __init__(self, mu=0.02, bins=10, momentum=0):
        super().__init__()
        self.mu = mu
        self.bins = bins
        self.edges = [float(x) / bins for x in range(bins+1)]
        self.edges[-1] = 1e3  # 怎么不是+=了？:是让所有点都落在这个范围内
        self.momentum = momentum
        if momentum > 0:
            self.acc_sum = [0.0 for _ in range(bins)]

    def calc(self, input, target):
        """ Args:
        input [batch_num, 4 (* class_num)]:
            The prediction of box regression layer. Channel number can be 4 or
            (4 * class_num) depending on whether it is class-agnostic.
        target [batch_num, 4 (* class_num)]:#batch,通道x？
            The target regression values with the same size of input.
        """
        mu = self.mu
        edges = self.edges
        mmt = self.momentum
        # if os.path.exists(debug_file):
        #     import ipdb
        #     ipdb.set_trace()
        # ASL1 loss
        diff = input - target
        loss = torch.sqrt(diff * diff + mu * mu) - mu
        loss = torch.sum(loss,1)

        # gradient length
        g = torch.abs(diff / torch.sqrt(mu * mu + diff * diff)).detach()  # 一个batch的所有梯度
        g = torch.sum(g,1) # 一个batch的所有梯度
        # print(g.size())
        weights = torch.zeros_like(g)  # β
        g/=(g.max()-g.min())

        # valid = mask > 0  # 【】mask怎么来
        # tot = max(mask.float().sum().item(), 1.0)  # total总样本数，至少要大于1
        tot = input.size(0)
        # tot = 1.0 #【see】
        n = 0  # n: valid bins#总有效样本数
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i+1])  #此处梯度是否属于该bin
            num_in_bin = inds.sum().item()#落在第i个bin里的样本数
            if num_in_bin > 0:
                n += 1
                if mmt > 0:#带动量的话就指数平均
                    self.acc_sum[i] = mmt * self.acc_sum[i] \
                        + (1 - mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]#更合理的β [10]
                else:
                    weights[inds] = tot / num_in_bin #【see】梯度在这一bin的才赋值，也就是该样本的β
        if n > 0:
            weights /= n

        loss = loss * weights / tot #不该元素乘积吗:这就是
        loss = loss.sum() /64/4096 #【see】
        return loss
