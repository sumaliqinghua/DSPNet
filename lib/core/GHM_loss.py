import torch
from torch import nn
import torch.nn.functional as F


class GHM_Loss(nn.Module):
    def __init__(self, bins, alpha):
        super(GHM_Loss, self).__init__()
        self._bins = bins  # 份数 10
        self._alpha = alpha  # 【】动量参数；取多少
        self._last_bin_count = None  # sj

    def _g2bin(self, g):  # g属于哪个ind区
        # g从哪儿来？为啥要减0.0001再向下取整：防止超过范围，但是g小于1吗
        return torch.floor(g * (self._bins - 0.0001)).long()  # 总份数x梯度=indx

    def _custom_loss(self, x, target, weight):
        raise NotImplementedError  # 怎么算

    def _custom_loss_grad(self, x, target):
        raise NotImplementedError

    def forward(self, x, target):
        g = torch.abs(self._custom_loss_grad(x, target)).detach()  # gi 梯度

        bin_idx = self._g2bin(g)  # 【see】bin_idx是整个batch中各自的；看看size

        bin_count = torch.zeros((self._bins))  # 一维0向量
        for i in range(self._bins):
            bin_count[i] = (bin_idx == i).sum().item()  # ind落在这区间的梯度有多少个

        N = (x.size(0) * x.size(1))  # 【】 batch x channel 为啥要乘通道数呢？

        if self._last_bin_count is None:
            self._last_bin_count = bin_count  # 这个是t-1时刻的sj 改良的Rj
        else:
            bin_count = self._alpha * self._last_bin_count + \
                (1 - self._alpha) * bin_count
            self._last_bin_count = bin_count  # t时刻的sj

        nonempty_bins = (bin_count > 0).sum().item()

        gd = bin_count * nonempty_bins  # GD 一维向量
        gd = torch.clamp(gd, min=0.0001)  # 防止为0#那gd也是10
        beta = N / gd  # 权值
        print(gd)#gd这么高？
        print(sum(bin_idx))#和0
        # print(beta,bin_idx.size())#[10]:[64,4096]:全0;
        print(sum(bin_count))#261270
        # print(N,nonempty_bins)#262144 7 N怎么这么大

        # return self._custom_loss(x, target, beta[bin_idx]) #【c】beta.size() bin_idx
        return self._custom_loss(x, target, beta) #【c】beta.size() bin_idxj


class GHMR_Loss(GHM_Loss):
    def __init__(self, bins, alpha, mu):
        super(GHMR_Loss, self).__init__(bins, alpha)
        self._mu = mu  # asl的参数

    def _custom_loss(self, x, target, weight):
        d = x - target
        mu = self._mu
        loss = torch.sqrt(d * d + mu * mu) - mu
        N = x.size(0) * x.size(1)
        return (loss * weight).sum() / N

    def _custom_loss_grad(self, x, target):
        d = x - target
        mu = self._mu
        return d / torch.sqrt(d * d + mu * mu)

class GHM_MSE(GHM_Loss):
    def __init__(self, bins, alpha):
        super(GHM_MSE, self).__init__(bins, alpha)

    def _custom_loss(self, x, target,weight):
        print(x.size())#[64,4096]
        x = x*weight
        return nn.MSELoss(reduction="mean")*weight

    def _custom_loss_grad(self, x, target):
        d = x - target
        return 2*d
