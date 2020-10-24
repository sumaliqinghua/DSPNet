# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------
'''
gradnorm function
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import os

import numpy as np
import torch

from core.evaluate import accuracy
from core.inference import get_final_preds
from utils.transforms import flip_back
from utils.vis import save_debug_images


logger = logging.getLogger(__name__)


def train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, lossweight,writer_dict,):
    # print(lossweight,type(lossweight))
    batch_time = AverageMeter()  # 【l】AverageMeter计算平均值用
    data_time = AverageMeter()
    losses = AverageMeter()  # 一个epoch重置一次
    acc = AverageMeter()
    # lossweight = [1,1,1,1,1,1,1,1,1,1,1]#传进来吧
    alpha = config.MODEL.alpha
    n_tasks = config.MODEL.NUM_JOINTS
    # switch to train mode
    model.train()  # 【see】

    end = time.time()
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # 本函数后面的全在for里
        # 【c】train_loader在哪儿定义的
        # measure data loading time
        data_time.update(time.time() - end)  #什么作用，和batchtime有啥不同:读取时间

        # compute output
        input = input.cuda()  # 【c】自己加的
        outputs = model(input)  # 由loader而得的input

        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

#放在要debug的位置前
        # import os
        # debug_file = './tmp/debug'#怎么写到config里
        # if os.path.exists(debug_file):
        #     import ipdb;
        #     ipdb.set_trace()

        if isinstance(outputs, list):  # 模型输出是列表？：中继损失
            loss = criterion(outputs[0], target, target_weight)#如果是列表的话三分支对应不同的权重或者三个的append到一起
            for output in outputs[1:]:
                loss += criterion(output, target, target_weight)  # 【】对应位置相加
        else:
            output = outputs
            task_loss = criterion(output, target, target_weight)
            task_loss = torch.stack(task_loss)#[c]

        weighted_task_loss = torch.mul(lossweight, task_loss)
        if i == 0:
            # set L(0)
            initial_task_loss = task_loss.data
            initial_task_loss = initial_task_loss.cpu().numpy()  # 【】干嘛的
        loss = torch.sum(weighted_task_loss)# 权重loss和
        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        print(type(lossweight))
        if i != 0:
            lossweight.grad.data = lossweight.grad.data * 0.0

        # get layer of shared weights
        W = model.module.get_last_shared_layer()  #【】就是最后一层特征图finallayer前的

        # get the gradient norms for each of the tasks
        # G^{(i)}_w(t)
        norms = []
        for i in range(len(task_loss)):  # 所以是列表？之前是
                # get the gradient of this task loss with respect to the shared parameters
            gygw = torch.autograd.grad(
                task_loss[i], W.parameters(), retain_graph=True)
            # compute the norm
            norms.append(torch.norm(
                torch.mul(lossweight[i], gygw[0])))  # weights怎来
        norms = torch.stack(norms)  # 压成一个
        # print('G_w(t): {}'.format(norms))

        # compute the inverse training rate r_i(t)
        # \curl{L}_i
        if torch.cuda.is_available():
            loss_ratio = task_loss.data.cpu().numpy() / initial_task_loss #【】为啥用cpu
        else:
            loss_ratio = task_loss.data.numpy() / initial_task_loss
            # r_i(t)
        inverse_train_rate = loss_ratio / np.mean(loss_ratio)
        # print('r_i(t): {}'.format(inverse_train_rate))

        # compute the mean norm \tilde{G}_w(t)
        if torch.cuda.is_available():
            mean_norm = np.mean(norms.data.cpu().numpy())
        else:
            mean_norm = np.mean(norms.data.numpy())
            # print('tilde G_w(t): {}'.format(mean_norm))

            # compute the GradNorm loss
            # this term has to remain constant
        # constant_term = torch.tensor(
            # mean_norm * (inverse_train_rate ** alpha), requires_grad=False)
        constant_term = torch.tensor(
            mean_norm * (inverse_train_rate ** alpha), requires_grad=False)
        if torch.cuda.is_available():
            constant_term = constant_term.cuda()
            # print('Constant term: {}'.format(constant_term))
            # this is the GradNorm loss itself
        # grad_norm_loss = torch.tensor(
        #     torch.sum(torch.abs(norms - constant_term)),requires_grad=True)
        grad_norm_loss = torch.sum(torch.abs(norms - constant_term))#[]gai
        # print('GradNorm loss {}'.format(grad_norm_loss))

        # compute the gradient for the weights
        print(grad_norm_loss.requires_grad,lossweight.requires_grad)
        lossweight.grad = torch.autograd.grad(
            grad_norm_loss, lossweight)[0]

        # lossweight是全局？
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))  # 添加损失和输入batch
        # 训练时一个batch就计算了下acc
        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                    target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)  # 【】统计到

        # measure elapsed time
        batch_time.update(time.time() - end)  # 一个batch模型训练的时间
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                'Speed {speed:.1f} samples/s\t' \
                'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                  # 【see】batchtime的输出
                    epoch, i, len(train_loader), batch_time=batch_time,
                    speed=input.size(0)/batch_time.val,
                    data_time=data_time, loss=losses, acc=acc)
            # epoch-迭代-总迭代数 当前迭代所花时间，整个epoch累计平均每迭代所花时间；
            # len(train_loader)train_loader是把整个dataset分成迭代元素了，一次传一个迭代给网络。总的长度是总的迭代个数向上取整
            logger.info(msg)  # 【c】之前好像用的其他方式统计log

            writer = writer_dict['writer']  # 【c】这在哪儿定义的
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val,
                          global_steps)  # 【c】writer是个什么类型
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1  # 更新总步数

            prefix = '{}_{}'.format(os.path.join(
                output_dir, 'train'), i)  # prefix保存debug图的路径
            save_debug_images(config, input, meta, target, pred*4, output,
                          prefix)  # 【c】pred*4干嘛，加强颜色？
    # print(lossweight,i)
    normalize_coeff = n_tasks / torch.sum(lossweight.data, dim=0) #【see】写在迭代还是epoch里
    lossweight.data = lossweight.data * normalize_coeff  # 迭代完一次
    return lossweight


def validate(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir,lossweight, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)  # 不用loader?用，这儿只是统计下总样本数
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )  # 【c】size
    all_boxes = np.zeros((num_samples, 6))  # 【c】
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():  # 后面的全是
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            # compute output
            outputs = model(input)
            if isinstance(outputs, list):
                output = outputs[-1]  # 为啥是-1:只取最后一个分支的输出
            else:
                output = outputs

            if config.TEST.FLIP_TEST:
                # this part is ugly, because pytorch has not supported negative index
                # input_flipped = model(input[:, :, :, ::-1])
                input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                input_flipped = torch.from_numpy(input_flipped).cuda()
                outputs_flipped = model(input_flipped)  # 【】怎么不用cuda了

                if isinstance(outputs_flipped, list):
                    output_flipped = outputs_flipped[-1]  # 【】为啥只要最后一个？不可以平均？
                else:
                    output_flipped = outputs_flipped

                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)  # 【see】将翻转过的输入的输出变成正常形式，左手-右手-右手(out)-左手（out）
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:  # 【】output_flipped size
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]  # 【】什么作用；翻转坐标？？

                output = (output + output_flipped) * 0.5  # 【see】上下翻转会不会提高

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            loss = criterion(output, target, target_weight) #【c】单分支和多分支都一样
            loss = torch.stack(loss)
            loss = torch.mul(lossweight, loss)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)  # 一次迭代的
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # 【c】meta是从哪儿获得的:valoader里来的；只有val有这个:不是，来自于mpii里的datasets里的gt_db
            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)  # 预测的坐标和值

            all_preds[idx:idx + num_images, :,
                      0:2] = preds[:, :, 0:2]  # 【c】preds不就是0:2吗
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)  # 【l】啥意思
            all_boxes[idx:idx + num_images, 5] = score  # 【】框的score?哪儿来的？什么作用？
            image_path.extend(meta['image'])

            idx += num_images  # 已训练图片数

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input, meta, target, pred*4, output,
                                  prefix)  # 【】？自动保存了？没看到啊
        # 整个epoch循环完了
        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )  # 【c】val_dataset是个什么类,evaluate函数在哪儿定义的什么作用？？

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:  # 【c】name_values
                _print_name_value(name_value, model_name)  # 【】干嘛的
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg, #【】
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator  # 【】这是个啥


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()  # 【c】是writer_dict吗
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )  # 【c】names是模型？
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
        ' |'
    )  # 【c】参数？


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val  # 当前迭代的平均
        self.sum += val * n
        self.count += n  # 当前epoch的平均
        self.avg = self.sum / self.count if self.count != 0 else 0  # 【see】
