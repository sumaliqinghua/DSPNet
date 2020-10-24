# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------
'''
six-scalepyramid function
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import cv2
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
scalelist=[0.5,0.75,1.0,1.25,1.5,1.75]
def multi_scale_aug(self, image,
        rand_scale=1):#rand_scale为放缩的倍数
    long_size = np.int(self.base_size * rand_scale + 0.5) #【】base_size是原图还是256；四舍五入
    h, w = image.shape[:2]
    if h > w:
        new_h = long_size
        new_w = np.int(w * long_size / h + 0.5) #【】保持比例
    else:
        new_w = long_size
        new_h = np.int(h * long_size / w + 0.5)
    
    image = cv2.resize(image, (new_w, new_h), #不行的话用torch.upsample
                        interpolation = cv2.INTER_LINEAR)
    return image

def train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        outputs = model(input)

        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        if isinstance(outputs, list):
            loss = criterion(outputs[0], target, target_weight)
            for output in outputs[1:]:
                loss += criterion(output, target, target_weight)
        else:
            output = outputs
            loss = criterion(output, target, target_weight)

        # loss = criterion(output, target, target_weight)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred*4, output,
                              prefix)


def validate(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)#2958
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    size = [64,64]
    num_joints =16
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(tqdm(val_loader)):
            # compute output

            output = torch.zeros([1, num_joints,
                                    size[0],size[1]]).cuda()
            for scale in scalelist:
                input = multi_scale_aug(input,scale)
                output_single = model(input)#<class 'torch.Tensor'> torch.Size([64, 16, 64, 64])
                if isinstance(outputs, list):
                    output_single = output_single[-1]#只输出最后一个

                if config.TEST.FLIP_TEST:
                    # this part is ugly, because pytorch has not supported negative index
                    # input_flipped = model(input[:, :, :, ::-1])
                    input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                    input_flipped = torch.from_numpy(input_flipped).cuda()          #torch.Size([64, 3, 256, 256])
                    outputs_flipped = model(input_flipped)#torch.Size([64, 16, 64, 64])

                    if isinstance(outputs_flipped, list):
                        output_flipped = outputs_flipped[-1]
                    else:
                        output_flipped = outputs_flipped

                    output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)#将翻转过的输入变成正常的输出
                    output_flipped = torch.from_numpy(output_flipped.copy()).cuda()


                    # feature is not aligned, shift flipped heatmap for higher accuracy
                #【】为啥翻转的没对齐
                    if config.TEST.SHIFT_HEATMAP:
                        output_flipped[:, :, :, 1:] = \
                            output_flipped.clone()[:, :, :, 0:-1] #【c】将0以后的图左移动

                    output_single = (output_single + output_flipped) * 0.5 #【see】妙啊
                    if output_single.size()[-2] != size[0] or output_single.size()[-1] != size[1]:
                        output_single = F.upsample(output_single, (size[-2], size[-1]),
                                   mode='bilinear')
                output += output_single #【】取平均吗？
            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)#target_weight是否可见

            loss = criterion(output, target, target_weight)
            # criterion(output, target, target_weight) #【see】

            num_images = input.size(0)#求平均值用
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()#meta只获取当前loader的 64个
            score = meta['score'].numpy()

            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals#(2958, 16, 3)
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)#沿着axis=1也就是×200后再平方的面积
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])#将路径信息传回meta

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)#len(val_loader)是总的迭代次数

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )#'output/mpii/pose_hrnet/w32_256x256_adam_lr1e-3/val_0'
                save_debug_images(config, input, meta, target, pred*4, output,
                                  prefix)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        ) #【】这儿的作用是？

        model_name = config.MODEL.NAME#'pose_hrnet'
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)#打印相关精度到终端
        else:
            _print_name_value(name_values, model_name)

        if writer_dict: #【】None 是不是显示损失用的，怎么用
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
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

    return perf_indicator


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


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
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
