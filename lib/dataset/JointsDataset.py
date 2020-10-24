# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# 【】上面这仨做咩的
import copy  # 【l】
import logging  # 【l】
import random  # 【l】

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset  # 【l】和tv.datasets有何不同
# utils文件下自己定义的包
from utils.transforms import get_affine_transform  # 【】
from utils.transforms import affine_transform  # 【】
from utils.transforms import fliplr_joints  # 【】


logger = logging.getLogger(__name__)  # 【l】记录日志


class JointsDataset(Dataset):
    # 【】不用写supper那个？
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        self.num_joints = 0
        self.pixel_std = 200  # 【】啥
        self.flip_pairs = []  # 【】配对节点？
        self.parent_ids = []  # 【】父节点？哪儿来的？

        self.is_train = is_train  # 是否训练集
        self.root = root
        self.image_set = image_set  # 数据集目录？

        # 【】cfg是外界传的，怎么不用self.cfg=cfg：因为cfg主要是在init里用可以直接用cfg
        self.output_path = cfg.OUTPUT_DIR  # 什么的输出：预测的mat输出
        self.data_format = cfg.DATASET.DATA_FORMAT

        self.scale_factor = cfg.DATASET.SCALE_FACTOR  # 【】 m:cfg应该是传的
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP
        self.num_joints_half_body = cfg.DATASET.NUM_JOINTS_HALF_BODY
        self.prob_half_body = cfg.DATASET.PROB_HALF_BODY  # 【c】这个参数里是啥
        self.color_rgb = cfg.DATASET.COLOR_RGB  # 【c】这个参数里是啥

        self.target_type = cfg.MODEL.TARGET_TYPE  # 【c】高斯？
        self.image_size = np.array(cfg.MODEL.IMAGE_SIZE)  # 【c】多大
        self.heatmap_size = np.array(cfg.MODEL.HEATMAP_SIZE)
        self.sigma = cfg.MODEL.SIGMA  # 【c】多大
        self.use_different_joints_weight = cfg.LOSS.USE_DIFFERENT_JOINTS_WEIGHT  # 【c】做手脚，有没有示例
        self.joints_weight = 1  # 【c】怎么用的，哪里用的

        self.transform = transform
        self.db = []  # 【c】干嘛的
        # self.cutpoint = self._cutpoint(64,11,)

    def _get_db(self):  # 【l】魔法函数吗？
        raise NotImplementedError  # 【l】啥意思

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        raise NotImplementedError  # 【】m:留来继承用，如果在父类上调用就报错,干嘛不在子类定义？统一全局，每一个子类都要有，但是又不一样？

    def half_body_transform(self, joints, joints_vis):  # 【】只取半身的关节点
        # 【c】传入的joints joints_vis值以及size()，后者是可见点吗，二者是一个人的还是多个人的？
        upper_joints = []
        lower_joints = []
        for joint_id in range(self.num_joints):  # 关节点都有特定的顺序
            if joints_vis[joint_id][0] > 0:  # 【】当前关节点可见就划分上下半身，不可见的呢？joints_vis存的是啥
                if joint_id in self.upper_body_ids:  # 【c】upper_body_ids这哪儿来的
                    upper_joints.append(joints[joint_id])  # 【】joints里应该是坐标
                else:
                    lower_joints.append(joints[joint_id])

        if np.random.randn() < 0.5 and len(upper_joints) > 2:  # 【see】
            selected_joints = upper_joints  # 【】选来干嘛，两个点以上代表着什么
        else:
            selected_joints = lower_joints \
                if len(lower_joints) > 2 else upper_joints  # 实在是没有就返回upper

        if len(selected_joints) < 2:  # 没找到符合条件的关节点
            return None, None

        selected_joints = np.array(
            selected_joints, dtype=np.float32)  # 【c】size()
        center = selected_joints.mean(axis=0)[:2]  # 半身关节点的中心

        # 【see】返回的是所有点里最左最上的坐标不是左上那个点
        left_top = np.amin(selected_joints, axis=0)
        right_bottom = np.amax(selected_joints, axis=0)

        w = right_bottom[0] - left_top[0]  # 半身关节点范围的宽
        h = right_bottom[1] - left_top[1]  # 高

        if w > self.aspect_ratio * h:  # 【c】aspect_ratio是啥，从哪儿来的，继承的？
            h = w * 1.0 / self.aspect_ratio  # 宽大了（比例）让高变大
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio  # 【see】宽小了，让宽变大；总之要满足比例，而且还是变大；保证框到的范围足够
        # 【】wh变了那左下右上不也变了？
        scale = np.array(
            [
                w * 1.0 / self.pixel_std,
                h * 1.0 / self.pixel_std  # 【】这啥
            ],
            dtype=np.float32
        )

        scale = scale * 1.5  # 【】缩放？

        return center, scale

    def __len__(self):
        return len(self.db)  # 【】db没赋值，子类里_get_db才赋值的，是单张图里人的个数？还是单张图里关节点数

    def __getitem__(self, idx):
        # 【c】db_rec是db的其中一个，是啥来着,一张图及其相关信息？
        db_rec = copy.deepcopy(self.db[idx])
        image_file = db_rec['image']  # db是数据集
        filename = db_rec['filename'] if 'filename' in db_rec else ''
        imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''  # 【c】总数？batch？

        if self.data_format == 'zip':  # 解压
            from utils import zipreader  # 【see】如果要用才导
            data_numpy = zipreader.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )  # 【l】
        else:
            data_numpy = cv2.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )  # 【】随便挑一个选项？

        if self.color_rgb:
            data_numpy = cv2.cvtColor(
                data_numpy, cv2.COLOR_BGR2RGB)  # 【l】为啥要转，不是该rgb2bgr?

        if data_numpy is None:
            logger.error('=> fail to read {}'.format(image_file))
            # 【see】语法不会报错但是完全影响了后面的结果，因此让其主动报错
            raise ValueError('Fail to read {}'.format(image_file))

        joints = db_rec['joints_3d']  # 【c】3d?
        joints_vis = db_rec['joints_3d_vis']  # 【】之前那个joints_vis就是从这儿获取的吧？

        c = db_rec['center']
        s = db_rec['scale']  # 数据集标注的
        # 【】谁的score，还是说暂时只用来说明非空
        score = db_rec['score'] if 'score' in db_rec else 1
        r = 0

        if self.is_train:  # 训练集才求半身
            if (np.sum(joints_vis[:, 0]) > self.num_joints_half_body  # 【】第0列元素求和；那么就是第一列为0,1？那么就是所有的点都有？
                    and np.random.rand() < self.prob_half_body):  # 【c】第二个是要采取半身的概率，为什么不在预处理做
                c_half_body, s_half_body = self.half_body_transform(
                    joints, joints_vis
                )

                if c_half_body is not None and s_half_body is not None:
                    c, s = c_half_body, s_half_body  # 取到了上半身或下半身的点就将c和s替换掉原标注的

            sf = self.scale_factor
            rf = self.rotation_factor  # 缩放旋转因子
            s = s * np.clip(np.random.randn()*sf + 1,
                            1 - sf, 1 + sf)  # 【l】取最大？
            r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
                if random.random() <= 0.6 else 0  # 【c】

            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]  # 将图像值水平翻转
                joints, joints_vis = fliplr_joints(
                    joints, joints_vis, data_numpy.shape[1], self.flip_pairs)  # GT坐标
                c[0] = data_numpy.shape[1] - c[0] - 1  # 最右-原==翻转过的因为宽比最右多1

        trans = get_affine_transform(
            c, s, r, self.image_size)  # 缩放旋转在transform里定义的，旋转空白怎么解决的？
        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)  # 【l】应用缩放旋转变换，input的size也变了吧？

        if self.transform:
            input = self.transform(input)  # 【c】还有另外的变换？从哪儿传入的哪儿定义的？
            # cut_trans = self._cutpoint(8, 1, 1, point)
            # input = cut_trans(input)
            #
        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:  # 【c】第一列不是0,1？有权重？只对可见点执行？还是说vis是未缺失有标记的点？
                # 【】对GT坐标也执行，怎么上面那个用的是warpAffine有何不同？
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)

        target, target_weight = self.generate_target(
            joints, joints_vis)  # 权重代表什么？
        # 【】上面都是在对numpy进行变换
        target = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)  # 【c】

        meta = {
            'image': image_file,
            'filename': filename,
            'imgnum': imgnum,
            'joints': joints,
            'joints_vis': joints_vis,
            'center': c,
            'scale': s,
            'rotation': r,
            'score': score
        }  # 【】有何用，日志？

        return input, target, target_weight, meta  # 【c】input是Tensor?

    def select_data(self, db):  # 【】干嘛的
        db_selected = []  # 和上面半身那儿的select没有联系
        for rec in db:
            num_vis = 0
            joints_x = 0.0
            joints_y = 0.0  # 【】定义声明；用来求vis的中心；为啥只要vis的
            for joint, joint_vis in zip(
                    rec['joints_3d'], rec['joints_3d_vis']):  # 【】selectdata是和getdb独立的？
                if joint_vis[0] <= 0:
                    continue
                num_vis += 1

                joints_x += joint[0]
                joints_y += joint[1]
            if num_vis == 0:
                continue

            joints_x, joints_y = joints_x / num_vis, joints_y / num_vis

            area = rec['scale'][0] * rec['scale'][1] * \
                (self.pixel_std**2)  # 【】后面一个是啥，单位？
            joints_center = np.array([joints_x, joints_y])  # 有啥用
            bbox_center = np.array(rec['center'])  # 【】那前面求的那个c呢，这个在何处用
            diff_norm2 = np.linalg.norm((joints_center-bbox_center), 2)
            ks = np.exp(-1.0*(diff_norm2**2) / ((0.2)**2*2.0*area))  # 【】这是啥？？

            metric = (0.2 / 16) * num_vis + 0.45 - 0.2 / 16  # 【】这又是啥？？
            if ks > metric:
                db_selected.append(rec)  # 将符合要求的添加进db_selectded

        logger.info('=> num db: {}'.format(len(db)))
        logger.info('=> num selected db: {}'.format(len(db_selected)))
        return db_selected

    def generate_target(self, joints, joints_vis):  # 给GT用的还是both?M:GT
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]  # 【c】是不是01

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'  # 【】还有其他选项？

        if self.target_type == 'gaussian':
            target = np.zeros((self.num_joints,
                               self.heatmap_size[1],  # 【】高？为什么和这儿反的
                               self.heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = self.sigma * 3  # 【】这啥

            for joint_id in range(self.num_joints):
                feat_stride = self.image_size / self.heatmap_size  # 【】什么的步长？
                # 【】对应heatmap上的位置，+0.5是向上取整,为啥要向上取整？
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds是不是所有的高斯都在范围内
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]  #
                br = [int(mu_x + tmp_size + 1),
                      int(mu_y + tmp_size + 1)]  # 【】为啥+1
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0  # 【】如果高斯点超出了边界就不要了，计算的时候不要吗
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]  # 【】为啥多一维
                x0 = y0 = size // 2  # 【】为嘛不用tmp_size
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) /
                           (2 * self.sigma ** 2))  # 【】这个值可以固定吧，不用每次都算

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:  # 【】大于0.5啥意思
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]
                          ]  # 【】把高斯值放到对应heatmap上，那是怎么还原到原图的呢？还有这个范围什么意思

        if self.use_different_joints_weight:  # 【】这是啥
            target_weight = np.multiply(
                target_weight, self.joints_weight)  # joints_weight默认1.干嘛的

        return target, target_weight