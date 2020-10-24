# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2


def cutout(mask_size, p, num_holes, mask_color=(0, 0, 0)):
    mask_size_half = mask_size // 2
    offset = 1 if mask_size % 2 == 0 else 0

    def _cutout(image):
        image = np.asarray(image).copy()

        if np.random.random() > p:
            return image

        h, w = image.shape[1:]

        cx_min, cx_max = mask_size_half, w + offset - mask_size_half
        cy_min, cy_max = mask_size_half, h + offset - mask_size_half
        for i in range(num_holes):
            cx = np.random.randint(cx_min, cx_max)
            cy = np.random.randint(cy_min, cy_max)
            x_min = cx - mask_size_half
            y_min = cy - mask_size_half
            x_max = x_min + mask_size
            y_max = y_min + mask_size
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w, x_max)
            y_max = min(h, y_max)
            mc = np.zeros((3, y_max - y_min, x_max - x_min))
            image[:, y_min:y_max, x_min:x_max] = mc  # 3,256,256
        return image

    return _cutout


def flip_back(output_flipped, matched_parts):  # 将heatmap左右翻转，并将output里配对通道的heatmap互换（左手变右手）
    '''
    ouput_flipped: numpy.ndarray(batch_size, num_joints, height, width)
    '''
    assert output_flipped.ndim == 4,\
        'output_flipped should be [batch_size, num_joints, height, width]'

    # 【c】output_flipped.size()，装的是对应16个关节坐标？
    output_flipped = output_flipped[:, :, :, ::-1]  # 【】np.flip不一样？为啥要翻转

    for pair in matched_parts:  # 【c】matched
        tmp = output_flipped[:, pair[0], :, :].copy()  # 64,16,64,64
        output_flipped[:, pair[0], :, :] = output_flipped[:, pair[1], :, :]
        output_flipped[:, pair[1], :, :] = tmp  # 【】output_flipped是什么

    return output_flipped


def fliplr_joints(joints, joints_vis, width, matched_parts):
    """
    flip coords
    """
    # Flip horizontal
    joints[:, 0] = width - joints[:, 0] - 1  # 【c】size

    # Change left-right parts
    # 【see】不是互换坐标，上面翻转过，坐标已经换了，现在是互换它们在joints中的对应部件，左手腕翻转后成了右手腕；中间部件的坐标已经变换了
    for pair in matched_parts:  # 【c】matched_parts是啥？M:flip_pairs
        joints[pair[0], :], joints[pair[1], :] = \
            joints[pair[1], :], joints[pair[0], :].copy()  # 【】为啥后面这个要copy
        joints_vis[pair[0], :], joints_vis[pair[1], :] = \
            joints_vis[pair[1], :], joints_vis[pair[0],
                                               :].copy()  # 【】这和上面的不一样？

    return joints*joints_vis, joints_vis  # 【】把不可见的置为0;其他地方也这么处理了吧要是关节在原点呢？为嘛不用之前的-1


def transform_preds(coords, center, scale, output_size):  # 【】啥时候用了
    target_coords = np.zeros(coords.shape)  # 【】coords是啥？多个点里取最大那个？
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)  # 【】干嘛的
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords  # 【】从heatmap到原图？


def get_affine_transform(
        center, scale, rot, output_size,
        shift=np.array([0, 0], dtype=np.float32), inv=0
):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])  # scale不是数组或列表就变成数组

    scale_tmp = scale * 200.0  # 【】为何200
    src_w = scale_tmp[0]  # 【】只要宽？
    dst_w = output_size[0]  # 【】和scale不冲突？
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)  # 【】这啥
    dst_dir = np.array([0, dst_w * -0.5], np.float32)  # 【】-0.5是干嘛的

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir  # 【】在干嘛哟

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:  # 【】啥
        trans = cv2.getAffineTransform(
            np.float32(dst), np.float32(src))  # 【】干嘛的
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T  # 【】1是干嘛的
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]  # 【c】前两行


def get_3rd_point(a, b):  # 【】这啥
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):  # 【】这啥
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def crop(img, center, scale, output_size, rot=0):
    trans = get_affine_transform(center, scale, rot, output_size)

    dst_img = cv2.warpAffine(
        img, trans, (int(output_size[0]), int(output_size[1])),
        flags=cv2.INTER_LINEAR
    )  # 【l】

    return dst_img
