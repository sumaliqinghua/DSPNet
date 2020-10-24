from thop import profile
import torch
import argparse
import os
import pprint
# import pose_resnet
import _init_paths
from config import cfg
from config import update_config
import models
def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()
    return args
def main():
    args = parse_args()
    update_config(cfg, args)
    model = eval('models.'+'multi_resnet'+'.get_pose_net')(
        cfg, is_train=False
    )#cichugenggaimoxing
    input = torch.randn(1,3,256,256)
    flops,params = profile(model,inputs=(input,))
    print(flops,params)

if __name__ == '__main__':
    main()
