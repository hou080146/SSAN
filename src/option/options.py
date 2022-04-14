# -*- coding: utf-8 -*-
"""
@author: zifyloo
"""

import argparse
import torch
import logging
import os
from utils.read_write_data import makedir

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class options():
    def __init__(self):
        self._par = argparse.ArgumentParser(description='options for Deep Cross Modal')

        self._par.add_argument('--model_name', type=str, help='experiment name')
        self._par.add_argument('--mode', type=str, default='', help='choose mode [train or test]')

        self._par.add_argument('--epoch', type=int, default=60, help='train epoch')
        self._par.add_argument('--epoch_decay', type=list, default=[20, 40], help='decay epoch')
        self._par.add_argument('--epoch_begin', type=int, default=5, help='when calculate the auto margin')

        self._par.add_argument('--batch_size', type=int, default=64, help='batch size')
        self._par.add_argument('--adam_alpha', type=float, default=0.9, help='momentum term of adam') # Adam的动量
        self._par.add_argument('--adam_beta', type=float, default=0.999, help='momentum term of adam')
        self._par.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam') #学习率
        self._par.add_argument('--margin', type=float, default=0.2, help='ranking loss margin')
        self._par.add_argument('--cr_beta', type=float, default=0.1, help12='ranking loss margin')

        self._par.add_argument('--vocab_size', type=int, default=5000, help='the size of vocab')
        self._par.add_argument('--feature_length', type=int, default=512, help='the length of feature')
        self._par.add_argument('--class_num', type=int, default=11000,
                               help='num of class for StarGAN training on second dataset')
        self._par.add_argument('--part', type=int, default=6, help='the num of image part')

        # 图像标注(Image Caption)也称为图片描述，是基于图片自动生成一段文字描述，包括其中的主要元素和环境信息等。
        self._par.add_argument('--caption_length_max', type=int, default=100, help='the max length of caption')

        self._par.add_argument('--save_path', type=str, default='./checkpoints/test',
                               help='save the result during training')
        self._par.add_argument('--GPU_id', type=str, default='2', help='choose GPU ID [0 1]')
        self._par.add_argument('--device', type=str, default='', help='cuda devie')
        self._par.add_argument('--dataset', type=str, help='choose the dataset ')
        self._par.add_argument('--dataroot', type=str,  help='data root of the Data')

        self.opt = self._par.parse_args()

        self.opt.device = torch.device('cuda:{}'.format(self.opt.GPU_id[0])) #用opt里GPU_id[0]替换{}
        # 上一句字符串组合后为 self.opt.device = torch.device('cuda:GPU_id[0]')
        # torch.device代表将torch.tensor分配到的设备的对象(简单点说，就是分配到你的CPU还是GPU上,以及哪块GPU上)。


def config(opt):

    log_config(opt)
    model_root = os.path.join(opt.save_path, 'model')
    # 路径不存在则生成路径
    if os.path.exists(model_root) is False:
        makedir(model_root)


def log_config(opt):
    # 存放log的目录
    logroot = os.path.join(opt.save_path, 'log')
    if os.path.exists(logroot) is False:
        makedir(logroot)
    filename = os.path.join(logroot, opt.mode + '.log')
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(logging.StreamHandler())
    logger.addHandler(handler)
    if opt.mode != 'test':
        logger.info(opt)



