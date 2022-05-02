# -*- coding: utf-8 -*-
"""
@author: zifyloo
"""

import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import os

# read_dict()在SSAN/dataset/utils下
from utils.read_write_data import read_dict
import cv2
import torchvision.transforms.functional as F
import random


def fliplr(img, dim):
    """
    flip horizontal
    :param img:
    :return:
    """
    inv_idx = torch.arange(img.size(dim) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(dim, inv_idx)
    return img_flip


class CUHKPEDEDataset(data.Dataset):  # 继承data.Dataset，所有数据集都是要继承重写data.Dataset的
    def __init__(self, opt, tran):  # opt：选项参数  tran：图像处理步骤

        self.opt = opt
        self.flip_flag = (self.opt.mode == 'train')  # 训练模式flag则为true

        # 反序列化，read_dict()在SSAN/dataset/utils下，返回值是个字典
        data_save = read_dict(os.path.join(opt.dataroot, 'processed_data', opt.mode + '_save.pkl'))

        self.img_path = [os.path.join(opt.dataroot, img_path) for img_path in data_save['img_path']]  # 取所有图片路径

        self.label = data_save['id']  # 标签为data_save里的key=id

        self.caption_code = data_save['lstm_caption_id']

        self.same_id_index = data_save['same_id_index']

        self.transform = tran

        self.num_data = len(self.img_path)  # 数据个数为img_path里所有图片路径个数

    def __getitem__(self, index):  # 迭代器，获取单个数据
        """
        :param index:单个下标
        :return: image and its label
        """

        image = Image.open(self.img_path[index])  # 获取一个图片
        image = self.transform(image)  # 对一张图进行预处理

        # np.array()的作用就是把列表转化为数组  long() 函数将数字或字符串转换为一个长整型。
        label = torch.from_numpy(np.array([self.label[index]])).long()
        caption_code, caption_length = self.caption_mask(self.caption_code[index])

        same_id_index = np.random.randint(len(self.same_id_index[index]))
        same_id_index = self.same_id_index[index][same_id_index]
        same_id_caption_code, same_id_caption_length = self.caption_mask(self.caption_code[same_id_index])

        return image, label, caption_code, caption_length, same_id_caption_code, same_id_caption_length

    def get_data(self, index, img=True):
        if img:
            image = Image.open(self.img_path[index])
            image = self.transform(image)
        else:
            image = 0

        label = torch.from_numpy(np.array([self.label[index]])).long()

        caption_code, caption_length = self.caption_mask(self.caption_code[index])

        return image, label, caption_code, caption_length

    def caption_mask(self, caption):
        # 对caption格式化
        caption_length = len(caption)

        # 转换成np数组，再转换成张量tensor   view()函数修改数组形状，这里改成一维,caption转换成一维张量
        caption = torch.from_numpy(np.array(caption)).view(-1).long()

        if caption_length < self.opt.caption_length_max:
            zero_padding = torch.zeros(self.opt.caption_length_max - caption_length).long()
            caption = torch.cat([caption, zero_padding], 0)  # 按维数0拼接,caption不满足最大长度，其余补零
        else:
            caption = caption[:self.opt.caption_length_max]  # 太长的截取最大长度
            caption_length = self.opt.caption_length_max

        return caption, caption_length

    def __len__(self):
        return self.num_data


class CUHKPEDE_img_dateset(data.Dataset):
    def __init__(self, opt, tran):
        self.opt = opt

        data_save = read_dict(os.path.join(opt.dataroot, 'processed_data', opt.mode + '_save.pkl'))

        self.img_path = [os.path.join(opt.dataroot, img_path) for img_path in data_save['img_path']]

        self.label = data_save['id']

        self.transform = tran

        self.num_data = len(self.img_path)

    def __getitem__(self, index):
        """
        :param index:
        :return: image and its label
        """

        image = Image.open(self.img_path[index])
        image = self.transform(image)

        label = torch.from_numpy(np.array([self.label[index]])).long()

        return image, label

    def __len__(self):
        return self.num_data


class CUHKPEDE_txt_dateset(data.Dataset):
    def __init__(self, opt):

        self.opt = opt

        data_save = read_dict(os.path.join(opt.dataroot, 'processed_data', opt.mode + '_save.pkl'))

        self.label = data_save['caption_label']
        self.caption_code = data_save['lstm_caption_id']

        self.num_data = len(self.caption_code)

    def __getitem__(self, index):
        """
        :param index:
        :return: image and its label
        """

        label = torch.from_numpy(np.array([self.label[index]])).long()

        caption_code, caption_length = self.caption_mask(self.caption_code[index])
        return label, caption_code, caption_length

    def caption_mask(self, caption):
        caption_length = len(caption)
        caption = torch.from_numpy(np.array(caption)).view(-1).float()
        if caption_length < self.opt.caption_length_max:
            zero_padding = torch.zeros(self.opt.caption_length_max - caption_length)
            caption = torch.cat([caption, zero_padding], 0)
        else:
            caption = caption[:self.opt.caption_length_max]
            caption_length = self.opt.caption_length_max

        return caption, caption_length

    def __len__(self):
        return self.num_data
