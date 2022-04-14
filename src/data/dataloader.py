# -*- coding: utf-8 -*-
"""
@author: zifyloo
"""

from torchvision import transforms
# tochvision主要处理图像数据，包含一些常用的数据集、模型、转换函数等。
from PIL import Image
import torch

# 从src/data/dataset.py中导入三个数据集处理函数
from data.dataset import CUHKPEDEDataset, CUHKPEDE_img_dateset, CUHKPEDE_txt_dateset


def get_dataloader(opt):
    """
    tranforms the image, downloads the image with the id by data.DataLoader
    """
    # mode分为训练（train）和测试（test）
    if opt.mode == 'train':
        transform_list = [
            transforms.RandomHorizontalFlip(), # 依据概率p对PIL图片进行水平翻转，p默认0.5
            # 重置图像分辨率，已(384, 128)分辨率输出图片，插值方式为BICUBIC
            transforms.Resize((384, 128), Image.BICUBIC),   # interpolation- 插值方法选择，默认为PIL.Image.BILINEAR
            # 将PIL Image或者 ndarray 转换为tensor，并且归一化至[0-1],归一化至[0-1]是直接除以255，若自己的ndarray数据尺度有变化，则需要自行修改。
            transforms.ToTensor(),
            # class torchvision.transforms.Normalize(mean, std)对数据按通道进行标准化，即先减均值，再除以标准差，注意是 hwc
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))]
        tran = transforms.Compose(transform_list) # compose()的参数是一个列表，列表里就是你想要执行的操作

        dataset = CUHKPEDEDataset(opt, tran)

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                                 shuffle=True, drop_last=True, num_workers=3)
        print('{}-{} has {} pohtos'.format(opt.dataset, opt.mode, len(dataset)))

        return dataloader

    else:
        tran = transforms.Compose([
            transforms.Resize((384, 128), Image.BICUBIC),  # interpolation
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))]
        )

        img_dataset = CUHKPEDE_img_dateset(opt, tran)

        img_dataloader = torch.utils.data.DataLoader(img_dataset, batch_size=opt.batch_size,
                                                 shuffle=False, drop_last=False, num_workers=3)

        txt_dataset = CUHKPEDE_txt_dateset(opt)

        txt_dataloader = torch.utils.data.DataLoader(txt_dataset, batch_size=opt.batch_size,
                                                 shuffle=False, drop_last=False, num_workers=3)

        print('{}-{} has {} pohtos, {} text'.format(opt.dataset, opt.mode, len(img_dataset), len(txt_dataset)))

        return img_dataloader, txt_dataloader
