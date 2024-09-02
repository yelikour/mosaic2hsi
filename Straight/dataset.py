# -*-coding: utf-8 -*-
# @Time    : 2024/8/29 15:28
# @Author  : YeLi
# @File    : dataset.py
# @Software: PyCharm
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import tifffile as tiff


class MyDataSet(Dataset):
    def __init__(self, dataPath, gtPath, transform=None, crop_size=(256, 256)):
        self.dataPath = dataPath
        self.gtPath = gtPath
        self.transform = transform
        self.crop_size = crop_size

        # 包含文件夹路径进行排序
        self.input_paths = sorted([os.path.join(root, f)
                                   for root, _, files in os.walk(dataPath)
                                   for f in files if f.endswith('.tiff') or f.endswith('.tif')])

        self.gt_paths = sorted([os.path.join(root, f)
                                for root, _, files in os.walk(gtPath)
                                for f in files if f.endswith('.npy')])

        for input_file, gt_file in zip(self.input_paths, self.gt_paths):
            print(f"Input file: {os.path.basename(input_file)}, GT file: {os.path.basename(gt_file)}")

        if len(self.input_paths) != len(self.gt_paths):
            print('Number of input: {}, number of gt: {}'.format(len(self.input_paths), len(self.gt_paths)))
            raise ValueError("The number of input and ground truth files do not match.")

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        # 获取第 idx 个文件的路径
        input_path = self.input_paths[idx]
        gt_path = self.gt_paths[idx]

        # 读取输入图像和地面真值数据
        input_data = tiff.imread(input_path)
        gt_data = np.load(gt_path, allow_pickle=True).astype(np.float32)

        # 归一化
        input_data = input_data / 65535.0  # 根据TIFF的16位图像数据类型进行归一化
        gt_data = gt_data / 65535.0

        # 获取图像的尺寸
        h, w = input_data.shape[:2]  # 只获取高度和宽度
        new_h, new_w = self.crop_size

        # 随机裁剪起始位置
        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)

        # 裁剪图像
        input_data = input_data[top:top + new_h, left:left + new_w]
        gt_data = gt_data[top:top + new_h, left:left + new_w, :]

        # 复制 input_data 并在第一维变成3通道
        input_data = np.stack([input_data] * 3, axis=0)

        if self.transform:
            input_data = self.transform(input_data)
            gt_data = self.transform(gt_data)

        input_data = torch.tensor(input_data, dtype=torch.float32)
        gt_data = torch.tensor(gt_data, dtype=torch.float32).permute(2, 0, 1)

        return {'mosaic': input_data, 'gd': gt_data}
