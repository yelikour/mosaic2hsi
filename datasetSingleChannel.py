import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

def create_mosaic(img):
    """Creates a mosaic image from an RGB image."""
    h, w, _ = img.shape # height * width * 3
    mosaic = np.zeros((h, w), dtype=img.dtype)  # Create single-channel mosaic

    mosaic[0::2, 0::2] = img[0::2, 0::2, 0]  # Red
    mosaic[0::2, 1::2] = img[0::2, 1::2, 1]  # Green
    mosaic[1::2, 0::2] = img[1::2, 0::2, 1]  # Green
    mosaic[1::2, 1::2] = img[1::2, 1::2, 2]  # Blue

    return mosaic

class MyDataSet(Dataset):
    def __init__(self, dataPath, gtPath, transform=None, crop_size=(256, 256)):
        self.dataPath = dataPath
        self.gtPath = gtPath
        self.transform = transform
        self.crop_size = crop_size
        self.input_paths = sorted([os.path.join(dataPath, f) for f in os.listdir(dataPath) if f.endswith('.png')])
        self.gt_paths = sorted([os.path.join(gtPath, f) for f in os.listdir(gtPath) if f.endswith('.npy')])

        if len(self.input_paths) != len(self.gt_paths):
            raise ValueError("The number of input and ground truth files do not match.")

        # 计算每张图像能裁剪出的子图片数量
        h, w = cv2.imread(self.input_paths[0]).shape[:2]
        new_h, new_w = self.crop_size
        self.num_crops_per_row = w // new_w
        self.num_crops_per_col = h // new_h
        self.num_crops_per_image = self.num_crops_per_row * self.num_crops_per_col

        # 更新数据集的总长度
        self.total_length = len(self.input_paths) * self.num_crops_per_image

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        # 获取第 idx 个文件的路径
        input_path = self.input_paths[idx // self.num_crops_per_image]
        gt_path = self.gt_paths[idx // self.num_crops_per_image]

        # 读取输入图像和地面真值数据
        input_data = cv2.imread(input_path)
        gt_data = np.load(gt_path, allow_pickle=True).astype(np.float32)

        # 应用马赛克转换
        input_data = create_mosaic(input_data)

        # 归一化
        input_data = input_data / 255.0
        gt_data = gt_data / 65535.0

        # 获取图像的尺寸
        h, w = input_data.shape[:2]  # 修改为使用[:2]，只获取高度和宽度
        new_h, new_w = self.crop_size

        # 计算裁剪起始位置
        crop_row = (idx % self.num_crops_per_image) // self.num_crops_per_row
        crop_col = (idx % self.num_crops_per_image) % self.num_crops_per_row

        top = crop_row * new_h
        left = crop_col * new_w

        # 检查裁剪区域是否超出图像边界，如果超出则返回 None
        if top + new_h > h or left + new_w > w:
            return None

        # 裁剪图像
        input_data = input_data[top:top + new_h, left:left + new_w]
        gt_data = gt_data[top:top + new_h, left:left + new_w, :]

        ## 模拟Measurement, 改为用单通道
        # # 复制 input_data 并在第一维变成3通道
        # input_data = np.stack([input_data] * 3, axis=0)

        if self.transform:
            input_data = self.transform(input_data)
            gt_data = self.transform(gt_data)

        input_data = torch.tensor(input_data, dtype=torch.float32)
        gt_data = torch.tensor(gt_data, dtype=torch.float32).permute(2, 0, 1)

        return {'mosaic': input_data, 'gd': gt_data}