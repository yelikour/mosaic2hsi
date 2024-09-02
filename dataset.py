import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

def create_mosaic(img):
    """Creates a 3-channel mosaic image from an RGB image."""
    h, w, _ = img.shape  # height * width * 3
    mosaic = np.zeros((h, w, 3), dtype=img.dtype)  # Create a 3-channel mosaic

    mosaic[0::2, 0::2, 0] = img[0::2, 0::2, 0]  # Red channel
    mosaic[0::2, 1::2, 1] = img[0::2, 1::2, 1]  # Green channel
    mosaic[1::2, 0::2, 1] = img[1::2, 0::2, 1]  # Green channel
    mosaic[1::2, 1::2, 2] = img[1::2, 1::2, 2]  # Blue channel

    return mosaic

class MyDataSet(Dataset):
    def __init__(self, dataPath, gtPath, transform=None, crop_size=(256, 256)):
        self.dataPath = dataPath
        self.gtPath = gtPath
        self.transform = transform
        self.crop_size = crop_size

        # 包含文件夹路径进行排序
        self.input_paths = sorted([os.path.join(root, f)
                                   for root, _, files in os.walk(dataPath)
                                   for f in files if f.endswith('.png')])

        self.gt_paths = sorted([os.path.join(root, f)
                                for root, _, files in os.walk(gtPath)
                                for f in files if f.endswith('.npy')])

        if len(self.input_paths) != len(self.gt_paths):
            print(f"The number of input: {len(self.input_paths)}, The number of ground truth files: {len(self.gt_paths)}")
            raise ValueError("The number of input and ground truth files do not match.")


    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        # 获取第 idx 个文件的路径
        input_path = self.input_paths[idx]
        gt_path = self.gt_paths[idx]

        # 读取输入图像和地面真值数据
        input_data = cv2.imread(input_path)
        gt_data = np.load(gt_path, allow_pickle=True).astype(np.float32)

        # 创建三通道马赛克图像
        input_data = create_mosaic(input_data)

        # 归一化
        input_data = input_data / 255.0
        gt_data = gt_data / 65535.0

        # 获取图像的尺寸
        h, w = input_data.shape[:2]  # 只获取高度和宽度
        new_h, new_w = self.crop_size

        # 随机裁剪起始位置
        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)

        # 裁剪图像
        input_data = input_data[top:top + new_h, left:left + new_w, :]
        gt_data = gt_data[top:top + new_h, left:left + new_w, :]

        if self.transform:
            input_data = self.transform(input_data)
            gt_data = self.transform(gt_data)

        input_data = torch.tensor(input_data, dtype=torch.float32).permute(2, 0, 1)  # (C, H, W)
        gt_data = torch.tensor(gt_data, dtype=torch.float32).permute(2, 0, 1)  # (C, H, W)

        return {'mosaic': input_data, 'gd': gt_data}
