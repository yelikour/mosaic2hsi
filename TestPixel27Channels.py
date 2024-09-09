# -*-coding: utf-8 -*-
# @Time    : 2024/9/9 11:16
# @Author  : YeLi
# @File    : TestPixel27Channels.py
# @Software: PyCharm
import cv2
import numpy as np
import torch
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from Hyper2RGBConverter import load_rgb_data
from model import MST27
from torch.utils.data import Dataset

def main():
    device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
    base_train_data_dir = '/home/wangruozhang/Hyper2Mosaic/Dataset/PixelRGB_LED'
    save_npy_folder = '/home/wangruozhang/mosaic2hsi/Output0909/npy'

    if not os.path.exists(save_npy_folder):
        os.makedirs(save_npy_folder)

    # Load the model
    model = MST27.MST(device).to(device)
    model.load_state_dict(torch.load('!!!!!!!!!!!!!!!!!!!!!!!!!!', map_location=device))
    model.eval()

    # Prepare the dataset and dataloader
    dataset = MyDataSet(dataPath=base_train_data_dir)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    # Process each image
    for i, data in enumerate(tqdm(data_loader, desc="Processing images")):
        inputs = data['mosaic'].to(device)

        # Forward pass
        with torch.no_grad():
            output = model(inputs)

        # Save the output as .npy file
        npy_filename = os.path.join(save_npy_folder, f"output_{i}.npy")
        np.save(npy_filename, output.cpu().numpy())


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
    def __init__(self, dataPath, gtPath, transform=None):
        self.dataPath = dataPath
        self.transform = transform

        # 包含文件夹路径进行排序
        self.input_paths = sorted([os.path.join(root, f)
                                   for root, _, files in os.walk(dataPath)
                                   for f in files if f.endswith('.png')])

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        # 获取第 idx 个文件的路径
        input_path = self.input_paths[idx]

        # 读取输入图像和地面真值数据
        input_data = cv2.imread(input_path)

        # 创建三通道马赛克图像
        input_data = create_mosaic(input_data)

        # 归一化
        input_data = input_data / 255.0

        if self.transform:
            input_data = self.transform(input_data)

        input_data = torch.tensor(input_data, dtype=torch.float32).permute(2, 0, 1)  # (C, H, W)

        return {'mosaic': input_data}


if __name__ == '__main__':
    main()
