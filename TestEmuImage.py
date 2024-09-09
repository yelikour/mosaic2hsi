# -*-coding: utf-8 -*-
# @Time    : 2024/9/3 15:51
# @Author  : YeLi
# @File    : TestEmuImage.py
# @Software: PyCharm
import cv2
import numpy as np
import torch
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from Hyper2RGBConverter import load_rgb_data
import tifffile
from model import MST3, MST, MST_Plus_Plus, ResidualNet, MultiscaleNet, ParallelMultiscaleNet

def main():
    device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
    base_train_data_dir = '/home/wangruozhang/Hyper2Mosaic/Dataset/RGBimages_LED'
    base_gt_data_dir = '/home/wangruozhang/Hyper2Mosaic/Dataset/Extracted_Data'
    save_npy_folder = '/home/wangruozhang/mosaic2hsi/Output0903/npy'
    output_rgb_folder = '/home/wangruozhang/mosaic2hsi/Output0903/RGB'
    rgb_curve_path = '/home/wangruozhang/Hyper2Mosaic/Dataset/CIE/RGBTransCurve'

    if not os.path.exists(save_npy_folder):
        os.makedirs(save_npy_folder)
    if not os.path.exists(output_rgb_folder):
        os.makedirs(output_rgb_folder)

    # Load the model
    model = MST3.MST(device).to(device)
    model.load_state_dict(torch.load('trainedWeights/MST3_final_(2024, 9, 3).pth', map_location=device))
    model.eval()

    # Load RGB data
    rgb_data = load_rgb_data(rgb_curve_path)

    # Prepare the dataset and dataloader
    dataset = MyDataSet(dataPath=base_train_data_dir, gtPath=base_gt_data_dir)
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
        # print(f"Model output saved as {npy_filename}")

        # Convert .npy file to RGB and save as TIFF
        convert_and_save_rgb(npy_filename, rgb_data, output_rgb_folder)



def convert_and_save_rgb(npy_file, rgb_data, output_rgb_folder):
    # Load the output from .npy file
    output_tensor = np.load(npy_file)
    output_tensor = output_tensor.squeeze()  # Remove batch dimension
    # Ensure output is in [c, h, w] format
    if output_tensor.shape[0] != 31:  # Assuming 31 channels for hyperspectral images
        print("output_tensor.shape", output_tensor.shape)
        raise ValueError(f"Expected 31 channels, but got {output_tensor.shape[0]}")

    selected_wavelengths = np.arange(400, 701, 10)
    red_values = rgb_data.loc[rgb_data['wavelength'].isin(selected_wavelengths), 'red'].values
    green_values = rgb_data.loc[rgb_data['wavelength'].isin(selected_wavelengths), 'green'].values
    blue_values = rgb_data.loc[rgb_data['wavelength'].isin(selected_wavelengths), 'blue'].values

    interpolated_data = np.load('/home/wangruozhang/Hyper2Mosaic/Dataset/CIE/normalized_wavelengths_and_integrals.npy')
    intensity_values = interpolated_data[:, 1]

    height, width = output_tensor.shape[1], output_tensor.shape[2]
    R = np.zeros((height, width))
    G = np.zeros((height, width))
    B = np.zeros((height, width))

    for i in range(len(selected_wavelengths)):
        R += output_tensor[i, :, :] * red_values[i] * intensity_values[i]
        G += output_tensor[i, :, :] * green_values[i] * intensity_values[i]
        B += output_tensor[i, :, :] * blue_values[i] * intensity_values[i]

    R = np.clip(R / np.max(R) * 255, 0, 255).astype(np.uint8)
    G = np.clip(G / np.max(G) * 255, 0, 255).astype(np.uint8)
    B = np.clip(B / np.max(B) * 255, 0, 255).astype(np.uint8)

    RGB_image = np.stack([R, G, B], axis=-1)

    base_filename = os.path.splitext(os.path.basename(npy_file))[0]
    output_tiff_file = os.path.join(output_rgb_folder, f"{base_filename}.tiff")

    tifffile.imwrite(output_tiff_file, RGB_image)
    # print(f"RGB image saved as TIFF at: {output_tiff_file}")

from torch.utils.data import Dataset


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
        self.gtPath = gtPath
        self.transform = transform

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

        if self.transform:
            input_data = self.transform(input_data)
            gt_data = self.transform(gt_data)

        input_data = torch.tensor(input_data, dtype=torch.float32).permute(2, 0, 1)  # (C, H, W)
        gt_data = torch.tensor(gt_data, dtype=torch.float32).permute(2, 0, 1)  # (C, H, W)

        return {'mosaic': input_data, 'gd': gt_data}


if __name__ == '__main__':
    main()
