# -*-coding: utf-8 -*-
# @Time    : 2024/8/12
# @Author  : YeLi
# @File    : Hyper2RGBConverter.py
# @Software: PyCharm
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from scipy.interpolate import interp1d
from tqdm import tqdm
import tifffile


def load_rgb_data(rgb_curve_path):
    # 定义文件路径
    red_file_path = os.path.join(rgb_curve_path, 'New_Red.csv')
    green_file_path = os.path.join(rgb_curve_path, 'New_Green.csv')
    blue_file_path = os.path.join(rgb_curve_path, 'New_Blue.csv')

    # 读取数据，并确保第二列是浮点数
    red_data = pd.read_csv(red_file_path, delimiter=',', header=None)
    green_data = pd.read_csv(green_file_path, delimiter=',', header=None)
    blue_data = pd.read_csv(blue_file_path, delimiter=',', header=None)

    # 检查数据格式
    if red_data.shape[1] != 2 or green_data.shape[1] != 2 or blue_data.shape[1] != 2:
        raise ValueError("Each CSV file is expected to have 2 columns: wavelength and proportion.")

    # 重命名列
    red_data.columns = ['wavelength', 'red_proportion']
    green_data.columns = ['wavelength', 'green_proportion']
    blue_data.columns = ['wavelength', 'blue_proportion']

    # 转换为浮点数类型
    red_data['red_proportion'] = pd.to_numeric(red_data['red_proportion'], errors='coerce')
    green_data['green_proportion'] = pd.to_numeric(green_data['green_proportion'], errors='coerce')
    blue_data['blue_proportion'] = pd.to_numeric(blue_data['blue_proportion'], errors='coerce')

    # 定义插值函数
    interp_wavelengths = np.arange(400, 701, 10)
    red_interp = interp1d(red_data['wavelength'], red_data['red_proportion'] / 100, kind='linear',
                          fill_value="extrapolate")
    green_interp = interp1d(green_data['wavelength'], green_data['green_proportion'] / 100, kind='linear',
                            fill_value="extrapolate")
    blue_interp = interp1d(blue_data['wavelength'], blue_data['blue_proportion'] / 100, kind='linear',
                           fill_value="extrapolate")

    # 计算插值后的数据
    red_values = red_interp(interp_wavelengths)
    green_values = green_interp(interp_wavelengths)
    blue_values = blue_interp(interp_wavelengths)

    # # 在 load_rgb_data 函数中添加以下代码进行调试
    # print("Original Blue Data:")
    # print(blue_data)
    #
    # # 打印插值后的 blue_values
    # print("Interpolated Blue Values:")
    # print(blue_values)

    # 创建插值后的DataFrame
    rgb_data = pd.DataFrame({
        'wavelength': interp_wavelengths,
        'red': red_values,
        'green': green_values,
        'blue': blue_values
    })

    return rgb_data


def process_image(image_file, rgb_data, output_folder):
    hyperspectral_image = np.load(image_file)
    selected_wavelengths = np.arange(400, 701, 10)
    selected_indices = rgb_data['wavelength'].isin(selected_wavelengths)

    # 提取与选定波长对应的red, green, blue值
    red_values = rgb_data.loc[selected_indices, 'red'].values
    green_values = rgb_data.loc[selected_indices, 'green'].values
    blue_values = rgb_data.loc[selected_indices, 'blue'].values

    # # 打印 blue_values
    # print("Blue Values Used in Image Processing:", blue_values)

    # 校验波长数量与图像的光谱维度是否匹配
    assert len(selected_wavelengths) == hyperspectral_image.shape[
        2], "Number of wavelengths and hyperspectral image slices do not match."

    # Use light intensity
    interpolated_data = np.load('/home/wangruozhang/Hyper2Mosaic/Dataset/CIE/normalized_wavelengths_and_integrals.npy')
    wavelengths_from_file = interpolated_data[:, 0]  # To en
    intensity_values = interpolated_data[:, 1]

    height, width = hyperspectral_image.shape[:2]
    R = np.zeros((height, width))
    G = np.zeros((height, width))
    B = np.zeros((height, width))

    # 对每个波长计算R, G, B，并显示进度条
    for i in tqdm(range(len(selected_wavelengths)), desc="Processing wavelengths"):
        R += hyperspectral_image[:, :, i] * red_values[i] * intensity_values[i]
        G += hyperspectral_image[:, :, i] * green_values[i] * intensity_values[i]
        B += hyperspectral_image[:, :, i] * blue_values[i] * intensity_values[i]
        # print(f"B channel after processing wavelength {selected_wavelengths[i]}:", B) # 8.26 10：15 检查B通道全为nan
        # 错误原因为被插值数据中前两行第一列数据相同，且第一行第二列中的数据存在字母，已修正sudo

    # 归一化
    R = R / np.max(R) * 255
    G = G / np.max(G) * 255
    B = B / np.max(B) * 255

    # 使用 np.clip 限制值在 [0, 255] 范围内
    R = np.clip(R, 0, 255)
    G = np.clip(G, 0, 255)
    B = np.clip(B, 0, 255)

    # 将 R, G, B 转换为 uint8 类型以保存为图像
    R = R.astype(np.uint8)
    G = G.astype(np.uint8)
    B = B.astype(np.uint8)

    # 堆叠 R, G, B 通道
    RGB_image = np.stack([R, G, B], axis=-1)

    # 保存RGB图像为TIFF文件
    output_file = os.path.join(output_folder, os.path.basename(image_file).replace('.npy', '.tiff'))
    tifffile.imwrite(output_file, RGB_image)
    print(f"RGB image saved as TIFF at: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Process hyperspectral images to RGB")
    parser.add_argument('--dataset_folder', type=str, default='/home/wangruozhang/Hyper2Mosaic/Dataset/Extracted_Data',
                        help="Path to the dataset folder containing the four subfolders")
    parser.add_argument('--RGBfolder', type=str, default='/home/wangruozhang/Hyper2Mosaic/Dataset/RGBimages_LED',
                        help="Folder to store output images")
    args = parser.parse_args()

    rgb_curve_path = '/home/wangruozhang/Hyper2Mosaic/Dataset/CIE/RGBTransCurve'
    rgb_data = load_rgb_data(rgb_curve_path)

    subfolders = [f.path for f in os.scandir(args.dataset_folder) if f.is_dir()]

    for subfolder in subfolders:
        subfolder_name = os.path.basename(subfolder)
        output_subfolder = os.path.join(args.RGBfolder, subfolder_name)
        if not os.path.exists(output_subfolder):
            os.makedirs(output_subfolder)

        # 遍历文件并显示进度条
        files = [file for root, _, files in os.walk(subfolder) for file in files if file.endswith('.npy')]
        for file in tqdm(files, desc=f"Processing folder {subfolder_name}"):
            image_file = os.path.join(subfolder, file)
            process_image(image_file, rgb_data, output_subfolder)


if __name__ == "__main__":
    main()
