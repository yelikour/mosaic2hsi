# -*-coding: utf-8 -*-
# @Time    : 2024/8/28 15:21
# @Author  : YeLi
# @File    : HSIIntensity.py
# @Software: PyCharm
import os
import numpy as np
import matplotlib.pyplot as plt

def save_intensity_curve_from_npy(npy_file, point, save_folder):
    """
    Save the intensity at a specific point for each channel in an npy file.

    Parameters:
    - npy_file: Path to the .npy file containing the hyperspectral image (shape: height x width x channels).
    - point: Tuple (x, y) representing the point in the image.
    - save_folder: Folder where the intensity curve will be saved.
    """
    # 加载npy文件
    hyperspectral_image = np.load(npy_file)

    if hyperspectral_image.ndim == 4:
        hyperspectral_image = hyperspectral_image[0]

    if hyperspectral_image.shape[0] == 31:
        hyperspectral_image = hyperspectral_image.transpose((1, 2, 0))
    # 检查形状是否符合预期
    assert hyperspectral_image.shape[2] == 31, "Expected 31 channels in the hyperspectral image."

    # 获取该点在所有通道上的强度值
    x, y = point
    intensity_values = hyperspectral_image[y, x, :]

    # 定义波长范围
    wavelengths = np.linspace(400, 700, 31)

    # 创建保存目录（如果不存在）
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # 绘制强度曲线
    plt.plot(wavelengths, intensity_values, 'o-', label=f'Point ({x}, {y})')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity')
    plt.title('Intensity at Point Across Wavelengths')
    plt.legend()

    # # 保存图像
    # save_path = os.path.join(save_folder, f'intensity_curve_{x}_{y}.png')
    # plt.savefig(save_path)
    # print(f"Intensity curve saved at: {save_path}")

    # 显示图像
    plt.show()

# 调用
npy_file = r'C:\Users\12970\OneDrive\桌面\Yuan\mosaic2hsi\Output\npy\output_0.npy'
point = (400, 400)  # 选择一个点
save_folder = r'..\TestOutput\Fig'
save_intensity_curve_from_npy(npy_file, point, save_folder)
