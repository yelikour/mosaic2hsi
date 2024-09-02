# -*-coding: utf-8 -*-
# @Time    : 2024/8/28 12:24
# @Author  : YeLi
# @File    : view_tiff_shape.py
# @Software: PyCharm
from PIL import Image
import numpy as np

# 文件路径
file_path = r"C:\Users\12970\OneDrive\桌面\Yuan\mosaic2hsi\kaist1_0025_5.tiff"

# 打开图像并查看其大小（形状）
with Image.open(file_path) as img:
    print(f"Image shape: {img.size} (Width x Height)")
    print(f"Number of channels: {len(img.getbands())}")

    # 将图像转换为 NumPy 数组
    img_array = np.array(img)

    # 检查图像的形状
    print(f"Image array shape: {img_array.shape} (Height x Width x Channels)")

    # 如果图像有多个通道，则分别处理每个通道
    if len(img_array.shape) == 3:
        for i, band in enumerate(img.getbands()):
            channel_matrix = img_array[:, :, i]
            max_value = np.max(channel_matrix)
            min_value = np.min(channel_matrix)
            mean_value = np.mean(channel_matrix)
            print(f"Channel {band} matrix:\n{channel_matrix}\n")
            print(f"Channel {band} statistics - Max: {max_value}, Min: {min_value}, Mean: {mean_value}\n")
