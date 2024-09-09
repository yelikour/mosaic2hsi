# -*-coding: utf-8 -*-
# @Time    : 2024/9/9 12:24
# @Author  : YeLi
# @File    : Disp27chHSIDiff.py
# @Software: PyCharm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

#############################################################
# base_file_path = r'ExampleData\27CompareSpectrum'            #
# name = '0000'                                               #
# 指定你要比较的点的坐标 (height, width)                        #
''''''''''''''''''''''''                                    #
point = (430, 220)                                          #
''''''''''''''''''''''''                                    #
#############################################################

# file2 = os.path.join(base_file_path, f'{name}.npy')
# file4 = os.path.join(base_file_path, f'{name}_final.npy')

# 读取 .npy 文件
original_data = np.load(r"C:\Users\12970\OneDrive\桌面\Yuan\mosaic2hsi\ExampleData\27CompareSpectrum'\0004.npy")  # shape: (1024, 1024, 31)
outputNpy = np.load(r"C:\Users\12970\OneDrive\桌面\Yuan\mosaic2hsi\ExampleData\27CompareSpectrum'\output_4.npy") # shape: (27, 1024, 1024)
image = Image.open(r"C:\Users\12970\OneDrive\桌面\Yuan\mosaic2hsi\ExampleData\CompareSpectrum\0004.png")

# 移除 data1 中的批次维度
outputNpy = outputNpy[0]

# 只保留前27个通道的数据
original_data = original_data[:, :, :27]
print(outputNpy.shape)

spectrum2 = original_data[point[0], point[1], :]  # 从 original_data 中提取
spectrum4 = outputNpy[:, point[0], point[1]]

spectrum2_normalized = spectrum2/ 65536



# 检查归一化后的光谱数据是否为零
if np.all(spectrum2_normalized == 0):
    print(f"Warning: All normalized values in spectrum2 at point {point} are zero.")


# 绘制归一化后的光谱图，x轴范围为400-700nm
wavelengths = np.linspace(400, 660, 27)

# 调整图像大小以拉长x轴
plt.figure(figsize=(12, 6))  # 将宽度设置为12，拉长x轴

plt.subplot(1, 2, 1)
plt.plot(wavelengths, spectrum2_normalized, label='Original (normalized)')
plt.plot(wavelengths, spectrum4, label='Final (normalized)')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Normalized Intensity')
plt.title(f'Normalized Spectrum Comparison at Point {point}')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.imshow(image)
plt.scatter([point[1]], [point[0]], c='r', s=3, label=f'Point {point}')
plt.legend()

plt.show()

