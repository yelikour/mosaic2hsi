# -*-coding: utf-8 -*-
# @Time    : 2024/9/2 17:04
# @Author  : YeLi
# @File    : TestCave0000Fig.py
# @Software: PyCharm
import numpy as np
import matplotlib.pyplot as plt

# 文件路径
file1 = '/home/wangruozhang/Hyper2Mosaic/Dataset/Extracted_Data/CAVE_ch166_expand/0000.npy'
file2 = '/home/wangruozhang/mosaic2hsi/TestOutput1/npy/output_0.npy'

# 读取 .npy 文件
data1 = np.load(file1)
data2 = np.load(file2)
data2 = data2.squeeze().transpose((1, 2, 0))

print(data1.shape)
print(data2.shape)

point = (200, 200)

# 提取该点的光谱数据
spectrum1 = data1[point[0], point[1], :]
spectrum2 = data2[point[0], point[1], :]

# 绘制光谱图
plt.figure(figsize=(10, 6))
plt.plot(spectrum1, label='File 1 - CAVE_ch166_expand/0000.npy')
plt.plot(spectrum2, label='File 2 - mosaic2hsi/output_0.npy')
plt.xlabel('Wavelength Index')
plt.ylabel('Intensity')
plt.title(f'Spectrum Comparison at Point {point}')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('Com.png')

print("Figure saved successfully")