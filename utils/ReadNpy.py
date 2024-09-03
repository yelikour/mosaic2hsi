# -*-coding: utf-8 -*-
# @Time    : 2024/9/2 16:20
# @Author  : YeLi
# @File    : ReadNpy.py
# @Software: PyCharm
import numpy as np
import matplotlib.pyplot as plt

# 指定要读取的 .npy 文件的路径
file_path = "../ExampleData/CompareSpectrum/output_0.npy"

# 使用 numpy 的 load 函数读取 .npy 文件
data = np.load(file_path)

# 提取两列数据：第一列是波长，第二列是对应的值
wavelengths = data[:, 0]
values = data[:, 1]

# 创建图表
plt.figure(figsize=(10, 6))
plt.plot(wavelengths, values, marker='o')

# 添加标签和标题
plt.xlabel("Wavelength (nm)")
plt.ylabel("Value")
plt.title("Wavelength vs. Value")

# 显示网格
plt.grid(True)

# 显示图表
plt.show()