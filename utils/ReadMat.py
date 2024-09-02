# -*-coding: utf-8 -*-
# @Time    : 2024/8/30 13:41
# @Author  : YeLi
# @File    : ReadMat.py
# @Software: PyCharm
import scipy.io

# 指定文件路径
file_path = r'C:\Users\12970\OneDrive\桌面\Yuan\mosaic2hsi\ExampleData\BGU\BGU_0522-1136.mat'

# 读取 .mat 文件
mat_data = scipy.io.loadmat(file_path)

# 打印 .mat 文件中的所有变量名
print("Variables in the .mat file:", mat_data.keys())

# 获取并打印特定变量的数据内容及其形状
# 假设变量名为 'data'，请根据实际变量名修改
variable_name = 'dataMat'  # 你需要根据实际情况修改这个变量名
if variable_name in mat_data:
    data = mat_data[variable_name]
    print(f"Data shape: {data.shape}")
    print(f"Data content:\n{data}")
else:
    print(f"Variable '{variable_name}' not found in the .mat file.")
