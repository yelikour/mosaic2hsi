# -*-coding: utf-8 -*-
# @Time    : 2024/8/30 11:30
# @Author  : YeLi
# @File    : MaxInImage.py
# @Software: PyCharm
from PIL import Image

# 替换为你的图像文件路径
image_path = r'G:\DataSets\CAVEdataset\balloons_ms\balloons_ms\balloons_rgb.bmp'

# 打开图像文件
image = Image.open(image_path)

# 获取图像数据
image_data = list(image.getdata())

# 获取最大像素值
max_value = max(image_data)

# 打印结果
print("图像最大值:", max_value)
