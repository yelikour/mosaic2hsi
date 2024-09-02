# -*-coding: utf-8 -*-
# @Time    : 2024/9/2 9:06
# @Author  : YeLi
# @File    : CompareRGB_Y.py
# @Software: PyCharm
import cv2
import numpy as np

# 读取图像文件
path_rgb32 = r"..\ExampleData\0828\Back_RGB32_92Hz.tiff"
path_rgb64 = r"..\ExampleData\0828\Back_RGB64_92Hz.tiff"
path_y16 = r"..\ExampleData\0828\Back_Y16_92Hz.tiff"

# 使用OpenCV读取图像
image_rgb32 = cv2.imread(path_rgb32, cv2.IMREAD_UNCHANGED)
image_rgb64 = cv2.imread(path_rgb64, cv2.IMREAD_UNCHANGED)
image_y16 = cv2.imread(path_y16, cv2.IMREAD_UNCHANGED)

# 检查图像的形状
# print(f"RGB32 Shape: {image_rgb32.shape}")
# print(f"RGB64 Shape: {image_rgb64.shape}")
# print(f"Y16 Shape: {image_y16.shape}")
'''
RGB32 Shape: (1080, 1440, 4)
RGB64 Shape: (1080, 1440, 4)
Y16 Shape: (1080, 1440)
'''

# 分别提取通道
r32, g32, b32, alpha32 = cv2.split(image_rgb32)
r64, g64, b64, alpha64 = cv2.split(image_rgb64)

# # Cal and Display the maximum of each channel
# max_r32 = np.max(r32)
# max_g32 = np.max(g32)
# max_b32 = np.max(b32)
#
# max_r64 = np.max(r64)
# max_g64 = np.max(g64)
# max_b64 = np.max(b64)
#
# max_y16 = np.max(image_y16)
#
# print("Max pixel values:")
# print(f"RGB32: R={max_r32}, G={max_g32}, B={max_b32}")
# print(f"RGB64: R={max_r64}, G={max_g64}, B={max_b64}")
# print(f"Y16: {max_y16}")
'''
Max pixel values:
RGB32: R=255, G=255, B=182
RGB64: R=65535, G=65535, B=47676
Y16: 65472
'''

# Normalization
r32 = r32 / 255
g32 = g32 / 255
b32 = b32 / 255

r64 = r64 / 65535
g64 = g64 / 65535
b64 = b64 / 65535

image_y16 = image_y16 / 65535

# # 检查Y16与RGB的指定位置是否相同
# match_red = np.all(image_y16[0::2, 0::2] == r32[0::2, 0::2])
# match_green_1 = np.all(image_y16[0::2, 1::2] == g32[0::2, 1::2])
# match_green_2 = np.all(image_y16[1::2, 0::2] == g32[1::2, 0::2])
# match_blue = np.all(image_y16[1::2, 1::2] == b32[1::2, 1::2])
#
# # 打印比较结果
# print(f"Red Channel Equality: {np.all(match_red)}")
# print(f"Green Channel 1 Equality: {np.all(match_green_1)}")
# print(f"Green Channel 2 Equality: {np.all(match_green_2)}")
# print(f"Blue Channel Equality: {np.all(match_blue)}")
'''
Y16 Shape: (1080, 1440)
Red Channel Equality: False
Green Channel 1 Equality: False
Green Channel 2 Equality: False
Blue Channel Equality: False
'''

# Define the module needed to be printed
module_size = 2
rows, cols = np.arange(module_size), np.arange(module_size)

# Print and compare the pixel value of the module
print("RGB32 vs Y16 Shape: ({}, {})".format(rows, cols))
for i in rows:
    for j in cols:
        print(f"Pixel ({i},{j}): R32={r32[i,j]}, G32={g32[i,j]}, B32={b32[i,j]}, Y16={image_y16[i,j]}")

# Print and compare the pixel value of the module
print("RGB64 vs Y16 Shape: ({}, {})".format(rows, cols))
for i in rows:
    for j in cols:
        print(f"Pixel ({i},{j}): R64={r64[i,j]}, G64={g64[i,j]}, B64={b64[i,j]}, Y16={image_y16[i,j]}")

