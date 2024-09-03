import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

base_file_path = 'ExampleData/CompareSpectrum'
name = 'kaist1_0004_0'
# 指定你要比较的点的坐标 (height, width)
point = (400, 100)  # 例如坐标为 (100, 200)

file1 = os.path.join(base_file_path, f'{name}_output.npy')
file2 = os.path.join(base_file_path, f'{name}.npy')
image1_path = os.path.join(base_file_path, f'{name}_output.tiff')
image2_path = os.path.join(base_file_path, f'{name}.png')

# 读取 .npy 文件
data1 = np.load(file1)  # shape: (1, 31, 1024, 1024)
data2 = np.load(file2)  # shape: (1024, 1024, 31)

# 移除 data1 中的批次维度
data1 = data1[0]  # 现在 shape 为 (31, 1024, 1024)

# 读取图像文件
image1 = Image.open(image1_path)
image2 = Image.open(image2_path)

# 提取该点的光谱数据
spectrum1 = data1[:, point[0], point[1]]  # 从 data1 中提取
spectrum2 = data2[point[0], point[1], :]  # 从 data2 中提取

# 归一化处理
spectrum1_min = spectrum1.min()
spectrum1_max = spectrum1.max()
spectrum1_normalized = (spectrum1 - spectrum1_min) / (spectrum1_max - spectrum1_min)

spectrum2_min = spectrum2.min()
spectrum2_max = spectrum2.max()
spectrum2_normalized = (spectrum2 - spectrum2_min) / (spectrum2_max - spectrum2_min)

# 检查归一化后的光谱数据是否为零
if np.all(spectrum1_normalized == 0):
    print(f"Warning: All normalized values in spectrum1 at point {point} are zero.")
if np.all(spectrum2_normalized == 0):
    print(f"Warning: All normalized values in spectrum2 at point {point} are zero.")

# 显示并标记图片
plt.figure(figsize=(12, 6))

# 显示 output_0.tiff 并标记点
plt.subplot(1, 2, 1)
plt.imshow(image1)
plt.scatter([point[1]], [point[0]], c='red', s=50, label=f'Point {point}')
plt.title('output_0.tiff')
plt.legend()

# 显示 0000.png 并标记点
plt.subplot(1, 2, 2)
plt.imshow(image2)
plt.scatter([point[1]], [point[0]], c='red', s=50, label=f'Point {point}')
plt.title('0000.png')
plt.legend()

plt.show()

# 绘制归一化后的光谱图，x轴范围为400-700nm
wavelengths = np.linspace(400, 700, 31)

plt.figure(figsize=(10, 6))
plt.plot(wavelengths, spectrum1_normalized, label='Output (normalized)')
plt.plot(wavelengths, spectrum2_normalized, label='Original (normalized)')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Normalized Intensity')
plt.title(f'Normalized Spectrum Comparison at Point {point}')
plt.legend()
plt.grid(True)
plt.show()
