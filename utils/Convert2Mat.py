import os
import numpy as np
import scipy.io as sio
from PIL import Image
from tqdm import tqdm

# 定义路径
extracted_data_path = '/home/wangruozhang/Hyper2Mosaic/Dataset/Extracted_Data'
mosaic_tiffs_path = '/home/wangruozhang/Hyper2Mosaic/Dataset/Mosaic_TIFFs'
save_path = '/home/wangruozhang/Hyper2Mosaic/Dataset/MAT_Files'

# 确保保存目录存在
os.makedirs(save_path, exist_ok=True)

# 获取所有子文件夹
extracted_subfolders = sorted([os.path.join(extracted_data_path, f) for f in os.listdir(extracted_data_path) if os.path.isdir(os.path.join(extracted_data_path, f))])
mosaic_subfolders = sorted([os.path.join(mosaic_tiffs_path, f) for f in os.listdir(mosaic_tiffs_path) if os.path.isdir(os.path.join(mosaic_tiffs_path, f))])

# 确保两个路径中的子文件夹数量一致
if len(extracted_subfolders) != len(mosaic_subfolders):
    raise ValueError("Extracted_Data 和 Mosaic_TIFFs 子文件夹数量不匹配。")

# 处理每个子文件夹中的文件
for extracted_subfolder, mosaic_subfolder in zip(extracted_subfolders, mosaic_subfolders):
    npy_files = sorted([os.path.join(extracted_subfolder, f) for f in os.listdir(extracted_subfolder) if f.endswith('.npy')])
    tiff_files = sorted([os.path.join(mosaic_subfolder, f) for f in os.listdir(mosaic_subfolder) if f.endswith('.tiff')])

    folder_name = os.path.basename(extracted_subfolder)

    # 逐个处理 .npy 和 .tiff 文件
    for npy_file, tiff_file in zip(npy_files, tiff_files):
        # 读取 .npy 文件数据
        npy_data = np.load(npy_file)

        # 读取 .tiff 文件数据
        img = np.array(Image.open(tiff_file))

        # 创建mosaic图像
        mosaic = np.zeros((img.shape[0], img.shape[1], 3), dtype=img.dtype)
        mosaic[0::2, 0::2, 0] = img[0::2, 0::2]  # Red channel
        mosaic[0::2, 1::2, 1] = img[0::2, 1::2]  # Green channel
        mosaic[1::2, 0::2, 1] = img[1::2, 0::2]  # Green channel
        mosaic[1::2, 1::2, 2] = img[1::2, 1::2]  # Blue channel

        # 创建.mat文件的数据结构
        mat_data = np.zeros((img.shape[0], img.shape[1], 3 + npy_data.shape[2]), dtype=np.float32)

        # 填充.mat文件的前三个通道
        mat_data[:, :, 0] = mosaic[:, :, 2]  # Blue channel
        mat_data[:, :, 1] = mosaic[:, :, 1]  # Green channel (combined)
        mat_data[:, :, 2] = mosaic[:, :, 0]  # Red channel

        # 将31个通道的数据添加到.mat文件中
        mat_data[:, :, 3:] = npy_data

        # 保存为.mat文件
        save_filename = os.path.join(save_path, f'{folder_name}_{os.path.splitext(os.path.basename(tiff_file))[0]}.mat')
        sio.savemat(save_filename, {'dataMat': mat_data})

        print(f"Saved {save_filename}")

print("All files have been processed and saved.")
