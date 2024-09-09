# -*-coding: utf-8 -*-
# @Time    : 2024/9/3 9:26
# @Author  : YeLi
# @File    : Disp650_700_error.py
# @Software: PyCharm
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 文件夹路径
gt_path = '/home/wangruozhang/Hyper2Mosaic/Dataset/Extracted_Data/CAVE_ch166_expand'
output_path = '/home/wangruozhang/mosaic2hsi/TestOutput1/npy'
save_path = '/home/wangruozhang/mosaic2hsi/utils/Compare540_590'

# 确保保存目录存在
os.makedirs(save_path, exist_ok=True)

# 获取所有文件名
gt_files = sorted([f for f in os.listdir(gt_path) if f.endswith('.npy') and f.startswith('00')])
output_files = sorted([f for f in os.listdir(output_path) if f.endswith('.npy') and f.startswith('output_')])

# 检查文件数量是否匹配
if len(gt_files) != len(output_files):
    raise ValueError("Ground truth 和 output 文件数量不匹配。")

# 一对一比较并使用 tqdm 可视化进度
for gt_file, output_file in tqdm(zip(gt_files, output_files), total=len(gt_files), desc="Processing files"):
    # 读取.npy 文件
    gt_data = np.load(os.path.join(gt_path, gt_file)).astype(np.float32)  # [0, 65535]
    output_data = np.load(os.path.join(output_path, output_file)).astype(np.float32)
    output_data = output_data.squeeze().transpose((1, 2, 0))  # [0, 1]

    # 归一化到 [0, 1]
    gt_data = (gt_data - gt_data.min()) / (gt_data.max() - gt_data.min())
    output_data = (output_data - output_data.min()) / (output_data.max() - output_data.min())

    # 提取第15个通道开始的6个通道
    gt_selected_channels = gt_data[:, :, 14:20]  # 15th channel is index 14 (0-based index)
    output_selected_channels = output_data[:, :, 14:20]

    # 创建子图
    fig, axs = plt.subplots(2, 6, figsize=(18, 6), constrained_layout=True)

    for i in range(6):
        # 计算差异
        difference = output_selected_channels[:, :, i] - gt_selected_channels[:, :, i]

        # 使用颜色映射显示 GT 和差异
        axs[0, i].imshow(gt_selected_channels[:, :, i], cmap='gray')
        axs[0, i].set_title(f'GT Channel {i + 15}')
        axs[0, i].axis('off')

        im = axs[1, i].imshow(difference, cmap='coolwarm', vmin=-1, vmax=1)
        axs[1, i].set_title(f'Diff Channel {i + 15}')
        axs[1, i].axis('off')

    # 添加颜色条
    cbar = fig.colorbar(im, ax=axs[1, :], orientation='horizontal', fraction=0.05, pad=0.2)
    cbar.ax.tick_params(labelsize=10)  # 调整颜色条刻度标签的大小

    # 添加整体标题
    fig.suptitle(f'Comparison between {gt_file} and {output_file}', fontsize=16)

    # 保存图像到指定目录
    save_filename = os.path.join(save_path, f'Compare_{gt_file[:-4]}.png')
    plt.savefig(save_filename)
    plt.close(fig)  # 关闭图像以释放内存
