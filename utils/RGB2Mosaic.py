# -*-coding: utf-8 -*-
# @Time    : 2024/8/28 16:55
# @Author  : YeLi
# @File    : RGB2Mosaic.py
# @Software: PyCharm
import os
import argparse
import numpy as np
import cv2
from tqdm import tqdm
import tifffile


def create_mosaic(img):
    """Creates a mosaic image from an RGB image."""
    h, w, _ = img.shape  # height * width * 3
    mosaic = np.zeros((h, w), dtype=img.dtype)  # Create single-channel mosaic

    mosaic[0::2, 0::2] = img[0::2, 0::2, 0]  # Red
    mosaic[0::2, 1::2] = img[0::2, 1::2, 1]  # Green
    mosaic[1::2, 0::2] = img[1::2, 0::2, 1]  # Green
    mosaic[1::2, 1::2] = img[1::2, 1::2, 2]  # Blue

    return mosaic


def convert_png_to_mosaic_tiff(input_folder, output_folder):
    """
    Converts all PNG files in the input folder to mosaic TIFF images and saves them in the output folder.

    Parameters:
    - input_folder: Path to the folder containing PNG files.
    - output_folder: Path to the folder where TIFF files will be saved.
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历文件并显示进度条
    files = [file for root, _, files in os.walk(input_folder) for file in files if file.endswith('.png')]
    for file in tqdm(files, desc=f"Processing folder {os.path.basename(input_folder)}"):
        image_path = os.path.join(input_folder, file)
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to load image: {image_path}")
            continue

        # 创建马赛克图像
        mosaic_img = create_mosaic(img)

        # 构建输出文件路径
        output_file = os.path.join(output_folder, file.replace('.png', '.tiff'))

        # 保存为TIFF格式
        tifffile.imwrite(output_file, mosaic_img)
        print(f"Mosaic image saved at: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Convert PNG images to mosaic TIFF images")
    parser.add_argument('--RGBfolder', type=str, default='/home/wangruozhang/Hyper2Mosaic/Dataset/RGBimages_LED',
                        help="Path to the folder containing the subfolders with PNG images")
    parser.add_argument('--output_folder', type=str, default='/home/wangruozhang/Hyper2Mosaic/Dataset/Mosaic_TIFFs',
                        help="Folder to store output TIFF images")
    args = parser.parse_args()

    subfolders = [f.path for f in os.scandir(args.RGBfolder) if f.is_dir()]

    for subfolder in subfolders:
        subfolder_name = os.path.basename(subfolder)
        output_subfolder = os.path.join(args.output_folder, subfolder_name)
        convert_png_to_mosaic_tiff(subfolder, output_subfolder)


if __name__ == "__main__":
    main()
