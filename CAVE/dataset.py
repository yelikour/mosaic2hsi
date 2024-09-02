import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

def create_mosaic(img):
    """Creates a mosaic image from an RGB image."""
    h, w, _ = img.shape  # height * width * 3
    mosaic = np.zeros((h, w), dtype=img.dtype)  # Create single-channel mosaic

    mosaic[0::2, 0::2] = img[0::2, 0::2, 0]  # Red
    mosaic[0::2, 1::2] = img[0::2, 1::2, 1]  # Green
    mosaic[1::2, 0::2] = img[1::2, 0::2, 1]  # Green
    mosaic[1::2, 1::2] = img[1::2, 1::2, 2]  # Blue

    return mosaic

class MyDataSet(Dataset):
    def __init__(self, root_dir, transform=None, crop_size=(256, 256)):
        self.root_dir = root_dir
        self.transform = transform
        self.crop_size = crop_size

        self.data_list = self._get_data_list()

    def _get_data_list(self):
        data_list = []
        for folder in sorted(os.listdir(self.root_dir)):
            folder_path = os.path.join(self.root_dir, folder)
            if os.path.isdir(folder_path):
                # 进入第二层文件夹
                for subfolder in sorted(os.listdir(folder_path)):
                    subfolder_path = os.path.join(folder_path, subfolder)
                    if os.path.isdir(subfolder_path):
                        base_name = folder.replace('_ms', '')
                        gt_images = []
                        input_image = None

                        # 搜集 名称_ms_01.png 到 名称_ms_31.png
                        for i in range(1, 32):
                            img_name = f"{base_name}_ms_{i:02d}.png"
                            img_path = os.path.join(subfolder_path, img_name)
                            if os.path.exists(img_path):
                                gt_images.append(img_path)
                            else:
                                print(f"Warning: {img_path} not found.")

                        # 搜集 名称_RGB.bmp
                        rgb_img_name = f"{base_name}_RGB.bmp"
                        rgb_img_path = os.path.join(subfolder_path, rgb_img_name)
                        if os.path.exists(rgb_img_path):
                            input_image = rgb_img_path
                        else:
                            print(f"Warning: {rgb_img_path} not found.")

                        if len(gt_images) == 31 and input_image:
                            data_list.append({
                                'input_path': input_image,
                                'gt_paths': gt_images
                            })
                        else:
                            print(f"Warning: Incomplete data in {subfolder_path}.")
        print(f"Total data loaded: {len(data_list)} samples.")
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_info = self.data_list[idx]
        input_path = data_info['input_path']
        gt_paths = data_info['gt_paths']

        # 读取输入图像
        input_data = cv2.imread(input_path)

        # 合并31张图片作为ground truth
        gt_data = [cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) for gt_path in gt_paths]
        gt_data = np.stack(gt_data, axis=-1).astype(np.float32)

        # 应用马赛克转换
        input_data = create_mosaic(input_data)

        # 归一化
        input_data = input_data / 255.0
        gt_data = gt_data / 65535.0

        # 获取图像的尺寸
        h, w = input_data.shape[:2]  # 只获取高度和宽度
        new_h, new_w = self.crop_size

        # 随机裁剪起始位置
        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)

        # 裁剪图像
        input_data = input_data[top:top + new_h, left:left + new_w]
        gt_data = gt_data[top:top + new_h, left:left + new_w, :]

        # 复制 input_data 并在第一维变成3通道
        input_data = np.stack([input_data] * 3, axis=0)

        if self.transform:
            input_data = self.transform(input_data)
            gt_data = self.transform(gt_data)

        input_data = torch.tensor(input_data, dtype=torch.float32)
        gt_data = torch.tensor(gt_data, dtype=torch.float32).permute(2, 0, 1)

        return {'mosaic': input_data, 'gd': gt_data}
