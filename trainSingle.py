# -*-coding: utf-8 -*-
# @Time    : 2024/8/8 15:04
# @Author  : YeLi
# @File    : train.py
# @Software: PyCharm
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset
import argparse
from model import MultiscaleNet, ParallelMultiscaleNet, MST, ResidualNet, MST3
from datasetSingleChannel import MyDataSet
from sklearn.model_selection import KFold
from tqdm import tqdm
import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def calculate_metrics(outputs, targets):
    psnr_values = []
    ssim_values = []

    max_pixel_value = 65535.0

    for i in range(outputs.shape[0]):  # outputs.shape[0] 是 batch_size
        output = outputs[i].cpu().detach().numpy()
        target = targets[i].cpu().detach().numpy()

        output = (output * max_pixel_value).astype(np.float32)
        target = (target * max_pixel_value).astype(np.float32)

        for c in range(output.shape[0]):  # 这里使用 shape[0] 或 shape[1] 都可以表示通道数
            output_image = output[c]  # 形状 (H, W)
            target_image = target[c]  # 形状 (H, W)

            psnr_value = psnr(target_image, output_image, data_range=max_pixel_value)
            ssim_value = ssim(target_image, output_image, data_range=max_pixel_value)

            psnr_values.append(psnr_value)
            ssim_values.append(ssim_value)

    psnr_mean = np.mean(psnr_values)
    psnr_var = np.var(psnr_values)
    ssim_mean = np.mean(ssim_values)
    ssim_var = np.var(ssim_values)

    return psnr_mean, psnr_var, ssim_mean, ssim_var


def main():
    device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    parser = argparse.ArgumentParser(description='Train different models.')
    parser.add_argument('--model', type=str, required=True, choices=['ResNet', 'MultiscaleNet', 'PMSNet', 'MST', 'MST3'],
                        help='The model to train: ResNet, MultiscaleNet, MST, or PMSNet')
    args = parser.parse_args()

    crop_size = (256, 256)
    batch_size = 8
    initial_lr = 0.001
    weight_decay = 0.0001
    num_epochs = 120
    lr_decay_step = 40
    lr_decay_gamma = 0.1
    num_folds = 5

    base_train_data_dir = '/home/wangruozhang/Hyper2Mosaic/Dataset/RGBimages_LED'
    base_gt_data_dir = '/home/wangruozhang/Hyper2Mosaic/Dataset/Extracted_Data'

    data_dirs = [os.path.join(base_train_data_dir, d) for d in os.listdir(base_train_data_dir) if os.path.isdir(os.path.join(base_train_data_dir, d))]
    gt_dirs = [os.path.join(base_gt_data_dir, d) for d in os.listdir(base_gt_data_dir) if os.path.isdir(os.path.join(base_gt_data_dir, d))]

    datasets = []
    for data_dir, gt_dir in zip(data_dirs, gt_dirs):
        dataset = MyDataSet(dataPath=data_dir, gtPath=gt_dir, crop_size=crop_size)  # Added crop size parameter
        if len(dataset) > 0:
            datasets.append(dataset)
        else:
            print(f"Warning: No data found in {data_dir} or {gt_dir}")

    if len(datasets) == 0:
        raise ValueError("No datasets found. Please check the data directories.")

    combined_dataset = ConcatDataset(datasets)

    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    # K-fold Cross Validation model evaluation
    for fold, (train_idx, val_idx) in enumerate(kfold.split(combined_dataset)):
        print(f'Fold {fold + 1}/{num_folds}')

        train_subset = Subset(combined_dataset, train_idx)
        val_subset = Subset(combined_dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)

        if args.model == 'ResNet':
            model = ResidualNet.ResNet().to(device)
        elif args.model == 'MultiscaleNet':
            model = MultiscaleNet.MultiscaleNet().to(device)
        elif args.model == 'PMSNet':
            model = ParallelMultiscaleNet.PMSNet().to(device)
        elif args.model == 'MST':
            model = MST.MST(device).to(device)
        elif args.model == 'MST3':
            model = MST3.MST(device).to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_gamma)

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch')

            start_time = time.time()  # Start time for epoch
            for i, data in enumerate(progress_bar):
                inputs = data['mosaic'].to(device)
                targets = data['gd'].to(device)

                optimizer.zero_grad()
                outputs = model(inputs)

                # print(f"outputs shape: {outputs.shape}")
                # print(f"targets shape: {targets.shape}")

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                progress_bar.set_postfix({'loss': running_loss / (i + 1)})

                # Calculate estimated time remaining
                elapsed_time = time.time() - start_time
                batches_done = i + 1
                batches_total = len(train_loader)
                estimated_total_time = elapsed_time * (batches_total / batches_done)
                remaining_time = estimated_total_time - elapsed_time

                # Format remaining time as hours, minutes, seconds
                hours, rem = divmod(remaining_time, 3600)
                minutes, seconds = divmod(rem, 60)
                progress_bar.set_postfix(remaining_time=f'{int(hours):02}:{int(minutes):02}:{int(seconds):02}')

            scheduler.step()

            # Calculate validation metrics
            model.eval()
            psnr_mean, psnr_var, ssim_mean, ssim_var = 0, 0, 0, 0
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    inputs = data['mosaic'].to(device)
                    targets = data['gd'].to(device)

                    outputs = model(inputs)
                    psnr_mean, psnr_var, ssim_mean, ssim_var = calculate_metrics(outputs, targets)

            print(
                f'Fold {fold + 1}, Epoch {epoch + 1}: PSNR Mean: {psnr_mean}, PSNR Var: {psnr_var}, SSIM Mean: {ssim_mean}, SSIM Var: {ssim_var}')

        model_path = f'/home/wangruozhang/mosaic2hsi/trainedWeights/{args.model}_fold{fold + 1}_CAVEselfCrop_Weights_trained.pth'
        torch.save(model.state_dict(), model_path)
        print(f'Model for fold {fold + 1} saved to {model_path}')

    print('Finished Cross-Validation Training')

def custom_collate_fn(batch):
    # 过滤掉 None 值
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


if __name__ == '__main__':
    main()