# -*-coding: utf-8 -*-
# @Time    : 2024/8/8 15:04
# @Author  : YeLi
# @File    : CAVECropTrain.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import argparse
from model import MultiscaleNet, ParallelMultiscaleNet, ResidualNet
from dataset import MyDataSet
from sklearn.model_selection import KFold
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import numpy as np


def calculate_metrics(outputs, targets):
    psnr_values = []
    ssim_values = []

    for i in range(outputs.shape[0]):
        output = outputs[i].cpu().detach().numpy().transpose(1, 2, 0)
        target = targets[i].cpu().detach().numpy().transpose(1, 2, 0)
        psnr_values.append(psnr(target, output))
        ssim_values.append(ssim(target, output, multichannel=True, data_range=output.max() - output.min()))

    psnr_mean = np.mean(psnr_values)
    psnr_var = np.var(psnr_values)
    ssim_mean = np.mean(ssim_values)
    ssim_var = np.var(ssim_values)

    return psnr_mean, psnr_var, ssim_mean, ssim_var


def random_crop(inputs, targets, crop_size=(128, 128)):
    _, _, h, w = inputs.size()
    new_h, new_w = crop_size
    top = np.random.randint(0, h - new_h)
    left = np.random.randint(0, w - new_w)
    inputs = inputs[:, :, top:top + new_h, left:left + new_w]
    targets = targets[:, :, top:top + new_h, left:left + new_w]
    return inputs, targets


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    parser = argparse.ArgumentParser(description='Train different models on CAVE dataset.')
    parser.add_argument('--model', type=str, required=True, choices=['ResNet', 'MultiscaleNet', 'PMSNet'],
                        help='The model to train: ResNet, MultiscaleNet, or PMSNet')
    args = parser.parse_args()

    batch_size = 32
    initial_lr = 0.001
    weight_decay = 0.0001
    num_epochs = 120
    lr_decay_step = 40
    lr_decay_gamma = 0.1
    num_folds = 5

    train_data_dir = '/home/wangruozhang/Hyper2Mosaic/Dataset/Output_images/CAVE_ch166_expand'
    gt_data_dir = '/home/wangruozhang/Hyper2Mosaic/Dataset/Extracted_Data/CAVE_ch166_expand'
    dataset = MyDataSet(dataPath=train_data_dir, gtPath=gt_data_dir)

    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    # K-fold Cross Validation model evaluation
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f'Fold {fold + 1}/{num_folds}')

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4)

        if args.model == 'ResNet':
            model = ResidualNet.ResNet().to(device)
        elif args.model == 'MultiscaleNet':
            model = MultiscaleNet.MultiscaleNet().to(device)
        elif args.model == 'PMSNet':
            model = ParallelMultiscaleNet.PMSNet().to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_gamma)

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch')

            for i, data in enumerate(progress_bar):
                inputs = data['masaic'].to(device)
                targets = data['gd'].to(device)

                # Random crop
                inputs, targets = random_crop(inputs, targets)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                progress_bar.set_postfix({'loss': running_loss / (i + 1)})

            scheduler.step()

            # Calculate validation metrics
            model.eval()
            psnr_mean, psnr_var, ssim_mean, ssim_var = 0, 0, 0, 0
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    inputs = data['masaic'].to(device)
                    targets = data['gd'].to(device)
                    outputs = model(inputs)
                    psnr_mean, psnr_var, ssim_mean, ssim_var = calculate_metrics(outputs, targets)

            print(
                f'Fold {fold + 1}, Epoch {epoch + 1}: PSNR Mean: {psnr_mean}, PSNR Var: {psnr_var}, SSIM Mean: {ssim_mean}, SSIM Var: {ssim_var}')

        model_path = f'/home/wangruozhang/mosaic2hsi/trainedWeights/{args.model}_fold{fold + 1}_CAVEselfCrop_Weights_trained.pth'
        torch.save(model.state_dict(), model_path)
        print(f'Model for fold {fold + 1} saved to {model_path}')

    print('Finished Cross-Validation Training')


if __name__ == '__main__':
    main()
