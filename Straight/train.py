# -*-coding: utf-8 -*-
# @Time    : 2024/8/29 15:29
# @Author  : YeLi
# @File    : train.py
# @Software: PyCharm
# -*-coding: utf-8 -*-
# @Time    : 2024/8/8 15:04
# @Author  : YeLi
# @File    : train.py
# @Software: PyCharm
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from model import MultiscaleNet, ParallelMultiscaleNet, ResidualNet, MST_Plus_Plus, MST3, MST3C
from dataset import MyDataSet
from tqdm import tqdm
import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt  # 用于绘制 loss 曲线
import datetime

# 获取当前日期
current_date = datetime.datetime.now().strftime('%Y-%m-%d')

def calculate_metrics(outputs, targets):
    psnr_values = []
    ssim_values = []

    max_pixel_value = 1

    for i in range(outputs.shape[0]):  # outputs.shape[0] 是 batch_size
        output_batch = outputs[i].cpu().detach().numpy().astype(np.float32)
        target_batch = targets[i].cpu().detach().numpy().astype(np.float32)

        # output = (output * max_pixel_value).astype(np.float32)
        # target = (target * max_pixel_value).astype(np.float32)

        for c in range(output_batch.shape[0]):  # 这里使用 shape[0] 或 shape[1] 都可以表示通道数
            output_image = output_batch[c]  # 形状 (H, W)
            target_image = target_batch[c]  # 形状 (H, W)

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
    device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    parser = argparse.ArgumentParser(description='Train different models.')
    parser.add_argument('--model', type=str, required=True, choices=['ResNet', 'MultiscaleNet', 'PMSNet', 'MST++', 'MST3', 'MST3C'],
                        help='The model to train: ResNet, MultiscaleNet, MST, MST3, MST3C, or PMSNet')
    args = parser.parse_args()

    crop_size = (256, 256)
    batch_size = 10
    initial_lr = 0.0002
    weight_decay = 0.0001
    num_epochs = 120
    lr_decay_step = 40
    lr_decay_gamma = 0.1

    base_train_data_dir = '/home/wangruozhang/Hyper2Mosaic/Dataset/Mosaic_TIFFs'
    base_gt_data_dir = '/home/wangruozhang/Hyper2Mosaic/Dataset/Extracted_Data'

    data_dirs = sorted([os.path.join(base_train_data_dir, d) for d in os.listdir(base_train_data_dir) if os.path.isdir(os.path.join(base_train_data_dir, d))])
    gt_dirs = sorted([os.path.join(base_gt_data_dir, d) for d in os.listdir(base_gt_data_dir) if os.path.isdir(os.path.join(base_gt_data_dir, d))])

    # 检查目录名称是否对应
    for d_dir, gt_dir in zip(data_dirs, gt_dirs):
        if os.path.basename(d_dir) != os.path.basename(gt_dir):
            raise ValueError(f"Mismatch between data directory {d_dir} and ground truth directory {gt_dir}.")

    datasets = []
    for data_dir, gt_dir in zip(data_dirs, gt_dirs):
        dataset = MyDataSet(dataPath=data_dir, gtPath=gt_dir, crop_size=crop_size)
        if len(dataset) > 0:
            datasets.append(dataset)
        else:
            print(f"Warning: No data found in {data_dir} or {gt_dir}")

    if len(datasets) == 0:
        raise ValueError("No datasets found. Please check the data directories.")

    combined_dataset = torch.utils.data.ConcatDataset(datasets)

    # 数据集划分
    total_length = len(combined_dataset)
    val_length = 30  # 最后30对作为验证集
    train_length = total_length - val_length

    train_subset, val_subset = torch.utils.data.random_split(combined_dataset, [train_length, val_length])

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)

    # 初始化模型
    if args.model == 'ResNet':
        model = ResidualNet.ResNet().to(device)
    elif args.model == 'MultiscaleNet':
        model = MultiscaleNet.MultiscaleNet().to(device)
    elif args.model == 'PMSNet':
        model = ParallelMultiscaleNet.PMSNet().to(device)
    elif args.model == 'MST++':
        model = MST_Plus_Plus.MST_Plus_Plus().to(device)
    elif args.model == 'MST3':
        model = MST3.MST(device).to(device)
    elif args.model == 'MST3C':
        model = MST3C.MST(device).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_gamma)

    epoch_losses = []  # 用于记录每个 epoch 的平均 loss
    best_psnr = -float('inf')
    best_epoch = -1
    best_model_path = None

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

        # 记录当前 epoch 的平均 loss
        avg_loss = running_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        print(f'Epoch {epoch + 1}/{num_epochs} - Training Loss: {avg_loss}')
        scheduler.step()

        # Calculate validation metrics
        model.eval()
        psnr_mean, psnr_var, ssim_mean, ssim_var = 0, 0, 0, 0
        total_loss = 0
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                inputs = data['mosaic'].to(device)
                targets = data['gd'].to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                psnr_mean, psnr_var, ssim_mean, ssim_var = calculate_metrics(outputs, targets)

                total_loss += loss.item()

        epoch_loss = total_loss / len(val_loader)
        print(f'Epoch {epoch + 1}: Loss: {epoch_loss}, PSNR Mean: {psnr_mean}, PSNR Var: {psnr_var}, SSIM Mean: {ssim_mean}, SSIM Var: {ssim_var}')

        # 如果当前PSNR值是最高的，则保存模型
        if psnr_mean > best_psnr:
            best_psnr = psnr_mean
            best_epoch = epoch + 1
            best_model_path = f'/home/wangruozhang/mosaic2hsi/trainedWeights/{args.model}_best_PSNR{best_psnr}_{current_date}.pth'
            torch.save(model.state_dict(), best_model_path)
            print(f'New best model found at epoch {best_epoch} with PSNR: {best_psnr}. Model saved to {best_model_path}')

    # 保存最终模型
    final_model_path = f'/home/wangruozhang/mosaic2hsi/trainedWeights/{args.model}_final_{current_date}.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f'Final model saved to {final_model_path}')

    # 绘制并保存 loss 曲线
    plt.figure()
    plt.plot(range(1, num_epochs + 1), epoch_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    loss_curve_path = f'/home/wangruozhang/mosaic2hsi/trainedWeights/{args.model}_loss_curve_{current_date}.png'
    plt.savefig(loss_curve_path)
    print(f'Loss curve saved to {loss_curve_path}')

    # 输出PSNR最高的epoch
    print(f'Best model found at epoch {best_epoch} with PSNR: {best_psnr}')

def custom_collate_fn(batch):
    # 过滤掉 None 值
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


class PSNRLoss(nn.Module):
    def __init__(self, max_val=1.0):
        super(PSNRLoss, self).__init__()
        self.max_val = max_val

    def forward(self, outputs, targets):
        mse = torch.mean((outputs - targets) ** 2)
        if mse == 0:
            # 如果 MSE 为 0，则 PSNR 无穷大，按照定义返回一个很大的数
            return torch.tensor(float('inf'))
        psnr_loss = 10 * torch.log10((self.max_val ** 2) / mse)
        return -psnr_loss  # 返回负的PSNR，因为我们需要最小化损失函数

if __name__ == '__main__':
    main()
