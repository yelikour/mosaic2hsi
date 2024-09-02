# -*-coding: utf-8 -*-
# @Time    : 2024/8/2 9:32
# @Author  : YeLi
# @File    : multiscale_result.py
# @Software: PyCharm
import torch
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from torch.utils.data import DataLoader
from model import MultiscaleNet, ParallelMultiscaleNet, ResidualNet
import dataset

dtype = 'float32'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# Net define
# Residual
PATH = './trainedWeights/ResNetWeights.pt'
Resnet = ResidualNet.ResNet().to(device)
Resnet.load_state_dict(torch.load(PATH, map_location=device))
Resnet.eval()

# Multiscale
PATH = './trainedWeights/MultiscaleNetWeights.pt'
MSnet = MultiscaleNet.MultiscaleNet().to(device)
MSnet.load_state_dict(torch.load(PATH, map_location=device))
MSnet.eval()

# PMS
PATH = './trainedWeights/PMSNetWeights.pt'
PMSnet = ParallelMultiscaleNet.PMSNet().to(device)
PMSnet.load_state_dict(torch.load(PATH, map_location=device))
PMSnet.eval()

# Validation
valSet = dataset.MyDataSet(r'.\ExampleData\camp')
valLoader = DataLoader(valSet, batch_size=1, shuffle=False, num_workers=0)

channels = list(range(31))  # 所有31个通道
with torch.no_grad():
    for i, data in enumerate(valLoader, 0):
        valmasaicPic, valgdPic, picName = data['masaic'].to(device), \
            data['gd'].to(device), \
            data['name'][0]
        Resvaloutput = Resnet(valmasaicPic)
        MSvaloutput = MSnet(valmasaicPic)
        PMSvaloutput = PMSnet(valmasaicPic)

        fig, axes = plt.subplots(len(channels), 7, figsize=(20, len(channels) * 4))
        for idx, channel in enumerate(channels):
            # Ground-truth reference images
            gdImg = valgdPic[0, channel, :, :].cpu().detach().numpy()
            axes[idx, 0].imshow(gdImg, cmap='gray')
            if idx == 0:
                axes[idx, 0].set_title('Reference')
            axes[idx, 0].set_ylabel(f'{400 + idx * 10}nm', rotation=0, labelpad=30)
            axes[idx, 0].axis('off')

            # Residual network results and error maps
            ResreImg = Resvaloutput[0, channel, :, :].cpu().detach().numpy()
            Respsnr = psnr(gdImg, ResreImg)
            axes[idx, 1].imshow(ResreImg, cmap='gray')
            if idx == 0:
                axes[idx, 1].set_title('Residual network')
            axes[idx, 1].axis('off')
            ResError = np.abs(gdImg - ResreImg)
            img1 = axes[idx, 2].imshow(ResError, cmap='hot', vmin=0, vmax=0.2)
            if idx == 0:
                axes[idx, 2].set_title('Residual network\nError map')
            axes[idx, 2].axis('off')

            # Multiscale network results and error maps
            MSreImg = MSvaloutput[0, channel, :, :].cpu().detach().numpy()
            MSpsnr = psnr(gdImg, MSreImg)
            axes[idx, 3].imshow(MSreImg, cmap='gray')
            if idx == 0:
                axes[idx, 3].set_title('Multiscale network')
            axes[idx, 3].axis('off')
            MSError = np.abs(gdImg - MSreImg)
            img2 = axes[idx, 4].imshow(MSError, cmap='hot', vmin=0, vmax=0.2)
            if idx == 0:
                axes[idx, 4].set_title('Multiscale network\nError map')
            axes[idx, 4].axis('off')

            # Parallel Multiscale network results and error maps
            PMSreImg = PMSvaloutput[0, channel, :, :].cpu().detach().numpy()
            PMSpsnr = psnr(gdImg, PMSreImg)
            axes[idx, 5].imshow(PMSreImg, cmap='gray')
            if idx == 0:
                axes[idx, 5].set_title('Parallel-multiscale network')
            axes[idx, 5].axis('off')
            PMSError = np.abs(gdImg - PMSreImg)
            img3 = axes[idx, 6].imshow(PMSError, cmap='hot', vmin=0, vmax=0.2)
            if idx == 0:
                axes[idx, 6].set_title('Parallel-multiscale network\nError map')
            axes[idx, 6].axis('off')

        fig.colorbar(img1, ax=axes[:, 2], fraction=0.046, pad=0.04)
        fig.colorbar(img2, ax=axes[:, 4], fraction=0.046, pad=0.04)
        fig.colorbar(img3, ax=axes[:, 6], fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(f'{picName}_multiscale_results.png')
        plt.show()
