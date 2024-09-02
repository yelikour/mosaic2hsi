# -*-coding: utf-8 -*-
# @Time    : 2024/8/30 9:35
# @Author  : YeLi
# @File    : 166HSIPlot.py
# @Software: PyCharm
import os
import numpy as np
import matplotlib.pyplot as plt

def save_intensity_curve_from_npy_166(npy_file, point, save_folder):
    """
    从npy文件中提取特定点在每个通道上的强度值，并保存为强度曲线图像

    参数：
    - npy_file：包含高光谱图像的.npy文件路径（形状：高度x宽度x通道）
    - point：图像中的点，以元组形式表示(x, y)
    - save_folder：保存强度曲线的文件夹路径
    """
    # 加载npy数据
    hyperspectral_image = np.load(npy_file)

    # 检查数据形状是否为预期的166通道
    assert hyperspectral_image.shape[2] == 166, "Expected 166 channels in the hyperspectral image."

    # 获取指定点的强度值
    x, y = point
    intensity_values = hyperspectral_image[y, x, :]

    # 从提供的波长列表中获取波长
    wavelengths = [400.0, 400.7, 401.3, 402.0, 402.7, 403.4, 404.1, 404.8, 405.5, 406.2, 407.0, 407.7, 408.4, 409.1, 409.9,
               410.6, 411.4, 412.1, 412.9, 413.7, 414.4, 415.2, 416.0, 416.8, 417.6, 418.4, 419.2, 420.0, 420.8, 421.6,
               422.5, 423.3, 424.2, 425.0, 425.9, 426.7, 427.6, 428.5, 429.4, 430.3, 431.2, 432.1, 433.0, 433.9, 434.9,
               435.8, 436.8, 437.7, 438.7, 439.7, 440.7, 441.6, 442.6, 443.7, 444.7, 445.7, 446.8, 447.8, 448.9, 449.9,
               451.0, 452.1, 453.2, 454.3, 455.4, 456.6, 457.7, 458.9, 460.1, 461.2, 462.4, 463.6, 464.9, 466.1, 467.3,
               468.6, 469.9, 471.2, 472.5, 473.8, 475.1, 476.5, 477.8, 479.2, 480.6, 482.0, 483.4, 484.9, 486.4, 487.8,
               489.3, 490.9, 492.4, 493.9, 495.5, 497.1, 498.7, 500.4, 502.0, 503.7, 505.4, 507.1, 508.9, 510.6, 512.4,
               514.3, 516.1, 518.0, 519.9, 521.8, 523.7, 525.7, 527.7, 529.8, 531.8, 533.9, 536.1, 538.2, 540.4, 542.7,
               544.9, 547.2, 549.5, 551.9, 554.3, 556.8, 559.3, 561.8, 564.3, 567.0, 569.6, 572.3, 575.0, 577.8, 580.7,
               583.5, 586.5, 589.5, 592.5, 595.6, 598.7, 602.0, 605.2, 608.5, 611.9, 615.3, 618.9, 622.4, 626.1, 629.8,
               633.6, 637.4, 641.4, 645.4, 649.5, 653.7, 658.0, 662.4, 666.8, 671.4, 676.1, 680.8, 685.7, 690.7, 695.9,
               701.1]

    # 创建保存目录（如果不存在）
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # 绘制强度曲线
    plt.plot(wavelengths, intensity_values, 'o-', label=f'Point ({x}, {y})')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity')
    plt.title('Intensity at Point Across Wavelengths')
    plt.legend()

    # # 保存图像
    # save_path = os.path.join(save_folder, f'intensity_curve_{x}_{y}.png')
    # plt.savefig(save_path)
    # print(f"Intensity curve saved at: {save_path}")

    # 显示图像
    plt.show()

# 示例调用
npy_file = r'C:\Users\12970\OneDrive\桌面\Yuan\mosaic2hsi\ExampleData\0000.npy'
point = (200, 200)  # 选择一个点
save_folder = r'..\TestOutput\Fig'
save_intensity_curve_from_npy_166(npy_file, point, save_folder)
