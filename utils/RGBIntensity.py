import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import tifffile


def load_rgb_data(rgb_curve_path):
    # 定义文件路径
    red_file_path = os.path.join(rgb_curve_path, 'New_Red.csv')
    green_file_path = os.path.join(rgb_curve_path, 'New_Green.csv')
    blue_file_path = os.path.join(rgb_curve_path, 'New_Blue.csv')

    # 读取数据
    red_data = pd.read_csv(red_file_path, delimiter=',', header=None, names=['wavelength', 'red_proportion'])
    green_data = pd.read_csv(green_file_path, delimiter=',', header=None, names=['wavelength', 'green_proportion'])
    blue_data = pd.read_csv(blue_file_path, delimiter=',', header=None, names=['wavelength', 'blue_proportion'])

    # 过滤波长在 400-700 nm 范围内的数据
    red_data = red_data[(red_data['wavelength'] >= 400) & (red_data['wavelength'] <= 700)]
    green_data = green_data[(green_data['wavelength'] >= 400) & (green_data['wavelength'] <= 700)]
    blue_data = blue_data[(blue_data['wavelength'] >= 400) & (blue_data['wavelength'] <= 700)]

    return red_data, green_data, blue_data


def interpolate_rgb_data(red_data, green_data, blue_data, wavelengths):
    # 定义插值函数
    red_interp = interp1d(red_data['wavelength'], red_data['red_proportion'], kind='linear', fill_value="extrapolate")
    green_interp = interp1d(green_data['wavelength'], green_data['green_proportion'], kind='linear',
                            fill_value="extrapolate")
    blue_interp = interp1d(blue_data['wavelength'], blue_data['blue_proportion'], kind='linear',
                           fill_value="extrapolate")

    # 插值到相同的波长
    red_values = red_interp(wavelengths)
    green_values = green_interp(wavelengths)
    blue_values = blue_interp(wavelengths)

    return red_values, green_values, blue_values


def plot_combined_intensity_curve(image_path, npy_file, rgb_curve_path, point, save_folder):
    """
    Plot and save the combined intensity curve for an RGB image and .npy file at a specific point.

    Parameters:
    - image_path: Path to the RGB image.
    - npy_file: Path to the .npy file containing the hyperspectral image (shape: height x width x channels).
    - rgb_curve_path: Path to the folder containing RGB curve data (CSV files).
    - point: Tuple (x, y) representing the point in the image.
    - save_folder: Folder where the intensity curve will be saved.
    """
    # 加载 RGB 图像
    rgb_image = tifffile.imread(image_path)

    # 加载npy文件
    hyperspectral_image = np.load(npy_file)

    hyperspectral_image = hyperspectral_image.squeeze().transpose((1, 2, 0))

    # 检查形状是否符合预期
    assert hyperspectral_image.shape[2] == 31, "Expected 31 channels in the hyperspectral image."

    # 获取点的坐标
    x, y = point

    # 获取该点在图像中的 RGB 值
    r_value = rgb_image[y, x, 0]
    g_value = rgb_image[y, x, 1]
    b_value = rgb_image[y, x, 2]

    # 加载 RGB 曲线数据
    red_data, green_data, blue_data = load_rgb_data(rgb_curve_path)

    # 定义波长范围
    wavelengths = np.linspace(400, 700, 31)

    # 插值 RGB 数据到相同波长范围
    r_intensity, g_intensity, b_intensity = interpolate_rgb_data(red_data, green_data, blue_data, wavelengths)

    # 计算每个波长对应的 RGB 强度
    r_intensity *= r_value
    g_intensity *= g_value
    b_intensity *= b_value

    # 将 RGB 强度相加，并归一化
    total_rgb_intensity = r_intensity + g_intensity + b_intensity
    total_rgb_intensity_normalized = total_rgb_intensity / np.max(total_rgb_intensity)

    # 获取该点在所有通道上的强度值并归一化
    npy_intensity_values = hyperspectral_image[y, x, :]
    npy_intensity_normalized = npy_intensity_values / np.max(npy_intensity_values)

    # 绘制强度曲线
    plt.plot(wavelengths, total_rgb_intensity_normalized, 'k-', label='Total RGB (Normalized)')
    plt.plot(wavelengths, npy_intensity_normalized, 'm-', label='Hyperspectral (Normalized)')

    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Normalized Intensity')
    plt.title(f'Combined Intensity at Point ({x}, {y})')
    plt.legend()

    # 创建保存目录（如果不存在）
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # 保存图像
    save_path = os.path.join(save_folder, f'combined_intensity_curve_point_{x}_{y}.png')
    plt.savefig(save_path)
    print(f"Combined intensity curve saved at: {save_path}")

    # 显示图像
    plt.show()


def main():
    image_path = r'../ExampleData/0828/Back_RGB32_92Hz.tiff'
    npy_file = r'../Output/npy/output_0.npy'
    rgb_curve_path = '../RGBTransCurve'
    point = (1000, 850)  # 选择一个点
    save_folder = '../TestOutput/Fig'

    plot_combined_intensity_curve(image_path, npy_file, rgb_curve_path, point, save_folder)


if __name__ == "__main__":
    main()
