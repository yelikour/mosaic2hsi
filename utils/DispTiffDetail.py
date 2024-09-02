from PIL import Image

# 读取 TIFF 图像
image = Image.open(r'C:\Users\12970\OneDrive\桌面\Yuan\mosaic2hsi\Output\RGB\output_0.tiff')

# 获取图像的通道数
num_channels = len(image.getbands())

# 创建一个空的二维数组，用于存储像素数据
pixel_data = [[0 for _ in range(image.size[1])] for _ in range(image.size[0])]
# 遍历通道并打印像素值
for channel in range(num_channels):
    print(f"\nChannel {channel}:")
    channel_data = image.getchannel(channel)
    pixels = channel_data.load()
    count = 0  # 初始化计数器
    for i in range(image.size[0]):
        for j in range(image.size[1]):
            # 将像素值存储到二维数组中
            pixel_data[i][j] = pixels[i, j]
            print(pixels[i, j], end=" ")
            count += 1  # 每次打印一个像素值，计数器加一
            if count == 100:  # 当打印数量达到 100 时停止
                break
        if count == 100:  # 当打印数量达到 100 时停止
            break

# 打印每个通道的最小值和最大值
for channel in range(num_channels):
    print(f"\nChannel {channel}:")
    channel_data = image.getchannel(channel)
    pixels = channel_data.load()
    min_pixel_value = 256  # 初始化最小值为一个较大的值
    max_pixel_value = 0  # 初始化最大值为一个较小的值
    for i in range(image.size[0]):
        for j in range(image.size[1]):
            # 更新最小值和最大值
            if pixels[i, j] < min_pixel_value:
                min_pixel_value = pixels[i, j]
            if pixels[i, j] > max_pixel_value:
                max_pixel_value = pixels[i, j]
    # 打印最小值和最大值
    print("Min pixel value:", min_pixel_value)
    print("Max pixel value:", max_pixel_value)

# print("Pixel data matrix:")
# # 打印像素数据矩阵
# for row in pixel_data:
#     print(row)
