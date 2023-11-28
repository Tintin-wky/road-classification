import matplotlib.pyplot as plt
import numpy as np

# 创建一个新的Matplotlib图形窗口
plt.figure(figsize=(12, 6))  # 设置窗口大小，可根据需要调整
formatted_numbers = [f'{i:03}' for i in range(1, 21)]

# 使用循环遍历每张图像并显示
for i in range(1,21):
    image_path='./data/'+formatted_numbers[i-1]+'F'+'.png'
    # 创建子图，将图像显示在子图中
    plt.subplot(4, 5, i)  # 创建4x5的子图布局，i+1是子图的位置
    image = plt.imread(image_path)  # 读取图像
    plt.imshow(image)
    plt.axis('off')  # 隐藏坐标轴
    plt.title(f'Image {i}')

# 调整子图之间的间距
plt.tight_layout()

# 显示图形窗口
plt.show()
