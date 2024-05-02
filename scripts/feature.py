import matplotlib.pyplot as plt
from PIL import Image

# 读取图片
image_paths = [
    "../feature/features_linear_acceleration_z_Frequency center.png",
    "../feature/features_linear_acceleration_y_Frequency center.png",
    "../feature/features_linear_acceleration_x_Frequency center.png",
    # '../feature/features_angular_velocity_z_Frequency center.png',
    # '../feature/features_angular_velocity_y_Frequency center.png',
    # '../feature/features_angular_velocity_x_Frequency center.png',
]

# 创建图像网格
fig, axs = plt.subplots(3, 1, figsize=(16, 12))

# 遍历每张图片并添加题注
for i, ax in enumerate(axs.flat):
    img = Image.open(image_paths[i])
    ax.imshow(img)
    ax.axis("off")  # 不显示坐标轴

# 调整子图之间的间距
plt.tight_layout(pad=0.2, w_pad=0, h_pad=0)

# 保存绘制好的图像
plt.savefig("feature.png")

# 显示图像
plt.show()