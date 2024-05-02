import matplotlib.pyplot as plt
from PIL import Image
import os

# 读取图片
image_paths = [
    # "../dataset/board1.jpg",
    "../dataset/board2.jpg",
    "../dataset/board3.jpg",
    "../dataset/brick12.jpg",
    # "../dataset/brick2.jpg",
    "../dataset/brick3.jpg",
    "../dataset/brick4.jpg",
    "../dataset/brick5.jpg",
    "../dataset/brick6.jpg",
    # "../dataset/brick7.jpg",
    # "../dataset/carroad1.jpg",
    # "../dataset/carroad2.jpg",
    "../dataset/carroad3.jpg",
    "../dataset/dirt.jpg",
    "../dataset/flat14.jpg",
    "../dataset/flat2.jpg",
    "../dataset/flat3.jpg",
    # "../dataset/flat4.jpg",
    # "../dataset/flat5.jpg",
    # "../dataset/flat6.jpg",
    "../dataset/floor-.jpg",
    "../dataset/grass.jpg",
    "../dataset/playground.jpg",
    "../dataset/rideroad.jpg",
    "../dataset/runway.jpg",
    "../dataset/stop.jpg",
]


def count_files(directory):
    """计算指定目录下的文件数量（不包括子目录中的文件）"""
    total_files = 0
    for entry in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, entry)):
            total_files += 1
    return total_files


# 创建一个2x4的图像网格
fig, axs = plt.subplots(6, 3, figsize=(15, 20))

# 遍历每张图片并添加题注
for i, ax in enumerate(axs.flat):
    img = Image.open(image_paths[i])
    ax.imshow(img)
    ax.axis("off")  # 不显示坐标轴
    type = image_paths[i].split("/")[-1].split(".")[0]
    fold = image_paths[i].split(".jpg")[0]
    ax.set_title(f"type:{type}  count:{count_files(fold)}", fontsize=28)

# 调整子图之间的间距
plt.tight_layout(pad=0.2, w_pad=0.2, h_pad=0.8)

# 保存绘制好的图像
plt.savefig("category.jpg")

# 显示图像
plt.show()
