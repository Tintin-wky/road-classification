import matplotlib.pyplot as plt
from PIL import Image

# 读取八张图片
image_paths = [
"../dataset/type_board.jpg",
"../dataset/type_brick-a.jpg",
"../dataset/type_brick-b.jpg",
"../dataset/type_brick-c.jpg",
"../dataset/type_brick-d.jpg",
"../dataset/type_brick-e.jpg",
"../dataset/type_carroad.jpg",
"../dataset/type_dirt.jpg",
"../dataset/type_floor.jpg",
"../dataset/type_grass.jpg",
"../dataset/type_playground.jpg",
"../dataset/type_runway.jpg"
]

def cut(str):
    name,_ = str.split(".jpg")
    _,type = name.split("_")
    return type

# 创建一个2x4的图像网格
fig, axs = plt.subplots(3, 4, figsize=(16, 10))

# 遍历每张图片并添加题注
for i, ax in enumerate(axs.flat):
    img = Image.open(image_paths[i])
    ax.imshow(img)
    ax.axis("off")  # 不显示坐标轴
    ax.set_title(cut(image_paths[i]), fontsize=36)

# 调整子图之间的间距
plt.tight_layout(pad=0.2, w_pad=0.6, h_pad=0)

# 保存绘制好的图像
plt.savefig("category.jpg")

# 显示图像
plt.show()
