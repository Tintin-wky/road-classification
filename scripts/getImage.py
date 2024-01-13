import io
from PIL import Image
import rosbag


bag = rosbag.Bag('../rosbag/018.bag', 'r')
count = 0
count_cut = 300
for topic, msg, t in bag.read_messages(topics=['/camera/left/image_raw/compressed']):
    obs_img = Image.open(io.BytesIO(msg.data))
    count+=1
    if count == count_cut:
        obs_img.save('CarRoad.jpg')
        break  # 只保存第一帧图像

bag.close()
