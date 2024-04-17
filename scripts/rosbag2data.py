import rosbag
import pandas as pd
import argparse
import os
import io
from PIL import Image

imu_topic = '/imu/data_raw'  # 125Hz
image_topic = '/camera/left/image_raw/compressed'
DATA_LEN = 200
# 设置要去除的秒数
start_trim = 5  # 开始的几秒
end_trim = 5    # 结束的几秒


def main(args: argparse.Namespace):
    data = []
    count = 0
    data_count = 1
    bag_file = args.rosbags
    bag_name, _ = os.path.splitext(os.path.basename(bag_file))
    type_name = bag_name.split('_')[0]
    # 打开ROS bag文件并提取数据
    with rosbag.Bag(bag_file, 'r') as bag:
        start_time = bag.get_start_time() + start_trim
        end_time = bag.get_end_time() - end_trim
        for topic, msg, t in bag.read_messages(topics=[image_topic]):
            if (start_time + end_time) / 2 <= t.to_sec():
                obs_img = Image.open(io.BytesIO(msg.data))
                img_file = os.path.join(args.dir, f'{type_name}.jpg')
                obs_img.save(img_file)
                print(f'Image from rosbag {bag_name} has been successfully saved to "{img_file}".')
                break  # 只保存第一帧图像
        for topic, msg, t in bag.read_messages(topics=[imu_topic]):
            if start_time <= t.to_sec() <= end_time:
                if count < DATA_LEN:
                    angular_velocity_x = msg.angular_velocity.x
                    angular_velocity_y = msg.angular_velocity.y
                    angular_velocity_z = msg.angular_velocity.z
                    linear_acceleration_x = msg.linear_acceleration.x
                    linear_acceleration_y = msg.linear_acceleration.y
                    linear_acceleration_z = msg.linear_acceleration.z

                    # 将数据添加到data列表
                    data.append(
                        [t.to_sec(), angular_velocity_x, angular_velocity_y, angular_velocity_z,
                         linear_acceleration_x, linear_acceleration_y, linear_acceleration_z])
                    # 将数据添加到data列表
                    count += 1
                else:
                    # 创建DataFrame
                    df = pd.DataFrame(data, columns=['timestamp', 'angular_velocity_x', 'angular_velocity_y',
                                                     'angular_velocity_z', 'linear_acceleration_x', 'linear_acceleration_y',
                                                     'linear_acceleration_z'])
                    # 将DataFrame保存为CSV文件
                    csv_file = os.path.join(args.dir, type_name, f"{bag_name}_{data_count:03}.csv")
                    directory = os.path.dirname(csv_file)
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    df.to_csv(csv_file, index=False)

                    print(f'Data from rosbag {bag_name} has been successfully converted to CSV and saved to "{csv_file}".')

                    count = 0
                    data_count += 1
                    data = []


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Code to run generate dataset from rosbag"
    )
    parser.add_argument(
        "--rosbags",
        "-i",
        type=str,
        help="path to rosbags",
    )
    parser.add_argument(
        "--dir",
        "-o",
        type=str,
        default="../dataset",
        help="path to csvdatas",
    )
    args = parser.parse_args()
    main(args)
