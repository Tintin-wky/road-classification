import rosbag
import pandas as pd
import argparse
import os

# 定义ROS bag文件路径和topic名称
bag_file = './data/018.bag' 
topic_name = '/imu/data_raw'    #3 125Hz 
DATA_LEN=200

def main(args: argparse.Namespace):
    data = []
    start_time = 1e100
    count = 0
    data_count = 1
    bag_file = args.rosbags
    bag_name, _ = os.path.splitext(os.path.basename(bag_file))
    # 打开ROS bag文件并提取数据
    with rosbag.Bag(bag_file, 'r') as bag:
        for topic, msg, timestamp in bag.read_messages(topics=[topic_name]):
            if count < DATA_LEN:
                # 在这里将数据转换为适当的格式，然后将其添加到data列表中
                if timestamp.to_sec() < start_time:
                    start_time = timestamp.to_sec()
                angular_velocity_x = msg.angular_velocity.x
                angular_velocity_y = msg.angular_velocity.y
                angular_velocity_z = msg.angular_velocity.z
                linear_acceleration_x = msg.linear_acceleration.x
                linear_acceleration_y = msg.linear_acceleration.y
                linear_acceleration_z = msg.linear_acceleration.z
                
                # 将数据添加到data列表
                data.append([timestamp.to_sec()-start_time, angular_velocity_x, angular_velocity_y, angular_velocity_z, linear_acceleration_x, linear_acceleration_y, linear_acceleration_z])
                count+=1
            else:
                # 创建DataFrame
                df = pd.DataFrame(data, columns=['timestamp', 'angular_velocity_x', 'angular_velocity_y', 'angular_velocity_z', 'linear_acceleration_x', 'linear_acceleration_y', 'linear_acceleration_z'])
                # 将DataFrame保存为CSV文件
                csv_file = args.dir + '/' + bag_name + '_' + f'{data_count:03}' + '.csv'
                df.to_csv(csv_file, index=False)

                print(f'Data from rosbag {bag_name} has been successfully converted to CSV and saved to "{csv_file}".')

                count=0
                data_count+=1
                data = []
                start_time = 1e100


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
        "-O",
        type=str,
        default="./data",
        help="path to csvdatas",
    )
    args = parser.parse_args()
    main(args)