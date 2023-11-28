import rosbag
import rospy

def extract_data_from_bag():
    bag_file = 'your_bag_file.bag'  # 替换为你的bag文件路径
    topic_name = '/your_topic'      # 替换为你感兴趣的topic名称
    start_time = rospy.Time(0)      # 替换为感兴趣时间段的开始时间
    end_time = rospy.Time(10)       # 替换为感兴趣时间段的结束时间

    with rosbag.Bag(bag_file, 'r') as bag:
        for topic, msg, timestamp in bag.read_messages(topics=[topic_name]):
            if start_time <= timestamp <= end_time:
                # 在这里处理提取到的数据，例如打印消息
                print(msg)

if __name__ == '__main__':
    rospy.init_node('bag_data_extraction_node')
    extract_data_from_bag()
