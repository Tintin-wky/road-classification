import argparse
import joblib
import numpy as np
import rospy
from collections import deque, Counter
from sensor_msgs.msg import Imu
from std_msgs.msg import String
from ImuSignal import ImuSignal

DATA_LEN = 200
FILTER_LEN = 5
signals = deque(maxlen=DATA_LEN)
filters = deque(maxlen=FILTER_LEN)


def get_most_frequent_recent_element(d):
    element_counts = Counter(d)
    max_occurrence = max(element_counts.values())
    for element in reversed(d):
        if element_counts[element] == max_occurrence:
            return element


def imu_callback(msg):
    global signal
    angular_velocity_x = msg.angular_velocity.x
    angular_velocity_y = msg.angular_velocity.y
    angular_velocity_z = msg.angular_velocity.z
    linear_acceleration_x = msg.linear_acceleration.x
    linear_acceleration_y = msg.linear_acceleration.y
    linear_acceleration_z = msg.linear_acceleration.z
    imu_data = [angular_velocity_x, angular_velocity_y, angular_velocity_z, linear_acceleration_x,
                linear_acceleration_y, linear_acceleration_z]
    signals.append(imu_data)


def main(args: argparse.Namespace):
    rospy.init_node('Classification', anonymous=True)

    scaler = joblib.load('../models/scaler.joblib')
    model = joblib.load(f'../models/{args.algorithm}.joblib')
    rospy.loginfo(f'Using {args.algorithm} Model {model}')

    rate = rospy.Rate(1)
    imu_data_subscriber = rospy.Subscriber("/imu/data_raw", Imu, imu_callback)
    road_type_publisher = rospy.Publisher("/road_type", String, queue_size=1)

    while not rospy.is_shutdown():
        global signals
        if len(signals) == DATA_LEN:
            imu_data = np.array(list(signals))
            features_matrix = []
            for i in range(imu_data.shape[1]):
                signal = ImuSignal(imu_data[:,i])
                features_matrix.append(list(signal.features.values()))
            features = np.array(features_matrix).flatten().reshape(1,-1)
            features_scaled = scaler.transform(features)
            filters.append(model.predict(features_scaled))
            # label = get_most_frequent_recent_element(filters)
            label = filters[-1]
            rospy.loginfo("Predicted class: %s", label)
            road_type = "%s" % label
            road_type_publisher.publish(road_type)
        rate.sleep()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Code to deploy model to real-time classification")
    parser.add_argument(
        "--algorithm",
        "-a",
        default="svm",
        type=str,
        help="algorithm chosen for classification (default: svm) ['knn', 'svm', 'rf', 'lr']",
    )
    args = parser.parse_args()
    main(args)
