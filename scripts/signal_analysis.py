import os
import random
import pandas as pd
import matplotlib.pyplot as plt
from ImuSignal import ImuSignal

count = 1
figures = 1


def visualize_frequency(chosen_label, chosen_feature, number=10, dataset_dir='../dataset'):
    global count, figures
    plt.subplot(4, figures // 4, count)
    for root, dirs, files in os.walk(dataset_dir):
        label = os.path.basename(root)
        if label != chosen_label:
            continue
        selected_files = random.sample(files, number)
        for file in selected_files:
            file_path = os.path.join(root, file)
            df = pd.read_csv(file_path)
            signal_data = df[chosen_feature]
            signal = ImuSignal(signal_data)
            plt.plot(signal.frequency, signal.magnitude)
    plt.title(f"{chosen_feature} of {chosen_label} in frequency domain")
    count += 1


def visualize_domain(chosen_label, chosen_feature, number=10, dataset_dir='../dataset'):
    global count, figures
    plt.subplot(4, figures // 4, count)
    for root, dirs, files in os.walk(dataset_dir):
        label = os.path.basename(root)
        if label != chosen_label:
            continue
        selected_files = random.sample(files, number)
        for file in selected_files:
            file_path = os.path.join(root, file)
            df = pd.read_csv(file_path)
            signal_data = df[chosen_feature]
            plt.plot(signal_data)
    plt.title(f"{chosen_feature} of {chosen_label} in domain")
    count += 1


if __name__ == "__main__":
    plt.figure(figsize=(30, 30))
    # chosen_labels = ['Stop', 'BrickRoad1', 'CarRoad', 'SpeedBump']
    chosen_labels = ['CarRoad', 'BrickRoad1', 'BrickRoad2', 'BrickRoad3']
    chosen_features = ['angular_velocity_z', 'linear_acceleration_z']
    figures = len(chosen_labels) * len(chosen_features) * 2
    for chosen_label in chosen_labels:
        for chosen_feature in chosen_features:
            visualize_domain(chosen_label, chosen_feature)
            visualize_frequency(chosen_label, chosen_feature)
    plt.savefig("signal_analysis")
    # plt.show()
