import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ImuSignal import ImuSignal

dimension_names = ["angular_velocity_x", "angular_velocity_y", "angular_velocity_z",
                   "linear_acceleration_x", "linear_acceleration_y", "linear_acceleration_z"]
# dimension_names = ["linear_acceleration_x", "linear_acceleration_y", "linear_acceleration_z"]
feature_names = [
    "Mean amplitude",
    "Square root amplitude",
    "Variance",
    "Peak",

    "Maximum value",
    "Minimum value",
    "Peak value",

    "Square mean root",
    "Root mean square",
    "Crest factor",
    "Clearance factor",
    "Kurtosis",
    "Standard deviation",
    "Skewness",
    "Waveform factor",
    "Pulse factor",
    "Residual factor",
    "Skewness factor",
    "Peak factor",
    "Yield factor",
    "Mean frequency",
    "Frequency center",
    "Variance of mean frequency",
    "Peak frequency",

    "Median frequency",
    "Root mean square frequency",
    "Mean square frequency",
    "Root mean frequency square"
]
# feature_names = [
#     "Kurtosis",
#     "Mean frequency",
#     "Frequency center"
# ]


class SIGNAL_DATASET():
    def __init__(self, chosen_labels=None, dataset_dir='../dataset'):
        features = []
        labels = []

        # Walk through the dataset directory to list all CSV files and their labels
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                if file.endswith('.csv'):
                    label = os.path.basename(root)
                    if chosen_labels is not None and label in chosen_labels or chosen_labels is None:
                        labels.append(label)
                    else:
                        continue
                    file_path = os.path.join(root, file)
                    df = pd.read_csv(file_path)
                    features_matrix = []

                    # linear_acceleration_z = ImuSignal(df['linear_acceleration_z'])
                    # features_matrix.extend([
                    #     # linear_acceleration_z.kurtosis(),
                    #     linear_acceleration_z.mean_frequency(),
                    #     linear_acceleration_z.frequency_center()
                    # ])
                    # linear_acceleration_y = ImuSignal(df['linear_acceleration_y'])
                    # features_matrix.extend([
                    #     linear_acceleration_y.kurtosis(),
                    #     linear_acceleration_y.mean_frequency(),
                    #     linear_acceleration_y.frequency_center()
                    # ])
                    # linear_acceleration_x = ImuSignal(df['linear_acceleration_x'])
                    # features_matrix.extend([
                    #     linear_acceleration_x.kurtosis(),
                    #     linear_acceleration_x.mean_frequency(),
                    #     # linear_acceleration_x.frequency_center()
                    # ])
                    # angular_velocity_z = ImuSignal(df['angular_velocity_z'])
                    # features_matrix.extend([
                    #     # angular_velocity_z.kurtosis(),
                    #     # angular_velocity_z.mean_frequency(),
                    #     angular_velocity_z.frequency_center()
                    # ])
                    # angular_velocity_y = ImuSignal(df['angular_velocity_y'])
                    # features_matrix.extend([
                    #     # angular_velocity_y.kurtosis(),
                    #     # angular_velocity_y.mean_frequency(),
                    #     angular_velocity_y.frequency_center()
                    # ])
                    # angular_velocity_x = ImuSignal(df['angular_velocity_x'])
                    # features_matrix.extend([
                    #     # angular_velocity_x.kurtosis(),
                    #     # angular_velocity_x.mean_frequency(),
                    #     angular_velocity_x.frequency_center()
                    # ])
                    # features.append(np.array(features_matrix).flatten())

                    for col in df.columns[1:]: # 全特征
                    # # # for col in df.columns[1:4]: # 角速度特征
                    # # # for col in df.columns[4:]: # 加速度特征
                    # # # for col in df.columns[6:]: # z加速度特征
                    # # # for col in [df.columns[i] for i in [1]]:
                    # # # for col in [df.columns[i] for i in [3, 4, 5, 6]]:
                        signal = ImuSignal(df[col])
                        features_matrix.append(list(signal.features.values()))
                    features.append(np.array(features_matrix).flatten())

        # self.labels = preprocessing.LabelEncoder().fit_transform(np.array(labels))
        self.labels = np.array(labels)
        self.features = np.array(features)

        scaler = StandardScaler()
        scaler.fit(self.features)
        joblib.dump(scaler, '../models/scaler.joblib')
        self.features_scaled = scaler.transform(self.features)

        pca = PCA(n_components=0.99)
        joblib.dump(pca, '../models/pca.joblib')
        self.features_scaled_pca = pca.fit_transform(self.features_scaled)

    def getDf(self):
        df = pd.DataFrame(self.features)
        df['label'] = self.labels
        return df

    def visualize(self, dimension, feature):
        column = dimension_names.index(dimension) * len(feature_names) + feature_names.index(feature)
        plt.figure(figsize=(16, 4))
        sns.boxplot(x='label', y=column, data=self.getDf())
        plt.title(f'Distribution of {dimension} {feature} Across Different Labels')
        plt.ylabel(f'{feature} Value')
        plt.xlabel('Label')
        # plt.show()
        plt.savefig(f'../feature/features_{dimension}_{feature}.png')
        plt.close()


if __name__ == "__main__":
    chosen_labels = ['stop', 'grass', 'dirt', 'floor-', 'playground', 'rideroad', 'runway']
    chosen_labels += ['carroad3']
    chosen_labels += ['board2', 'board3']
    chosen_labels += ['flat14', 'flat2', 'flat3']
    chosen_labels += ['brick12', 'brick3', 'brick4', 'brick5', 'brick6']
    dataset = SIGNAL_DATASET(chosen_labels)
    for feature in feature_names:
        for dimension in dimension_names:
            dataset.visualize(dimension=dimension, feature=feature)
    # dataset.visualize(dimension="linear_acceleration_z", feature="Mean amplitude")
