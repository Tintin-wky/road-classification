import joblib
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ImuSignal import ImuSignal

dimension_names = ["angular_velocity_x", "angular_velocity_y", "angular_velocity_z", "linear_acceleration_x",
                   "linear_acceleration_y", "linear_acceleration_z"]
feature_names = [
    "Mean amplitude",
    "Square root amplitude",
    "Maximum value",
    "Minimum value",
    "Peak",
    "Peak value",
    "Square mean root",
    "Root mean square",
    "Crest factor",
    "Clearance factor",
    "Kurtosis",
    "Variance",
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
    "Median frequency",
    "Peak frequency",
    "Root mean square frequency",
    "Mean square frequency",
    "Root mean frequency square"
]


class SIGNAL_DATASET(Dataset):
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
                    # signal = ImuSignal(df['angular_velocity_z'])
                    # features_matrix.append(list(signal.features.values()))
                    # signal = ImuSignal(df['linear_acceleration_z'])
                    # features_matrix.append(list(signal.features.values()))
                    for col in df.columns[1:]:
                    # for col in df.columns[3:]:
                    # for col in df.columns[4:]:
                    # for col in [df.columns[i] for i in [1, 2, 3, 6]]:
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

    def __getitem__(self, index):
        # Convert to torch tensors
        features_tensor = torch.tensor(self.features[index], dtype=torch.float32)
        label_tensor = torch.tensor(self.labels[index])

        return features_tensor, label_tensor

    def __len__(self):
        return len(self.features)

    def getDf(self):
        df = pd.DataFrame(self.features)
        df['label'] = self.labels
        return df

    def visualize(self, dimension, feature):
        column = dimension_names.index(dimension) * len(feature_names) + feature_names.index(feature)
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='label', y=column, data=self.getDf())
        plt.title(f'Distribution of {dimension} {feature} Across Different Labels')
        plt.ylabel(f'{feature} Value')
        plt.xlabel('Label')
        plt.show()
        # plt.savefig(f'../dataset/features_{dimension}_{feature}.png')
        # plt.close()

if __name__ == "__main__":
    dataset_dir = '../dataset'
    dataset = SIGNAL_DATASET()
    # for feature in feature_names:
    #     for dimension in dimension_names:
    #         dataset.visualize(dimension=dimension, feature=feature)
    # dataset.visualize(dimension="linear_acceleration_z", feature="Mean amplitude")
    # print(dataset.__len__())
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    # for feature, label in dataloader:
    #     print(feature, label)
    #     break
