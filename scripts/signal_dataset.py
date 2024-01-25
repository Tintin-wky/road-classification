import numpy as np
from sklearn import preprocessing
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from ImuSignal import ImuSignal
import torch


class SIGNAL_DATASET(Dataset):
    def __init__(self, chosen_labels, dataset_dir='../dataset'):
        features = []
        labels = []

        # Walk through the dataset directory to list all CSV files and their labels
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                label = os.path.basename(root)
                if label in chosen_labels:
                    labels.append(label)
                else:
                    continue

                if file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    df = pd.read_csv(file_path)
                    features_matrix = []
                    # signal = ImuSignal(df['angular_velocity_z'])
                    # features_matrix.append(list(signal.features.values()))
                    # signal = ImuSignal(df['linear_acceleration_z'])
                    # features_matrix.append(list(signal.features.values()))
                    for col in df.columns[1:]:
                        signal = ImuSignal(df[col])
                        features_matrix.append(list(signal.features.values()))
                    features.append(np.array(features_matrix).flatten())

        # self.labels = preprocessing.LabelEncoder().fit_transform(np.array(labels))
        self.labels = np.array(labels)
        self.features = np.array(features)

    def __getitem__(self, index):
        # Convert to torch tensors
        features_tensor = torch.tensor(self.features[index], dtype=torch.float32)
        label_tensor = torch.tensor(self.labels[index])

        return features_tensor, label_tensor

    def __len__(self):
        return len(self.features)


if __name__ == "__main__":
    dataset_dir = '../dataset'
    chosen_labels = ['Stop', 'BrickRoad1', 'BrickRoad2', 'BrickRoad3', 'CarRoad']
    dataset = SIGNAL_DATASET(chosen_labels=chosen_labels, dataset_dir=dataset_dir)
    print(dataset.__len__())
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    # for feature, label in dataloader:
    #     print(feature, label)
    #     break
