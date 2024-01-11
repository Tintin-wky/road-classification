from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from ImuSignal import ImuSignal


class SIGNAL_DATASET(Dataset):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.data_files = []
        self.labels = []

        # Walk through the dataset directory to list all CSV files and their labels
        for root, dirs, files in os.walk(self.dataset_dir):
            for file in files:
                if file.endswith('.csv'):
                    self.data_files.append(os.path.join(root, file))
                    self.labels.append(os.path.basename(root))

    def __getitem__(self, index):
        # Get the file path and label for the requested index
        file_path = self.data_files[index]
        label = self.labels[index]

        # Read the signal from the CSV file
        signal = ImuSignal(pd.read_csv(file_path)['angular_velocity_z'])
        features = list(signal.features.values())

        return features, label

    def __len__(self):
        return len(self.data_files)


if __name__ == "__main__":
    dataset_dir = '../dataset'
    dataset = SIGNAL_DATASET(dataset_dir)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    for features, labels in dataloader:
        print(features, labels)
