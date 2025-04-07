import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from utils import normalization


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['y_norm'])

    def __getitem__(self, idx):
        return self.data['m'][idx], self.data['y_norm'][idx]



def load_data(file_path, batch_size):
    # Load data
    df = pd.read_excel(file_path, engine='openpyxl')
    data_y = df.pivot(index="unit", columns="time", values="y").to_numpy()
    data_m = 1 - df.pivot(index="unit", columns="time", values="d").to_numpy()

    # Introduce missing data
    miss_data_y = data_y.copy()
    miss_data_y[data_m == 0] = np.nan

    # Normalization
    norm_data_y, norm_parameters = normalization(miss_data_y)
    norm_data_y = np.nan_to_num(norm_data_y, 0)

    data = {
        'm': torch.tensor(data_m, dtype=torch.float),
        'y_norm': torch.tensor(norm_data_y, dtype=torch.float),
    }

    dataloader = DataLoader(dataset=MyDataset(data), batch_size=batch_size)

    return df, dataloader, norm_parameters