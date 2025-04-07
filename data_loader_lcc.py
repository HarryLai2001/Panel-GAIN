import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['y'])

    def __getitem__(self, idx):
        return self.data['d'][idx], self.data['y'][idx], self.data['y_norm'][idx], self.data['group'][idx]


def load_lcc_data(path, batch_size):
    df = pd.read_excel(path, engine='openpyxl')
    agg = df.groupby(['policy', '年度']).agg(y_mean = ("co2", np.mean),
                                             y_std = ("co2", np.std))

    df = pd.merge(df, agg, how='left', on=['policy', '年度'])
    df['y_norm'] = (df["co2"] - df['y_mean']) / (df['y_std'] + 1e-8)

    d = []
    y = []
    y_norm = []
    group = []

    for unit, unit_fea in df.groupby('id'):
        d.append(np.expand_dims(unit_fea['lccpost'].to_numpy(), axis=0))
        y.append(np.expand_dims(unit_fea["co2"].to_numpy(), axis=0))
        y_norm.append(np.expand_dims(unit_fea['y_norm'].to_numpy(), axis=0))
        group.append(unit_fea['group'].iloc[0])

    d = np.concatenate(d, axis=0)
    y = np.concatenate(y, axis=0)
    y_norm = np.concatenate(y_norm, axis=0)

    data = {
        'd': torch.tensor(d, dtype=torch.float),
        'y': torch.tensor(y, dtype=torch.float),
        'y_norm': torch.tensor(y_norm, dtype=torch.float),
        'group': torch.tensor(group, dtype=torch.long)
    }

    dataloader = DataLoader(dataset=MyDataset(data), batch_size=batch_size)

    return df, dataloader