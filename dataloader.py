import numpy as np
import pandas as pd
import os, glob
import os.path
import torch
import random
from torch.utils.data import Dataset, DataLoader
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset


def get_global_values(df, args):
    if args.scaler == 'minmax' : 
        p1, p2 = 0,100
    elif args.scaler == 'percentile':
        p1, p2 = 10, 90
    min_value = np.percentile(df, p1, axis=0)
    max_value = np.percentile(df, p2, axis=0)

    return min_value, max_value


def minmax_scaling(x, min_value, max_value):
    x = (x - min_value)/(max_value - min_value)
    
    return x



class loader(Dataset):
    def __init__(self, targetdir, device, col, seq_len, pred_len, loader_type='train', args=None, transform=None, encoding=False):
        self.loader_type = loader_type
        if self.loader_type == 'test':
            file_name = f'device{device}_anomaly_{loader_type}.csv'
        else: file_name = f'device{device}_{loader_type}.csv'
        data = pd.read_csv(os.path.join(targetdir, file_name), index_col=0)
        data.index = pd.to_datetime(data.index)
        data = data.resample('1T').first().dropna()
        if self.loader_type == 'test':
            self.label = data.label
            data.drop('label', axis=1, inplace=True)
        data.drop('REG_DTIME.1', axis=1, inplace=True)
        
        train_data = pd.read_csv(os.path.join(targetdir,  f'device{device}_train.csv' ), index_col=0)
        train_data.index = pd.to_datetime(train_data.index)
        train_data = train_data.resample('1T').first().dropna()
        train_data.drop('REG_DTIME.1', axis=1, inplace=True)
        min_value, max_value = get_global_values(train_data, args)
        self.min_value = min_value
        self.max_value = max_value

        self.data = minmax_scaling(data, min_value, max_value)
        self.time = torch.tensor(data.index.hour)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.transform = transform
        self.encoding = encoding
        print(args.multimodal)
        if args.multimodal:
            self.data = self.data[['TEMP', 'HUMIDITY']]
        else: self.data = self.data[col]
        indices = []
        for i in range(len(self.data)):
            indices.append(i)
        self.indices = indices[:-(self.seq_len + self.pred_len)]
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        idx = self.indices[index]
        x = self.data.iloc[idx : idx + self.seq_len].values.transpose()
        y = self.data.iloc[idx + self.seq_len : idx + self.seq_len + self.pred_len].values.transpose()
        hour = self.time[idx : idx + self.seq_len] / 24.
        x = torch.cat((torch.tensor(x), hour.unsqueeze(0)))

        return x, y

if __name__ == "__main__":
    traindir = r'/nas/datahub/iot_device/container'
    testdir = r'/nas/datahub/iot_device/container'

    file_list = sorted(os.listdir(testdir))
    data_list = [file for file in file_list if file.endswith(".csv")]

    tr_dataset = loader(traindir, 4, 'TEMP', 100, 100, loader_type = 'train')
    tr_dl = DataLoader(tr_dataset, shuffle=False, batch_size=10, pin_memory=False)

    for i in range(len(tr_dl)):
        x, y = next(iter(tr_dl))