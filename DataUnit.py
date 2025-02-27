import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class DataUnit(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]