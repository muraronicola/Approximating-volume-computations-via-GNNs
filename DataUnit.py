import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch_geometric.data import Data

class DataUnit(Dataset):
    def __init__(self, x, y):

        converted_data = []
        for index in range(len(x)):
            this_x = x[index]
            this_y = y[index]
            
            edge_index = []
            for i in range(this_x.shape[0]):
                for j in range(this_x.shape[0]):
                    if i != j:
                        edge_index.append([i, j])
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            
            torch_x = torch.tensor(this_x.astype('float64'), dtype=torch.float)
            torch_y = torch.tensor(this_y, dtype=torch.float)
            converted_data.append(Data(x=torch_x, edge_index=edge_index, y=torch_y))
        
        self.data = converted_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]