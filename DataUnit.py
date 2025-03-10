import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch_geometric.data import Data, HeteroData

class DataUnit(Dataset):
    def __init__(self, x, y, conversion="constraints"):

        if conversion != "constraints" and conversion != "dimensions" and conversion != "heterogeneous":
            raise ValueError("Invalid conversion type")
        
        if conversion == "constraints" or conversion == "dimensions":
            self.data = self.convert_data(x, y, conversion)
        else:
            self.data = self.heterogeneus_data(x, y)
    
    
    def heterogeneus_data(self, x, y):
        
        converted_data = []
        for index in range(len(x)):
            this_x = x[index]
            this_y = y[index]
            #print("This is the x: ", this_x)
            #print("This is the y: ", this_y)
            
            torch_y = torch.tensor(this_y, dtype=torch.float)
            data_i = HeteroData(y=torch_y)
            
            #data_a = torch.tensor(this_x[:, :-1].astype('float64'), dtype=torch.float)
            data_a = []
            for i in range(this_x.shape[1]-1):
                data_a.append([])
                
                for j in range(this_x.shape[0]):
                    data_a[i].append(this_x[j, i])
            
            for i in range(len(data_a)):
                data_one_dimention = torch.tensor(data_a[i], dtype=torch.float)
                data_i['a_{}'.format(i)].x = data_one_dimention
            
            data_b = torch.tensor(this_x[:, -1].astype('float64'), dtype=torch.float)
            data_i['b'].x = data_b

            
            #Edges between same dimension
            edge_between_same_dim = []
            
            for i in range(this_x.shape[0]):
                for j in range(this_x.shape[0]):
                    if i != j:
                        edge_between_same_dim.append([i, j])
            
            for i in range(this_x.shape[1]-1):
                edge_index = torch.tensor(edge_between_same_dim, dtype=torch.long).t().contiguous()
                data_i['a_{}'.format(i), "a_{}".format(i), 'a_{}'.format(i)].edge_index = edge_index
            
            data_edge_a_b = []
            for i in range(this_x.shape[0]):
                data_edge_a_b.append([i, i])
                
            edge_index = torch.tensor(data_edge_a_b, dtype=torch.long).t().contiguous()
            
            for i in range(this_x.shape[1]-1):
                data_i['a_{}'.format(i), "a_b_{}".format(i), 'b'].edge_index = edge_index
            
            data_i['b', 'b', 'b'].edge_index = torch.tensor(edge_between_same_dim, dtype=torch.long).t().contiguous()
            
            """print("This is the data_i: ", data_i)
            print("This is the data_i['a_0']: ", data_i['a_0'])
            print("This is the data_i['a_1']: ", data_i['a_1'])
            print("This is the data_i['b']: ", data_i['b'])
            print("This is the data_i['a_0', 'a_0, 'a_0']: ", data_i['a_0', 'a_0', 'a_0'])
            print("This is the data_i['b', 'b', 'b']: ", data_i['b', 'b', 'b'])
            print("This is the data_i['a_0', 'a_b_0', 'b']: ", data_i['a_0', 'a_b_0', 'b'])"""
            
            converted_data.append(data_i)
            #exit(0)
            
            """
            edge_index = []
            for i in range(this_x.shape[0]):
                for j in range(this_x.shape[0]):
                    if i != j:
                        edge_index.append([i, j])
            edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            
            torch_x = torch.tensor(this_x.astype('float64'), dtype=torch.float)
            torch_y = torch.tensor(this_y, dtype=torch.float)
            #torch_y = torch.tensor(100, dtype=torch.float)
            converted_data.append(Data(x=torch_x, edge_index=edge_index, y=torch_y))
            """

        return converted_data

        
    def convert_data(self, x, y, conversion): #Second try
        converted_data = []
        for index in range(len(x)):
            this_x = x[index]
            this_y = y[index]
            
            if conversion == "dimensions":
                this_x = this_x.transpose()
            
            edge_index = []
            for i in range(this_x.shape[0]):
                for j in range(this_x.shape[0]):
                    if i != j:
                        edge_index.append([i, j])
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            
            torch_x = torch.tensor(this_x.astype('float64'), dtype=torch.float)
            torch_y = torch.tensor(this_y, dtype=torch.float)
            #torch_y = torch.tensor(100, dtype=torch.float)
            converted_data.append(Data(x=torch_x, edge_index=edge_index, y=torch_y))

        return converted_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]