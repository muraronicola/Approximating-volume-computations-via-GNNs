import torch
from torch_geometric.data import Data
import numpy as np


def convert(x):
    print("x.shape", x.shape)
    #sample
    #constraint
    #dim (dim + 1 (causa b))
    
    #samples = []
    #edge_indexs = []
    
    converted_data = []
    for sample in x:
        edge_index = []
        for i in range(sample.shape[0]):
            for j in range(sample.shape[0]):
                if i != j:
                    edge_index.append([i, j])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        sample = sample.astype('float64')
        this_x = torch.tensor(sample, dtype=torch.float)
        #samples.append(this_x)
        #edge_indexs.append(edge_index)
        #this_x = sample
        converted_data.append(Data(x=this_x, edge_index=edge_index))
    
    
    #edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    #x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

    #data = Data(x=x, edge_index=edge_index)
    
    #converted_data =  torch.stack(converted_data, dim=0)
    #print("converted_data.shape", converted_data.shape)
    return converted_data

    #return samples, edge_indexs