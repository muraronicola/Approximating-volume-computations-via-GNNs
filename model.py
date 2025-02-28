import torch
from torch.nn import Linear, Dropout
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, global_mean_pool


class GCN(torch.nn.Module):
    
    def __init__(self, node_features, hidden_channels, p_drop=0.3, seed=0):
        super(GCN, self).__init__()
        torch.manual_seed(seed)
        
        self.conv1 = GraphConv(node_features, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.conv4 = GraphConv(hidden_channels, hidden_channels)
        self.conv5 = GraphConv(hidden_channels, hidden_channels)
        self.conv6 = GraphConv(hidden_channels, hidden_channels)
        self.conv7 = GraphConv(hidden_channels, hidden_channels)
        
        self.dropout = Dropout(p=p_drop)
        self.out = Linear(hidden_channels, 1)
        
        torch.nn.init.uniform_(self.out.weight) 

    def forward(self, x, edge_index, batch, train=True):
        # 1. Obtain node embeddings
        
        """print("x", x)
        print("edge_index", edge_index)
        print("batch", batch)"""
        
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = self.conv4(x, edge_index)
        x = x.relu()
        x = self.conv5(x, edge_index)
        x = x.relu()
        x = self.conv6(x, edge_index)
        x = x.relu()
        x = self.conv7(x, edge_index)

        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        if train:
            x = self.dropout(x)
        
        x = self.out(x)
        x = torch.flatten(x)
        
        return x