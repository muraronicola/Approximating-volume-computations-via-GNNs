import torch
from torch.nn import Linear, Dropout
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, global_mean_pool, HeteroConv, GCNConv, SAGEConv, GATConv, Linear, GraphConv, ResGatedGraphConv
import copy

class NewModel(torch.nn.Module):
    
    def __init__(self, node_features, hidden_channels, n_releations, targhet_shape, conversion, n_layers, p_drop=0.3, seed=0):
        super(NewModel, self).__init__()
        
        torch.manual_seed(seed)
        self.convs = torch.nn.ModuleList()
        hidden_channels_linear = hidden_channels *3 #3 is the number of different node types (v3)
        
        for _ in range(n_layers):
            conv = HeteroConv({
                ('x', 'a', 'c'):  ResGatedGraphConv((-1, -1), hidden_channels, edge_dim=1),
                ('b', 'b', 'c'):  ResGatedGraphConv((-1, -1), hidden_channels, edge_dim=1),
                
                ('c', 'a', 'x'):  ResGatedGraphConv((-1, -1), hidden_channels, edge_dim=1),
                ('c', 'b', 'b'):  ResGatedGraphConv((-1, -1), hidden_channels, edge_dim=1),
                }, aggr='sum')
            self.convs.append(conv)
            
        
        self.dropout = Dropout(p=p_drop)
        
        self.linear1 = Linear(hidden_channels_linear, hidden_channels_linear)
        self.linear2 = Linear(hidden_channels_linear, hidden_channels_linear)
        self.linear3 = Linear(hidden_channels_linear, hidden_channels_linear)
        self.out = Linear(hidden_channels_linear, 1)
        
        torch.nn.init.uniform_(self.out.weight) 

    def forward(self, x, edge_index, edge_attr, batch, train=True):
        i = 0
        
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr) #GraphConv
            x = {key: fra.relu() for key, fra in x.items()}
            i += 1

        mean_representation = []
        for key in x.keys():
            mean_representation.append(global_mean_pool(x[key], batch[key]))
        
        x = torch.cat(mean_representation, dim=1)

        if train:
            x = self.dropout(x)
        
        x = self.linear1(x)
        x = x.relu()
        x = self.linear2(x)
        x = x.relu()
        x = self.linear3(x)
        x = x.relu()
        
        x = self.out(x)
        x = torch.flatten(x)
        
        return x
    
