import torch
from torch.nn import Linear, Dropout
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, global_mean_pool, HeteroConv, GCNConv, SAGEConv, GATConv, Linear, GraphConv, ResGatedGraphConv
import copy

class NewModel(torch.nn.Module):
    
    def __init__(self, node_features, hidden_channels, n_releations, targhet_shape, conversion, n_layers, p_drop=0.3, seed=0):
        super(NewModel, self).__init__()
        torch.manual_seed(seed)
        
        
        # We need to use a convolutional layer to obtain the node embeddings (for the inhomogeneous case)
        # https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.RGCNConv.html#torch_geometric.nn.conv.RGCNConv
        # https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html
        # https://pytorch-geometric.readthedocs.io/en/latest/cheatsheet/gnn_cheatsheet.html#heterogeneous-graph-neural-network-operators 
        # https://pytorch-geometric.readthedocs.io/en/latest/tutorial/heterogeneous.html
        
        
        
        self.convs = torch.nn.ModuleList()
        hidden_channels_linear = hidden_channels
        
        """
        #For H2:
        for _ in range(n_layers):
            conv = HeteroConv({
                ('a_0', 'a_columns', 'a_0'):  GraphConv(-1, hidden_channels),
                ('a_1', 'a_columns', 'a_1'):  GraphConv(-1, hidden_channels),
                ('a_0', 'a_b', 'b'):  GraphConv(-1, hidden_channels),
                ('a_1', 'a_b', 'b'):  GraphConv(-1, hidden_channels),
                ('b', 'b', 'b'):  GraphConv(-1, hidden_channels),
                ('a_0', 'a_rows', 'a_1'):  GraphConv(-1, hidden_channels),
                ('a_1', 'a_rows', 'a_0'):  GraphConv(-1, hidden_channels),
            }, aggr='sum')
            self.convs.append(conv)
        """
        
        for _ in range(n_layers):
            conv = HeteroConv({
                ('x', 'a', 'c'):  ResGatedGraphConv((-1, -1), hidden_channels, edge_dim=1),
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
        # 1. Obtain node embeddings
        
        #print("x", x)
        #print("edge_index", edge_index)
        #print("edge_attr", edge_attr)
        
        """x["x"] = x["x"][:2]
        x["c"] = x["c"][:3]
        x["b"] = x["b"][:3]
        
        edge_attr[("x", "a", "c")] = edge_attr[("x", "a", "c")][:3]
        edge_attr[("c", "b", "b")] = edge_attr[("c", "b", "b")][:6]
        
        edge_index[("x", "a", "c")] = edge_index[("x", "a", "c")][:3]
        edge_index[("c", "b", "b")] = edge_index[("c", "b", "b")][:6]"""

        i = 0
        for conv in self.convs:
            print(f"Iteration {i}")
            print("x before conv:", x)
            print(x["c"].shape)
            print(x["b"].shape)
            print(x["x"].shape)
            print(edge_index[("x", "a", "c")].shape)
            print(edge_attr[("x", "a", "c")].shape)
            x = conv(x, edge_index, edge_attr)
            print("x after conv:", x)
            x = {key: fra.relu() for key, fra in x.items()}
            print("x after ReLU:", x)
            i += 1

        mean_representation = []
        for key in x.keys():
            mean_representation.append(global_mean_pool(x[key], batch[key]))
        x = torch.cat(mean_representation, dim=1)
        #print("x.shape", x.shape)

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