import torch
from torch.nn import Linear, Dropout
from torch_geometric.nn import global_mean_pool, HeteroConv, Linear, ResGatedGraphConv



#Class for the implementation of the GNN
class GNN_Model(torch.nn.Module):
    
    def __init__(self, hidden_channels, n_layers, p_drop=0.3, seed=0):
        super(GNN_Model, self).__init__()
        
        torch.manual_seed(seed)
        self.convs = torch.nn.ModuleList()
        hidden_channels_linear = hidden_channels * 3 #3 is the number of different node types (v3)
        
        # Initialize the layers of the GNN
        for _ in range(n_layers):
            conv = HeteroConv({
                ('x', 'a', 'c'):  ResGatedGraphConv((-1, -1), hidden_channels, edge_dim=1),
                ('b', 'b', 'c'):  ResGatedGraphConv((-1, -1), hidden_channels, edge_dim=1),
                
                ('c', 'a', 'x'):  ResGatedGraphConv((-1, -1), hidden_channels, edge_dim=1),
                ('c', 'b', 'b'):  ResGatedGraphConv((-1, -1), hidden_channels, edge_dim=1),
                }, aggr='sum')
            self.convs.append(conv)
        
        self.dropout = Dropout(p=p_drop)
        
        # Initialize the linear layers
        self.linear1 = Linear(hidden_channels_linear, hidden_channels_linear)
        self.linear2 = Linear(hidden_channels_linear, hidden_channels_linear)
        self.linear3 = Linear(hidden_channels_linear, hidden_channels_linear)
        self.out = Linear(hidden_channels_linear, 1)
        
        torch.nn.init.uniform_(self.out.weight) 



    def forward(self, x, edge_index, edge_attr, batch, train=True):
        
        # Forward pass of the GNN
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr) #GraphConv
            x = {key: fra.relu() for key, fra in x.items()}

        cat_representation = []
        for key in x.keys():
            cat_representation.append(global_mean_pool(x[key], batch[key]))
        
        # Concatenate the mean representations of all node types
        x = torch.cat(cat_representation, dim=1)

        if train: #Disable dropout during inference
            x = self.dropout(x)
        
        # Pass through the linear layers
        x = self.linear1(x)
        x = x.relu()
        x = self.linear2(x)
        x = x.relu()
        x = self.linear3(x)
        x = x.relu()
        
        x = self.out(x)
        x = torch.flatten(x)
        
        return x
