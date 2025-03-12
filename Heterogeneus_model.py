import torch
from torch.nn import Linear, Dropout
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, global_mean_pool, HeteroConv, GCNConv, SAGEConv, GATConv, Linear, GraphConv

class Heterogeneus(torch.nn.Module):
    
    def __init__(self, node_features, hidden_channels, n_releations, p_drop=0.3, seed=0):
        super(Heterogeneus, self).__init__()
        torch.manual_seed(seed)
        
        
        # We need to use a convolutional layer to obtain the node embeddings (for the inhomogeneous case)
        # https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.RGCNConv.html#torch_geometric.nn.conv.RGCNConv
        # https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html
        # https://pytorch-geometric.readthedocs.io/en/latest/cheatsheet/gnn_cheatsheet.html#heterogeneous-graph-neural-network-operators 
        # https://pytorch-geometric.readthedocs.io/en/latest/tutorial/heterogeneous.html
        
        
        """self.conv1 = GraphConv(node_features, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.conv4 = GraphConv(hidden_channels, hidden_channels)
        self.conv5 = GraphConv(hidden_channels, hidden_channels)
        self.conv6 = GraphConv(hidden_channels, hidden_channels)
        self.conv7 = GraphConv(hidden_channels, hidden_channels)"""
        
        num_layers = 7
        
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('a_1', 'a_columns_1', 'a_1'):  GraphConv(-1, hidden_channels),
                ('a_0', 'a_b_0', 'b'):  GraphConv(-1, hidden_channels),
                ('a_1', 'a_b_1', 'b'):  GraphConv(-1, hidden_channels),
                ('b', 'b', 'b'):  GraphConv(-1, hidden_channels),
                ('a_0', 'a_rows_0', 'a_1'):  GraphConv(-1, hidden_channels),
                ('a_1', 'a_rows_1', 'a_0'):  GraphConv(-1, hidden_channels),
            }, aggr='sum')
            self.convs.append(conv)
            
        """('a_1', 'a_columns_1', 'a_1'):  GraphConv(node_features, hidden_channels),
        ('a_0', 'a_b_0', 'b'):  GraphConv(node_features, hidden_channels),
        ('a_1', 'a_b_1', 'b'):  GraphConv(node_features, hidden_channels),
        ('b', 'b', 'b'):  GraphConv(node_features, hidden_channels),
        ('a_0', 'a_rows_0', 'a_1'):  GraphConv(node_features, hidden_channels),
        ('a_1', 'a_rows_1', 'a_0'):  GraphConv(node_features, hidden_channels),"""
        
        
        """self.conv1 = RGCNConv(node_features, hidden_channels, n_releations)
        self.conv2 = RGCNConv(hidden_channels, hidden_channels, n_releations)
        self.conv3 = RGCNConv(hidden_channels, hidden_channels, n_releations)
        self.conv4 = RGCNConv(hidden_channels, hidden_channels, n_releations)
        self.conv5 = RGCNConv(hidden_channels, hidden_channels, n_releations)
        self.conv6 = RGCNConv(hidden_channels, hidden_channels, n_releations)
        self.conv7 = RGCNConv(hidden_channels, hidden_channels, n_releations)"""
        
        self.dropout = Dropout(p=p_drop)
        
        hidden_channels_linear = hidden_channels * 3
        self.linear1 = Linear(hidden_channels_linear, hidden_channels_linear)
        self.linear2 = Linear(hidden_channels_linear, hidden_channels_linear)
        self.linear3 = Linear(hidden_channels_linear, hidden_channels_linear)
        self.out = Linear(hidden_channels_linear, 1)
        
        torch.nn.init.uniform_(self.out.weight) 

    def forward(self, x, edge_index, batch, train=True):
        # 1. Obtain node embeddings
        
        """print("x", x)
        print("edge_index", edge_index)
        print("batch", batch)
        
        
        print("x['a_0']", x['a_0'].shape)
        print("x['a_1']", x['a_1'].shape)
        print("x['b']", x['b'].shape)
        print("edge_index[('a_0', 'a_columns_0', 'a_0')]", edge_index[('a_0', 'a_columns_0', 'a_0')].shape)
        print("-"*50)"""

        i = 0
        for conv in self.convs:
            #print(x)
            #print(i)
            i += 1
            x = conv(x, edge_index)
            x = {key: x.relu() for key, x in x.items()}
            
        """x = self.conv1(x, edge_index)
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
        x = self.conv7(x, edge_index)"""
        
        #print("x", x)

        #x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        
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
        
        #print("output shape", x.shape)
        return x