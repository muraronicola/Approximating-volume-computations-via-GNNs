import torch
from torch.nn import Linear, Dropout
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, global_mean_pool, HeteroConv, GCNConv, SAGEConv, GATConv, Linear, GraphConv
import copy

class Heterogeneus(torch.nn.Module):
    
    def __init__(self, node_features, hidden_channels, n_releations, targhet_shape, conversion, n_layers, p_drop=0.3, seed=0):
        super(Heterogeneus, self).__init__()
        torch.manual_seed(seed)
        
        
        # We need to use a convolutional layer to obtain the node embeddings (for the inhomogeneous case)
        # https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.RGCNConv.html#torch_geometric.nn.conv.RGCNConv
        # https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html
        # https://pytorch-geometric.readthedocs.io/en/latest/cheatsheet/gnn_cheatsheet.html#heterogeneous-graph-neural-network-operators 
        # https://pytorch-geometric.readthedocs.io/en/latest/tutorial/heterogeneous.html
        
        
        
        self.convs = torch.nn.ModuleList()
        dictionary = {}
        hidden_channels_linear = hidden_channels
        
        if conversion == "h1":
            #print("\n\n")
            #print("-"*50)
            #print("I'm in the model")
            
            
            
            for i in range(targhet_shape[0]):
                for j in range(targhet_shape[0]):
                    if i != j:
                        dictionary[('a_'+str(i), 'a_columns', 'a_'+str(j))] = GraphConv(-1, hidden_channels)
                        #print("a_"+ str(i) + " --- " + "a_columns" + " --- " + "a_"+str(j))
                        
            for i in range(targhet_shape[0]):
                dictionary[('a_'+str(i), 'a_b', 'b')] = GraphConv(-1, hidden_channels)
                #print("a_"+ str(i) + " --- " + "a_b" + " --- " + "b")
            
            
            dictionary[('b', 'b', 'b')] = GraphConv(-1, hidden_channels)
            #print("b --- b --- b")
            
            for i in range(targhet_shape[0]):
                dictionary[('a_'+str(i), 'a_rows', 'a_'+str(i))] = GraphConv(-1, hidden_channels)
                #print("a_"+ str(i) + " --- " + "a_rows" + " --- " + "a_"+str(i))
            
            hidden_channels_linear = hidden_channels * (targhet_shape[0] + 1) # 4 is for h1, 3 is for h2
            
            """#For H1:
            for _ in range(n_layers):
                conv = HeteroConv({
                    ('a_0', 'a_columns', 'a_1'):  GraphConv(-1, hidden_channels),
                    ('a_0', 'a_columns', 'a_2'):  GraphConv(-1, hidden_channels),
                    ('a_1', 'a_columns', 'a_0'):  GraphConv(-1, hidden_channels),
                    ('a_1', 'a_columns', 'a_2'):  GraphConv(-1, hidden_channels),
                    ('a_2', 'a_columns', 'a_0'):  GraphConv(-1, hidden_channels),
                    ('a_2', 'a_columns', 'a_1'):  GraphConv(-1, hidden_channels),
                    ('a_0', 'a_b', 'b'):  GraphConv(-1, hidden_channels),
                    ('a_1', 'a_b', 'b'):  GraphConv(-1, hidden_channels),
                    ('a_2', 'a_b', 'b'):  GraphConv(-1, hidden_channels),
                    ('b', 'b', 'b'):  GraphConv(-1, hidden_channels),
                    ('a_0', 'a_row', 'a_0'):  GraphConv(-1, hidden_channels),
                    ('a_1', 'a_row', 'a_1'):  GraphConv(-1, hidden_channels),
                    ('a_2', 'a_row', 'a_2'):  GraphConv(-1, hidden_channels),
                }, aggr='sum')
                self.convs.append(conv)
                """
            
            #hidden_channels_linear = hidden_channels  # 4 is for h1, 3 is for h2
            
        else:
            #print("\n\n")
            #print("-"*50)
            #print("I'm in the model")
            
            for i in range(targhet_shape[1]-1):
                dictionary[('a_'+str(i), 'a_columns', 'a_'+str(i))] = GraphConv(-1, hidden_channels)
                #print("a_"+ str(i) + " --- " + "a_columns" + " --- " + "a_"+str(i))
            
            for i in range(targhet_shape[1]-1):
                dictionary[('a_'+str(i), 'a_b', 'b')] = GraphConv(-1, hidden_channels)
                #print("a_"+ str(i) + " --- " + "a_b" + " --- " + "b")
                
            dictionary[('b', 'b', 'b')] = GraphConv(-1, hidden_channels)
            #print("b --- b --- b")
            
            
            for i in range(targhet_shape[1]-1):
                for j in range(targhet_shape[1]-1):
                    if i != j:
                        dictionary[('a_'+str(i), 'a_rows', 'a_'+str(j))] = GraphConv(-1, hidden_channels)
                        #print("a_"+ str(i) + " --- " + "a_rows" + " --- " + "a_"+str(j))
            
            
            
            hidden_channels_linear = hidden_channels * targhet_shape[1]  # 4 is for h1, 3 is for h2
            
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


        #For H2 (old old version):
        """
        for _ in range(n_layers):
            conv = HeteroConv({
                NE MANCA 1....
                ('a_1', 'a_columns_1', 'a_1'):  GraphConv(-1, hidden_channels),
                ('a_0', 'a_b_0', 'b'):  GraphConv(-1, hidden_channels),
                ('a_1', 'a_b_1', 'b'):  GraphConv(-1, hidden_channels),
                ('b', 'b', 'b'):  GraphConv(-1, hidden_channels),
                ('a_0', 'a_rows_0', 'a_1'):  GraphConv(-1, hidden_channels),
                ('a_1', 'a_rows_1', 'a_0'):  GraphConv(-1, hidden_channels),
            }, aggr='sum')
            self.convs.append(conv)
        """
        
        for _ in range(n_layers):
            conv = HeteroConv(copy.deepcopy(dictionary), aggr='sum')
            self.convs.append(conv)
        
        #print(dictionary)
        """print("-"*50)
        print({
                    ('a_0', 'a_columns', 'a_1'):  GraphConv(-1, hidden_channels),
                    ('a_0', 'a_columns', 'a_2'):  GraphConv(-1, hidden_channels),
                    ('a_1', 'a_columns', 'a_0'):  GraphConv(-1, hidden_channels),
                    ('a_1', 'a_columns', 'a_2'):  GraphConv(-1, hidden_channels),
                    ('a_2', 'a_columns', 'a_0'):  GraphConv(-1, hidden_channels),
                    ('a_2', 'a_columns', 'a_1'):  GraphConv(-1, hidden_channels),
                    ('a_0', 'a_b', 'b'):  GraphConv(-1, hidden_channels),
                    ('a_1', 'a_b', 'b'):  GraphConv(-1, hidden_channels),
                    ('b', 'b', 'b'):  GraphConv(-1, hidden_channels),
                    ('a_0', 'a_row', 'a_0'):  GraphConv(-1, hidden_channels),
                    ('a_1', 'a_row', 'a_1'):  GraphConv(-1, hidden_channels),
                    ('a_2', 'a_row', 'a_2'):  GraphConv(-1, hidden_channels),
                })
        """
        self.dropout = Dropout(p=p_drop)
        
        self.linear1 = Linear(hidden_channels_linear, hidden_channels_linear)
        self.linear2 = Linear(hidden_channels_linear, hidden_channels_linear)
        self.linear3 = Linear(hidden_channels_linear, hidden_channels_linear)
        self.out = Linear(hidden_channels_linear, 1)
        
        torch.nn.init.uniform_(self.out.weight) 

    def forward(self, x, edge_index, batch, train=True):
        # 1. Obtain node embeddings
        
        #print("edge_index", edge_index)
        
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
        
        return x