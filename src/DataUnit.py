import torch
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData


# Class for the implementation of the data unit
class DataUnit(Dataset):
    def __init__(self, x, y):
        
        self.x = x
        self.y = y
        
        self.data = self.convert_data()
        
        print("DataUnit init")
        print("len(x)", len(x))
    
    def __len__(self): 
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    
    # Convert the data from the input format to a graph format
    def convert_data(self):
        debug = False
        converted_data = []
        
        for index in range(len(self.x)):
            this_x = self.x[index]
            this_y = self.y[index]
            
            if debug:
                print("\n\nThis is the x: ", this_x)
                print("This is the y: ", this_y)
                print("-"*50)
                print("\n")
            
            torch_y = torch.tensor(this_y, dtype=torch.float)
            data_i = HeteroData(y=torch_y)
            
            
            # Create node types
            data_x = []
            data_c = []
            data_b = []
            
            for i in range(this_x.shape[1]-1):
                data_x.append(0)
            
            for i in range(this_x.shape[0]):
                data_c.append(0) 
                data_b.append(0)
            
            data_x = torch.tensor(data_x, dtype=torch.float)
            data_i["x"].x = data_x.unsqueeze(1)
            
            data_c = torch.tensor(data_c, dtype=torch.float)
            data_i["c"].x = data_c.unsqueeze(1)
            
            data_b = torch.tensor(data_b, dtype=torch.float)
            data_i["b"].x = data_b.unsqueeze(1)
            
            
            
            #Edges between dimensions and constraints
            edge_attr = []
            edge_index = []
            edge_index_reverse = []
            
            for i in range(this_x.shape[1] - 1):
                for j in range(this_x.shape[0]):
                    edge_index.append([i, j])
                    edge_attr.append(this_x[j, i])
                    edge_index_reverse.append([j, i])
            
            data_i[("x", "a", "c")].edge_attr = torch.tensor(edge_attr, dtype=torch.float).unsqueeze(1) #ResGatedGraphConv
            data_i[("x", "a", "c")].edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            
            data_i[("c", "a", "x")].edge_attr = torch.tensor(edge_attr, dtype=torch.float).unsqueeze(1) #V1
            data_i[("c", "a", "x")].edge_index = torch.tensor(edge_index_reverse, dtype=torch.long).t().contiguous() #V1
            
            
            
            #Edges between constraints and b
            edge_attr = []
            edge_index = []
            
            edge_attr.append(0)
            edge_index.append([0, 0])
            
            for i in range(this_x.shape[0]):
                edge_index.append([i, i])
                edge_attr.append(this_x[i, -1])
            
            data_i[("b", "b", "c")].edge_attr = torch.tensor(edge_attr, dtype=torch.float).unsqueeze(1)
            data_i[("b", "b", "c")].edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            
            data_i[("c", "b", "b")].edge_attr = torch.tensor(edge_attr, dtype=torch.float).unsqueeze(1) #V1
            data_i[("c", "b", "b")].edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() #V1
            
            if debug:
                debug = False
                print("\n")
                print("This is the data_i: ", data_i)
                print("This is data_i[('x', 'a', 'c')]", data_i[("x", "a", "c")])
                print("This is data_i[('c', 'b', 'b')]", data_i[("c", "b", "b")])

            converted_data.append(data_i)
        
        return converted_data
    
