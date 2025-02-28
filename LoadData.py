import numpy as np
import torch
from sklearn.model_selection import train_test_split
import DataUnit as du
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import StandardScaler

class LoadData():
    def __init__(self, base_path="./data/", exact_polytopes=True):
        self.base_path = base_path
        
        self.file_name = "/exact_politopes" if exact_polytopes else "/all_politopes"
        
        self.x_train = np.array([])
        self.y_train = np.array([])
        self.x_test = np.array([])
        self.y_test = np.array([])



    def add_dataset(self, folder_name, train_data=True): #I can train on r3 and test on r4. Right now i can't train on r4 and r3
        x = np.load(self.base_path + folder_name + self.file_name + "_x.npy", allow_pickle=True)
        y = np.load(self.base_path + folder_name + self.file_name+ "_y.npy", allow_pickle=True)
        
        
        x = x[:100]
        y = y[:100]
        
        """x = x[:10]
        y = y[:10]
        
        print("x", x)
        print("y", y)
        
        e1 = [[1, 3, 15],
            [2, 1, 5],
            [2, 9, 31]]
        
        e2 = [[6, 7, 52],
            [7, 11, 5],
            [9, 8, 12]]
        
        x = np.array([e1, e2, e1, e2, e1, e2, e1, e2, e1, e2])
        y = np.array([1, 100, 1, 100, 1, 100, 1, 100, 1, 100])
        
        print("x", x)
        print("y", y)"""
        
        if train_data:
            self.x_train = x #np.append(self.x_train, x, axis=0)
            self.y_train = y #np.append(self.y_train, y, axis=0)
        else:
            self.x_test = x #np.append(self.x_test, x, axis=0)
            self.y_test = y #np.append(self.y_test, y, axis=0)
    

    def get_node_features(self):
        return self.x_train[0].shape


    def get_dataloaders(self, dev_split_size=0.2, test_split_size=0.2, train_batch_size=16, eval_batch_size=32):
        
        if self.y_test.size == 0:
            first_cut = dev_split_size+test_split_size
            second_cut = dev_split_size/first_cut

            x_train, x_test, y_train, y_test = train_test_split(self.x_train, self.y_train, test_size=first_cut, random_state=0, shuffle=False)
            x_test, x_dev, y_test, y_dev = train_test_split(x_test, y_test, test_size=second_cut, random_state=0)
        else:
            x_train, x_dev, y_train, y_dev = train_test_split(self.x_train, self.y_train, test_size=dev_split_size, random_state=0, shuffle=False)
            x_test = self.x_test
            y_test = self.y_test
        
        
        """
        scalers = {}
        for i in range(x_train.shape[1]):
            scalers[i] = StandardScaler()
            x_train[:, i, :] = scalers[i].fit_transform(x_train[:, i, :]) 

        for i in range(x_dev.shape[1]):
            x_dev[:, i, :] = scalers[i].transform(x_dev[:, i, :])

        for i in range(x_test.shape[1]):
            x_test[:, i, :] = scalers[i].transform(x_test[:, i, :]) 
        """

        
        du_train = du.DataUnit(x_train, y_train)
        du_dev = du.DataUnit(x_dev, y_dev)
        du_test = du.DataUnit(x_test, y_test)
        
        train_loader = DataLoader(du_train, batch_size=train_batch_size, shuffle=False)
        dev_loader = DataLoader(du_dev, batch_size=train_batch_size, shuffle=False)
        test_loader = DataLoader(du_test, batch_size=eval_batch_size, shuffle=False)
        
        return train_loader, dev_loader, test_loader