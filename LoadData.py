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



    def add_dataset(self, folder_name, train_data=True, n_samples=1000000, cutoff=-1): #I can train on r3 and test on r4. Right now i can't train on r4 and r3
        x = np.load(self.base_path + folder_name + self.file_name + "_x.npy", allow_pickle=True)
        y = np.load(self.base_path + folder_name + self.file_name+ "_y.npy", allow_pickle=True)
        
        #x[:,:,-1] = x[:,:,-1]*-1 #b should be negative
        
        if cutoff > 0:
            #remove all the samples that have a value higher than the cutoff
            mask = y < cutoff
            x = x[mask]
            y = y[mask]
        
        x = x[:n_samples]
        y = y[:n_samples]
        #x = x[:1000]
        #y = y[:1000]
        
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


    def get_dataloaders(self, dev_split_size=0.2, test_split_size=0.2, train_batch_size=16, eval_batch_size=32, normalize=False, conversions="constraints"):
        
        if self.y_test.size == 0:
            first_cut = dev_split_size+test_split_size
            second_cut = dev_split_size/first_cut

            x_train, x_test, y_train, y_test = train_test_split(self.x_train, self.y_train, test_size=first_cut, random_state=0, shuffle=True)
            x_test, x_dev, y_test, y_dev = train_test_split(x_test, y_test, test_size=second_cut, random_state=0)
        else:
            x_train, x_dev, y_train, y_dev = train_test_split(self.x_train, self.y_train, test_size=dev_split_size, random_state=0, shuffle=True)
            x_test = self.x_test
            y_test = self.y_test
        
        
        if normalize:
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape)
            x_dev = scaler.transform(x_dev.reshape(-1, x_dev.shape[-1])).reshape(x_dev.shape)
            x_test = scaler.transform(x_test.reshape(-1, x_test.shape[-1])).reshape(x_test.shape)
        
        
        du_train = du.DataUnit(x_train, y_train, conversion=conversions)
        du_dev = du.DataUnit(x_dev, y_dev, conversion=conversions)
        du_test = du.DataUnit(x_test, y_test, conversion=conversions)
        
        train_loader = DataLoader(du_train, batch_size=train_batch_size, shuffle=False)
        dev_loader = DataLoader(du_dev, batch_size=train_batch_size, shuffle=False)
        test_loader = DataLoader(du_test, batch_size=eval_batch_size, shuffle=False)
        
        return train_loader, dev_loader, test_loader