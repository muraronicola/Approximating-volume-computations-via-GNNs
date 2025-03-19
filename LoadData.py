import numpy as np
import torch
from sklearn.model_selection import train_test_split
import DataUnit as du
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import StandardScaler

class LoadData():
    def __init__(self, base_path="./data/", exact_polytopes=True, shape=(4,4), rng=None):
        self.base_path = base_path
        self.rng = rng
        
        self.file_name = "/exact_politopes" if exact_polytopes else "/all_politopes"
        
        shape = (0, shape[0], shape[1])
        self.shape = shape
        
        self.x_train = np.empty(self.shape)
        self.y_train = np.empty(0)
        self.x_test = np.empty(self.shape)
        self.y_test = np.empty(0)
        
        
    def convert_to_target_shape(self, x):
        if x.shape[1] < self.shape[1]:
            extra_column = np.zeros((x.shape[0], self.shape[1] - x.shape[1], x.shape[2]))
            x = np.concatenate((x, extra_column), axis=1)

        if x.shape[2] < self.shape[2]:
            extra_column = np.zeros((x.shape[0], x.shape[1], self.shape[2] - x.shape[2]))
            x = np.concatenate((x, extra_column), axis=2)
        
        return x 


    def add_dataset(self, folder_name, train_data=True, cutoff=-1): #I can train on r3 and test on r4. Right now i can't train on r4 and r3
        x = np.load(self.base_path + folder_name + self.file_name + "_x.npy", allow_pickle=True)
        y = np.load(self.base_path + folder_name + self.file_name+ "_y.npy", allow_pickle=True)
        
        x = self.convert_to_target_shape(x)
        #x[:,:,-1] = x[:,:,-1]*-1 #b should be negative
        
        if cutoff > 0:
            #remove all the samples that have a value higher than the cutoff
            mask = y < cutoff
            x = x[mask]
            y = y[mask]
        
        
        """print("self.x_train.shape", self.x_train.shape)
        print("x.shape", x.shape)
        
        print("self.y_train.shape", self.y_train.shape)
        print("y.shape", y.shape)"""
        
        
        if train_data:
            self.x_train = np.concatenate((self.x_train, x)) 
            self.y_train = np.concatenate((self.y_train, y))
        else:
            self.x_test = np.concatenate((self.x_test, x))
            self.y_test = np.concatenate((self.y_test, y))
    

    def get_node_features(self):
        return self.x_train[0].shape


    def get_dataloaders(self, dev_split_size=0.2, test_split_size=0.2, train_batch_size=16, eval_batch_size=32, normalize=False, conversions="constraints", n_max_samples=100000, only_inference=False):
        du_train, du_dev, du_test = self.get_data(dev_split_size, test_split_size, normalize, conversions, n_max_samples, only_inference)
        
        train_loader = DataLoader(du_train, batch_size=train_batch_size, shuffle=False)
        dev_loader = DataLoader(du_dev, batch_size=eval_batch_size, shuffle=False)
        test_loader = DataLoader(du_test, batch_size=eval_batch_size, shuffle=False)
        
        return train_loader, dev_loader, test_loader
    
    
    def get_data(self, dev_split_size=0.2, test_split_size=0.2, normalize=False, conversions="constraints", n_max_samples=100000, only_inference=False):
        
        
        #Shuffle the data
        p = self.rng.permutation(len(self.x_train))
        self.x_train = self.x_train[p]
        self.y_train = self.y_train[p]
        
        self.x_train = self.x_train[:n_max_samples]
        self.y_train = self.y_train[:n_max_samples]
        
        if only_inference:
            x_train = self.x_test
            y_train = self.y_train
        else:
            if self.y_test.size == 0:
                first_cut = dev_split_size+test_split_size
                second_cut = dev_split_size/first_cut
            
                x_train, x_test, y_train, y_test = train_test_split(self.x_train, self.y_train, test_size=first_cut, random_state=0, shuffle=True)
                x_test, x_dev, y_test, y_dev = train_test_split(x_test, y_test, test_size=second_cut, random_state=0)
            else:
                self.x_test = self.x_test[:n_max_samples]
                self.y_test = self.y_test[:n_max_samples]
                
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
        
        return du_train, du_dev, du_test