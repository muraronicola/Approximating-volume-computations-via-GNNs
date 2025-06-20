import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import StandardScaler
import random

import src.DataUnit as du

#Class used to load the data from the various numpy files
class LoadData():
    def __init__(self, base_path="./data/", dev_split_size=0.2, test_split_size=0.2, seed=0):
        self.base_path = base_path
        self.rng = np.random.default_rng(seed)
        random.seed(seed)
        
        self.file_name = "/exact_politopes"
        
        self.dev_split_size = dev_split_size
        self.test_split_size = test_split_size

        self.x_train = []
        self.y_train = []
        self.x_dev = []
        self.y_dev = []
        self.x_test = []
        self.y_test = []

    # Function to add a dataset to the loader
    def add_dataset(self, folder_name, split): 
        x = np.load(self.base_path + folder_name + "/exact_politopes_x.npy", allow_pickle=True)
        y = np.load(self.base_path + folder_name + "/exact_politopes_y.npy", allow_pickle=True)
        
        #Shuffle the data (in order to have a different distribution between train/dev/test)
        p = self.rng.permutation(len(x))
        x = x[p]
        y = y[p]
        
        # Calculate the number of elements to split for dev and test sets
        n_elements_split_dev = int(len(x) * self.dev_split_size)
        n_elements_split_test = int(len(x) * self.test_split_size)
        
        #Split the data according to the specified split
        if split == "tr":
            self.x_train.append(x)
            self.y_train.append(y)

        elif split == "de":
            self.x_dev.append(x)
            self.y_test.append(y)

        elif split == "te":
            self.x_test.append(x)
            self.y_test.append(y)

        elif split == "tr-de":
            self.x_train.append(x[:len(x) - n_elements_split_dev])
            self.y_train.append(y[:len(y) - n_elements_split_dev])
            self.x_dev.append(x[len(x) - n_elements_split_dev:])
            self.y_dev.append(y[len(y) - n_elements_split_dev:])

        elif split == "tr-te":
            self.x_train.append(x[:len(x) - n_elements_split_test])
            self.y_train.append(y[:len(y) - n_elements_split_test])
            self.x_test.append(x[len(x) - n_elements_split_test:])
            self.y_test.append(y[len(y) - n_elements_split_test:])

        elif split == "de-te":
            self.x_dev.append(x[:n_elements_split_dev])
            self.y_dev.append(y[:n_elements_split_dev])
            self.x_test.append(x[len(x) - n_elements_split_test:])
            self.y_test.append(y[len(y) - n_elements_split_test:])

        elif split == "tr-de-te":
            self.x_train.append(x[:len(x) - (n_elements_split_dev+n_elements_split_test)])
            self.y_train.append(y[:len(y) - (n_elements_split_dev+n_elements_split_test)])
            self.x_dev.append(x[len(x) - (n_elements_split_dev+n_elements_split_test) : len(x) - n_elements_split_test])
            self.y_dev.append(y[len(y) - (n_elements_split_dev+n_elements_split_test) : len(y) - n_elements_split_test])
            self.x_test.append(x[len(x) - n_elements_split_test:])
            self.y_test.append(y[len(y) - n_elements_split_test:])
            
        else:
            raise ValueError("Invalid split value. Use 'tr', 'te', 'tr-de', 'tr-te', 'de-te', or 'tr-de-te'.")


    # Function to flatten the list of lists
    def flatten(self, xss):
        return [x for xs in xss for x in xs]


    # Function to get the dataloaders for training, dev, and test sets
    def get_dataloaders(self, train_batch_size=16, eval_batch_size=32, normalize=False, n_max_samples=None):
        
        x_train = self.flatten(self.x_train)
        x_dev = self.flatten(self.x_dev)
        x_test = self.flatten(self.x_test)
        
        y_train = self.flatten(self.y_train)
        y_dev = self.flatten(self.y_dev)
        y_test = self.flatten(self.y_test)
        
        
        if n_max_samples is not None:
            if n_max_samples < len(x_train):
                #Shuffle necessary in order not to cut the data from only the last datasets
                temp = list(zip(x_train, y_train))  
                random.shuffle(temp) 
                x_train, y_train = zip(*temp)  
                
                x_train = [x for x in x_train[:n_max_samples]]
                y_train = [y for y in y_train[:n_max_samples]]
                
        
        if normalize:
            for i in range(len(x_train)):
                x_train[i] = x_train[i] / np.linalg.norm(x_train[i])
            
            for i in range(len(x_dev)):
                x_dev[i] = x_dev[i] / np.linalg.norm(x_dev[i])
                
            for i in range(len(x_test)):
                x_test[i] = x_test[i] / np.linalg.norm(x_test[i])
        
        
        #Create the DataUnit objects for training, dev, and test sets
        du_train = du.DataUnit(x_train, y_train)
        du_dev = du.DataUnit(x_dev, y_dev)
        du_test = du.DataUnit(x_test, y_test)
        
        #Get the dataloaders for training, dev, and test sets
        train_loader = DataLoader(du_train, batch_size=train_batch_size, shuffle=False)
        dev_loader = DataLoader(du_dev, batch_size=eval_batch_size, shuffle=False)
        test_loader = DataLoader(du_test, batch_size=eval_batch_size, shuffle=False)
        
        return train_loader, dev_loader, test_loader