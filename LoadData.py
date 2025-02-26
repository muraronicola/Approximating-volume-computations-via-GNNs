import numpy as np
import torch
from sklearn.model_selection import train_test_split

class LoadData:
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
        
        if train_data:
            self.x_train = x #np.append(self.x_train, x, axis=0)
            self.y_train = x #np.append(self.y_train, y, axis=0)
        else:
            self.x_test = x #np.append(self.x_test, x, axis=0)
            self.y_test = y #np.append(self.y_test, y, axis=0)
    
    def get_data_split(self, dev_split_size=0.2, test_split_size=0.2):
        
        print("self.y_test.size",  self.y_test.size)
        print("self.y_test",  self.y_test)
        
        if self.y_test.size == 0:
            first_cut = dev_split_size+test_split_size
            second_cut = dev_split_size/first_cut

            print("first_cut", first_cut)
            print("second_cut", second_cut)
        
            x_train, x_test, y_train, y_test = train_test_split(self.x_train, self.y_train, test_size=first_cut, random_state=0)
            x_test, x_dev, y_test, y_dev = train_test_split(x_test, y_test, test_size=second_cut, random_state=0)
            print("first case")
        else:
            x_train, x_dev, y_train, y_dev = train_test_split(self.x_train, self.y_train, test_size=dev_split_size, random_state=0)
            x_test = self.x_test
            y_test = self.y_test
            print("second case")
            
        print("x_train", x_train.shape)
        print("x_dev", x_dev.shape)
        print("x_test", x_test.shape)
        print("y_train", y_train.shape)
        print("y_dev", y_dev.shape)
        print("y_test", y_test.shape)
        
        return x_train, x_dev, x_test, y_train, y_dev, y_test