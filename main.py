from utils import mean_relative_error, mean_squared_error, mean_absolute_error, find_filename, get_paths
import torch
from LoadData import LoadData
import configurations.configurations as conf
import argparse
from GNN_Model import GNN_Model
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import copy
import shutil

def train(model, train_loader, optimizer, loss_function, device="cpu"):
    model.train()

    loss_array = []
    for data in train_loader: 
        data.to(device)
        
        out = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict, data.batch_dict, train=True)
        
        loss = loss_function(out, data.y) 
        loss_array.append(loss.item())
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        data.to("cpu")
    
    return loss_array


def evaluate(model, eval_loader, device="cpu"):
    model.eval()
    
    pred_mean = 0
    array_pred = np.empty(0)
    array_y = np.empty(0)

    with torch.no_grad():    
        for data in eval_loader:
            data.to(device)
            
            out = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict, data.batch_dict, train=False)
            
            array_pred = np.concatenate((array_pred, out.detach().cpu().numpy().flatten()), axis=None)
            array_y = np.concatenate((array_y, data.y.detach().cpu().numpy().flatten()), axis=None)
            
            data.to("cpu")
    
    mse = mean_squared_error(array_pred, array_y)
    mae = mean_absolute_error(array_pred, array_y)
    
    pred_mean = np.mean(array_pred)
    pred_std = np.std(array_pred)
    
    mre = mean_relative_error(array_pred, array_y)
    
    return mse, mae, mre, pred_mean, pred_std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-path_configuration", type=str, default="./configurations/default.json", help="specify the path to the configuration file")

    args = parser.parse_args()
    path_configuration = args.path_configuration
    
    configuration = conf.get_configuration(path_configuration)
    
    out_conf_path, out_data_path, out_data_model = get_paths(configuration["out_path"], configuration["base_filename"])
    
    shutil.copyfile(path_configuration, out_conf_path)
    
    device = configuration["device"]
    conf_data = configuration["data"]
    conf_train = configuration["train"]
    
    load_data = LoadData(base_path=conf_data["base_path"], exact_polytopes=conf_data["exact_polytopes"], dev_split_size=conf_data["train-dev-split"], test_split_size=conf_data["train-test-split"], seed=configuration["seed"])
    load_data.add_dataset(conf_data["data"][0], split=conf_data["data-split"][0])
    
    for i in range(1, len(conf_data["data"])):
        load_data.add_dataset(conf_data["data"][i], split=conf_data["data-split"][i])
    
    train_loader, dev_loader, test_loader = load_data.get_dataloaders(train_batch_size=conf_train["train_batch_size"], eval_batch_size=conf_train["eval_batch_size"], normalize=conf_data["normalize"], n_max_samples=conf_data["max_samples"])
    
    #Save txt file with configuration
    

    model = GNN_Model(hidden_channels=conf_train["hidden_channels"], n_layers=conf_train["n_layers"]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=conf_train["learning_rate"])
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.000005)
    
    if conf_train["loss"] == "mse":
        loss_function = torch.nn.MSELoss()
    elif conf_train["loss"] == "l1":
        loss_function = torch.nn.L1Loss()
    else:
        raise ValueError("Loss not supported")

    results = pd.DataFrame(columns=["epoch", "loss_train", "mse_train", "mse_dev", "mse_test", "mae_train", "mae_dev", "mae_test", "mre_train", "mre_dev", "mre_test",  "pred_mean_train", "pred_mean_dev", "pred_mean_test", "pred_std_train", "pred_std_dev", "pred_std_test"])
    
    iterator = tqdm(range(1, conf_train["train_epochs"]+1))
    best_eval = float("inf")
    best_epoch_eval = 0
    best_model = None
    
    for epoch in iterator:
        loss = train(model, train_loader, optimizer, loss_function, device=device)
        
        mse_train, mae_train, mre_train, pred_mean_train, pred_std_train = evaluate(model, train_loader, device=device)
        mse_dev, mae_dev, mre_dev, pred_mean_dev, pred_std_dev = evaluate(model, dev_loader, device=device)
        mse_test, mae_test, mre_test, pred_mean_test, pred_std_test = evaluate(model, test_loader, device=device)
        
        
        if conf_train["loss"] == "mse":
            if mse_dev < best_eval:
                best_eval = mse_dev
                best_epoch_eval = epoch
                best_model = copy.deepcopy(model)
        else:
            if mae_dev < best_eval:
                best_eval = mae_dev
                best_epoch_eval = epoch
                best_model = copy.deepcopy(model)
        
        if epoch - best_epoch_eval > conf_train["early_stopping"]:
            break
        
        new_data = pd.DataFrame([[epoch, np.mean(loss), mse_train, mse_dev, mse_test, mae_train, mae_dev, mae_test, mre_train, mre_dev, mre_test, pred_mean_train, pred_mean_dev, pred_mean_test, pred_std_train, pred_std_dev, pred_std_test]], columns=["epoch", "loss_train", "mse_train", "mse_dev", "mse_test", "mae_train", "mae_dev", "mae_test", "mre_train", "mre_dev", "mre_test",  "pred_mean_train", "pred_mean_dev", "pred_mean_test", "pred_std_train", "pred_std_dev", "pred_std_test"])
        results = pd.concat([results, new_data])
        
        iterator.set_description(f'Mean absolute error dev: { mae_dev:.4f}')

    results.to_csv(out_data_path)
    torch.save(best_model.state_dict(), out_data_model)


if __name__ == "__main__":
    main()