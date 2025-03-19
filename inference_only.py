import torch
from LoadData import LoadData
import configurations.configurations as conf
import argparse
from Heterogeneus_model import Heterogeneus
from Homogeneus_model import Homogeneus
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

def r2_accuracy(pred_y, y):
    score = r2_score(y, pred_y)
    return round(score, 2)*100

def calculate_se(out, y):
    se_global = 0
    
    for i in range(len(out)):
        se = (out[i] - y[i])**2
        se_global += se
        
        #print(out[i], y[i], mse)
        #print("+++++")
    
    return se_global / len(out)

def calculate_error(out, y):
    error_global = 0
    
    for i in range(len(out)):
        error = abs(out[i] - y[i])
        error_global += error
    
    return error_global / len(out)


def train(model, train_loader, optimizer, criterion, heterogeneus=False, device="cpu"):
    model.train()

    loss_array = []
    for data in train_loader: 
        data.to(device)
        
        if heterogeneus:
            out = model(data.x_dict, data.edge_index_dict, data.batch_dict)
        else:
            out = model(data.x, data.edge_index, data.batch) 
        
        loss = criterion(out, data.y) 
        loss_array.append(loss.item())
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    #print("Loss: ", loss.item())
    #print("out: ", out.detach().cpu().numpy()[0])
    return loss_array


def evaluate(model, eval_loader, heterogeneus = False, device="cpu"):
    model.eval()

    with torch.no_grad():    
        pred_mean = 0
        array_pred = np.empty(0)
        array_y = np.empty(0)
        for data in eval_loader:
            data.to(device)
            
            if heterogeneus:
                out = model(data.x_dict, data.edge_index_dict, data.batch_dict, train=False)
            else:
                out = model(data.x, data.edge_index, data.batch, train=False) 
                
            #print("out: ", flatten_out)
            #print("out: ", flatten_out.detach().cpu().numpy())
            #print("data: ", data.y.detach().cpu().numpy())
            
            #print(out[0], data.y[0], this_mse)
            #pred_sum += np.sum(np.array(out.detach().cpu().numpy()))
            array_pred = np.concatenate((array_pred, out.detach().cpu().numpy().flatten()), axis=None)
            array_y = np.concatenate((array_y, data.y.detach().cpu().numpy().flatten()), axis=None)
    
    mse = calculate_se(array_pred, array_y)
    mean_error = calculate_error(array_pred, array_y)
    
    pred_mean = np.mean(array_pred)
    pred_std = np.std(array_pred)
    
    return mse, mean_error, pred_mean, pred_std, array_y, array_pred
    

def find_filename(base_filename):
    i = 1
    while True:
        file_name = base_filename + "__" + "inference" + "_" + str(i)
        if not os.path.exists("./runs/" + file_name + ".csv"):
            return file_name
        i += 1

def main():
    seed = 0

    parser = argparse.ArgumentParser()
    parser.add_argument("-path_configuration", type=str, default="./configurations/inference_only.json", help="specify the path to the configuration file")

    args = parser.parse_args()
    path_configuration = args.path_configuration
    
    configuration = conf.get_configuration(path_configuration)
    model_name = configuration["model_name"]
    
    file_name = find_filename(model_name)
    print("Saving results in: ", file_name)
    
    rng = np.random.default_rng(seed)
    
    device = configuration["device"]

    conf_data = configuration["data"]
    conf_model = configuration["model"]
    conf_train = configuration["train"]
    
    conf_data["target_shape"][1] += 1
    
    load_data = LoadData(base_path=conf_data["base_path"], exact_polytopes=conf_data["exact_polytopes"], shape=(conf_data["target_shape"][0], conf_data["target_shape"][1]), rng=rng)
    load_data.add_dataset(conf_data["train-test-data"][0], train_data=True, cutoff=conf_data["cutoff"])
    
    for i in range(1, len(conf_data["train-test-data"])):
        load_data.add_dataset(conf_data["train-test-data"][i], train_data=True)
    
    node_features = load_data.get_node_features()[1]
    
    
    if (conf_data["conversion"] == "constraints" or conf_data["conversion"] == "dimensions") and conf_model["heterogeneus"]:
        raise ValueError("Invalid combination of converstion and heterogeneus")
    
    if (conf_data["conversion"] == "h1" or conf_data["conversion"] == "h2") and not conf_model["heterogeneus"]:
        raise ValueError("Invalid combination of converstion and heterogeneus")
    
    
    train_loader, _, _ = load_data.get_dataloaders(test_split_size=conf_data["train-test-split"], dev_split_size=conf_data["train-eval-split"], train_batch_size=conf_train["train_batch_size"], eval_batch_size=conf_train["eval_batch_size"], normalize=conf_data["normalize"], conversions=conf_data["conversion"], n_max_samples=conf_data["max_samples"], only_inference=True)
    
    
    #Save txt file with configuration
    file_config = open("./runs/" + file_name + ".txt", "w")
    #It's a json i want a new line for each key
    json_str = str(configuration).replace(", ", ",\n").replace("{", "{\n").replace("}", "\n}")
    file_config.write(json_str)
    file_config.close()
    
    
    if conf_model["heterogeneus"]:
        model = Heterogeneus(node_features=node_features, hidden_channels=conf_train["hidden_channels"], n_releations=conf_model["n_releations"], p_drop=conf_train["dropout"], targhet_shape=conf_data["target_shape"], conversion=conf_data["conversion"], n_layers=conf_train["n_layers"]).to(device)
    else:
        model = Homogeneus(node_features=node_features, hidden_channels=conf_train["hidden_channels"], p_drop=conf_train["dropout"]).to(device)
    
    model = torch.load("./runs/" + model_name + ".pt", weights_only=False)
    test_mse, mean_error_test, mean_pred_test, std_pred_test, _, _ = evaluate(model, train_loader, heterogeneus=conf_model["heterogeneus"], device=device)
    
    results = pd.DataFrame(columns=["epoch", "train_mse", "dev_mse", "test_mse", "mean_error_train", "mean_error_dev", "mean_error_test", "mean_pred_train", "mean_pred_dev", "mean_pred_test", "std_pred_train", "std_pred_dev", "std_pred_test"])
    new_result = pd.DataFrame(columns=["epoch", "train_mse", "dev_mse", "test_mse", "mean_error_train", "mean_error_dev", "mean_error_test", "mean_pred_train", "mean_pred_dev", "mean_pred_test", "std_pred_train", "std_pred_dev", "std_pred_test"], data=[[0, 0, 0, test_mse, 0, 0, mean_error_test, 0, 0, mean_pred_test, 0, 0, std_pred_test]])
    results = pd.concat([results, new_result])
    
    results.to_csv("./runs/" + file_name + ".csv")


if __name__ == "__main__":
    main()