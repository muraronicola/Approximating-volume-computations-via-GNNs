import torch
from LoadData import LoadData
import configurations.configurations as conf
import argparse
from model import GCN
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


def train(model, train_loader, optimizer, criterion, device="cpu"):
    model.train()

    loss_array = []
    for data in train_loader: 
        data.to(device)
        out = model(data.x, data.edge_index, data.batch) 
        loss = criterion(out, data.y) 
        loss_array.append(loss.item())
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    #print("Loss: ", loss.item())
    #print("out: ", out.detach().cpu().numpy()[0])
    return loss_array


def evaluate(model, eval_loader, device="cpu"):
    model.eval()

    with torch.no_grad():    
        pred_mean = 0
        array_pred = np.empty(0)
        array_y = np.empty(0)
        for data in eval_loader:
            data.to(device)
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
        file_name = base_filename + "_" + "run" + "_" + str(i)
        if not os.path.exists("./runs/" + file_name + ".csv"):
            return file_name
        i += 1

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-path_configuration", type=str, default="./configurations/default.json", help="specify the path to the configuration file")
    parser.add_argument("-out_filename", type=str, default="default", help="specify the output file name")

    args = parser.parse_args()
    path_configuration = args.path_configuration
    base_filename = args.out_filename
    
    configuration = conf.get_configuration(path_configuration)
    
    file_name = find_filename(base_filename)
    print("Saving results in: ", file_name)
    
    device = configuration["device"]

    conf_data = configuration["data"]
    load_data = LoadData(base_path=conf_data["base_path"], exact_polytopes=conf_data["exact_polytopes"], shape=(conf_data["target_shape"][0], conf_data["target_shape"][1]))
    load_data.add_dataset(conf_data["train-test-data"][0], train_data=conf_data["train-test-data-train"][0], cutoff=conf_data["cutoff"])
    
    if len(conf_data["train-test-data"]) == 2:
        load_data.add_dataset(conf_data["train-test-data"][1], train_data=conf_data["train-test-data-train"][1])
    
    node_features = load_data.get_node_features()[1]
    
    
    conf_train = configuration["train"]
    train_loader, dev_loader, test_loader = load_data.get_dataloaders(test_split_size=conf_data["train-test-split"], dev_split_size=conf_data["train-eval-split"], train_batch_size=conf_train["train_batch_size"], eval_batch_size=conf_train["eval_batch_size"], normalize=conf_data["normalize"], conversions=conf_data["conversion"], n_max_samples=conf_data["max_samples"])
    
    
    #Save txt file with configuration
    file_config = open("./runs/" + file_name + ".txt", "w")
    #It's a json i want a new line for each key
    json_str = str(configuration).replace(", ", ",\n").replace("{", "{\n").replace("}", "\n}")
    file_config.write(json_str)
    file_config.close()
    
    model = GCN(node_features=node_features, hidden_channels=conf_train["hidden_channels"], p_drop=conf_train["dropout"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=conf_train["learning_rate"])
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.000005)
    
    if conf_train["loss"] == "mse":
        criterion = torch.nn.MSELoss()
    elif conf_train["loss"] == "l1":
        criterion = torch.nn.L1Loss()
    else:
        raise ValueError("Loss not supported")

    results = pd.DataFrame(columns=["epoch", "train_loss", "train_mse", "test_mse", "dev_mse", "mean_error_train", "mean_error_dev", "mean_error_test", "mean_pred_train", "mean_pred_dev", "mean_pred_test", "std_pred_train", "std_pred_dev", "std_pred_test"])
    
    iterator = tqdm(range(1, conf_train["train_epochs"]+1))
    best_eval = 100000000
    best_epoch_eval = 0
    for epoch in iterator:
        loss = train(model, train_loader, optimizer, criterion, device=device)
        
        train_mse, mean_error_train, mean_pred_train, std_pred_train, _, _ = evaluate(model, train_loader, device=device)
        dev_mse, mean_error_dev, mean_pred_dev, std_pred_dev, _, _ = evaluate(model, dev_loader, device=device)
        test_mse, mean_error_test, mean_pred_test, std_pred_test, _, _ = evaluate(model, test_loader, device=device)
        
        
        if conf_train["loss"] == "mse":
            if dev_mse < best_eval:
                best_eval = dev_mse
                best_epoch_eval = epoch
        else:
            if mean_error_dev < best_eval:
                best_eval = mean_error_dev
                best_epoch_eval = epoch
        
        if epoch - best_epoch_eval > conf_train["early_stopping"]:
            break
        
        new_data = pd.DataFrame([[epoch, np.mean(loss), train_mse, test_mse, dev_mse, mean_error_train, mean_error_dev, mean_error_test, mean_pred_train, mean_pred_dev, mean_pred_test, std_pred_train, std_pred_dev, std_pred_test]], columns=["epoch", "train_loss", "train_mse", "test_mse", "dev_mse", "mean_error_train", "mean_error_dev", "mean_error_test", "mean_pred_train", "mean_pred_dev", "mean_pred_test", "std_pred_train", "std_pred_dev", "std_pred_test"])
        results = pd.concat([results, new_data])
        
        iterator.set_description(f'Train mean loss: { np.mean(loss):.4f}')

    results.to_csv("./runs/" + file_name + ".csv")
    
    
    figure2, ax2 = plt.subplots(5, 3, figsize=(18, 20))

    sns.lineplot(data=results, x="epoch", y="train_mse", ax=ax2[0, 0])
    sns.lineplot(data=results, x="epoch", y="dev_mse", ax=ax2[0, 1])
    sns.lineplot(data=results, x="epoch", y="test_mse", ax=ax2[0, 2])

    sns.lineplot(data=results, x="epoch", y="train_mse", ax=ax2[1, 0])
    sns.lineplot(data=results, x="epoch", y="dev_mse", ax=ax2[1, 1])
    sns.lineplot(data=results, x="epoch", y="test_mse", ax=ax2[1, 2])
    ax2[1, 0].set_yscale('log')
    ax2[1, 1].set_yscale('log')
    ax2[1, 2].set_yscale('log')


    sns.lineplot(data=results, x="epoch", y="mean_error_train", ax=ax2[2, 0])
    sns.lineplot(data=results, x="epoch", y="mean_error_dev", ax=ax2[2, 1])
    sns.lineplot(data=results, x="epoch", y="mean_error_test", ax=ax2[2, 2])

    sns.lineplot(data=results, x="epoch", y="mean_pred_train", ax=ax2[3, 0])
    sns.lineplot(data=results, x="epoch", y="mean_pred_dev", ax=ax2[3, 1])
    sns.lineplot(data=results, x="epoch", y="mean_pred_test", ax=ax2[3, 2])

    sns.lineplot(data=results, x="epoch", y="std_pred_train", ax=ax2[4, 0])
    sns.lineplot(data=results, x="epoch", y="std_pred_dev", ax=ax2[4, 1])
    sns.lineplot(data=results, x="epoch", y="std_pred_test", ax=ax2[4, 2])

    figure2.tight_layout()
    figure2.savefig("./runs/" + file_name + ".png")
    
    
    
    _, _, _, _, y, y_pred = evaluate(model, train_loader, device=device)
    train_predictions = pd.DataFrame(columns=["y", "y_pred"])
    train_predictions["y"] = y
    train_predictions["y_pred"] = y_pred
    train_predictions.to_csv("./runs/" + file_name + "_train_predictions.csv")
    
    _, _, _, _, y, y_pred = evaluate(model, dev_loader, device=device)
    dev_predictions = pd.DataFrame(columns=["y", "y_pred"])
    dev_predictions["y"] = y
    dev_predictions["y_pred"] = y_pred
    dev_predictions.to_csv("./runs/" + file_name + "_dev_predictions.csv")
    
    _, _, _, _, y, y_pred = evaluate(model, test_loader, device=device)
    test_predictions = pd.DataFrame(columns=["y", "y_pred"])
    test_predictions["y"] = y
    test_predictions["y_pred"] = y_pred
    test_predictions.to_csv("./runs/" + file_name + "_test_predictions.csv")


if __name__ == "__main__":
    main()