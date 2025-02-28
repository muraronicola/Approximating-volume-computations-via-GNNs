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

def r2_accuracy(pred_y, y):
    score = r2_score(y, pred_y)
    return round(score, 2)*100

def calculate_se_over_batch(out, y):
    se_global = 0
    
    for i in range(len(out)):
        se = (out[i].item() - y[i])**2
        se_global += se
        
        #print(out[i], y[i], mse)
        #print("+++++")
    
    return se_global


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
        se = []
        pred_mean = 0
        pred_std = 0
        acc = 0
        for data in eval_loader:
            data.to(device)
            out = model(data.x, data.edge_index, data.batch, train=False)
            this_se = calculate_se_over_batch(out.detach().cpu().numpy(), data.y.detach().cpu().numpy())
            se.append(this_se)
            
            #print("out: ", flatten_out)
            #print("out: ", flatten_out.detach().cpu().numpy())
            #print("data: ", data.y.detach().cpu().numpy())
            
            this_acc = r2_accuracy(out.detach().cpu().numpy(), data.y.detach().cpu().numpy())
            #print(out[0], data.y[0], this_mse)
            acc += this_acc
            pred_mean += np.mean(np.array(out.detach().cpu().numpy()))
            pred_std += np.std(np.array(out.detach().cpu().numpy()))
            
            
    
    mse = sum(se) / len(eval_loader.dataset)
    acc = acc / len(eval_loader.dataset)
    
    pred_mean = pred_mean / len(eval_loader.dataset)
    pred_std = pred_std / len(eval_loader.dataset)
    
    return mse, acc, pred_mean, pred_std
    


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-path_configuration", type=str, default="./configurations/default.json", help="specify the path to the configuration file"
    )

    args = parser.parse_args()

    path_configuration = args.path_configuration
    configuration = conf.get_configuration(path_configuration)

    device = configuration["device"]

    conf_data = configuration["data"]
    load_data = LoadData(base_path=conf_data["base_path"], exact_polytopes=conf_data["exact_polytopes"])
    load_data.add_dataset(conf_data["train-test-data"][0], train_data=True)
    #load_data.add_dataset(conf_data["train-test-data"][1], train_data=False)
    
    node_features = load_data.get_node_features()[1]
    
    conf_train = configuration["train"]
    train_loader, dev_loader, test_loader = load_data.get_dataloaders(test_split_size=conf_data["train-test-split"], dev_split_size=conf_data["train-eval-split"], train_batch_size=conf_train["train_batch_size"], eval_batch_size=conf_train["eval_batch_size"])
    
    """print(train_loader)
    print("----------------")
    
    for batch in train_loader:
        print(batch)
        print("+++++")"""
        
    
    model = GCN(node_features=node_features, hidden_channels=256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.000005)
    criterion = torch.nn.MSELoss()

    results = pd.DataFrame(columns=["epoch", "train_loss", "train_mse", "test_mse", "dev_mse", "acc_train", "acc_test", "acc_dev", "mean_pred_train", "mean_pred_dev", "mean_pred_test", "std_pred_train", "std_pred_dev", "std_pred_test"])
    
    iterator = tqdm(range(1, 10000))
    for epoch in iterator:
        loss = train(model, train_loader, optimizer, criterion, device=device)
        
        train_mse, acc_train, mean_pred_train, std_pred_train = evaluate(model, train_loader, device=device)
        test_mse, acc_test, mean_pred_dev, std_pred_dev = evaluate(model, test_loader, device=device)
        dev_mse, acc_dev, mean_pred_test, std_pred_test = evaluate(model, dev_loader, device=device)
        
        new_data = pd.DataFrame([[epoch, np.mean(loss), train_mse, test_mse, dev_mse, acc_train, acc_test, acc_dev, mean_pred_train, mean_pred_dev, mean_pred_test, std_pred_train, std_pred_dev, std_pred_test]], columns=["epoch", "train_loss", "train_mse", "test_mse", "dev_mse", "acc_train", "acc_test", "acc_dev", "mean_pred_train", "mean_pred_dev", "mean_pred_test", "std_pred_train", "std_pred_dev", "std_pred_test"])
        results = pd.concat([results, new_data])
        
        iterator.set_description(f'Train mean loss: { np.mean(loss):.4f}')

    results.to_csv("./runs/run01.csv")
    
    
    figure, ax = plt.subplots(2, 3, figsize=(18, 8))
    sns.lineplot(data=results, x="epoch", y="train_mse", ax=ax[0, 0])
    sns.lineplot(data=results, x="epoch", y="dev_mse", ax=ax[0, 1])
    sns.lineplot(data=results, x="epoch", y="test_mse", ax=ax[0, 2])
    sns.lineplot(data=results, x="epoch", y="acc_train", ax=ax[1, 0])
    sns.lineplot(data=results, x="epoch", y="acc_dev", ax=ax[1, 1])
    sns.lineplot(data=results, x="epoch", y="acc_test", ax=ax[1, 2])
    
    figure.tight_layout()
    figure.savefig("./runs/run01.png")

if __name__ == "__main__":
    main()
    