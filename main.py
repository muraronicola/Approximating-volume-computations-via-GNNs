import torch
from LoadData import LoadData
import configurations.configurations as conf
import argparse
from model import GCN
from sklearn.metrics import r2_score


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

    for data in train_loader: 
        data.to(device)
        out = model(data.x, data.edge_index, data.batch) 
        loss = criterion(out, data.y) 
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    print("Loss: ", loss.item())
    #print("out: ", out.detach().cpu().numpy()[0])


def evaluate(model, eval_loader, device="cpu"):
    model.eval()

    with torch.no_grad():    
        se = []
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
            
    
    mse = sum(se) / len(eval_loader.dataset)
    acc = acc / len(eval_loader.dataset)
    return mse, acc
    


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
    load_data.add_dataset(conf_data["train-test-data"][1], train_data=True)
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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.000005)
    criterion = torch.nn.MSELoss()

    for epoch in range(1, 1000000):
        train(model, train_loader, optimizer, criterion, device=device)
        
        train_mse, acc_train = evaluate(model, train_loader, device=device)
        test_mse, acc_test = evaluate(model, test_loader, device=device)
        dev_mse, acc_dev = evaluate(model, dev_loader, device=device)
        
        #print(f'Epoch: {epoch:03d}, Train MSE: {train_mse:.4f}, Dev MSE: {dev_mse:.4f}, Test MSE: {test_mse:.4f}')
        #print(f'Epoch: {epoch:03d}, Train acc: {acc_train:.4f}, Dev acc: {acc_dev:.4f}, Test acc: {acc_test:.4f}')


if __name__ == "__main__":
    main()
    