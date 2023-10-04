import pandas as pd 
import numpy as np
import math 

import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader

import os 
from tqdm import tqdm 
from torch.utils.tensorboard import SummaryWriter
from sklearn.feature_selection import SelectKBest, f_regression

import yfinance as yf


#data related 
class FinanceDataset(Dataset): 
    def __init__(self, data , target = None):
        self.data = torch.FloatTensor(data) if data is not None else None
        self.target = torch.FloatTensor(target) if target is not None else None

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.target is None:
            return self.data[idx]
        else:
            return self.data[idx], self.target[idx]

class DataSetTool(): 

    def __init__(self):
        return None

    def yf_data_gripper(self,firm , start_date, end_date = None):
        ticker = yf.Ticker(firm)
        df = ticker.history(start = start_date, end = end_date)
        return df
        
    def sequence_processing(self, df, day_back):# input a dataframe, output a dataframe with day_back days of data
        df2 = df.copy(deep = True)
        for day in range(1, day_back + 1):
            df2 = pd.concat([df2, df.shift(day).add_suffix(f'_{day}')], axis = 1)
        df2.dropna(inplace = True)
        return df2
    
    def move_target(self, df, target): # move target to the fisrt column
        target = df[target]
        df.drop(['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits'], axis = 1, inplace = True)
        df.insert(0, target.name, target)
        return df
    
    def data_split(self, train_data, test_data, config):
        #move target to the first column
        train_data = self.move_target(train_data, config["target"])
        test_data = self.move_target(test_data, config["target"])
        train_data = train_data.values
        test_data = test_data.values


        #train valid split
        valid = int(config["valid_ratio"]*len(train_data))
        train_set_size = len(train_data) - valid
        train_set, valid_set = torch.utils.data.random_split(train_data, [train_set_size, valid], generator = torch.Generator().manual_seed(config["seed"]))
        train_set = np.array(train_set)
        valid_set = np.array(valid_set)
     
        #X,y split 
        X_train, y_train = train_set[:,1:], train_set[:,0]
        X_valid, y_valid = valid_set[:,1:], valid_set[:,0]
        X_test, y_test = test_data[:,1:], test_data[:,0]

        #feature selection 
        selector = SelectKBest(f_regression, k = config["KBest"])
        selector.fit(X_train,y_train)
        mask = selector.get_support()
        X_train = X_train[:,mask]
        X_valid = X_valid[:,mask]
        X_test = X_test[:,mask]    
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        return X_train, y_train, X_valid, y_valid, X_test, y_test

    def train_valid_split(self, data_set, test_ratio = 0.15, seed = 10086):
        test = int(test_ratio*len(data_set))
        train_set_size = len(data_set) - test
        train_set, test_set = torch.utils.data.random_split(data_set, [train_set_size, test])
        return np.array(train_set), np.array(test_set)
    
    def feature_selection(self,X,y,K):
        selector = SelectKBest(f_regression, k = K )
        selector.fit(X, y)
        mask = selector.get_support()
        #new_features = X.columns[mask]
        return mask 

#model related 

class My_model(nn.Module):
    def __init__(self, input_dim):
        super(My_model, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(1)
        return x



def trainer(train_load, valid_loader, model, config, device = "cuda" if torch.cuda.is_available() else "cpu"):
    criterion = nn.MSELoss(reduction = "mean")
    optimizer = torch.optim.Adam(model.parameters(), lr = config["learning_rate"])

    writer = SummaryWriter()

    if not os.path.isdir('./models'):
        os.mkdir("./models")
    
    n_epochs, best_loss, step ,early_stop_count = config["n_epochs"], math.inf, 0, config["early_stop"]

    for epoch in range(n_epochs):
        model.train()
        loss_record = []

        train_pbar = tqdm(train_load, position = 0 , leave = True) 

        for x, y in train_pbar: 
            optimizer.zero_grad()
            x,y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            
            loss.backward()
            optimizer.step()
            step += 1
            loss_record.append(loss.detach().item())

            train_pbar.set_description(f"Epoch [{epoch + 1}/{n_epochs}]")
            train_pbar.set_postfix({"loss": loss.detach().item()})
        
        mean_train_loss = sum(loss_record)/len(loss_record)
        writer.add_scalar("Loss/train", mean_train_loss, step)
        
        model.eval()
        loss_record = []
        for x,y in valid_loader: 
            x,y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)
            loss_record.append(loss.detach().item())    
        
        mean_valid_loss = sum(loss_record)/len(loss_record)
        print(f"Epoch [{epoch+1}/{n_epochs}]: Train loss: {mean_train_loss}, valid loss: {mean_valid_loss:.4f}")
        writer.add_scalar("loss/valid", mean_valid_loss, step)

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(),config["save_path"])
            print("Saving model with loss {:.3f}...".format(best_loss))
            early_stop_count = 0 
        else:
            early_stop_count += 1
        
        if early_stop_count >= config['early_stop']:
            print("\n Model is not improving, so we halt the training process.")
            return 
    

#other tools

def same_seed(seed):
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

#prediction 
def predict(test_loader, model, device = "cuda" if torch.cuda.is_available() else "cpu"):
    model.eval() 
    preds = []
    for x in tqdm(test_loader):
        x = x.to(device)                        
        with torch.no_grad():                   
            pred = model(x)                     
            preds.append(pred.detach().cpu())   
    preds = torch.cat(preds, dim=0).numpy()  
    return preds

