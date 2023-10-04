




import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns 

import torch.nn as nn 
import torch.functional as F 
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm 
from torch.utils.tensorboard import SummaryWriter

from sklearn.preprocessing import MinMaxScaler

import yfinance as yf

from lstm.yfinance_stock_predicting.AnalysingTool import *  

dst = DataSetTool()


config = {
    "seed": 10086, 
    "valid_ratio": 0.15, 
    "KBest":10,
    "target": "Close",
    "n_epochs": 1000, 
    "batch_size": 256,
    "learning_rate": 0.001,
    "early_stop": 100, 
    "save_path": "./models/model.ckpt"
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





train_data_raw = dst.yf_data_gripper("AAPL", "1986-01-01", "2022-12-31")
train_data = dst.sequence_processing(train_data_raw, 50)
test_data_raw = dst.yf_data_gripper("AAPL", "2022-12-31", None)
test_data = dst.sequence_processing(test_data_raw, 50)


X_train, y_train, X_valid, y_valid, X_test, y_test = dst.data_split(train_data,test_data, config)

train_dataset = FinanceDataset(X_train, y_train)
valid_dataset = FinanceDataset(X_valid, y_valid)
test_dataset = FinanceDataset(X_test)
same_seed(config["seed"])
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=config["batch_size"], shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=True, pin_memory=True)


class Linear(nn.Module):
    def __init__(self, input_dim):
        super(My_model, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
    
    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(1)
        return x

model = Linear(input_dim = X_train.shape[1]).to(device)
trainer(train_loader, valid_loader, model, config)


model = Linear( X_train.shape[1]).to(device)
model.load_state_dict(torch.load(config["save_path"]))
preds = predict(test_loader, model, device)
plt.plot(np.arange(len(preds)), preds, label="preds")
plt.plot(np.arange(len(preds)), y_test, label="y_test")

