


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

#from sklearn.preprocessing import MinMaxScaler

import yfinance as yf

from AnalysingTool import *  




dst = DataSetTool()





TDR1 = yf.Ticker("AAPL").history(start = "1985-01-01", end = "2022-01-01")
TDR2 = yf.Ticker("MSFT").history(period="max")
TDR3 = yf.Ticker("GOOG").history(period="max")
TDR4 = yf.Ticker("AMZN").history(period="max")
TDR5 = yf.Ticker("IBM").history(period="max")
TDR6 = yf.Ticker("TSLA").history(period="max")
TDR7 = yf.Ticker("NVDA").history(period="max")
TDR8 = yf.Ticker("PYPL").history(period="max")
TDR9 = yf.Ticker("ADBE").history(period="max")
TDR10 = yf.Ticker("INTC").history(period="max")
dataset1 = dst.sequence_processing(TDR1, 5)
dataset2 = dst.sequence_processing(TDR2, 5)
dataset3 = dst.sequence_processing(TDR3, 5)
dataset4 = dst.sequence_processing(TDR4, 5)
dataset5 = dst.sequence_processing(TDR5, 5)
dataset6 = dst.sequence_processing(TDR6, 5)
dataset7 = dst.sequence_processing(TDR7, 5)
dataset8 = dst.sequence_processing(TDR8, 5)
dataset9 = dst.sequence_processing(TDR9, 5)
dataset10 = dst.sequence_processing(TDR10, 5)
#add the firm name to the dataset
#import sklearn onehotencoder
#use encoder to encode the firm name




train_data = pd.concat((dataset1, dataset2, dataset3, dataset4, dataset5, dataset6, dataset7, dataset8, dataset9, dataset10), axis = 0)

test_data_raw = dst.yf_data_gripper("AAPL", "2022-12-31", None)
#add the firm name to the dataset
#test_data_raw["firm"] = [1,0,0,0,0,0,0,0,0,0]
test_data = dst.sequence_processing(test_data_raw, 5)
print(train_data.shape)

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


X_train, y_train, X_valid, y_valid, X_test, y_test = dst.data_split(train_data,test_data, config)


from sklearn.metrics import mean_squared_error as mse

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(max_features="sqrt", n_estimators=1000, n_jobs=-1, oob_score=True)
model.fit(X_train, y_train)
preds = model.predict(X_test)
plt.plot(np.arange(len(preds)), preds, label="preds")
plt.plot(np.arange(len(preds)), y_test, label="y_test")
plt.legend()
#In []



mse(preds, y_test)



from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.01)
model.fit(X_train, y_train)
preds = model.predict(X_test)
plt.plot(np.arange(len(preds)), preds, label="preds")
plt.plot(np.arange(len(preds)), y_test, label="y_test")
plt.legend()


print("randomforest mse: ",mse(preds, y_test))



from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)
plt.plot(np.arange(len(preds)), preds, label="preds")
plt.plot(np.arange(len(preds)), y_test, label="y_test")
_ = plt.legend()





mse(preds, y_test)





from sklearn.ensemble import AdaBoostRegressor
model = AdaBoostRegressor(n_estimators=1000, learning_rate=0.1)
model.fit(X_train, y_train)
preds = model.predict(X_test)
plt.plot(np.arange(len(preds)), preds, label="preds")
plt.plot(np.arange(len(preds)), y_test, label="y_test")
plt.legend()

print("AdaBoost mse: ",mse(preds, y_test))







