import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from prettytable import PrettyTable
import numpy as np

'''
BIG NOTE: данный код написал deepseek потому что за pytorch мы не шарим
          в идеале его подредачить, когда будем собирать данные       
'''

df = pd.read_csv("data/Student_Performance.csv", header=None)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#Magic from deepseek 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_t = torch.FloatTensor(X_train)
X_test_t = torch.FloatTensor(X_test)
y_train_t = torch.FloatTensor(y_train)
y_test_t = torch.FloatTensor(y_test)

class RegressionNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

input_size = X_train.shape[1]
loss_f = nn.MSELoss() #(!!!) функция потерь 

#NOTE: у Momentum и Nestero сейчас другие параметры потому что с 0.01 они не работают
optimizers = {
    "SGD": optim.SGD,
    "Momentum": lambda params: optim.SGD(params, lr=0.001, momentum=0.9), 
    "Nesterov": lambda params: optim.SGD(params, lr=0.001, momentum=0.9, nesterov=True),
    "AdaGrad": lambda params: optim.Adagrad(params, lr=0.01),
    "RMSProp": lambda params: optim.RMSprop(params, lr=0.01, alpha=0.99),
    "Adam": lambda params: optim.Adam(params, lr=0.01, betas=(0.9, 0.999))
}

results_table = PrettyTable()
results_table.field_names = ["Optimizer", "Test Loss", "R² Score"]
results_table.float_format = ".4"

for opt_name, opt_func in optimizers.items():
    model = RegressionNN(input_size)
    optimizer = opt_func(model.parameters())
    
    #Само обучение 
    #----
    epochs = 100
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = loss_f(outputs, y_train_t)
        loss.backward()
        optimizer.step()
    #----
    
    with torch.no_grad():
        predictions = model(X_test_t)
        test_loss = loss_f(predictions, y_test_t).item()
        
        y_true = y_test_t.numpy().flatten()
        y_pred = predictions.numpy().flatten()
        r2 = r2_score(y_true, y_pred)
        
        results_table.add_row([opt_name, test_loss, r2])
        
        print(f"\nOptimizer: {opt_name}")
        print(f"Test Loss: {test_loss:.6f}")
        print(f"R^2 score: {r2}")
        print("="*50)

print("\nSummary Results:")
print(results_table)