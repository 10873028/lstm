import torch
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from preprocessing import StandardScaler, CreateMatrix, Split
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

timesteps = 90
start = 0
future = 89

df = pd.read_csv('bdi.csv').fillna(method='ffill')
data = df['bdi'].values
x, y = CreateMatrix(data, timesteps)
x_train, y_train, x_val, y_val, x_test, y_test = Split(x, y)

scaler = StandardScaler(y_train)
x_train = scaler.transform(x_train)
y_train = scaler.transform(y_train).reshape(-1, 1)
x_val = scaler.transform(x_val)
y_val = scaler.transform(y_val).reshape(-1, 1)
x_test = scaler.transform(x_test)
y_test = y_test.reshape(-1, 1)
x_test = torch.FloatTensor(x_test).to(device)

model = torch.load('checkpoint.pt')
tmp = x_test[start].unsqueeze(dim=0)
model.eval()
with torch.no_grad():
    pred = model(tmp, future=future).cpu().numpy().reshape(-1, 1)
    pred = scaler.inverse_transform(pred)
plt.plot(y_test)
plt.plot(pred)
plt.show()
print(metrics.mean_absolute_error(y_test[start:(start+future+1)], pred))
