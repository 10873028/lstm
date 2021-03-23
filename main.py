import torch
from torch import nn
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from preprocessing import StandardScaler, CreateMatrix, Split
from model import Net
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

timesteps = 90
batch_size = 512
epochs = 100
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
print(x_train.shape, y_train.shape, x_val.shape, y_val.shape, x_test.shape, y_test.shape)

#%%
x_train = torch.FloatTensor(x_train).to(device)
y_train = torch.FloatTensor(y_train).to(device)
x_val = torch.FloatTensor(x_val).to(device)
y_val = torch.FloatTensor(y_val).to(device)
x_test = torch.FloatTensor(x_test).to(device)

dataset = Data.TensorDataset(x_train, y_train)
dataloader = Data.DataLoader(dataset=dataset, batch_size=batch_size)

model = Net().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

history = {'loss': [], 'val_loss': []}
best_model = np.inf

for epoch in range(epochs):
    for x_batch, y_batch in dataloader:
        model.train()
        y_pred = model(x_batch)
        optimizer.zero_grad()
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        y_pred = model(x_val)
        val_loss = criterion(y_pred, y_val)
        print(f'epoch:{epoch:02d}-loss:{loss.item():.7f}-val_loss:{val_loss.item():.7f}')
        history['loss'].append(loss.item())
        history['val_loss'].append(val_loss.item())
        if val_loss.item() <= best_model:
            best_model = val_loss.item()
            torch.save(model, 'checkpoint.pt')
            print('weight saved')

model = torch.load('checkpoint.pt')
model.eval()
with torch.no_grad():
    pred = model(x_test)
    pred = pred.cpu().numpy()
    pred = scaler.inverse_transform(pred)
plt.plot(y_test, label='actual')
plt.plot(pred, label='predict')
plt.legend()
plt.show()
plt.plot(history['loss'], label='loss')
plt.plot(history['val_loss'], label='val_loss')
plt.legend()
plt.show()
print(metrics.mean_absolute_error(pred, y_test))

#%%
start = 0
future = 749
tmp = x_test[start].unsqueeze(dim=0)
model.eval()
with torch.no_grad():
    pred = model(tmp, future=future).cpu().numpy().reshape(-1, 1)
    pred = scaler.inverse_transform(pred)
plt.plot(y_test)
plt.plot(pred)
plt.show()
print(metrics.mean_absolute_error(y_test, pred))