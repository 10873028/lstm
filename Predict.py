import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from preprocessing import StandardScaler, CreateMatrix, Split
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

timesteps = 90
future = int(input('future'))
data = pd.read_csv('bdi.csv').fillna(method='ffill')['bdi'].values
errors = 0
x, y = CreateMatrix(data, timesteps)
x_train, y_train, x_val, y_val, x_test, y_test = Split(x, y)
y_test = y_test[:, -1:]
scaler = StandardScaler(y_train)
x_test = scaler.transform(x_test)
inputs = np.array([x_test[start] for start in range(749-future)])
inputs = torch.FloatTensor(inputs).to(device)
model = torch.load(f'checkpoint.pt')
model.eval()
with torch.no_grad():
    pred = model(inputs, future=future)[:, -1-future:]
    pred = scaler.inverse_transform(pred.cpu().numpy())

errors = 0
plt.plot(y_test)
for start in range(749-future):
    plt.plot(range(start, start+future+1), pred[start], ':')
    error = metrics.mean_absolute_error(y_test[start:start+future+1], pred[start])
    errors += error
    print(error)
plt.title(errors/(749 - future))
plt.show()

