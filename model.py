import torch
from torch import nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Net(nn.Module):
    def __init__(self, hidden=64, dropout=0.1):
        super(Net, self).__init__()
        self.hidden = hidden
        self.lstm = nn.LSTMCell(1, hidden)
        self.lstm2 = nn.LSTMCell(hidden, hidden)
        self.lstm3 = nn.LSTMCell(hidden, hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden, hidden//2)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(hidden//2, 1)

    def forward(self, x, future=0):
        outputs = []
        N = x.size(0)
        h_t = torch.zeros(N, self.hidden).to(device)
        c_t = torch.zeros(N, self.hidden).to(device)
        h_t2 = torch.zeros(N, self.hidden).to(device)
        c_t2 = torch.zeros(N, self.hidden).to(device)
        h_t3 = torch.zeros(N, self.hidden).to(device)
        c_t3 = torch.zeros(N, self.hidden).to(device)

        for i in x.split(1, dim=1):
            h_t, c_t = self.lstm(i, (h_t, c_t))
            output = self.dropout(h_t)
            h_t2, c_t2 = self.lstm2(output, (h_t2, c_t2))
            output = self.dropout(h_t2)
            h_t3, c_t3 = self.lstm3(output, (h_t3, c_t3))
            output = self.dropout(h_t3)
            output = self.fc1(output)
            output = self.leakyrelu(output)
            output = self.fc2(output)
            outputs.append(output)

        for i in range(future):
            h_t, c_t = self.lstm(output, (h_t, c_t))
            output = self.dropout(h_t)
            h_t2, c_t2 = self.lstm2(output, (h_t2, c_t2))
            output = self.dropout(h_t2)
            h_t3, c_t3 = self.lstm3(output, (h_t3, c_t3))
            output = self.dropout(h_t3)
            output = self.fc1(output)
            output = self.leakyrelu(output)
            output = self.fc2(output)
            outputs.append(output)

        outputs = torch.cat(outputs, dim=1)
        return outputs

