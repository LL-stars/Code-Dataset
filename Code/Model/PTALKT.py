import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from Constant import Constant as C

class SimpleAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SimpleAttention, self).__init__()
        self.W = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, hidden_size*2)
        energy = torch.tanh(self.W(x))
        attention_weights = F.softmax(energy, dim=1)
        output = attention_weights * x
        output = F.relu6(output)
        return output

class PTALKT(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(PTALKT, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim
        self.lstm_net = nn.LSTM(1, hidden_dim, layer_dim, batch_first=True)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.at = SimpleAttention(hidden_dim)
        self.att = nn.MultiheadAttention(embed_dim=hidden_dim*2, num_heads=1)
        self.drop = nn.Dropout(p=0.1)
        self.fc1 = nn.Linear(self.hidden_dim*2, self.output_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.output_dim)
        self.fc4 = nn.Linear(self.output_dim, self.output_dim)
        self.sig = nn.Tanh()
        self.sig1 = nn.Sigmoid()

    def forward(self, x, n):
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
        c0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
        statein = (h0,c0)
        out0, _ = self.lstm(x, statein)

        h1 = Variable(torch.zeros(self.layer_dim, n.size(0), self.hidden_dim))
        c1 = Variable(torch.zeros(self.layer_dim, n.size(0), self.hidden_dim))
        out_net, _ = self.lstm_net(n)

        attention_output1 = self.at(out0)
        attention_output2 = self.at(out_net)
        combined_output = attention_output1 + attention_output2
        attention_output = self.fc2(combined_output)
        attention_output = self.fc3(attention_output)
        y = self.sig(torch.abs(attention_output))
        y = torch.abs(y)
        return y
