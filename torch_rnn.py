import torch
import matplotlib.pyplot as plt
from torch import nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.utils.data as Data

EPOCH = 5           # 训练整批数据多少次, 为了节约时间, 我们只训练一次
BATCH_SIZE = 64
TIME_STEP = 28      # rnn 时间步数 / 图片高度
INPUT_SIZE = 28     # rnn 每步输入值 / 图片每行像素
LR = 0.01           # learning rate
DOWNLOAD_MNIST = False  #

train_data = dsets.MNIST(
    'mnist', train=True, download=DOWNLOAD_MNIST, transform=transforms.ToTensor())

test_data = dsets.MNIST(
    'mnist', train=False, download=DOWNLOAD_MNIST)

train_loader = Data.DataLoader(
    dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[
    :2000] / 255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_y = test_data.test_labels[:2000]


class RNN(nn.Module):
    """docstring for RNN"""

    def __init__(self, *args):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(input_size=28,
                           hidden_size=64,     # rnn hidden unit
                           num_layers=2,       # 有几层 RNN layers
                           batch_first=True)
        self.out = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)   LSTM 有两个 hidden states, h_n 是分线, h_c 是主线
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)   # None 表示 hidden state 会用全0的
        out = self.out(r_out[:, -1, :])
        return out


rnn = None

try:
    rnn = torch.load('torch_rnn.pkl')
except Exception as e:
    rnn = RNN()
    optimizer = torch.optim.Adam(
        rnn.parameters(), lr=LR)   # optimize all parameters
    loss_func = nn.CrossEntropyLoss()   # the target label is not one-hotted
    # training and testing
    for epoch in range(EPOCH):
        for step, (x, b_y) in enumerate(train_loader):   # gives batch data
            # reshape x to (batch, time_step, input_size)
            b_x = x.view(-1, 28, 28)

            output = rnn(b_x)               # rnn output
            loss = loss_func(output, b_y)   # cross entropy loss
            optimizer.zero_grad()           # clear gradients for this training step
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients

    torch.save(rnn, 'torch_rnn.pkl')
else:
    pass
finally:
    pass


def print_accuracy_score(y_pred, y):
    prob, index = torch.max(y_pred, dim=1)
    sum = torch.sum(index == y).type(torch.float32)
    accuracy = torch.div(sum, y.size()[0]).data.numpy()
    print('score is ', accuracy)
    return accuracy

print(rnn)


test_output = rnn(test_x[:2000].view(-1, 28, 28))
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'prediction number')
print(test_y[:2000], 'real number')

print_accuracy_score(test_output, test_y[:2000])


train_x = torch.unsqueeze(train_data.train_data[:2000], dim=1).type(
    torch.FloatTensor) / 255.

train_output = rnn(train_x.view(-1, 28, 28))

print_accuracy_score(train_output, train_data.train_labels[:2000])
