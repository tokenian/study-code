#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-12-27
# @Author  : tangxc (tangxc1987@mail.126.com)
# @Link    : http://arxiv.org
# @Version : 0.1


# import pandas as pd

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Sequential, Linear, Softmax,ReLU
import matplotlib.pyplot as plt
import numpy as np

n_data = torch.ones(100, 2)
x0 = torch.normal(2 * n_data, 1)
y0 = torch.zeros(100)               # class0 y data (tensor), shape=(100, 1)
x1 = torch.normal(-2 * n_data, 1)     # class1 x data (tensor), shape=(100, 2)
y1 = torch.ones(100)                # class1 y data (tensor), shape=(100, 1)
# shape (200, 2) FloatTensor = 32-bit floating
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
# shape (200,) LongTensor = 64-bit integer
y = torch.cat((y0, y1), ).type(torch.LongTensor)

# plt.figure(figsize=(10,8))
# plt.scatter(x.numpy()[:,0],x.numpy()[:,1],c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
# plt.show()
# print(type(x))
x = Variable(x)
y = Variable(y)


class Net(torch.nn.Module):
    """docstring for Net"""

    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        # x = F.sigmoid(self.predict(x))
        x = self.predict(x)
        return x


# net = Net(2, 10, 2)
net = Sequential(Linear(2,10),ReLU(),Linear(10,2))
print(net)
print(net.state_dict())

optimizer = torch.optim.SGD(net.parameters(), lr=0.05)
loss_func = torch.nn.CrossEntropyLoss()

plt.ion()

for i in range(100):
    out = net(x)
    loss = loss_func(out, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 2 == 0:
        plt.cla()       # plot and show learning process
        prediction = torch.max(F.softmax(out, dim=1), 1)[1]
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)


plt.ioff()
plt.show()

torch.save(net, 'torch_classify.pkl')
torch.save(net.state_dict(),'torch_classify_paras.pkl') # save network parameters

# load trained networks from file
# net = torch.load("torch_classify.pkl")

#load paras from file ,then kick it into networks
# net.load_state_dict(torch.load("torch_classify_paras.pkl"))
# net(x)

