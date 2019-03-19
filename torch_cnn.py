import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.utils.data as Data


torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 1           # 训练整批数据多少次, 为了节约时间, 我们只训练一次
BATCH_SIZE = 100
LR = 0.005          # 学习率
DOWNLOAD_MNIST = False

train_data = torchvision.datasets.MNIST(
    'mnist', train=True, download=DOWNLOAD_MNIST, transform=torchvision.transforms.ToTensor())

test_data = torchvision.datasets.MNIST(
    'mnist', train=False, download=DOWNLOAD_MNIST)

train_loader = Data.DataLoader(
    dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[
    :2000] / 255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_y = test_data.test_labels[:2000]


class CNN(nn.Module):
    """docstring for CNN"""

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))

        self.conv2 = nn.Sequential(  # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),  # output shape (32, 14, 14)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (32, 7, 7)
        )

        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        return self.out(x)

cnn = None

try:
	cnn = torch.load('torch_cnn.pkl')
except FileNotFoundError as e:
	print('we need to train it')
	cnn = CNN()
	optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
	loss_func = nn.CrossEntropyLoss()

	for epoch in range(EPOCH):
		for step, (batch_x,batch_y) in enumerate(train_loader):
			print('Epoch: ', epoch, '| Step: ', step)
			y_pred = cnn(batch_x)
			optimizer.zero_grad()
			loss = loss_func(y_pred, batch_y)
			loss.backward()
			optimizer.step()

	torch.save(cnn,'torch_cnn.pkl')


def print_accuracy_score(y_pred, y):
	prob, index = torch.max(y_pred,dim=1)
	sum = torch.sum(index == y).type(torch.float32)
	accuracy = torch.div(sum,y.size()[0]).data.numpy()
	print('score is ', accuracy)
	return accuracy

print(cnn)
# print('test shape', test_y.size())
# test_output = cnn(test_x[:10])
# prob, index = torch.max(test_output,dim=1)
# print('pred number',index.numpy())
# print('real number',test_y[:10].data.numpy()) 
test_out = cnn(test_x)
print_accuracy_score(test_out, test_y)