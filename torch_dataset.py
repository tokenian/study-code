import torch
import torch.utils.data as Data

torch.manual_seed(1)    # reproducible

BATCH_SIZE = 5      # 批训练的数据个数

x = torch.linspace(1, 10, 10)       # x data (torch tensor)
y = torch.linspace(10, 1, 10)

torch_dataset = Data.TensorDataset(x, y)

loader = Data.DataLoader(
    torch_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
if __name__ == '__main__':
	for epoch in range(3):   # 训练所有!整套!数据 3 次
		# 每一步 loader 释放一小批数据用来学习
	    for step, (batch_x, batch_y) in enumerate(loader):
	        # 假设这里就是你训练的地方...

	        # 打出来一些数据
	        print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
	              batch_x.numpy(), '| batch y: ', batch_y.numpy())

# torch.optim.Adam