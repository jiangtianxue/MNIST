import numpy as np
import torch
from torchvision.datasets import mnist # 导入 pytorch 内置的 mnist 数据
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os


# 使用内置函数下载 mnist 数据集
train_set = mnist.MNIST('./data', train=True, download=True)
test_set = mnist.MNIST('./data', train=False, download=True)

# 对数据进行处理
def data_tf(x):
	# 标准化
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5 
    # 拉平，变成一行
    x = x.reshape((-1,)) 
    # 将ndarray转化为pytorch中的tensor
    x = torch.from_numpy(x)
    return x


# 对产生的loss值和accuracy值进行保存    
def text_write(filename, data):
	file = open(filename, 'a')
	for i in range(len(data)):
		s = str(data[i]) + ' '
		file.writelines(s)
	file.write('\n')
	file.close()


# 重新载入数据集，申明定义的数据变换
train_set = mnist.MNIST('./data', train=True, transform=data_tf, download=True) 
test_set = mnist.MNIST('./data', train=False, transform=data_tf, download=True)
# 使用DataLoader定义数据迭代器
train_data = DataLoader(train_set, batch_size=64, shuffle=True)
test_data = DataLoader(test_set, batch_size=128, shuffle=False)


# 使用sequential定义4层神经网络，就是简单的BP
net = nn.Sequential(
	nn.Linear(784, 400),
	nn.ReLU(),
	nn.Linear(400, 200),
	nn.ReLU(),
	nn.Linear(200, 100),
	nn.ReLU(),
	nn.Linear(100, 10),	
	)


# 定义 loss 函数
criterion = nn.CrossEntropyLoss()
# 使用随机梯度下降，学习率 0.1
optimizer = torch.optim.SGD(net.parameters(), lr=1e-1) 
# optimizer = nn.DataParallel(optimizer, device_id=device_id)


# 开始训练
losses = []
acces = []
eval_losses = []
eval_acces = []

for e in range(20):
    train_loss = 0
    train_acc = 0
    net.train()
    for im, label in train_data:
    	# 只有变成Variable才会被求梯度，因为只有定义的Variable
    	# 的梯度放在.grad属性中
        im = Variable(im)
        label = Variable(label)
        # 前向传播
        out = net(im)
        # 在58行定义的交叉熵函数
        loss = criterion(out, label)
        print(loss)
        # 反向传播，梯度初始化为0
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 记录误差
        train_loss += loss.item()
        # 计算分类的准确率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / im.shape[0]
        train_acc += acc
        
    losses.append(train_loss / len(train_data))
    acces.append(train_acc / len(train_data))
    # 在测试集上检验效果
    eval_loss = 0
    eval_acc = 0
    net.eval() # 将模型改为预测模式
    for im, label in test_data:
        im = Variable(im)
        label = Variable(label)
        out = net(im)
        loss = criterion(out, label)
        # 记录误差
        eval_loss += loss.item()
        # 记录准确率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / im.shape[0]
        eval_acc += acc
        
    eval_losses.append(eval_loss / len(test_data))
    eval_acces.append(eval_acc / len(test_data))
    print('epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f}'
          .format(e+1, train_loss / len(train_data), train_acc / len(train_data), 
                     eval_loss / len(test_data), eval_acc / len(test_data)))


params = list(net.parameters())
k = 0
for i in params:
	l = 1
	print('该层结构：' + str(list(i.size())))
	for j in i.size():
		l *= j
	print('该层参数和：' + str(l))
	k = k + l
print('总参数和：' + str(k))


if os.path.exists('loss_accus.txt'):
	os.remove('loss_accus.txt')
text_write('loss_accus.txt', losses)
text_write('loss_accus.txt', acces)
text_write('loss_accus.txt', eval_losses)
text_write('loss_accus.txt', eval_acces)


