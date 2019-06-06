import torch
import torch.nn.functional as f
from torch.autograd import Variable
import matplotlib.pyplot as plt




loss_fn = torch.nn.CrossEntropyLoss()
input = Variable(torch.randn(1, 2)) # (batch_size, C)
target = Variable(torch.LongTensor(1).random_(2))
print('0', input)
print('1', target)
loss = loss_fn(input, target)
print(input); print(target); print(loss)

print('------------------------------')


# 建造数据集
data = torch.ones((100, 2))
x0 = torch.normal(2 * data, 1)
y0 = torch.zeros(100)  # y0是标签  shape(100,),是一维
x1 = torch.normal(-2 * data, 1)
y1 = torch.ones(100)  # y1也是标签 shape(100,)，是一维
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # 参数0表示维度，在纵向方向将x0,x1合并，合并后shape(200, 2）)
y = torch.cat((y0, y1), 0).type(torch.LongTensor)  # 标签是0或1，类型为整数，LongTensor = 64-bit integer,
x, y = Variable(x), Variable(y)  # 训练神经网络只能接受变量输入，故要把x, y转化为变量



# 建造神经网络模型
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = f.relu(self.hidden(x))
        y = self.out(x)
        return y


# 定义神经网络
net = Net(n_feature=2, n_hidden=10, n_output=2)
# n_output=2,因为它返回一个元素为2的列表。[0, 1]表示学习到的内容为标签1，[1, 0]表示学习到的内容为标签0。
print(net)

# 训练神经网络模型并将训练过程可视化
optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
loss_func = torch.nn.CrossEntropyLoss()
plt.ion()
for i in range(1):
    out = net(x)
    print(out)
    print(y)
    print(len(out))
    print(len(y))
    loss = loss_func(out, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
