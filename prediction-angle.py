import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler


# 超参数
input_size = 3
output_size = 1
n_hidden = 3

num_epochs = 12000
learning_rate = 0.01

# 数据集
data = pd.read_csv("data/balancer_1.csv")
train_data = data.values.astype(np.float32)
X = train_data[:,0:3]
y = train_data[:,3]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)

# 数据标准化
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# y_train = np.expand_dims(y_train, axis=1)
# y_test = np.expand_dims(y_test, axis=1)

# 线性回归模型
class LinearRegression(nn.Module):

    def __init__(self, input_size, utput_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.linear(x)
        return out

# 线性回归模型 添加中间层
class TrainNet(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        """定义每层用的形式
        Args:
            n_feature：输入特征
            n_hidden：隐藏层
            n_output：输出层
        """
        super(TrainNet, self).__init__()
        self.hidden = nn.Linear(n_feature, n_hidden)   # 隐藏层
        self.predict = nn.Linear(n_hidden, n_output)   # 输出层

    def forward(self, x):
        """"""
        # x = F.relu(self.hidden(x))      # 隐藏层激活函数
        x = torch.tanh(self.hidden(x))  #
        return self.predict(x)             # 线性输出



# model = LinearRegression(input_size, output_size)
model = TrainNet(input_size, n_hidden, output_size)

# 损失优化
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

def train(X_train, y_train):
    inputs = torch.from_numpy(X_train).float()
    targets = torch.from_numpy(y_train).float()

    optimizer.zero_grad()
    outputs = model(inputs)

    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    return loss.item()

def valid(X_test, y_test):
    inputs = torch.from_numpy(X_test).float()
    targets = torch.from_numpy(y_test).float()

    outputs = model(inputs)
    val_loss = criterion(outputs, targets)

    return val_loss.item()

# 训练模型
loss_list = []
val_loss_list = []
for epoch in range(num_epochs):
    # data shuffle
    perm = np.arange(X_train.shape[0])
    np.random.shuffle(perm)
    X_train = X_train[perm]
    y_train = y_train[perm]

    loss = train(X_train, y_train)
    val_loss = valid(X_test, y_test)

    if epoch % 200 == 0:
        print('epoch %d, loss: %.2f val_loss: %.2f' % (epoch, loss, val_loss))

    loss_list.append(loss)
    val_loss_list.append(val_loss)

# 绘制曲线图
plt.plot(range(num_epochs), loss_list, 'r-', label='train_loss')
plt.plot(range(num_epochs), val_loss_list, 'b-', label='val_loss')
plt.legend()
plt.show()