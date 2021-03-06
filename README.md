
## 采样设备

- 砂轮重量：896克
- 平衡仪传感器：3个HX711压力传感器，间隔120度，均匀分布圆周
- 支撑点圆直径：366mm，上面加支架，支撑位采用钢珠


## 数据源：

取一片砂轮，手动置于平衡仪上：

1. 读取3个传感器数据记为：sensor_0,sensor_1,sensor_2；
2. 估算对应角度记为：angle

拿起砂轮旋转约20度再次放置于平衡仪上，同上取数，重复若干次，得到数据集

数据集如下：

| sensor_0 | sensor_1 | sensor_2 | angle  |
| -------- | -------- | -------- | ------ |
| 1430802  | 2087086  | 1551057  | 0      |
| 1455911  | 2138408  | 1556609  | 20     |
| 1525408  | 2350695  | 1641848  | 40     |
| 1450539  | 2115891  | 1577330  | 60     |
| 1462667  | 2162498  | 1531984  | 80     |
| ......   | ......   | ......   | ...... |


## 需求

通过上述数据集，搭建机器学习模型，预测角度


## 模型

采用`pytorch`搭建线性回归模型模型，前三列为特征，最后一列为标记。

测试不同的隐藏层、激活函数、学习率、训练集、梯度下降算法


- 训练集：2/3
- 测试集：1/3
- 训练模型：线性回归
- 输入特征：3
- 隐藏层：3
- 输出层：1
- 损失函数：nn.MSELoss()(均方差)
- 梯度下降：Adam
- 激活函数：torch.tanh()



## 精度

![](https://img2018.cnblogs.com/blog/720033/201901/720033-20190125182222980-570181807.png)


## 结论

误差还是很大，原因

1. 数据量少,脏
2. 角度读数偏差大
3. 优化模型不够优秀

10000步已经收敛,趋势很好。

## 下一步：

1. 更换干净数据集
2. 优化模型