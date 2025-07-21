#导入相关库
import torch                         # 导入PyTorch库
import torch.nn as nn                # 导入神经网络模块
import torch.optim as optim          # 导入优化器模块
import numpy as np                   # 导入NumPy库
import pandas as pd
import tushare as ts                 # 导入tushare库，用于获取股票数据
from tqdm import tqdm                # 导入tqdm库，用于显示进度条
import matplotlib.pyplot as plt      # 导入matplotlib库，用于绘制图表
from copy import deepcopy as copy    # 导入deepcopy函数，用于深拷贝对象
from torch.utils.data import DataLoader, TensorDataset  # 导入DataLoader和TensorDataset类，用于加载数据
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class GetData:
    def __init__(self, stock_id, save_path):
        self.stock_id = stock_id
        self.save_path = save_path
        self.data = None

    def getData(self):
        should_be_save = True
        # 判断data.csv文件是否存在
        if os.path.exists(self.save_path):
            df = pd.read_csv(self.save_path)
            if not df.empty:  # 判断文件是否有数据
                self.data = df
                should_be_save = False
            else:
                import tushare as ts
                self.data = ts.get_hist_data(self.stock_id).iloc[:-1]
        else:
            import tushare as ts
            self.data = ts.get_hist_data(self.stock_id).iloc[:-1]

        self.data = self.data[["open", "close", "high", "low", "volume"]]
        self.close_min = self.data["volume"].min()
        self.close_max = self.data["volume"].max()
        self.data = self.data.apply(lambda x: (x - min(x)) / (max(x) - min(x)))
        if should_be_save:
            self.data.to_csv(self.save_path, index=False)
        return self.data

    def process_data(self, n):
        """
        将数据分为特征(feature)和标签(label),
        并划分为训练集和测试集
        """
        if self.data is None:
            self.getData()
        feature = [self.data.iloc[i: i + n].values.tolist()
                   for i in range(len(self.data) - n + 2)
                   if i + n < len(self.data)]
        label = [
            self.data.close.values[i + n]
            for i in range(len(self.data) - n + 2)
            if i + n < len(self.data)
        ]
        train_x = feature[:500]
        test_x = feature[500:]
        train_y = label[:500]
        test_y = label[500:]
        return train_x, test_x, train_y, test_y


class Model(nn.Module):
    """
    定义了一个神经网络模块
    包含一个LSTM层和一个线性层
    """
    def __init__(self, n):
        """
        @param n: 输入序列中每个实践步的特征维度
        """
        super(Model, self).__init__()
        # hidden_size LSTM单元中隐藏状态h和细胞状态c的维度
        self.lstm_layer = nn.LSTM(input_size=n, hidden_size=128, batch_first=True)
        # out_features 模型最终输出的维度
        self.linear_layer = nn.Linear(in_features=128, out_features=1, bias=True)

    def forward(self, x):
        """
        前向传播方法中，通过LSTM层处理输入x，得到输出out1和隐藏状态h_n、h_c
        然后通过线性层处理h_n得到最终输出out2
        """
        out, (h_n, h_c) = self.lstm_layer(x)
        out = self.linear_layer(out[:, -1, :])
        return out
    
# 定义一个名为 Model 的神经网络模块
# class Model(nn.Module):
#     def __init__(self, n):  # 初始化方法，接收一个参数 n
#         super(Model, self).__init__()  # 调用父类的初始化方法
#         # 创建一个 LSTM 层，输入大小为 n，隐藏大小为 256，批次优先为 True
#         self.lstm_layer = nn.LSTM(input_size=n, hidden_size=256,
#                                   batch_first=True)

#         # 创建一个线性层，输入特征数为 256，输出特征数为 1，有偏差
#         self.linear_layer = nn.Linear(in_features=256, out_features=1, bias=True)

#     def forward(self, x):  # 前向传播方法，接收一个输入 x
#         # 通过 LSTM 层处理 x，得到输出 out1 和隐藏状态 h_n、h_c
#         out1, (h_n, h_c) = self.lstm_layer(x)
#         a, b, c = h_n.shape  # 获取 h_n 的形状信息
#         # 将 h_n 重塑为(a*b, c)的形状后，通过线性层处理，得到输出 out2
#         out2 = self.linear_layer(h_n.reshape(a * b, c))
#         return out2  # 返回最终的输出 out2
    

def train_model(epoch, train_dataLoader, test_dataLoader):
    """
    @param epoch: 训练轮数
    @param train_dataLoader: 训练数据加载器
    @param test_dataLoader:  测试数据加载器
    """
    # 最佳模型
    best_model = None
    # 训练损失
    train_loss = 0
    # 测试损失
    test_loss = 0
    # 最佳损失
    best_loss = 100
    # 论述计数器
    epoch_cnt = 0
    for _ in range(epoch):
        # 训练总损失
        total_train_loss = 0
        # 训练样本总数
        total_train_num = 0
        # 测试总损失
        total_test_loss = 0
        # 测试样本总数
        total_test_num = 0
        
        for x, y in tqdm(train_dataLoader,desc='Epoch: {}| Train Loss: {}| Test Loss: {}'.format(_, train_loss, test_loss)):
            # x的数量
            x_num = len(x)
            # 模型预测结果
            p = model(x)
            # print(len(p[0]))
            # 计算损失
            loss = loss_func(p, y)
            # 梯度清零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            # 累计训练损失
            total_train_loss += loss.item()
            # 累计训练样本数
            total_train_num += x_num
        
        # 计算平均训练损失
        train_loss = total_train_loss / total_train_num
        
        for x, y in test_dataLoader:
            # x 的数量
            x_num = len(x)
            # 模型预测
            p = model(x)
            # 计算损失
            loss = loss_func(p, y)
            # 测试集不应该更新参数
            # 验证结果失真
            # 过拟合
            # 梯度清零
            # optimizer.zero_grad()
            # # 反向传播
            # loss.backward()
            # # 更新参数
            # optimizer.step()
            # 累计测试损失
            total_test_loss += loss.item()
            # 累计测试样本
            total_test_num += x_num
        
        # 计算平均测试损失
        test_loss = total_test_loss / total_test_num
        
        # 如果当前测试损失小于最佳损失
        if best_loss > test_loss:
            # 更新最佳损失
            best_loss = test_loss
            # 赋值当前模型
            best_model = copy(model)
            # 轮数计数器清零
            epoch_cnt = 0
        else:
            # 轮数计数器加一
            epoch_cnt += 1

        # 如果轮数计数器大于提前停止的轮数
        if epoch_cnt > early_stop:
            # 保存最佳模型的状态字典
            torch.save(best_model.state_dict(), './lstm_.pth')
            # 中断训练
            break

def test_model(test_dataLoader_):
    # 预测值列表
    pred = []
    # 真实值列表
    label = []
    model_ = Model(5)
    # 加载模型的状态字典
    model_.load_state_dict(torch.load("./lstm_.pth"))
    # 将模型设置为评估模式
    model_.eval()
    # 测试总损失
    total_test_loss = 0
    # 测试样本总数
    total_test_num = 0
    for x, y in test_dataLoader_:
        x_num = len(x)
        p = model_(x)
        loss = loss_func(p, y)
        total_test_loss += loss.item()
        total_test_num += x_num
        # 将预测值添加到列表中
        pred.extend(p.data.squeeze(1).tolist())
        # 将真实标签添加到列表中
        label.extend(y.tolist())

    # 计算平均测试损失
    test_loss = total_test_loss / total_test_num

    return pred, label, test_loss

def plot_img(data, pred):
    """
    @param data: 真实值
    @param pred: 预测值
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(12, 7))
    plt.plot(range(len(pred)), pred, color='green')
    plt.plot(range(len(data)), data, color='blue')
    for i in range(0, len(pred)-3, 5):
        price = [data[i]+pred[j]-pred[i] for j in range(i, i+3)]
        plt.plot(range(i, i+3), price, color='red')
    plt.xticks(fontproperties = 'Times New Roman', size = 15)
    plt.yticks(fontproperties = 'Times New Roman', size = 15)
    plt.xlabel('日期', fontsize=18)
    plt.ylabel('收盘价', fontsize=18)
    plt.show()

if __name__ == '__main__':
    #超参数
    days_num = 5
    epoch = 20
    fea = 5
    batch_size = 20
    early_stop = 5
 
    #初始化模型
    model = Model(fea)
 
    #数据处理
    GD = GetData(stock_id='601398', save_path='ch05\data.csv')
    x_train, x_test, y_train, y_test = GD.process_data(days_num)
    x_train = torch.tensor(x_train).float()
    x_test = torch.tensor(x_test).float()
    y_train = torch.tensor(y_train).float()
    y_test = torch.tensor(y_test).float()
    train_data = TensorDataset(x_train, y_train)
    train_dataLoader = DataLoader(train_data, batch_size=batch_size)
    test_data = TensorDataset(x_test, y_test)
    test_dataLoader = DataLoader(test_data, batch_size=batch_size)
 
    #损失函数、优化器
    loss_func = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_model(epoch, train_dataLoader, test_dataLoader)
    p, y, test_loss = test_model(test_dataLoader)
 
    #绘制折线图
    pred = [ele * (GD.close_max - GD.close_min) + GD.close_min for ele in p]
    data = [ele * (GD.close_max - GD.close_min) + GD.close_min for ele in y]
    plot_img(data, pred)

    #输出模型损失
    print('模型损失：',test_loss)