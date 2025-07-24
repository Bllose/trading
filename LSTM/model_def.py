import torch                         # 导入PyTorch库
from torch import Tensor
import torch.nn as nn                # 导入神经网络模块
import torch.optim as optim          # 导入优化器模块
from tqdm import tqdm                # 导入tqdm库，用于显示进度条
from copy import deepcopy as copy    # 导入deepcopy函数，用于深拷贝对象

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

def train_model(epoch, train_dataLoader, test_dataLoader, 
                model, loss_func, optimizer, early_stop):
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
            return True, None
    return False, best_model

def test_model(test_dataLoader_, loss_func):
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

def test_model_trainning(test_dataLoader_):
    # 预测值列表
    pred = []
    # 真实值列表
    label = []
    model_ = Model(5)
    # 加载模型的状态字典
    model_.load_state_dict(torch.load("./lstm_.pth"))
    # 将模型设置为评估模式
    model_.eval()

    trend_right_counter = 0
    trend_wrong_counter = 0
    # 测试样本总数
    total_test_num = 0
    for x, y in test_dataLoader_:
        x_num = len(x)
        p = model_(x)
        total_test_num += x_num
        # 将预测值添加到列表中
        pred.extend(p.data.squeeze(1).tolist())
        # 将真实标签添加到列表中
        label.extend(y.tolist())
        if (y[-1] > y[-2]).item() == (p[-1] > p[-2]).item():
            trend_right_counter += 1
        else:
            trend_wrong_counter += 1
    return pred, label, trend_right_counter, trend_wrong_counter

def predict_model(newest_data: Tensor, n):
    """
    @param newest_data: shape([20, 5, 5]) 
    20 = batch_size + 
    """
    model_ = Model(n)
    # 加载模型的状态字典
    model_.load_state_dict(torch.load("./lstm_.pth"))
    # 将模型设置为评估模式
    model_.eval()
    p = model_(newest_data)
    return p
