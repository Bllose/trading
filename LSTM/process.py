import torch                         # 导入PyTorch库
import torch.nn as nn                # 导入神经网络模块
import torch.optim as optim          # 导入优化器模块
import matplotlib.pyplot as plt      # 导入matplotlib库，用于绘制图表
from torch.utils.data import DataLoader, TensorDataset  # 导入DataLoader和TensorDataset类，用于加载数据
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from data_source import GetData
from model_def import Model, train_model, test_model, predict_model, test_model_trainning

def plot_img(data, pred):
    """
    @param data: 真实值
    @param pred: 预测值
    """
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.figure(figsize=(12, 7))
    plt.plot(range(len(pred)), pred, color='green')
    plt.plot(range(len(data)), data, color='blue')
    for i in range(0, len(pred)-3, 5):
        price = [data[i]+pred[j]-pred[i] for j in range(i, i+3)]
        plt.plot(range(i, i+3), price, color='red')
    plt.xticks(fontproperties = 'Times New Roman', size = 15)
    plt.yticks(fontproperties = 'Times New Roman', size = 15)
    plt.xlabel('DATE', fontsize=18)
    plt.ylabel('CLOSE', fontsize=18)
    plt.show()

def train_task():
    #超参数
    days_num = 5
    epoch = 20
    fea = 5
    batch_size = 20
    early_stop = 5
 
    #初始化模型
    model = Model(fea)
 
    #数据处理
    GD = GetData(save_path=r'LSTM\resources\ochlv.csv')
    x_train, x_test, y_train, y_test = GD.process_data(days_num, 0.7)
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
    saved, best_model = train_model(epoch, train_dataLoader, test_dataLoader, model, loss_func, optimizer, early_stop)
    if not saved:
        torch.save(best_model.state_dict(), './lstm_.pth')
    p, y, test_loss = test_model(test_dataLoader, loss_func)
 
    #绘制折线图
    pred = [ele * (GD.close_max - GD.close_min) + GD.close_min for ele in p]
    data = [ele * (GD.close_max - GD.close_min) + GD.close_min for ele in y]
    plot_img(data, pred)

    #输出模型损失
    print('模型损失：',test_loss)

def test_predict_task():
    #超参数
    days_num = 5
    # epoch = 20
    # fea = 5
    batch_size = 20
    # early_stop = 5
 
    # #初始化模型
    # model = Model(fea)
 
    #数据处理
    GD = GetData(save_path=r'LSTM\resources\ochlv.csv')
    x_train, x_test, y_train, y_test = GD.process_data(days_num, 0.7)
    # x_train = torch.tensor(x_train).float()
    x_test = torch.tensor(x_test).float()
    # y_train = torch.tensor(y_train).float()
    y_test = torch.tensor(y_test).float()
    test_data = TensorDataset(x_test, y_test)
    test_dataLoader = DataLoader(test_data, batch_size=batch_size)
 
    p, y = test_model_trainning(test_dataLoader)
 
    #绘制折线图
    pred = [ele * (GD.close_max - GD.close_min) + GD.close_min for ele in p]
    data = [ele * (GD.close_max - GD.close_min) + GD.close_min for ele in y]
    plot_img(data, pred)

    #输出模型损失
    print('模型损失：')

def predict_task():
    #超参数
    days_num = 5
    batch_size = 20

     #数据处理
    GD = GetData(save_path=r'LSTM\resources\ochlv.csv')

    features = GD.getNewestData(days_num, batch_size)
    x_features = torch.tensor(features).float()

    p = predict_model(x_features, days_num)
    # pred = [ele * (GD.close_max - GD.close_min) + GD.close_min for ele in p]

    return p[-1] * (GD.close_max - GD.close_min) + GD.close_min, p[-1]>p[-2]

if __name__ == '__main__':
    
    # train_task()
    test_predict_task()

    # print(predict_task())
    print('DONE')
