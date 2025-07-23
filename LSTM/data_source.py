import os
import pandas as pd

class GetData():
    def __init__(self, save_path):
        self.save_path = save_path
        self.data = None

    def getData(self):
        # 判断data.csv文件是否存在
        if os.path.exists(self.save_path):
            df = pd.read_csv(self.save_path)
            if not df.empty:  # 判断文件是否有数据
                self.data = df

        self.data = self.data[["open", "close", "high", "low", "volume"]]
        self.close_max = max(self.data.close)
        self.close_min = min(self.data.close)
        # Normalization
        self.data = self.data.apply(lambda x: (x - min(x)) / (max(x) - min(x)))
        return self.data

    def process_data(self, n, r):
        """
        将数据分为特征(feature)和标签(label),
        并划分为训练集和测试集
        @param n: 步长
        @param r: 训练比例
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

        total = len(feature)
        train_num = int(total * r)

        train_x = feature[:train_num]
        test_x = feature[train_num:]
        train_y = label[:train_num]
        test_y = label[train_num:]
        return train_x, test_x, train_y, test_y
    
    def getNewestData(self, n:int,  num: int):
        """
        获取最新的数据
        @param num: 最新数据的行数
        @param n: 步长
        """
        if self.data is None:
            self.getData()
        return [self.data.iloc[i: i + n].values.tolist()
                   for i in range(len(self.data) - num - n, len(self.data))
                   if i + n < len(self.data)]
    
if __name__ == '__main__':
    GD = GetData(r'LSTM\resources\ochlv.csv')
    # train_x, test_x, train_y, test_y = GD.process_data(5, 0.7)
    # print(len(train_x), len(test_x), len(train_y), len(test_y))
    features = GD.getNewestData(5, 20)
    print(len(features))
    print(features)
