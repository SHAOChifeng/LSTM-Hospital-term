from pandas import read_csv
import numpy as np
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
#
def getData(corpusFile,sequence_length,batchSize):
    # 数据预处理
    people_data = read_csv(corpusFile)
    people_data.drop('date_time', axis=1, inplace=True)  # 删除’date_time‘
    close_max = people_data['people'].max() #队伍人数的最大值
    close_min = people_data['people'].min() #队伍人数的最小值
    df = people_data.apply(lambda x: (x - min(x)) / (max(x) - min(x)))  # min-max标准化

    # 构造X和Y
    #根据前半小时的数据，预测未来半小时的队伍人数(people)
    sequence = sequence_length
    X = []
    Y = []
    for i in range(df.shape[0] - sequence):
        X.append(np.array(df.iloc[i:(i + sequence), ].values, dtype=np.float32))
        Y.append(np.array(df.iloc[(i + sequence), 0], dtype=np.float32))

    # 构建batch
    total_len = len(Y)
    # print(total_len)

    trainx, trainy = X[:int(0.9875 * total_len)], Y[:int(0.9875 * total_len)]
    testx, testy = X[int(0.9875 * total_len):], Y[int(0.9875 * total_len):]
    train_loader = DataLoader(dataset=Mydataset(trainx, trainy, transform=transforms.ToTensor()), batch_size=batchSize,
                              shuffle=True)
    test_loader = DataLoader(dataset=Mydataset(testx, testy), batch_size=batchSize, shuffle=False)
    return close_max,close_min,train_loader,test_loader



class Mydataset(Dataset):
    def __init__(self, xx, yy, transform=None):
        self.x = xx
        self.y = yy
        self.tranform = transform

    def __getitem__(self, index):
        x1 = self.x[index]
        y1 = self.y[index]
        if self.tranform != None:
            return self.tranform(x1), y1
        return x1, y1

    def __len__(self):
        return len(self.x)
