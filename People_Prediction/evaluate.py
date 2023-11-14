from LSTMModel import lstm
from dataset import getData
from parser_my import args
import torch
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
def eval():
    # model = torch.load(args.save_file)
    model = lstm(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.layers , output_size=1)
    model.to(args.device)
    checkpoint = torch.load(args.save_file)
    model.load_state_dict(checkpoint['state_dict'])
    preds = []
    labels = []
    close_max, close_min, train_loader, test_loader = getData(args.corpusFile, args.sequence_length, args.batch_size)
    for idx, (x, label) in enumerate(test_loader):
        if args.useGPU:
            x = x.squeeze(1).cuda()  # batch_size,seq_len,input_size
        else:
            x = x.squeeze(1)
        pred = model(x)
        list = pred.data.squeeze(1).tolist()
        preds.extend(list[-1])
        labels.extend(label.tolist())
    a=[]
    b=[]
    for i in range(len(preds)):
        print('预测值是%.2f,真实值是%.2f' % (
        preds[i][0] * (close_max - close_min) + close_min, labels[i] * (close_max - close_min) + close_min))
        a.append(preds[i][0] * (close_max - close_min) + close_min)
        b.append(labels[i] * (close_max - close_min) + close_min)

    # 绘图
    plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

    L1,=plt.plot(a,marker = 'o',)
    L2,=plt.plot(b,marker = '*')
    plt.legend([L1, L2], ['Predict', 'Real'], loc='upper right')
    plt.xlabel('/5mins')
    plt.ylabel('People_Count')
    plt.savefig(args.save_evaluate, dpi=1200,bbox_inches='tight')
    plt.show()

eval()